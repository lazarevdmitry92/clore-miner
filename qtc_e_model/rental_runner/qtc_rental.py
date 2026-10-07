#!/usr/bin/env python3
"""Atlas-only, fail-closed orchestration for the bounded QTC E compute rental.

This module never imports a platform client.  Paid actions are delegated to
``action_bridge.py``, whose implementation calls only the project's public
``ops.market_actions.place``, ``ops.drop`` and ``ops.pilots`` entry points.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request


PILOT = "qtc_own_miner"
BY = "qtc-e-rental"
EXPECTED_MODEL = "rtx 4090"
DEFAULT_SERVER = "69333"
DEFAULT_OWNER = "4727"
OWNER_DENY = "570"
LOG_PORT = 21559
MAX_MARGIN_USD_DAY = 1.0
MAX_COST_USD = 1.0
MAX_PRICE_USD_HOUR = 2.0
RUN_DEADLINE_S = 30 * 60
PILOT_UNTIL = "35m"
IMAGE_RE = re.compile(
    r"^(?P<repo>ghcr\.io/lazarevdmitry92/clore-miner):"
    r"(?P<tag>qtcbench-cuda-(?P<commit>[0-9a-f]{40}))@"
    r"(?P<digest>sha256:[0-9a-f]{64})$"
)
RATE_KEYS = {
    "USD-Blockchain": "usd-blockchain",
    "bitcoin": "bitcoin",
    "CLORE-Blockchain": "clore-ai",
}
RUNNER_FILES = ("qtc_rental.py", "action_bridge.py", "watchdog.py")
RUNNER_INVENTORY = "RUNNER_SHA256SUMS"


class Refusal(RuntimeError):
    pass


class ActionFailure(Refusal):
    def __init__(self, message: str, code: str | None = None, possibly_accepted: bool = False,
                 outcome: str | None = None, operation_id: str | None = None, request_sent: bool | None = None):
        super().__init__(message)
        self.code = code
        self.possibly_accepted = possibly_accepted
        self.outcome = outcome
        self.operation_id = operation_id
        self.request_sent = request_sent


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@contextlib.contextmanager
def db_session(path: str | Path):
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def image_parts(image: str) -> dict[str, str]:
    match = IMAGE_RE.fullmatch(image)
    if not match:
        raise Refusal(
            "image must be the immutable project image "
            "ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<40 hex>@sha256:<64 hex>"
        )
    return match.groupdict()


def verify_runner_package(root: Path, expected_inventory_sha256: str) -> dict:
    """Bind the atlas scripts to the independently reviewed frozen inventory."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_inventory_sha256):
        raise Refusal("runner inventory SHA-256 must be 64 lowercase hex characters")
    inventory_path = root / RUNNER_INVENTORY
    body = inventory_path.read_bytes()
    actual_inventory = hashlib.sha256(body).hexdigest()
    if actual_inventory != expected_inventory_sha256:
        raise Refusal(f"runner inventory digest mismatch: {actual_inventory} != {expected_inventory_sha256}")
    entries = {}
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refusal("runner inventory is not UTF-8") from exc
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            raise Refusal(f"malformed runner inventory line: {line!r}")
        digest, name = match.groups()
        if name not in RUNNER_FILES or name in entries:
            raise Refusal(f"unexpected or duplicate runner file: {name!r}")
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise Refusal(f"runner file does not match reviewed inventory: {name}")
        entries[name] = digest
    if set(entries) != set(RUNNER_FILES):
        raise Refusal("runner inventory does not cover the exact executable package")
    return {"inventorySha256": actual_inventory, "files": entries}


def placement_image(image: str) -> str:
    """Clore accepts a tag, not Docker's ``tag@digest`` reference."""
    parts = image_parts(image)
    return f"{parts['repo']}:{parts['tag']}"


def verify_remote_digest(image: str, crane: str = "crane") -> str:
    """Resolve the tag remotely and require it to equal the supplied digest."""
    parts = image_parts(image)
    tag_ref = f"{parts['repo']}:{parts['tag']}"
    try:
        got = subprocess.run(
            [crane, "digest", tag_ref], check=True, capture_output=True, text=True, timeout=90
        ).stdout.strip().splitlines()[-1]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError) as exc:
        raise Refusal(f"cannot inspect immutable image tag {tag_ref}: {exc}") from exc
    if got != parts["digest"]:
        raise Refusal(f"image digest mismatch: tag resolves to {got!r}, requested {parts['digest']!r}")
    return got


def _one_offer(conn: sqlite3.Connection, server_id: str) -> tuple[dict, str]:
    row = conn.execute(
        "SELECT data, updated_at FROM current_items WHERE source='offers-clore' AND item_key=?", (server_id,)
    ).fetchone()
    if row is None:
        raise Refusal(f"candidate {server_id} is absent from collector current_items")
    try:
        return json.loads(row["data"]), row["updated_at"]
    except (TypeError, json.JSONDecodeError) as exc:
        raise Refusal(f"candidate {server_id} has malformed collector JSON") from exc


def _model(offer: dict) -> str:
    return str(offer.get("gpu_key") or offer.get("gpu_model") or "").strip().lower()


def _open_order_rows(conn: sqlite3.Connection, server_id: str) -> list[sqlite3.Row]:
    if not table_exists(conn, "order_versions"):
        return []
    return conn.execute(
        "SELECT * FROM order_versions WHERE platform='clore' AND server_id=? AND valid_to IS NULL "
        "AND status IN ('active','paused') AND NOT coalesce(julianday(expires_at) <= julianday('now'), 0)",
        (server_id,),
    ).fetchall()


def _blocked(conn: sqlite3.Connection, server_id: str, owner: str, model: str) -> sqlite3.Row | None:
    if not table_exists(conn, "bans"):
        return None
    keys = (("server", server_id), ("host", owner), ("host_gpu", f"{owner}:{model}"), ("gpu", model))
    for scope, key in keys:
        row = conn.execute(
            "SELECT * FROM bans WHERE (platform='clore' OR scope='gpu') AND scope=? AND key=? "
            "AND state='banned' AND until > ?",
            (scope, key, iso_now()),
        ).fetchone()
        if row is not None:
            return row
    return None


def _rate(conn: sqlite3.Connection, currency: str) -> float:
    key = RATE_KEYS.get(currency)
    if not key:
        raise Refusal(f"unsupported Clore currency {currency!r}")
    row = conn.execute("SELECT usd FROM rates WHERE currency=? AND valid_to IS NULL", (key,)).fetchone()
    if row is None or row[0] is None or float(row[0]) <= 0:
        raise Refusal(f"no current positive USD rate for {currency} ({key})")
    return float(row[0])


def _price_terms(conn: sqlite3.Connection, offer: dict, minutes: int) -> dict:
    try:
        prices = json.loads(offer["price_json"])["on_demand"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise Refusal("candidate has no valid on-demand price_json") from exc
    choices = []
    for currency, amount in (prices or {}).items():
        if amount is None or float(amount) <= 0:
            continue
        rate = _rate(conn, currency)
        fee = conn.execute(
            "SELECT fee_pct, creation_fee FROM platform_fees WHERE platform='clore' AND order_type='on-demand' "
            "AND currency=? AND valid_to IS NULL ORDER BY valid_from DESC LIMIT 1",
            (currency,),
        ).fetchone()
        if fee is None:
            raise Refusal(f"no current Clore on-demand fee for {currency}")
        raw_daily = float(amount) * rate
        multiplier = 1.0 + float(fee["fee_pct"]) / 100.0
        daily = raw_daily * multiplier
        creation = float(fee["creation_fee"] or 0.0) * rate
        cost = creation + daily * minutes / (24.0 * 60.0)
        choices.append((
            0 if currency == "USD-Blockchain" else 1,
            cost, currency, float(amount), rate, multiplier, raw_daily, daily, creation,
        ))
    if not choices:
        raise Refusal("candidate accepts no supported positive on-demand currency")
    _, cost, currency, price, rate, multiplier, raw_daily, daily, creation = min(choices)
    if daily / 24.0 > MAX_PRICE_USD_HOUR:
        raise Refusal(f"price ${daily / 24.0:.4f}/h exceeds ${MAX_PRICE_USD_HOUR:.2f}/h")
    if cost > MAX_COST_USD:
        raise Refusal(f"projected {minutes}m cost ${cost:.4f} exceeds ${MAX_COST_USD:.2f}")
    return {
        "currency": currency,
        "price": price,
        "rateUsd": rate,
        "feeMultiplier": multiplier,
        "priceUsdDayRaw": raw_daily,
        "priceUsdDayWithFee": daily,
        "creationFeeUsd": creation,
        "projectedCostUsd": cost,
    }


def _valuation_meta_gate(conn: sqlite3.Connection, terms: dict) -> dict:
    if not table_exists(conn, "server_eval_meta"):
        raise Refusal("server_eval_meta is absent; cannot prove a current coherent margin")
    rows = conn.execute("SELECT * FROM server_eval_meta WHERE platform='clore'").fetchall()
    if len(rows) != 1:
        raise Refusal(f"expected one clore server_eval_meta row, found {len(rows)}")
    meta = rows[0]
    try:
        built = dt.datetime.fromisoformat(meta["built_at"]).astimezone(dt.timezone.utc)
        snapshot = dt.datetime.fromisoformat(meta["snapshot_at"]).astimezone(dt.timezone.utc)
        fetched = dt.datetime.fromisoformat(meta["fetched_at"]).astimezone(dt.timezone.utc)
    except (TypeError, ValueError) as exc:
        raise Refusal("server_eval_meta timestamps are invalid") from exc
    age = (utc_now() - built).total_seconds()
    if age < -5 or age > 120:
        raise Refusal(f"server_eval margin build is stale ({age:.1f}s old)")
    if snapshot > built + dt.timedelta(seconds=5) or fetched > built + dt.timedelta(seconds=5):
        raise Refusal("server_eval margin build predates its offer snapshot")
    try:
        rates = json.loads(meta["rates"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise Refusal("server_eval_meta rates are invalid") from exc
    rate_key = RATE_KEYS[terms["currency"]]
    if not isinstance(rates, dict) or rate_key not in rates or not math.isclose(
        float(rates[rate_key]), terms["rateUsd"], rel_tol=0, abs_tol=1e-12
    ):
        raise Refusal("server_eval margin used a different exchange rate")
    if meta["od_fee"] is None or not math.isclose(
        float(meta["od_fee"]), terms["feeMultiplier"], rel_tol=0, abs_tol=1e-12
    ):
        raise Refusal("server_eval margin used a different on-demand fee")
    if not meta["version"]:
        raise Refusal("server_eval margin build has no input version")
    return {"builtAt": meta["built_at"], "version": meta["version"]}


def candidate_plan(
    conn: sqlite3.Connection,
    server_id: str,
    expected_owner: str,
    expected_model: str = EXPECTED_MODEL,
    minutes: int = 35,
) -> dict:
    """Re-read collector-owned state and return a placement plan or fail closed."""
    server_id, expected_owner, expected_model = str(server_id), str(expected_owner), expected_model.lower()
    offer, updated_at = _one_offer(conn, server_id)
    owner = str(offer.get("owner"))
    model = _model(offer)
    if owner == OWNER_DENY:
        raise Refusal(f"candidate owner {OWNER_DENY} is explicitly denied")
    if owner != expected_owner:
        raise Refusal(f"owner changed: expected {expected_owner}, collector has {owner}")
    if model != expected_model:
        raise Refusal(f"model changed: expected {expected_model}, collector has {model or '?'}")
    if int(offer.get("gpu_count") or 0) != 1:
        raise Refusal(f"candidate must have exactly one GPU, collector has {offer.get('gpu_count')!r}")
    cuda_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(offer.get("cuda_version") or ""))
    if not cuda_match or tuple(map(int, cuda_match.groups())) < (12, 6):
        raise Refusal(f"candidate must advertise host CUDA compatibility >=12.6, got {offer.get('cuda_version')!r}")
    if bool(offer.get("rented")) or bool(offer.get("spot_rented")):
        raise Refusal("candidate is not free in collector current_items")
    if _open_order_rows(conn, server_id):
        raise Refusal("candidate already has our current order")
    if table_exists(conn, "pilot_servers") and conn.execute(
        "SELECT 1 FROM pilot_servers WHERE platform='clore' AND server_id=? AND valid_to IS NULL", (server_id,)
    ).fetchone():
        raise Refusal("candidate already has an open pilot row")
    ban = _blocked(conn, server_id, owner, model)
    if ban is not None:
        raise Refusal(f"candidate is banned by {ban['scope']}:{ban['key']}")

    eval_rows = conn.execute(
        "SELECT * FROM server_eval WHERE platform='clore' AND server_id=?", (server_id,)
    ).fetchall()
    if len(eval_rows) != 1:
        raise Refusal(f"candidate needs one current server_eval row, found {len(eval_rows)}")
    ev = eval_rows[0]
    if str(ev["owner"]) != owner or int(ev["n"] or 0) != 1 or int(ev["offer_gpus"] or 0) != 1:
        raise Refusal("owner/card count is not stable between offer and server_eval")
    label = f"{ev['gpu_raw'] or ''} {ev['label'] or ''}".lower()
    if "4090" not in label or int(ev["rented"] or 0):
        raise Refusal("model/free state is not stable between offer and server_eval")
    if ev["od_margin"] is None:
        raise Refusal("current on-demand fleet margin is unknown; cannot prove it is <= $1/day")
    terms = _price_terms(conn, offer, minutes)
    valuation_meta = _valuation_meta_gate(conn, terms)
    if str(ev["od_currency"]) != terms["currency"]:
        raise Refusal("server_eval currency is stale relative to the current offer")
    if ev["od_price"] is None or not math.isclose(float(ev["od_price"]), terms["price"], rel_tol=0, abs_tol=1e-9):
        raise Refusal("server_eval price is stale relative to the current offer")
    if ev["od_usd"] is None or not math.isclose(
        float(ev["od_usd"]), terms["priceUsdDayRaw"], rel_tol=0, abs_tol=0.00011
    ):
        raise Refusal("server_eval USD price is stale relative to the current rate/offer")
    margins = [float(v) for v in (ev["od_margin"], ev["sp_margin"]) if v is not None]
    if margins and max(margins) > MAX_MARGIN_USD_DAY:
        raise Refusal(f"candidate is a profitable fleet target: max margin ${max(margins):.4f}/day > $1/day")
    if ev["outcome"] in ("ours", "banned"):
        raise Refusal(f"server_eval outcome is {ev['outcome']}")

    stable = {
        "serverId": server_id,
        "owner": owner,
        "model": model,
        "gpuRaw": offer.get("gpu_raw"),
        "cudaVersion": offer.get("cuda_version"),
        "priceJson": offer.get("price_json"),
        "evalCurrency": ev["od_currency"],
        "evalPrice": ev["od_price"],
        "evalUsd": ev["od_usd"],
        "evalOdMargin": ev["od_margin"],
        "evalSpMargin": ev["sp_margin"],
        "evalBuiltAt": valuation_meta["builtAt"],
        "evalVersion": valuation_meta["version"],
    }
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return {
        **stable,
        **terms,
        "valuation": valuation_meta,
        "offerUpdatedAt": updated_at,
        "maxMarginUsdDay": max(margins) if margins else None,
        "pilot": PILOT,
        "until": PILOT_UNTIL,
        "exempt": "all",
        "deadlineSeconds": RUN_DEADLINE_S,
        "fingerprint": fingerprint,
    }


def current_order_snapshot(conn: sqlite3.Connection, order_id: str, server_id: str) -> dict | None:
    """The order object only from collector's current_items; never asks Clore."""
    rows = conn.execute("SELECT data FROM current_items WHERE source='orders-clore'").fetchall()
    for row in rows:
        try:
            obj = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            continue
        if str(obj.get("id")) == str(order_id) and str(obj.get("si")) == str(server_id) and not obj.get("expired"):
            return obj
    return None


def discover_pilot_order(
    conn: sqlite3.Connection,
    server_id: str,
    pilot: str,
    armed_at: str | None = None,
    image: str | None = None,
) -> str | None:
    """Recover the exact order id after Clore accepted it but place lost the id."""
    if not all(
        table_exists(conn, name) for name in ("order_versions", "order_terms")
    ):
        return None
    row = conn.execute(
        "SELECT o.order_id FROM order_versions o JOIN order_terms t ON t.platform=o.platform "
        "AND t.server_id=o.server_id AND t.order_id=o.order_id "
        "WHERE o.platform='clore' AND o.server_id=? AND o.valid_to IS NULL "
        "AND o.status IN ('active','paused') AND t.pilot=? ORDER BY o.valid_from DESC LIMIT 1",
        (server_id, pilot),
    ).fetchone()
    if row is not None:
        return str(row[0])
    if not armed_at:
        return None
    row = conn.execute(
        "SELECT o.order_id FROM order_versions o WHERE o.platform='clore' AND o.server_id=? "
        "AND o.valid_to IS NULL AND o.status IN ('active','paused') "
        "AND EXISTS (SELECT 1 FROM order_terms t WHERE t.platform='clore' AND t.server_id=o.server_id "
        "AND t.pilot=? AND (? IS NULL OR t.image=?) "
        "AND julianday(t.placed_at) >= julianday(?) - 5.0/86400.0 "
        "AND abs(julianday(t.placed_at)-julianday(o.created_at))*86400 <= 120) "
        "ORDER BY o.valid_from DESC LIMIT 1",
        (server_id, pilot, image, image, armed_at),
    ).fetchone()
    return None if row is None else str(row[0])


def endpoint_from_order(order: dict, internal_port: int = LOG_PORT) -> str:
    clusters = order.get("pub_cluster") or []
    if len(clusters) != 1 or not isinstance(clusters[0], str) or not clusters[0]:
        raise Refusal(f"collector order has no unique pub_cluster: {clusters!r}")
    outside = []
    for mapping in order.get("tcp_ports") or []:
        try:
            inside, public = str(mapping).split(":", 1)
        except ValueError:
            continue
        if inside == str(internal_port) and public.isdigit():
            outside.append(int(public))
    if len(outside) != 1:
        raise Refusal(f"collector order has {len(outside)} mappings for internal port {internal_port}")
    return f"http://{clusters[0]}:{outside[0]}"


class ActionClient:
    def __init__(self, bridge: Path, project_root: Path, python: str = sys.executable):
        self.bridge, self.project_root, self.python = bridge, project_root, python

    def call(self, *args: str, timeout: int = 150) -> dict:
        cmd = [self.python, str(self.bridge), "--project-root", str(self.project_root), *map(str, args)]
        done = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        if done.returncode:
            lines = [line for line in done.stdout.splitlines() if line.strip()]
            payload = None
            if lines:
                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(lines[-1])
            if isinstance(payload, dict) and payload.get("ok") is False:
                raise ActionFailure(
                    str(payload.get("error") or "public action refused"),
                    code=payload.get("code"),
                    possibly_accepted=bool(payload.get("possiblyAccepted")),
                    outcome=payload.get("outcome"),
                    operation_id=payload.get("operationId"),
                    request_sent=payload.get("requestSent"),
                )
            raise ActionFailure(
                f"action failed ({done.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n{done.stderr.strip()}",
                possibly_accepted=True,
            )
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        if not lines:
            raise Refusal("action returned no JSON")
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise Refusal(f"action returned malformed JSON: {lines[-1]!r}") from exc

    def place(self, plan: dict, image: str) -> dict:
        return self.call(
            "place", "--server", plan["serverId"], "--currency", plan["currency"],
            "--price", str(plan["price"]), "--image", image,
        )

    def drop(self, server_id: str, order_id: str) -> dict:
        return self.call("drop", "--server", server_id, "--order", order_id)

    def end(self, server_id: str) -> dict:
        return self.call("end", "--server", server_id)


def fetch(url: str, timeout: int = 10, limit: int = 64 * 1024 * 1024) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if len(data) > limit:
        raise Refusal(f"artifact exceeds {limit} bytes: {url}")
    return data


ARTIFACT_ROOT_FILES = {
    "package.sha256",
    "qtc_sim2_batch",
    "qtc_sim2_batch.sha256",
    "parity-reps3.bin",
    "parity-reps6.bin",
    "parity.json",
    "result.json",
}
PARITY_HEADER = struct.Struct("<8s7I")
PARITY_TOLERANCE = 1e-12
MAX_ARTIFACT_FILES = 32
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


def parity_batch_jobs(body: bytes, expected_reps: int) -> int:
    if len(body) < PARITY_HEADER.size:
        raise Refusal("parity batch is truncated before its header")
    magic, version, count, length, seeds, nwarps, reps, fields = PARITY_HEADER.unpack_from(body)
    if magic != b"QTCE4GPU" or version != 1 or fields != 5 or reps != expected_reps:
        raise Refusal(f"parity reps={expected_reps} batch header is invalid")
    if not (1 <= count <= 1_000_000 and 1 <= length <= 65_535 and 1 <= seeds <= 4_096 and 1 <= nwarps <= 8):
        raise Refusal(f"parity reps={expected_reps} batch dimensions are invalid")
    jobs = count * seeds
    expected_size = PARITY_HEADER.size + count * length * fields + seeds * nwarps * 2 + jobs * 8
    if jobs > 10_000_000 or len(body) != expected_size:
        raise Refusal(f"parity reps={expected_reps} batch size is invalid")
    values = struct.unpack_from(f"<{jobs}d", body, expected_size - jobs * 8)
    if any(not math.isfinite(value) for value in values):
        raise Refusal(f"parity reps={expected_reps} authoritative scores are non-finite")
    return jobs


def parse_artifact_inventory(body: bytes, *, require_complete: bool = True) -> dict[str, str]:
    """Parse the frozen sha256sum inventory with a strict path allowlist."""
    entries: dict[str, str] = {}
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refusal("ARTIFACTS.sha256 is not UTF-8") from exc
    if len(lines) > MAX_ARTIFACT_FILES:
        raise Refusal(f"artifact inventory has more than {MAX_ARTIFACT_FILES} entries")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (/var/log/qtc-e/([A-Za-z0-9._/-]+))", line)
        if not match:
            raise Refusal(f"malformed artifact inventory line: {line!r}")
        digest, absolute, relative = match.groups()
        path = PurePosixPath(relative)
        if ".." in path.parts or str(path) != relative or relative.startswith("/"):
            raise Refusal(f"unsafe artifact path: {absolute!r}")
        allowed = relative in ARTIFACT_ROOT_FILES or relative.startswith("run/")
        if not allowed or relative in entries:
            raise Refusal(f"unexpected or duplicate artifact path: {absolute!r}")
        entries[relative] = digest
    if not entries:
        raise Refusal("artifact inventory is empty")
    if require_complete:
        required = ARTIFACT_ROOT_FILES | {"run/manifest.json"}
        missing = sorted(required - entries.keys())
        if missing:
            raise Refusal(f"artifact inventory is missing required files: {missing}")
    return entries


def collect_verified_inventory(
    endpoint: str,
    inventory_body: bytes,
    local_root: Path,
    *,
    require_complete: bool = True,
    require_success: bool = True,
    fetch_one=None,
) -> list[dict]:
    entries = parse_artifact_inventory(inventory_body, require_complete=require_complete)
    getter = fetch_one or (lambda url, limit: fetch(url, limit=limit))
    collected = []
    total_bytes = 0
    for relative, expected in sorted(entries.items()):
        remaining = MAX_ARTIFACT_BYTES - total_bytes
        if remaining <= 0:
            raise Refusal(f"artifact set exceeds {MAX_ARTIFACT_BYTES} bytes")
        body = getter(endpoint + "/" + relative, remaining)
        if body is None:
            raise Refusal(f"inventory artifact is missing: /{relative}")
        total_bytes += len(body)
        if total_bytes > MAX_ARTIFACT_BYTES:
            raise Refusal(f"artifact set exceeds {MAX_ARTIFACT_BYTES} bytes")
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise Refusal(f"inventory hash mismatch for /{relative}: {actual} != {expected}")
        target = local_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        collected.append({"path": relative, "bytes": len(body), "sha256": actual})

    if not require_success:
        return collected

    def load_json(relative: str) -> dict:
        try:
            value = json.loads((local_root / relative).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Refusal(f"invalid JSON artifact: {relative}") from exc
        if not isinstance(value, dict):
            raise Refusal(f"JSON artifact is not an object: {relative}")
        return value

    binary_hash = entries["qtc_sim2_batch"]
    package_text = (local_root / "package.sha256").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", package_text):
        raise Refusal("package.sha256 does not contain exactly one SHA-256 digest")
    binary_sidecar = (local_root / "qtc_sim2_batch.sha256").read_text(encoding="utf-8").split()
    if not binary_sidecar or binary_sidecar[0] != binary_hash:
        raise Refusal("CUDA binary sidecar does not match the inventoried binary")
    parity = load_json("parity.json")
    if parity.get("schema") != "qtc-e-cuda-parity-v2" or parity.get("gpu_binary_sha256") != binary_hash:
        raise Refusal("parity artifact does not bind the inventoried CUDA binary")
    model_hash = parity.get("authoritative_model_sha256")
    if not isinstance(model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise Refusal("parity artifact has no valid authoritative model binding")
    if type(parity.get("tolerance")) not in (int, float) or parity["tolerance"] != PARITY_TOLERANCE:
        raise Refusal("parity artifact has an invalid tolerance")
    runs = parity.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise Refusal("parity artifact must contain two runs")
    parity_batches = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("reps") not in (3, 6):
            raise Refusal("parity artifact has an invalid repetition mode")
        reps = int(run["reps"])
        expected = entries[f"parity-reps{reps}.bin"]
        if run.get("authoritative_batch_sha256") != expected:
            raise Refusal(f"parity reps={reps} does not bind its inventoried batch")
        maximum = run.get("max_abs_difference")
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or not 0 <= maximum <= PARITY_TOLERANCE:
            raise Refusal(f"parity reps={reps} did not pass the frozen tolerance")
        batch_jobs = parity_batch_jobs((local_root / f"parity-reps{reps}.bin").read_bytes(), reps)
        if type(run.get("jobs")) is not int or run["jobs"] != batch_jobs:
            raise Refusal(f"parity reps={reps} job count does not match its batch")
        parity_batches[str(reps)] = expected
    if set(parity_batches) != {"3", "6"}:
        raise Refusal("parity artifact does not cover reps=3 and reps=6")

    manifest = load_json("run/manifest.json")
    if manifest.get("schema") != "qtc-e-rental-run-v1" or manifest.get("status") != "complete":
        raise Refusal("run manifest is not a complete qtc-e-rental-run-v1 artifact")
    if manifest.get("gpu_binary_sha256") != binary_hash:
        raise Refusal("run manifest does not bind the inventoried CUDA binary")
    if manifest.get("parity_artifact_sha256") != entries["parity.json"]:
        raise Refusal("run manifest does not bind the inventoried parity artifact")
    if manifest.get("authoritative_model_sha256") != model_hash:
        raise Refusal("run manifest authoritative model disagrees with parity.json")
    if manifest.get("authoritative_batches_sha256") != parity_batches:
        raise Refusal("run manifest parity batch bindings disagree with parity.json")
    batches = manifest.get("batches")
    if not isinstance(batches, list) or not batches:
        raise Refusal("run manifest contains no scored batches")
    for item in batches:
        if not isinstance(item, dict):
            raise Refusal("run manifest contains a malformed batch entry")
        for name_key, hash_key in (("batch", "batch_sha256"), ("results", "results_sha256")):
            name = item.get(name_key)
            if not isinstance(name, str) or PurePosixPath(name).name != name:
                raise Refusal("run manifest contains an unsafe batch filename")
            relative = "run/" + name
            if entries.get(relative) != item.get(hash_key):
                raise Refusal(f"run manifest hash mismatch for {relative}")
    result = load_json("result.json")
    if result.get("archive_manifest_sha256") != entries["run/manifest.json"]:
        raise Refusal("result does not bind the inventoried run manifest")
    if result.get("scorer") != "cuda":
        raise Refusal("result is not from the CUDA scorer")
    return collected


def collect_failure_evidence(endpoint: str, out: Path) -> list[dict]:
    """Best-effort bounded collection before teardown; never turns failure into success."""
    root = out / "failure-evidence"
    root.mkdir(parents=True, exist_ok=True)
    collected, errors = [], []
    total_bytes = 0
    deadline = time.monotonic() + 45

    def bounded_fetch(relative: str, requested_limit: int = 64 * 1024 * 1024) -> bytes | None:
        nonlocal total_bytes
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("45-second failure evidence budget exhausted")
        remaining_bytes = MAX_ARTIFACT_BYTES - total_bytes
        if remaining_bytes <= 0:
            raise Refusal(f"failure evidence exceeds {MAX_ARTIFACT_BYTES} bytes")
        body = fetch(
            endpoint + "/" + relative,
            timeout=max(1, min(5, int(remaining + 0.999))),
            limit=min(requested_limit, remaining_bytes),
        )
        if body is not None:
            total_bytes += len(body)
        return body
    known = (
        "failed",
        "run.log",
        "compute-timeout.log",
        "package.sha256",
        "qtc_sim2_batch.sha256",
        "parity.json",
        "result.json",
        "run/manifest.json",
        "ARTIFACTS.sha256",
    )
    bodies = {}
    for relative in known:
        try:
            body = bounded_fetch(relative)
        except Exception as exc:
            errors.append({"path": relative, "error": str(exc)})
            continue
        if body is None:
            continue
        bodies[relative] = body
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        collected.append({"path": relative, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})

    inventory = bodies.get("ARTIFACTS.sha256")
    if inventory is not None:
        try:
            collected.extend(collect_verified_inventory(
                endpoint,
                inventory,
                root / "inventoried",
                require_complete=False,
                require_success=False,
                fetch_one=lambda url, limit: bounded_fetch(url.removeprefix(endpoint + "/"), limit),
            ))
        except Exception as exc:
            errors.append({"path": "ARTIFACTS.sha256", "error": str(exc)})

    manifest_body = bodies.get("run/manifest.json")
    if manifest_body is not None:
        try:
            manifest = json.loads(manifest_body)
            batches = manifest.get("batches") if isinstance(manifest, dict) else None
            if not isinstance(batches, list):
                raise ValueError("manifest batches are absent")
            for item in batches:
                for name_key, hash_key in (("batch", "batch_sha256"), ("results", "results_sha256")):
                    name = item.get(name_key) if isinstance(item, dict) else None
                    if not isinstance(name, str) or PurePosixPath(name).name != name:
                        raise ValueError("unsafe manifest artifact name")
                    relative = "run/" + name
                    body = bounded_fetch(relative)
                    if body is None:
                        raise ValueError(f"missing {relative}")
                    actual = hashlib.sha256(body).hexdigest()
                    if actual != item.get(hash_key):
                        raise ValueError(f"hash mismatch for {relative}")
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(body)
                    collected.append({"path": relative, "bytes": len(body), "sha256": actual})
        except Exception as exc:
            errors.append({"path": "run/manifest.json", "error": str(exc)})
    atomic_json(root / "collection.json", {"at": iso_now(), "files": collected, "errors": errors})
    return collected


def require_watchdog_alive(watchdog: subprocess.Popen) -> None:
    code = watchdog.poll()
    if code is not None:
        raise Refusal(f"independent watchdog exited unexpectedly with status {code}")


def wait_for_order(
    db: Path,
    order_id: str,
    server_id: str,
    timeout_s: int = 150,
    watchdog: subprocess.Popen | None = None,
) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if watchdog is not None:
            require_watchdog_alive(watchdog)
        with db_session(db) as conn:
            order = current_order_snapshot(conn, order_id, server_id)
        if order is not None:
            return order
        time.sleep(3)
    raise Refusal("collector did not expose the placed order in current_items before timeout")


def verify_cleanup(db: Path, server_id: str, order_id: str, timeout_s: int = 180) -> None:
    end = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < end:
        with db_session(db) as conn:
            pilot_open = bool(conn.execute(
                "SELECT 1 FROM pilot_servers WHERE platform='clore' AND server_id=? AND valid_to IS NULL", (server_id,)
            ).fetchone())
            billable = bool(_open_order_rows(conn, server_id))
            snapshot = current_order_snapshot(conn, order_id, server_id)
        last = {"pilotOpen": pilot_open, "billableOrder": billable, "orderSnapshot": snapshot is not None}
        if not any(last.values()):
            return
        time.sleep(3)
    raise Refusal(f"cleanup did not converge in collector DB: {last}")


def update_state(path: Path, **changes) -> dict:
    state = read_json(path)
    state.update(changes)
    state["updatedAt"] = iso_now()
    atomic_json(path, state)
    return state


def stop_watchdog(watchdog: subprocess.Popen, cleanup_verified: bool) -> bool:
    """Reap the guard only after cleanup convergence; otherwise detach it."""
    if not cleanup_verified:
        return False
    try:
        watchdog.terminate()
        watchdog.wait(timeout=10)
    except Exception:
        with contextlib.suppress(Exception):
            watchdog.kill()
    return True


def cleanup(actions: ActionClient, db: Path, state_path: Path, strict_verify: bool = True) -> list[dict]:
    state = read_json(state_path)
    results = []
    server_id, order_id = state["serverId"], state.get("orderId")
    recovered_order = False
    uncertain = state.get("placementOutcome") in ("pending", "uncertain", "possibly_accepted")
    needs_discovery = uncertain or state.get("phase") in ("placed", "running", "complete", "guarding")
    if not order_id and needs_discovery:
        # market_actions may report code=lost after Clore accepted a paid order
        # but before its id appeared.  Keep both the pilot row and watchdog
        # alive while collector converges, then drop that exact recovered id.
        deadline = time.monotonic() + float(state.get("discoveryTimeoutSeconds", 120))
        while time.monotonic() < deadline and not order_id:
            with db_session(db) as conn:
                order_id = discover_pilot_order(
                    conn,
                    server_id,
                    state["pilot"],
                    state.get("armedAt"),
                    state.get("image"),
                )
            if not order_id:
                time.sleep(3)
        if order_id:
            recovered_order = True
            update_state(state_path, orderId=order_id, recoveredOrderId=True)
        elif uncertain:
            update_state(state_path, phase="guarding", cleanupPending=True)
            raise Refusal("placement outcome is uncertain and no exact order id is visible; watchdog remains armed")
    update_state(state_path, phase="cleanup")
    if order_id:
        try:
            results.append({"drop": actions.drop(server_id, order_id)})
        except Exception as exc:
            results.append({"dropError": str(exc)})
    try:
        results.append({"end": actions.end(server_id)})
    except Exception as exc:
        results.append({"endError": str(exc)})
    if order_id and (strict_verify or recovered_order):
        verify_cleanup(db, server_id, order_id)
    update_state(state_path, phase="cleaned", cleanup=results)
    return results


def run(args) -> dict:
    project_root = Path(args.project_root).resolve()
    db = Path(args.db).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    runner_provenance = verify_runner_package(here, args.runner_inventory_sha256)
    parts = image_parts(args.image)
    place_image = placement_image(args.image)
    with db_session(db) as conn:
        plan = candidate_plan(conn, args.server, args.expected_owner)
    plan.update({
        "immutableImage": args.image,
        "placementImage": place_image,
        "expectedDigest": parts["digest"],
        "mode": "run" if args.yes else "plan",
        "plannedAt": iso_now(),
        "runner": runner_provenance,
    })
    atomic_json(out / "plan.json", plan)
    if not args.yes:
        return plan

    live_db = (project_root / "data" / "store.db").resolve()
    if db != live_db:
        raise Refusal(f"paid run must use atlas collector DB {live_db}, not {db}")

    verified_digest = verify_remote_digest(args.image, args.crane)
    plan["verifiedDigest"] = verified_digest
    atomic_json(out / "plan.json", plan)

    actions = ActionClient(here / "action_bridge.py", project_root, args.python)
    state_path = out / "state.json"
    armed_at = iso_now()
    state = {
        "phase": "armed",
        "parentPid": os.getpid(),
        "serverId": plan["serverId"],
        "expectedOwner": plan["owner"],
        "expectedModel": plan["model"],
        "pilot": PILOT,
        "armedAt": armed_at,
        "billingStartedAt": armed_at,
        "deadlineSeconds": RUN_DEADLINE_S,
        "maxCostUsd": MAX_COST_USD,
        "priceUsdDayWithFee": plan["priceUsdDayWithFee"],
        "creationFeeUsd": plan["creationFeeUsd"],
        "image": place_image,
        "expectedDigest": parts["digest"],
        "resolvedDigest": verified_digest,
        "actionBridge": str(here / "action_bridge.py"),
        "python": args.python,
        "runner": runner_provenance,
        "placementOutcome": "not_started",
        "discoveryTimeoutSeconds": 120,
    }
    atomic_json(state_path, state)
    watchdog_log = (out / "watchdog.log").open("ab", buffering=0)
    watchdog = subprocess.Popen(
        [args.python, str(here / "watchdog.py"), "--state", str(state_path), "--db", str(db),
         "--project-root", str(project_root), "--interval", "15", "--stall-seconds", str(args.stall_seconds)],
        stdout=watchdog_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    placed = False
    caught: Exception | None = None
    try:
        require_watchdog_alive(watchdog)
        # Second collector read after the independent watchdog is alive closes the planning/placement gap.
        with db_session(db) as conn:
            final_plan = candidate_plan(conn, args.server, args.expected_owner)
        if final_plan["fingerprint"] != plan["fingerprint"]:
            raise Refusal("candidate owner/model/price changed between plan and placement")
        require_watchdog_alive(watchdog)
        update_state(state_path, phase="placing", placementOutcome="pending")
        try:
            placement = actions.place(final_plan, place_image)
        except ActionFailure as exc:
            update_state(
                state_path,
                placementOutcome="possibly_accepted" if exc.possibly_accepted else "refused",
                placementErrorCode=exc.code,
                placementError=str(exc),
                placementActionOutcome=exc.outcome,
                placementOperationId=exc.operation_id,
                placementRequestSent=exc.request_sent,
            )
            raise
        order_id = str(placement["orderId"])
        placed = True
        update_state(
            state_path,
            phase="placed",
            placementOutcome="accepted",
            orderId=order_id,
            placedAt=iso_now(),
            placement=placement,
        )

        require_watchdog_alive(watchdog)
        order = wait_for_order(db, order_id, final_plan["serverId"], watchdog=watchdog)
        endpoint = endpoint_from_order(order)
        update_state(state_path, phase="running", endpoint=endpoint, endpointDiscoveredAt=iso_now())

        billing_started = dt.datetime.fromisoformat(read_json(state_path)["billingStartedAt"]).astimezone(dt.timezone.utc)
        deadline = time.monotonic() + max(
            0.0,
            RUN_DEADLINE_S - (dt.datetime.now(dt.timezone.utc) - billing_started).total_seconds(),
        )

        def guarded_fetch(url: str, limit: int = 64 * 1024 * 1024) -> bytes | None:
            require_watchdog_alive(watchdog)
            if (out / "abort.json").exists():
                raise Refusal(f"watchdog aborted during artifact collection: {(out / 'abort.json').read_text().strip()}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Refusal("30-minute paid deadline reached during artifact collection")
            return fetch(url, timeout=max(1, min(10, int(remaining + 0.999))), limit=limit)

        endpoint_ready_deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            require_watchdog_alive(watchdog)
            if (out / "abort.json").exists():
                raise Refusal(f"watchdog aborted run: {(out / 'abort.json').read_text().strip()}")
            try:
                failed = fetch(endpoint + "/failed")
                log = fetch(endpoint + "/run.log")
                done_marker = fetch(endpoint + "/done")
            except (OSError, urllib.error.URLError):
                if time.monotonic() < endpoint_ready_deadline:
                    time.sleep(5)
                    continue
                raise Refusal("mapped compute endpoint did not become reachable within 120 seconds")
            if failed is not None:
                (out / "failed.remote").write_bytes(failed)
                raise Refusal("rental endpoint reported /failed")
            if log is not None:
                (out / "run.log").write_bytes(log)
            if done_marker is not None:
                break
            time.sleep(5)
        else:
            raise Refusal("hard 30-minute compute deadline reached")

        done_body = guarded_fetch(endpoint + "/done")
        inventory_body = guarded_fetch(endpoint + "/ARTIFACTS.sha256")
        if done_body is None or inventory_body is None:
            raise Refusal("done was present but terminal marker or artifact inventory is missing")
        (out / "done").write_bytes(done_body)
        (out / "ARTIFACTS.sha256").write_bytes(inventory_body)
        terminal_failed = guarded_fetch(endpoint + "/failed")
        if terminal_failed is not None:
            (out / "failed.remote").write_bytes(terminal_failed)
            raise Refusal("terminal recheck found /failed alongside /done")
        atomic_json(out / "failed.absent.json", {"at": iso_now(), "endpoint": endpoint + "/failed"})
        inventory = collect_verified_inventory(
            endpoint,
            inventory_body,
            out / "artifacts",
            fetch_one=guarded_fetch,
        )
        done_fields = {}
        for line in done_body.decode("utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                done_fields[key] = value
        manifest_hash = hashlib.sha256((out / "artifacts/run/manifest.json").read_bytes()).hexdigest()
        if done_fields.get("status") != "complete" or done_fields.get("manifest_sha256") != manifest_hash:
            raise Refusal("done marker does not bind the verified complete manifest")
        require_watchdog_alive(watchdog)
        if (out / "abort.json").exists() or time.monotonic() >= deadline:
            raise Refusal("run ceased to be valid before terminal acceptance")
        atomic_json(out / "collected.json", {"at": iso_now(), "endpoint": endpoint, "files": inventory})
        update_state(state_path, phase="complete")
        return {"status": "complete", "serverId": final_plan["serverId"], "orderId": order_id,
                "output": str(out), "files": len(inventory)}
    except Exception as exc:
        caught = exc
        with contextlib.suppress(Exception):
            failure_endpoint = read_json(state_path).get("endpoint")
            if failure_endpoint:
                collect_failure_evidence(failure_endpoint, out)
        atomic_json(out / "runner-failed.json", {"at": iso_now(), "error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        cleanup_verified = False
        try:
            cleanup(actions, db, state_path, strict_verify=placed)
            cleanup_verified = read_json(state_path).get("phase") == "cleaned"
        except Exception as cleanup_exc:
            atomic_json(out / "cleanup-failed.json", {"at": iso_now(), "error": str(cleanup_exc)})
            if caught is None:
                raise
        finally:
            # Stop the independent guard only after collector convergence was
            # proved.  If coordinator cleanup failed, leave the new session
            # alive: once this parent exits it performs the second exact-order
            # cleanup instead of abandoning a potentially billable order.
            stop_watchdog(watchdog, cleanup_verified)
            watchdog_log.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=os.getcwd())
    p.add_argument("--db", default="data/store.db")
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--expected-owner", default=DEFAULT_OWNER)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--crane", default="crane")
    p.add_argument("--runner-inventory-sha256", required=True)
    p.add_argument("--stall-seconds", type=int, default=75)
    p.add_argument("--yes", action="store_true", help="perform paid placement; without it only write the dry-run plan")
    return p


def main() -> None:
    args = parser().parse_args()
    if not Path(args.db).is_absolute():
        args.db = str(Path(args.project_root) / args.db)
    try:
        result = run(args)
    except Refusal as exc:
        raise SystemExit(f"NO-GO: {exc}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
