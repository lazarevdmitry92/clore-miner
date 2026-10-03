"""HTTP /summary for the collector (stdlib http.server). The cards come from nvidia-smi whatever the backend; without
nvidia-smi the list is empty and gpu_telemetry_error says why. A card process of the supervisor (--relay) answers
relay_summary() instead: its own counters only, the supervisor adds the telemetry."""
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import VERSION
from .pool import DIFF1

log = logging.getLogger("api")

SMI_FIELDS = ["index", "name", "pci.bus_id", "uuid", "compute_cap", "driver_version", "clocks_throttle_reasons.active",
              "power.draw", "power.limit", "clocks.sm", "clocks.max.sm", "clocks.mem", "temperature.gpu", "fan.speed",
              "utilization.gpu"]
SMI_TEXTS = ["uuid", "compute_cap", "driver"]
SMI_NUMBERS = ["power_w", "power_limit_w", "core_clock_mhz", "clock_max_mhz", "mem_clock_mhz", "temperature_c",
               "fan_pct", "util_pct"]
# NVML clocks_throttle_reasons bits (nvml.h nvmlClocksThrottleReason*)
THROTTLE_BITS = {0x1: "idle", 0x2: "app_clocks", 0x4: "power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "power_brake", 0x100: "display_clocks"}
CUDA_RE = re.compile(r"CUDA Version:\s*([0-9.]+)")


def _na(v: str) -> bool:
    return v.startswith("[") or v in ("", "N/A")


def parse_smi_csv(text: str) -> list[dict]:
    """nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits -> one dict per card, in index order.
    Fields nvidia-smi gives as [N/A] / [Not Supported] are None; throttle_mask is the reasons bit mask."""
    out = []
    for line in text.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(SMI_FIELDS):
            raise ValueError(f"nvidia-smi line has {len(vals)} fields, want {len(SMI_FIELDS)}: {line!r}")
        row = {"index": int(vals[0]), "name": vals[1], "pci_bus_id": vals[2]}
        for key, v in zip(SMI_TEXTS, vals[3:6]):
            row[key] = None if _na(v) else v
        row["throttle_mask"] = None if _na(vals[6]) else int(vals[6], 16)
        for key, v in zip(SMI_NUMBERS, vals[7:]):
            try:
                row[key] = float(v)
            except ValueError:   # [N/A], [Not Supported]
                row[key] = None
        out.append(row)
    return sorted(out, key=lambda r: r["index"])


def throttle_words(mask: int | None) -> list[str] | None:
    if mask is None:
        return None
    words = [name for bit, name in THROTTLE_BITS.items() if mask & bit]
    rest = mask & ~sum(THROTTLE_BITS)
    return words + ([f"0x{rest:x}"] if rest else [])


def _ratio(a, b):
    return a / b if a is not None and b else None


def k_of(hashrate: float | None, sm_count: int | None, clock_mhz: float | None) -> float | None:
    """MAC per SM per clock (metrics.md M4)."""
    if hashrate is None or not sm_count or not clock_mhz:
        return None
    return hashrate / (sm_count * clock_mhz * 1e6)


def card_fields(c: dict | None, hashrate: float | None, sm_count: int | None) -> dict:
    """The telemetry of one card for /summary: nvidia-smi values plus throttle words, ratios to the limits and k."""
    c = c or {}
    return {"uuid": c.get("uuid"), "compute_cap": c.get("compute_cap"), "driver": c.get("driver"),
            "cuda": c.get("cuda"), **{k: c.get(k) for k in SMI_NUMBERS},
            "throttle": throttle_words(c.get("throttle_mask")),
            "power_ratio": _ratio(c.get("power_w"), c.get("power_limit_w")),
            "clock_ratio": _ratio(c.get("core_clock_mhz"), c.get("clock_max_mhz")),
            "k_actual": k_of(hashrate, sm_count, c.get("core_clock_mhz"))}


class Telemetry:
    def __init__(self, ttl: float = 5.0, smi: str | None = None):
        self.smi = smi or shutil.which("nvidia-smi")
        self.ttl = ttl
        self._at, self._cards, self._error = 0.0, [], None
        self._cuda, self._cuda_read = None, False
        self._lock = threading.Lock()

    def _read_cuda(self) -> str | None:
        """CUDA version of the driver: only the header of plain nvidia-smi has it (no --query-gpu field)."""
        try:
            res = subprocess.run([self.smi], capture_output=True, text=True, timeout=10, check=True)
        except (subprocess.SubprocessError, OSError) as e:
            log.warning("nvidia-smi (CUDA version): %s", e)
            return None
        m = CUDA_RE.search(res.stdout)
        if m is None:
            log.warning("nvidia-smi: no 'CUDA Version' in its header")
            return None
        return m.group(1)

    def get(self) -> tuple[list[dict], str | None]:
        """(cards, error): the cards of parse_smi_csv (+ "cuda"), or [] and why there are none."""
        if not self.smi:
            return [], "nvidia-smi not found in PATH"
        with self._lock:
            if not self._cuda_read:
                self._cuda, self._cuda_read = self._read_cuda(), True
            if time.time() - self._at > self.ttl:
                try:
                    res = subprocess.run([self.smi, f"--query-gpu={','.join(SMI_FIELDS)}",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5, check=True)
                    self._cards, self._error = parse_smi_csv(res.stdout), None
                    for c in self._cards:
                        c["cuda"] = self._cuda
                except (subprocess.SubprocessError, OSError, ValueError) as e:
                    log.warning("nvidia-smi: %s", e)
                    self._cards, self._error = [], f"nvidia-smi: {e}"
                self._at = time.time()
            return self._cards, self._error


class StageClock:
    """stage -> (stage, since): since is when the stage last changed."""

    def __init__(self):
        self.stage, self.since = None, time.time()
        self._lock = threading.Lock()

    def __call__(self, stage: str) -> tuple[str, float]:
        with self._lock:
            if stage != self.stage:
                self.stage, self.since = stage, time.time()
            return self.stage, self.since


def miner_stage(miner, pool) -> str:
    """starting (no job yet) -> kernel_init (the first kernel call in flight: init, the library's self-check) ->
    mining; no_job -- the pool connection is lost; rejected -- the pool rejected our authorize."""
    if getattr(pool, "rejected", False):
        return "rejected"
    if miner.calls == 0:
        return "kernel_init" if miner.call_started is not None else "starting"
    return "mining" if miner.latest() is not None else "no_job"


def summary(miner, pool, stats, backend, telemetry: Telemetry, stage: StageClock) -> dict:
    """gpus -- the cards of nvidia-smi, each with the work of the backend device on it (0 if none: cpu-ref);
    cpu -- the backend's CPU devices together (None if it has none); hashrate -- all devices, 0 without a job."""
    job = miner.latest()
    target = miner.share_target(job) if job else None
    rate = (lambda d=None: stats.hashrate(d)) if job else (lambda d=None: 0.0)
    cards, error = telemetry.get()
    devices = backend.devices()
    on_card = {d["nvidia_index"]: d for d in devices if d["nvidia_index"] is not None}
    lost = sorted(set(on_card) - {c["index"] for c in cards})
    if lost:
        error = "; ".join(filter(None, [error, f"backend devices on nvidia indexes {lost} are not in nvidia-smi"]))
    gpus = []
    for c in cards:
        d = on_card.get(c["index"])
        n = stats.dev[d["id"]] if d else {"accepted": 0, "rejected": 0, "compute_errors": 0}
        hs = rate(d["id"]) if d else 0.0
        sm = d["sm_count"] if d else None
        gpus.append({"id": c["index"], "name": c["name"], "pci_bus_id": c["pci_bus_id"], "sm_count": sm,
                     "hashrate": hs, **n, **card_fields(c, hs, sm), "kernel": backend.kernel if d else None,
                     "variant": backend.variant if d else None})
    on_cpu = [d for d in devices if d["nvidia_index"] is None]
    if on_cpu and backend.card_note:
        error = "; ".join(filter(None, [error, f"backend device not tied to a card ({backend.card_note}): "
                                               f"its work is under cpu"]))
    cpu = None
    if on_cpu:
        cpu = {"name": ", ".join(d["name"] for d in on_cpu),
               "hashrate": sum(rate(d["id"]) for d in on_cpu),
               **{k: sum(stats.dev[d["id"]][k] for d in on_cpu) for k in ("accepted", "rejected", "compute_errors")},
               "kernel": backend.kernel}
    st, since = stage(miner_stage(miner, pool))
    return {
        "miner": "own",
        "version": VERSION,
        "uptime": int(time.time() - stats.started),
        "stage": st,
        "stage_since": int(since),
        "backend": backend.name,
        "pool": {**pool.status(), "difficulty": DIFF1 / target if target else None},
        "hashrate": rate(),
        "accepted": stats.accepted,
        "rejected": stats.rejected,
        "stale": stats.stale,
        "dropped": stats.dropped,
        "gpus": gpus,
        "gpu_telemetry_error": error,
        "cpu": cpu,
    }


def relay_summary(miner, stats, backend) -> dict:
    """A card process of the supervisor: its counters, its one device and its kernel variants; no telemetry."""
    devices = backend.devices()
    if len(devices) != 1:
        raise ValueError(f"a card process has one backend device, got {len(devices)}")
    d = devices[0]
    return {"uptime": time.time() - stats.started, "stage": miner_stage(miner, None),
            "hashrate": stats.hashrate() if miner.latest() else 0.0, "found": stats.found,
            "accepted": stats.accepted, "rejected": stats.rejected, "stale": stats.stale, "dropped": stats.dropped,
            "compute_errors": stats.dev[d["id"]]["compute_errors"], "sm_count": d["sm_count"],
            "device_name": d["name"], "kernel": backend.kernel, "variants": backend.variants,
            "variant": backend.variant}


def serve(port: int, host: str, build) -> ThreadingHTTPServer:
    """Start the server in a daemon thread; build() -> the summary dict."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/summary", "/"):
                self.send_error(404)
                return
            body = json.dumps(build()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, name="api", daemon=True).start()
    return srv
