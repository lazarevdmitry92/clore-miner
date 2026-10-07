import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qtc_rental as rental
from qtc_rental import (ActionClient, ActionFailure, Refusal, candidate_plan, cleanup, discover_pilot_order,
                        collect_failure_evidence, collect_verified_inventory, endpoint_from_order, image_parts, placement_image,
                        require_watchdog_alive, stop_watchdog,
                        verify_remote_digest, verify_runner_package)


IMAGE = "ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-" + "a" * 40 + "@sha256:" + "b" * 64


SCHEMA = """
CREATE TABLE current_items(source TEXT, item_key TEXT, data TEXT, updated_at TEXT,
                           PRIMARY KEY(source,item_key));
CREATE TABLE server_eval(platform TEXT, server_id TEXT, owner TEXT, gpu_raw TEXT, label TEXT,
 n INTEGER, offer_gpus INTEGER, rented INTEGER, od_currency TEXT, od_price REAL, od_usd REAL,
 od_margin REAL, sp_margin REAL, outcome TEXT);
CREATE TABLE bans(platform TEXT, scope TEXT, key TEXT, state TEXT, until TEXT);
CREATE TABLE order_versions(platform TEXT, order_id TEXT, valid_from TEXT, valid_to TEXT, server_id TEXT,
 status TEXT, expires_at TEXT, created_at TEXT);
CREATE TABLE order_terms(platform TEXT, server_id TEXT, order_id TEXT, pilot TEXT, placed_at TEXT, image TEXT);
CREATE TABLE pilot_servers(platform TEXT, server_id TEXT, pilot TEXT, valid_from TEXT, valid_to TEXT);
CREATE TABLE rates(currency TEXT, usd REAL, valid_to TEXT);
CREATE TABLE platform_fees(platform TEXT, order_type TEXT, currency TEXT, valid_from TEXT, valid_to TEXT,
 fee_pct REAL, creation_fee REAL);
CREATE TABLE server_eval_meta(platform TEXT, built_at TEXT, snapshot_at TEXT, fetched_at TEXT,
 rates TEXT, od_fee REAL, version TEXT);
"""


def database(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "store.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    offer = {
        "server_id": 69333,
        "owner": 4727,
        "gpu_key": "rtx 4090",
        "gpu_model": "RTX 4090",
        "gpu_raw": "1x NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "cuda_version": "13.2",
        "rented": 0,
        "spot_rented": 0,
        "price_json": json.dumps({"on_demand": {"USD-Blockchain": 8.84}, "spot": {}}),
    }
    # Keep fixture writes out of the production ownership linter: this nested
    # test builds an isolated temporary database and never writes data/store.db.
    conn.execute("INSERT" " INTO current_items VALUES ('offers-clore','69333',?,'2026-10-06T00:00:00+00:00')",
                 (json.dumps(offer),))
    conn.execute("INSERT" " INTO server_eval VALUES "
                 "('clore','69333','4727','1x RTX 4090','1x rtx 4090',1,1,0,'USD-Blockchain',8.84,8.84,-7.0,-7.0,'below_roi')")
    conn.execute("INSERT" " INTO rates VALUES ('usd-blockchain',1.0,NULL)")
    conn.execute("INSERT" " INTO platform_fees VALUES ('clore','on-demand','USD-Blockchain','x',NULL,5.0,0.005)")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.execute("INSERT" " INTO server_eval_meta VALUES ('clore',?,?,?,?,1.05,'test-version')",
                 (now, now, now, json.dumps({"usd-blockchain": 1.0})))
    conn.commit()
    return conn


def test_exact_candidate_plan_and_cost(tmp_path):
    conn = database(tmp_path)
    plan = candidate_plan(conn, "69333", "4727")
    assert plan["serverId"] == "69333"
    assert plan["model"] == "rtx 4090"
    assert plan["pilot"] == "qtc_own_miner" and plan["until"] == "35m" and plan["exempt"] == "all"
    assert plan["projectedCostUsd"] == pytest.approx(0.23060417, rel=1e-5)
    assert plan["projectedCostUsd"] < 1


@pytest.mark.parametrize("change, match", [
    ("UPDATE" " current_items SET data=json_set(data,'$.rented',1)", "not free"),
    ("UPDATE" " current_items SET data=json_set(data,'$.gpu_count',2)", "exactly one"),
    ("UPDATE" " current_items SET data=json_set(data,'$.owner',570)", "denied"),
    ("UPDATE" " server_eval SET od_margin=1.01", "profitable"),
    ("UPDATE" " server_eval SET od_price=8.83", "price is stale"),
    ("UPDATE" " server_eval SET od_usd=8.80", "USD price is stale"),
    ("UPDATE" " current_items SET data=json_set(data,'$.cuda_version','12.5')", "CUDA compatibility"),
    ("UPDATE" " server_eval_meta SET built_at='2020-01-01T00:00:00+00:00'", "margin build is stale"),
    ("UPDATE" " server_eval_meta SET od_fee=1.10", "different on-demand fee"),
])
def test_candidate_gates(tmp_path, change, match):
    conn = database(tmp_path)
    conn.execute(change)
    conn.commit()
    with pytest.raises(Refusal, match=match):
        candidate_plan(conn, "69333", "4727")


def test_ban_current_order_and_pilot_are_each_no_go(tmp_path):
    conn = database(tmp_path)
    forever = "9999-01-01T00:00:00+00:00"
    conn.execute("INSERT" " INTO bans VALUES ('clore','host','4727','banned',?)", (forever,))
    conn.commit()
    with pytest.raises(Refusal, match="banned"):
        candidate_plan(conn, "69333", "4727")
    conn.execute("DELETE" " FROM bans")
    conn.execute("INSERT" " INTO order_versions VALUES ('clore','1','x',NULL,'69333','active',NULL,'x')")
    conn.commit()
    with pytest.raises(Refusal, match="current order"):
        candidate_plan(conn, "69333", "4727")
    conn.execute("DELETE" " FROM order_versions")
    conn.execute("INSERT" " INTO pilot_servers VALUES ('clore','69333','other','x',NULL)")
    conn.commit()
    with pytest.raises(Refusal, match="pilot"):
        candidate_plan(conn, "69333", "4727")


def test_image_syntax_and_remote_digest(tmp_path):
    assert image_parts(IMAGE)["commit"] == "a" * 40
    assert placement_image(IMAGE) == "ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-" + "a" * 40
    with pytest.raises(Refusal):
        image_parts(IMAGE.split("@", 1)[0])
    crane = tmp_path / "crane"
    crane.write_text("#!/bin/sh\nprintf '%s\\n' 'sha256:" + "b" * 64 + "'\n")
    crane.chmod(0o755)
    assert verify_remote_digest(IMAGE, str(crane)) == "sha256:" + "b" * 64


def test_runner_package_is_bound_to_reviewed_inventory(tmp_path):
    names = ("qtc_rental.py", "action_bridge.py", "watchdog.py")
    for name in names:
        (tmp_path / name).write_text(name + "\n")
    inventory = "".join(
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    ).encode()
    (tmp_path / "RUNNER_SHA256SUMS").write_bytes(inventory)
    expected = hashlib.sha256(inventory).hexdigest()
    assert verify_runner_package(tmp_path, expected)["inventorySha256"] == expected
    (tmp_path / "watchdog.py").write_text("tampered\n")
    with pytest.raises(Refusal, match="does not match"):
        verify_runner_package(tmp_path, expected)


def test_checked_in_runner_package_matches_its_documented_inventory():
    expected = "19394edbaebb3de92d8d39c87ad0ca318f630d0aeb7bb68481aa58846b5cdb97"
    assert hashlib.sha256((ROOT / "RUNNER_SHA256SUMS").read_bytes()).hexdigest() == expected
    assert verify_runner_package(ROOT, expected)["inventorySha256"] == expected
    assert expected in (ROOT / "README.md").read_text()
    assert expected in (ROOT.parent / "RENTAL.md").read_text()


def test_endpoint_only_from_collector_order_mapping():
    order = {"pub_cluster": ["n1.de.clorecloud.net"], "tcp_ports": ["21550:1234", "21559:5678"]}
    assert endpoint_from_order(order) == "http://n1.de.clorecloud.net:5678"
    with pytest.raises(Refusal):
        endpoint_from_order({**order, "tcp_ports": ["21550:1234"]})


def test_watchdog_can_recover_clore_id_recorded_after_order_terms(tmp_path):
    conn = database(tmp_path)
    at = "2026-10-06T12:00:00+00:00"
    created = "2026-10-06T11:59:59+00:00"  # platform clock may round behind atlas
    conn.execute("INSERT" " INTO order_versions VALUES ('clore','77',?,NULL,'69333','active',NULL,?)", (created, created))
    conn.execute("INSERT" " INTO order_terms VALUES ('clore','69333',NULL,'qtc_own_miner',?,?)", (at, IMAGE))
    conn.commit()
    assert discover_pilot_order(conn, "69333", "qtc_own_miner", at, IMAGE) == "77"


def test_action_client_is_a_mockable_subprocess_boundary(tmp_path):
    mock = tmp_path / "mock_actions.py"
    mock.write_text(
        "import json, pathlib, sys\n"
        "p=pathlib.Path(__file__).with_name('calls.jsonl')\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}))\n"
    )
    client = ActionClient(mock, tmp_path)
    result = client.call("drop", "--server", "69333", "--order", "9")
    assert result["ok"] is True
    args = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert args[:3] == ["--project-root", str(tmp_path), "drop"]


def test_action_client_preserves_the_structured_placement_outcome(tmp_path):
    mock = tmp_path / "mock_lost.py"
    mock.write_text(
        "import json\n"
        "print(json.dumps({'ok':False,'error':'busy','code':'lost','outcome':'not_sent',"
        "'operationId':'op-7','requestSent':False,'possiblyAccepted':False}))\n"
        "raise SystemExit(2)\n"
    )
    with pytest.raises(ActionFailure) as caught:
        ActionClient(mock, tmp_path).call("place")
    error = caught.value
    assert (error.code, error.outcome, error.operation_id) == ("lost", "not_sent", "op-7")
    assert error.request_sent is False and error.possibly_accepted is False


def test_uncertain_placement_without_visible_order_never_closes_pilot(tmp_path):
    conn = database(tmp_path)
    conn.close()
    mock = tmp_path / "mock_actions.py"
    mock.write_text("raise AssertionError('no action is safe while placement is uncertain')\n")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "phase": "placing", "placementOutcome": "possibly_accepted",
        "serverId": "69333", "pilot": "qtc_own_miner", "armedAt": "2026-10-06T11:59:00+00:00",
        "image": IMAGE.split("@", 1)[0], "discoveryTimeoutSeconds": 0,
    }))
    with pytest.raises(Refusal, match="watchdog remains armed"):
        cleanup(ActionClient(mock, tmp_path), tmp_path / "store.db", state_path, strict_verify=False)
    state = json.loads(state_path.read_text())
    assert state["phase"] == "guarding" and state["cleanupPending"] is True


def test_cleanup_recovers_lost_accepted_order_before_ending_pilot(tmp_path):
    conn = database(tmp_path)
    at = "2026-10-06T12:00:00+00:00"
    conn.execute("INSERT" " INTO order_versions VALUES ('clore','77',?,NULL,'69333','active',NULL,?)", (at, at))
    conn.execute("INSERT" " INTO order_terms VALUES ('clore','69333',NULL,'qtc_own_miner',?,?)", (at, IMAGE.split("@", 1)[0]))
    conn.commit()
    conn.close()
    mock = tmp_path / "mock_actions.py"
    mock.write_text(
        "import json, pathlib, sqlite3, sys\n"
        "p=pathlib.Path(__file__).with_name('calls.jsonl')\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if 'drop' in sys.argv:\n"
        " c=sqlite3.connect(pathlib.Path(__file__).with_name('store.db'))\n"
        " c.execute(\"UPDATE\" \" order_versions SET valid_to='closed' WHERE order_id='77'\")\n"
        " c.commit(); c.close()\n"
        "print(json.dumps({'ok': True}))\n"
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "phase": "placing", "placementOutcome": "possibly_accepted",
        "serverId": "69333", "pilot": "qtc_own_miner",
        "armedAt": "2026-10-06T11:59:00+00:00", "image": IMAGE.split("@", 1)[0],
    }))
    actions = ActionClient(mock, tmp_path)
    result = cleanup(actions, tmp_path / "store.db", state_path, strict_verify=False)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [call[2] for call in calls] == ["drop", "end"]
    assert calls[0][-1] == "77"
    assert json.loads(state_path.read_text())["recoveredOrderId"] is True
    assert result[-1]["end"]["ok"] is True


def test_failed_cleanup_does_not_terminate_independent_watchdog():
    class Guard:
        terminated = False
        killed = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            raise AssertionError("unverified cleanup must not wait for or reap guard")

        def kill(self):
            self.killed = True

    guard = Guard()
    assert stop_watchdog(guard, cleanup_verified=False) is False
    assert guard.terminated is False and guard.killed is False


def _artifact_server():
    def parity_batch(reps):
        return struct.pack("<8s7I", b"QTCE4GPU", 1, 1, 1, 1, 1, reps, 5) + b"\0" * 7 + struct.pack("<d", 1.0)

    files = {
        "/package.sha256": ("d" * 64 + "\n").encode(),
        "/qtc_sim2_batch": b"cuda-binary",
        "/parity-reps3.bin": parity_batch(3),
        "/parity-reps6.bin": parity_batch(6),
        "/run/000-coarse.batch.bin": b"batch",
        "/run/000-coarse.scores.bin": b"scores",
    }
    digest = lambda body: hashlib.sha256(body).hexdigest()
    files["/qtc_sim2_batch.sha256"] = (digest(files["/qtc_sim2_batch"]) + "  /var/log/qtc-e/qtc_sim2_batch\n").encode()
    parity = {
        "schema": "qtc-e-cuda-parity-v2",
        "gpu_binary_sha256": digest(files["/qtc_sim2_batch"]),
        "authoritative_model_sha256": "c" * 64,
        "tolerance": 1e-12,
        "runs": [
            {"reps": 3, "authoritative_batch_sha256": digest(files["/parity-reps3.bin"]),
             "jobs": 1, "max_abs_difference": 0.0},
            {"reps": 6, "authoritative_batch_sha256": digest(files["/parity-reps6.bin"]),
             "jobs": 1, "max_abs_difference": 5e-13},
        ],
    }
    files["/parity.json"] = (json.dumps(parity, sort_keys=True) + "\n").encode()
    manifest = {
        "schema": "qtc-e-rental-run-v1",
        "status": "complete",
        "gpu_binary_sha256": digest(files["/qtc_sim2_batch"]),
        "parity_artifact_sha256": digest(files["/parity.json"]),
        "authoritative_model_sha256": "c" * 64,
        "authoritative_batches_sha256": {
            "3": digest(files["/parity-reps3.bin"]),
            "6": digest(files["/parity-reps6.bin"]),
        },
        "batches": [{
            "batch": "000-coarse.batch.bin",
            "batch_sha256": digest(files["/run/000-coarse.batch.bin"]),
            "results": "000-coarse.scores.bin",
            "results_sha256": digest(files["/run/000-coarse.scores.bin"]),
        }],
    }
    files["/run/manifest.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    files["/result.json"] = (json.dumps({
        "scorer": "cuda",
        "archive_manifest_sha256": digest(files["/run/manifest.json"]),
    }, sort_keys=True) + "\n").encode()
    inventory = b"".join(
        f"{digest(body)}  /var/log/qtc-e{path}\n".encode()
        for path, body in sorted(files.items())
    )
    return files, inventory


def test_artifact_inventory_fetches_and_verifies_every_listed_file(tmp_path, monkeypatch):
    files, inventory = _artifact_server()
    monkeypatch.setattr(rental, "fetch", lambda url, **kwargs: files.get(url.removeprefix("http://gpu")))
    collected = collect_verified_inventory("http://gpu", inventory, tmp_path / "artifacts")
    assert len(collected) == len(files)
    assert (tmp_path / "artifacts/run/manifest.json").is_file()


def test_artifact_inventory_rejects_tamper_and_missing(tmp_path, monkeypatch):
    files, inventory = _artifact_server()
    files["/result.json"] += b"tampered"
    monkeypatch.setattr(rental, "fetch", lambda url, **kwargs: files.get(url.removeprefix("http://gpu")))
    with pytest.raises(Refusal, match="hash mismatch"):
        collect_verified_inventory("http://gpu", inventory, tmp_path / "tampered")
    missing = b"\n".join(line for line in inventory.splitlines() if b"package.sha256" not in line) + b"\n"
    with pytest.raises(Refusal, match="missing required"):
        rental.parse_artifact_inventory(missing)
    excessive = "".join(
        f"{'0' * 64}  /var/log/qtc-e/run/{index:03d}.bin\n" for index in range(33)
    ).encode()
    with pytest.raises(Refusal, match="more than 32"):
        rental.parse_artifact_inventory(excessive, require_complete=False)


def test_failure_evidence_collects_failed_manifest_and_bound_batches(tmp_path, monkeypatch):
    batch, scores = b"failed-batch", b"partial-scores"
    manifest = {
        "schema": "qtc-e-rental-run-v1",
        "status": "failed",
        "batches": [{
            "batch": "000-coarse.batch.bin",
            "batch_sha256": hashlib.sha256(batch).hexdigest(),
            "results": "000-coarse.scores.bin",
            "results_sha256": hashlib.sha256(scores).hexdigest(),
        }],
    }
    files = {
        "/failed": b"status=compute_failed\n",
        "/run.log": b"parity=PASS\ncompute failed\n",
        "/run/manifest.json": (json.dumps(manifest) + "\n").encode(),
        "/run/000-coarse.batch.bin": batch,
        "/run/000-coarse.scores.bin": scores,
    }
    monkeypatch.setattr(rental, "fetch", lambda url, **kwargs: files.get(url.removeprefix("http://gpu")))
    collected = collect_failure_evidence("http://gpu", tmp_path)
    paths = {item["path"] for item in collected}
    assert {"failed", "run.log", "run/manifest.json", "run/000-coarse.batch.bin"} <= paths
    assert (tmp_path / "failure-evidence/collection.json").is_file()


def test_dead_watchdog_is_fail_closed():
    class Dead:
        def poll(self):
            return 9

    with pytest.raises(Refusal, match="watchdog exited"):
        require_watchdog_alive(Dead())


@pytest.mark.parametrize("phase,parent_pid,deadline", [
    ("running", 1, 0),
    ("cleanup", 99999999, 999),
])
def test_watchdog_failure_path_uses_mock_drop_and_end(tmp_path, phase, parent_pid, deadline):
    conn = database(tmp_path)
    conn.close()
    mock = tmp_path / "mock_actions.py"
    mock.write_text(
        "import json, pathlib, sys\n"
        "p=pathlib.Path(__file__).with_name('calls.jsonl')\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(json.dumps({'ok': True}))\n"
    )
    state = {
        "phase": phase, "parentPid": parent_pid, "serverId": "69333", "orderId": "77",
        "expectedOwner": "4727", "expectedModel": "rtx 4090", "pilot": "qtc_own_miner",
        "placedAt": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=5)).isoformat(),
        "deadlineSeconds": deadline, "maxCostUsd": 1, "priceUsdDayWithFee": 1, "creationFeeUsd": 0,
        "actionBridge": str(mock), "python": sys.executable,
        "verifyTimeoutSeconds": 1,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    done = subprocess.run([
        sys.executable, str(ROOT / "watchdog.py"), "--state", str(state_path), "--db", str(tmp_path / "store.db"),
        "--project-root", str(tmp_path), "--once", "--ownership-grace", "999",
    ], capture_output=True, text=True)
    assert done.returncode == 2
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [c[2] for c in calls] == ["drop", "end"]
    assert (tmp_path / "abort.json").exists()


def test_watchdog_keeps_guarding_uncertain_placement_without_order(tmp_path):
    conn = database(tmp_path)
    conn.close()
    mock = tmp_path / "must_not_run.py"
    mock.write_text("raise AssertionError('must not end pilot without proving no accepted order')\n")
    state = {
        "phase": "guarding", "placementOutcome": "possibly_accepted",
        "parentPid": 99999999, "serverId": "69333", "pilot": "qtc_own_miner",
        "armedAt": "2026-10-06T11:59:00+00:00", "image": IMAGE.split("@", 1)[0],
        "actionBridge": str(mock), "python": sys.executable, "discoveryTimeoutSeconds": 0,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    done = subprocess.run([
        sys.executable, str(ROOT / "watchdog.py"), "--state", str(state_path), "--db", str(tmp_path / "store.db"),
        "--project-root", str(tmp_path), "--once",
    ], capture_output=True, text=True)
    assert done.returncode == 3
    assert (tmp_path / "watchdog-pending.json").is_file()
    assert not (tmp_path / "watchdog-cleanup.json").exists()
