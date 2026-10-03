"""HTTP /summary for the collector (stdlib http.server). The cards come from nvidia-smi whatever the backend; without
nvidia-smi the list is empty and gpu_telemetry_error says why."""
import json
import logging
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import VERSION
from .pool import DIFF1

log = logging.getLogger("api")

SMI_FIELDS = ["index", "name", "pci.bus_id", "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu", "fan.speed",
              "utilization.gpu"]
SMI_NUMBERS = ["power_w", "core_clock_mhz", "mem_clock_mhz", "temperature_c", "fan_pct", "util_pct"]


def parse_smi_csv(text: str) -> list[dict]:
    """nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits -> one dict per card, in index order."""
    out = []
    for line in text.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(SMI_FIELDS):
            raise ValueError(f"nvidia-smi line has {len(vals)} fields, want {len(SMI_FIELDS)}: {line!r}")
        row = {"index": int(vals[0]), "name": vals[1], "pci_bus_id": vals[2]}
        for key, v in zip(SMI_NUMBERS, vals[3:]):
            try:
                row[key] = float(v)
            except ValueError:   # [N/A], [Not Supported]
                row[key] = None
        out.append(row)
    return sorted(out, key=lambda r: r["index"])


class Telemetry:
    def __init__(self, ttl: float = 5.0):
        self.smi = shutil.which("nvidia-smi")
        self.ttl = ttl
        self._at, self._cards, self._error = 0.0, [], None
        self._lock = threading.Lock()

    def get(self) -> tuple[list[dict], str | None]:
        """(cards, error): the cards of parse_smi_csv, or [] and why there are none."""
        if not self.smi:
            return [], "nvidia-smi not found in PATH"
        with self._lock:
            if time.time() - self._at > self.ttl:
                try:
                    res = subprocess.run([self.smi, f"--query-gpu={','.join(SMI_FIELDS)}",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5, check=True)
                    self._cards, self._error = parse_smi_csv(res.stdout), None
                except (subprocess.SubprocessError, OSError, ValueError) as e:
                    log.warning("nvidia-smi: %s", e)
                    self._cards, self._error = [], f"nvidia-smi: {e}"
                self._at = time.time()
            return self._cards, self._error


def summary(miner, pool, stats, backend, telemetry: Telemetry) -> dict:
    """gpus -- the cards of nvidia-smi, each with the work of the backend device on it (0 if none: cpu-ref);
    cpu -- the backend's CPU devices together (None if it has none); hashrate -- all devices."""
    job = miner.latest()
    target = miner.share_target(job) if job else None
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
        gpus.append({"id": c["index"], "name": c["name"], "pci_bus_id": c["pci_bus_id"],
                     "sm_count": d["sm_count"] if d else None,
                     "hashrate": stats.hashrate(d["id"]) if d else 0.0, **n,
                     **{k: c[k] for k in SMI_NUMBERS}, "kernel": backend.kernel if d else None})
    on_cpu = [d for d in devices if d["nvidia_index"] is None]
    cpu = None
    if on_cpu:
        cpu = {"name": ", ".join(d["name"] for d in on_cpu),
               "hashrate": sum(stats.hashrate(d["id"]) for d in on_cpu),
               **{k: sum(stats.dev[d["id"]][k] for d in on_cpu) for k in ("accepted", "rejected", "compute_errors")},
               "kernel": backend.kernel}
    return {
        "miner": "own",
        "version": VERSION,
        "uptime": int(time.time() - stats.started),
        "backend": backend.name,
        "pool": {"url": pool.url, "connected": bool(pool.connected),
                 "difficulty": DIFF1 / target if target else None},
        "hashrate": stats.hashrate(),
        "accepted": stats.accepted,
        "rejected": stats.rejected,
        "stale": stats.stale,
        "dropped": stats.dropped,
        "gpus": gpus,
        "gpu_telemetry_error": error,
        "cpu": cpu,
    }


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
