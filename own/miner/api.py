"""HTTP /summary for the collector (stdlib http.server). Card telemetry from nvidia-smi, null without it."""
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

SMI_FIELDS = ["index", "temperature.gpu", "power.draw", "fan.speed", "clocks.sm", "clocks.mem"]
SMI_KEYS = [None, "temperature_c", "power_w", "fan_pct", "core_clock_mhz", "mem_clock_mhz"]
TELEMETRY_KEYS = [k for k in SMI_KEYS if k]


def parse_smi_csv(text: str) -> dict[int, dict]:
    """nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits -> {index: telemetry}."""
    out = {}
    for line in text.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(SMI_FIELDS):
            raise ValueError(f"nvidia-smi line has {len(vals)} fields: {line!r}")
        row = {}
        for key, v in zip(SMI_KEYS[1:], vals[1:]):
            try:
                row[key] = float(v)
            except ValueError:   # [N/A], [Not Supported]
                row[key] = None
        out[int(vals[0])] = row
    return out


class Telemetry:
    def __init__(self, ttl: float = 5.0):
        self.smi = shutil.which("nvidia-smi")
        self.ttl = ttl
        self._at, self._data = 0.0, {}
        self._lock = threading.Lock()

    def get(self) -> dict[int, dict]:
        if not self.smi:
            return {}
        with self._lock:
            if time.time() - self._at > self.ttl:
                try:
                    res = subprocess.run([self.smi, f"--query-gpu={','.join(SMI_FIELDS)}",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5, check=True)
                    self._data = parse_smi_csv(res.stdout)
                except (subprocess.SubprocessError, OSError, ValueError) as e:
                    log.warning("nvidia-smi: %s", e)
                    self._data = {}
                self._at = time.time()
            return self._data


def summary(miner, pool, stats, backend, telemetry: Telemetry) -> dict:
    job = miner.latest()
    target = miner.share_target(job) if job else None
    smi = telemetry.get()
    gpus = []
    for d in backend.devices():
        t = smi.get(d["nvidia_index"], {}) if d["nvidia_index"] is not None else {}
        c = stats.dev[d["id"]]
        gpus.append({"id": d["id"], "name": d["name"], "pci_bus_id": d["pci_bus_id"], "sm_count": d["sm_count"],
                     "hashrate": stats.hashrate(d["id"]), "accepted": c["accepted"], "rejected": c["rejected"],
                     "compute_errors": c["compute_errors"], **{k: t.get(k) for k in TELEMETRY_KEYS},
                     "kernel": backend.kernel})
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
