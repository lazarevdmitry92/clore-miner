"""Own Pearl miner (host side).

    python -m miner.main --wallet prl1... --worker own --backend cpu-ref --dry-run
    python -m miner.main --pool stratum+tcp://127.0.0.1:3333 --wallet x --nbits-override 0x1e00ffff
"""
import argparse
import datetime
import logging
import signal
import threading
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import VERSION
from .api import Telemetry, serve, summary
from .jobs import TILES, Miner, Stats
from .kernel import BACKENDS, SO_PREFIX, Backend, is_backend_name, make_backend
from .pool import IDLE_TIMEOUT, TLS_NAME, Pool

log = logging.getLogger("main")


def positive(cast):
    def parse(s):
        v = cast(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, got {s}")
        return v
    return parse


def backend_name(s):
    if is_backend_name(s):
        return s
    raise argparse.ArgumentTypeError(f"unknown backend {s!r}; known: {', '.join(BACKENDS)}, {SO_PREFIX}<path>")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="miner", description=f"own Pearl miner {VERSION}")
    p.add_argument("--pool", default="stratum+ssl://pearl.herominers.com:1200",
                   help="stratum+ssl://host:port (TLS) or stratum+tcp://host:port")
    p.add_argument("--tls-name", default=TLS_NAME, help="certificate name to check (HM: pearl.herominers.com)")
    p.add_argument("--wallet", required=True)
    p.add_argument("--worker", default="own")
    p.add_argument("--backend", default="cpu-ref", type=backend_name,
                   help=f"{' | '.join(BACKENDS)} | {SO_PREFIX}/path/lib.so (a kernel library over ctypes)")
    p.add_argument("--api-port", type=int, default=21550, help="0 = any free port")
    p.add_argument("--api-host", default="0.0.0.0")
    p.add_argument("--dry-run", action="store_true", help="never submit: log a verified share instead")
    p.add_argument("--nbits-override", type=lambda s: int(s, 0), default=None,
                   help="share target as compact nbits instead of the pool target (local tests)")
    p.add_argument("--idle-timeout", type=positive(float), default=IDLE_TIMEOUT,
                   help="s without a line from the pool before reconnecting (HM sends a job every ~35 s)")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--tile", choices=list(TILES), default="8x16",
                   help="hash tile: 8x16 (s0 §6), 16x16 contiguous (the SOAT kernel, soat_backend/; "
                        "--portion-rows and --n multiples of 256 run without padding) or v100 (kernels/v100: "
                        "m8n8k4 rows period 32, cols period 64)")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--portion-rows", type=positive(int), default=256)
    p.add_argument("--matrices", choices=["zero", "random"], default="zero")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until a signal)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


@dataclass
class App:
    backend: Backend
    stats: Stats
    miner: Miner
    pool: Pool
    srv: ThreadingHTTPServer
    threads: list

    def start(self):
        for t in self.threads:
            t.start()

    def wait(self, stop: threading.Event, duration: float = 0, tick: float = 1.0):
        """Until stop, duration or a failure; a fatal pool error or a dead thread -> SystemExit (non-zero)."""
        deadline = time.time() + duration if duration else None
        last = time.time()
        failure = None
        try:
            while not stop.wait(tick):
                if self.pool.fatal:
                    failure = f"fatal: {self.pool.fatal}"
                    break
                dead = [t.name for t in self.threads if not t.is_alive()]
                if dead:
                    failure = f"thread died: {', '.join(dead)}"
                    break
                if deadline and time.time() >= deadline:
                    break
                if time.time() - last >= 30:
                    last = time.time()
                    s = self.stats
                    log.info("hashrate %.3g MAC/s found %d accepted %d rejected %d stale %d dropped %d "
                             "pool connected=%s", s.hashrate(), s.found, s.accepted, s.rejected, s.stale, s.dropped,
                             self.pool.connected)
        finally:
            self.shutdown()
        if failure:
            log.error(failure)
            raise SystemExit(failure)

    def shutdown(self):
        self.miner.stop()
        self.pool.stop()
        self.srv.shutdown()
        self.threads[0].join(5)   # the pool thread closes its log on the way out


def build(args) -> App:
    backend = make_backend(args.backend)
    stats = Stats(backend.devices())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    miner = Miner(backend, None, stats, k=args.k, m=args.m, n=args.n, portion_rows=args.portion_rows,
                  matrices=args.matrices, nbits_override=args.nbits_override, dry_run=args.dry_run, tile=args.tile)
    pool = Pool(args.pool, args.wallet, args.worker, miner.on_job, args.log_dir / f"pool-{stamp}.jsonl",
                tls_name=args.tls_name, idle_timeout=args.idle_timeout)
    miner.pool = pool
    telemetry = Telemetry()
    srv = serve(args.api_port, args.api_host, lambda: summary(miner, pool, stats, backend, telemetry))
    threads = [threading.Thread(target=pool.run, name="pool", daemon=True),
               threading.Thread(target=miner.run, name="mining", daemon=True),
               threading.Thread(target=miner.run_submitter, name="submit", daemon=True)]
    return App(backend, stats, miner, pool, srv, threads)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = build(args)
    log.info("miner %s backend %s dry_run=%s api :%d k=%d m=%d n=%d tile %s", VERSION, app.backend.name,
             args.dry_run, app.srv.server_address[1], args.k, args.m, args.n, args.tile)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    app.start()
    app.wait(stop, args.duration)


if __name__ == "__main__":
    main()
