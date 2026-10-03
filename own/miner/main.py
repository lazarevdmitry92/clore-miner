"""Own Pearl miner (host side).

    python -m miner.main --wallet prl1... --worker c123 --all-gpus --kernels-dir /opt/own/kernels   # every card
    python -m miner.main --wallet prl1... --worker own --backend cpu-ref --dry-run                   # one process
    python -m miner.main --pool stratum+tcp://127.0.0.1:3333 --wallet x --backend cpu-ref --nbits-override 0x1e00ffff

--all-gpus runs the supervisor (supervisor.py): one card process per nvidia-smi card (this module with --relay), one
pool connection, one /summary. Exit codes: 0 -- a signal or --duration; 1 -- a failure; 2 -- the kernel returned an
error (KernelError); 3 -- a kernel call did not return in time (the card is stuck).
"""
import argparse
import datetime
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import VERSION
from .api import StageClock, Telemetry, relay_summary, serve, summary
from .jobs import TILES, Miner, Stats
from .kernel import BACKENDS, SO_PREFIX, Backend, KernelError, is_backend_name, make_backend
from .pool import IDLE_TIMEOUT, TLS_NAME, Pool, parse_pool_arg
from .relay import RelayPool, protocol_stdout

log = logging.getLogger("main")

EXIT_FAIL, EXIT_KERNEL, EXIT_STALLED = 1, 2, 3
SINGLE_SHAPE = {"tile": "8x16", "k": 2048, "m": 1024, "n": 1024, "portion_rows": 256}
CALL_TIMEOUT = 120.0    # s, a kernel call (a v100 portion of 65 536 rows is ~5-6 s)
INIT_TIMEOUT = 600.0    # s, the first call: device init, the library's self-check and autotune


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


def pool_urls(s):
    try:
        return parse_pool_arg(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="miner", description=f"own Pearl miner {VERSION}")
    p.add_argument("--pool", type=pool_urls, default=pool_urls(f"stratum+ssl://{TLS_NAME}:1200"),
                   help="stratum+ssl://host:port (TLS) or stratum+tcp://host:port, several comma-separated (official "
                        "nodes of one pool: the fastest answering is used), or `hm` -- every HeroMiners node")
    p.add_argument("--tls-name", default=TLS_NAME, help="certificate name to check (HM: pearl.herominers.com)")
    p.add_argument("--wallet")
    p.add_argument("--worker", default="own", help="one per server: c<server_id> (collector reads the pool by it)")
    p.add_argument("--backend", type=backend_name, default=None,
                   help=f"{' | '.join(BACKENDS)} | {SO_PREFIX}/path/lib.so; with --all-gpus the kernel of each card "
                        f"comes from the registry and only cpu-ref may be forced (tests)")
    p.add_argument("--all-gpus", action="store_true", help="the supervisor: one card process per nvidia-smi card")
    p.add_argument("--kernels-dir", type=Path, help="--all-gpus: the directory with the kernel libraries (registry.py)")
    p.add_argument("--api-port", type=int, default=21550, help="0 = any free port")
    p.add_argument("--api-host", default="0.0.0.0")
    p.add_argument("--dry-run", action="store_true", help="never submit: log a verified share instead")
    p.add_argument("--nbits-override", type=lambda s: int(s, 0), default=None,
                   help="share target as compact nbits instead of the pool target (local tests)")
    p.add_argument("--idle-timeout", type=positive(float), default=IDLE_TIMEOUT,
                   help="s without a line from the pool before reconnecting (HM sends a job every ~35 s)")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--tile", choices=list(TILES), default=None,
                   help="hash tile: 8x16 (s0 §6), 16x16 contiguous (the SOAT kernel, soat_backend/; "
                        "--portion-rows and --n multiples of 256 run without padding) or v100 (kernels/v100: "
                        "m8n8k4 rows period 32, cols period 64)")
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--portion-rows", type=positive(int), default=None)
    p.add_argument("--matrices", choices=["zero", "random"], default="zero")
    p.add_argument("--call-timeout", type=positive(float), default=CALL_TIMEOUT,
                   help="s a kernel call may take before the card counts as stuck (exit 3)")
    p.add_argument("--init-timeout", type=positive(float), default=INIT_TIMEOUT,
                   help="s the first kernel call may take (device init, self-check, autotune)")
    p.add_argument("--tune-hold", type=float, default=120.0, help="--all-gpus: s each kernel variant mines in a "
                                                                  "tune (0 = no tune)")
    p.add_argument("--tune-every", type=positive(float), default=3600.0, help="--all-gpus: s between tunes")
    p.add_argument("--restart-pause", type=float, default=10.0, help="--all-gpus: s before a crashed card restarts")
    p.add_argument("--max-restarts", type=positive(int), default=5,
                   help="--all-gpus: crashes of a card within --restart-window that turn it off")
    p.add_argument("--restart-window", type=positive(float), default=3600.0)
    p.add_argument("--slow-after", type=positive(float), default=600.0,
                   help="--all-gpus: s of k_actual < 0.8 k_expected before the slow_card flag")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until a signal)")
    p.add_argument("--relay", action="store_true", help=argparse.SUPPRESS)       # a card process of the supervisor
    p.add_argument("--nonce-lane", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--nonce-lanes", type=positive(int), default=1, help=argparse.SUPPRESS)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    try:
        check_args(args)
    except ValueError as e:
        p.error(str(e))
    return args


def check_args(args):
    if args.relay and args.all_gpus:
        raise ValueError("--relay is a card process of --all-gpus, not both")
    if not args.relay and not args.wallet:
        raise ValueError("--wallet is required")
    if args.all_gpus:
        if args.backend is not None and args.backend != "cpu-ref":
            raise ValueError("--all-gpus takes each card's kernel from the registry: only --backend cpu-ref may be "
                             "forced")
        if args.backend is None:
            if args.kernels_dir is None:
                raise ValueError("--all-gpus needs --kernels-dir (or --backend cpu-ref)")
            if args.tile is not None or args.k is not None:
                raise ValueError("--all-gpus: --tile and --k come from the registry with the kernel")
        return
    if args.backend is None:
        raise ValueError("choose --backend (cpu-ref | so:<path>) or --all-gpus")
    for key, default in SINGLE_SHAPE.items():
        if getattr(args, key) is None:
            setattr(args, key, default)


def _guarded(name, fn, errors):
    def run():
        try:
            fn()
        except BaseException as e:
            errors[name] = e
            log.exception("thread %s died", name)
    return run


@dataclass
class App:
    backend: Backend
    stats: Stats
    miner: Miner
    pool: Pool | RelayPool
    srv: ThreadingHTTPServer
    threads: list
    errors: dict = field(default_factory=dict)
    call_timeout: float = CALL_TIMEOUT
    init_timeout: float = INIT_TIMEOUT

    def start(self):
        for t in self.threads:
            t.start()

    def stalled(self) -> float | None:
        """Seconds the kernel call in flight has run past its timeout, None if it has not."""
        started = self.miner.call_started
        if started is None:
            return None
        limit = self.init_timeout if self.miner.calls == 0 else self.call_timeout
        took = time.monotonic() - started
        return took if took > limit else None

    def wait(self, stop: threading.Event, duration: float = 0, tick: float = 1.0):
        """Until stop, duration or a failure; a fatal pool error or a dead thread -> SystemExit (non-zero): code 2
        for a kernel error, 3 for a kernel call that did not return in time."""
        deadline = time.time() + duration if duration else None
        last = time.time()
        failure, code = None, EXIT_FAIL
        try:
            while not stop.wait(tick):
                if self.pool.fatal:
                    failure = f"fatal: {self.pool.fatal}"
                    break
                dead = [t.name for t in self.threads if not t.is_alive()]
                if dead:
                    failure = f"thread died: {', '.join(dead)}"
                    err = self.errors.get("mining")
                    if isinstance(err, KernelError):
                        failure, code = f"kernel error: {err}", EXIT_KERNEL
                    break
                took = self.stalled()
                if took is not None:
                    failure, code = f"kernel call {self.miner.calls + 1} has not returned in {took:.0f}s", EXIT_STALLED
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
            raise SystemExit(failure if code == EXIT_FAIL else code)

    def shutdown(self):
        self.miner.stop()
        self.pool.stop()
        self.srv.shutdown()
        self.threads[0].join(5)   # the pool thread closes its log on the way out


def build(args, relay_out=None, relay_in: int = 0) -> App:
    backend = make_backend(args.backend)
    stats = Stats(backend.devices())
    miner = Miner(backend, None, stats, k=args.k, m=args.m, n=args.n, portion_rows=args.portion_rows,
                  matrices=args.matrices, nbits_override=args.nbits_override, dry_run=args.dry_run, tile=args.tile,
                  nonce_lane=args.nonce_lane, nonce_lanes=args.nonce_lanes)
    if args.relay:
        pool = RelayPool(miner.on_job, miner.clear_job, relay_in, relay_out)
        srv = serve(args.api_port, args.api_host, lambda: relay_summary(miner, stats, backend))
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        pool = Pool(args.pool, args.wallet, args.worker, miner.on_job, args.log_dir / f"pool-{stamp}.jsonl",
                    tls_name=args.tls_name, idle_timeout=args.idle_timeout, on_lost=miner.clear_job)
        telemetry, stage = Telemetry(), StageClock()
        srv = serve(args.api_port, args.api_host, lambda: summary(miner, pool, stats, backend, telemetry, stage))
    miner.pool = pool
    errors = {}
    threads = [threading.Thread(target=_guarded("pool", pool.run, errors), name="pool", daemon=True),
               threading.Thread(target=_guarded("mining", miner.run, errors), name="mining", daemon=True),
               threading.Thread(target=_guarded("submit", miner.run_submitter, errors), name="submit", daemon=True)]
    return App(backend, stats, miner, pool, srv, threads, errors, args.call_timeout, args.init_timeout)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if args.all_gpus:
        from .supervisor import Supervisor
        Supervisor(args).run(stop, args.duration)
        return
    out = protocol_stdout() if args.relay else None
    app = build(args, relay_out=out)
    log.info("miner %s backend %s dry_run=%s api :%d k=%d m=%d n=%d tile %s%s", VERSION, app.backend.name,
             args.dry_run, app.srv.server_address[1], args.k, args.m, args.n, args.tile,
             f" relay lane {args.nonce_lane}/{args.nonce_lanes}" if args.relay else "")
    app.start()
    if args.relay:
        app.pool.send({"ready": {"api_port": app.srv.server_address[1]}})
    try:
        app.wait(stop, args.duration)
    except SystemExit as e:
        if e.code == EXIT_STALLED:   # the mining thread is stuck inside the library: do not wait for it
            logging.shutdown()
            os._exit(EXIT_STALLED)
        raise


if __name__ == "__main__":
    main()
