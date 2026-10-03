"""--all-gpus: one card process per nvidia-smi card, one pool connection for the server, one /summary.

Cards. The supervisor reads the cards from nvidia-smi and gives each its kernel by compute capability (registry.py);
a card without one is listed in /summary with "kernel": null and "error" and is not mined. A card process is
`miner.main --relay` with CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<index> and its nonce lane: it mines
and answers its own /summary on 127.0.0.1; jobs and submits go over its stdin/stdout (relay.py).

Worker and connection. One pool connection and one worker (--worker c<server_id>) for the whole server: collector
and the monitor read the pool by that worker (collector/parse/herominers.worker_server: c/v + a number, any other
name is not ours), so a worker per card would hide the server from them; a connection per card would multiply the
connections HM sees from one host and drop a card's connection on every restart of its process. The cards differ in
their nonce lanes (jobs.Miner), not in the pool's job.

Restarts. A card process that exits is restarted after --restart-pause; two crashes in a row on one kernel variant
exclude it and the next variant runs; --max-restarts crashes within --restart-window, or every variant excluded, turn
the card off (state kernel_crash). A kernel call that does not return (exit 3) -> stalled, own share check failures
(compute_errors) -> errors: off at once, no restart. With no card left the supervisor exits (code 1).

Tune. Variants: the library's pearl_variants export when it has one, else the registry's list (PEARL_VARIANT, read by
the library itself or set by pearl_set_variant). Each variant mines --tune-hold s in real work under the card's real
limits (restart with PEARL_VARIANT=<v>), the best H/s wins (under a power cap it is also the fewest J/TH), again every
--tune-every s. k_expected = the winner's MAC/(SM*clock); k_actual below 0.8 of it for --slow-after s -> slow_card
(a shared card, a substitute or a hidden ceiling: the monitor decides). The card's mode (clocks, power limit, fans)
is never touched.
"""
import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pearl_ref as R
from . import VERSION
from .api import StageClock, Telemetry, card_fields, k_of, serve
from .main import EXIT_KERNEL, EXIT_STALLED, SINGLE_SHAPE
from .pool import DIFF1, Pool, PoolError
from .registry import NoKernel, kernel_for
from .relay import encode, job_to_wire

log = logging.getLogger("supervisor")

ROOT = Path(__file__).resolve().parent.parent      # where `python -m miner.main` runs
COUNTERS = ("accepted", "rejected", "stale", "dropped", "compute_errors")
TERMINAL = ("no_kernel", "errors", "stalled", "kernel_crash")
SWITCH_AFTER = 2         # crashes in a row on one variant before the next variant
SLOW_RATIO = 0.8
READY_TIMEOUT = 120.0    # s from spawn to the card process's "ready"
STOP_TIMEOUT = 10.0      # s from SIGTERM to SIGKILL
WINDOW = 60.0            # s, the hashrate window of a card process (jobs.Stats)


class Card:
    def __init__(self, t: dict, lane: int, spec=None, lib=None, error=None):
        self.index, self.name, self.pci = t["index"], t["name"], t["pci_bus_id"]
        self.lane, self.spec, self.lib = lane, spec, lib
        self.error = error
        self.state = "no_kernel" if error else "run"     # run | restarting | TERMINAL
        self.flags = {"no_kernel"} if error else set()
        self.proc = self.err_thread = None
        self.api_port = self.child = self.sm_count = None
        self.base = dict.fromkeys(COUNTERS, 0)
        self.crashes: list[float] = []
        self.consecutive: dict[str | None, int] = {}
        self.bad: set[str | None] = set()
        self.restarts = 0
        self.last_line = None
        self.spawned_at = self.ready_at = self.restart_at = None
        self.variant: str | None = None
        self.variants: list[str] | None = list(spec.variants) if spec and spec.variants else None
        self.variants_learned = False
        self.tuning = False
        self.tune: dict[str, float | None] = {}
        self.k_tune: dict[str, float | None] = {}
        self.tune_queue: list[str] = []
        self.tuned_at = self.next_tune = None
        self.k_expected = spec.k_passport if spec else None
        self.slow_since = None
        self.stdin_lock = threading.Lock()

    def good_variants(self) -> list[str]:
        return [v for v in self.variants or [] if v not in self.bad]


class Supervisor:
    def __init__(self, args, telemetry: Telemetry | None = None, tick: float = 1.0):
        self.args, self.tick = args, tick
        self.tele = telemetry or Telemetry()
        cards, err = self.tele.get()
        if not cards:
            raise SystemExit(f"--all-gpus: no cards ({err or 'nvidia-smi lists none'})")
        self.cards = [self._card(t, lane) for lane, t in enumerate(cards)]
        if all(c.state in TERMINAL for c in self.cards):
            raise SystemExit("--all-gpus: no card has a kernel: " + "; ".join(f"gpu {c.index}: {c.error}"
                                                                            for c in self.cards))
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.pool = Pool(args.pool, args.wallet, args.worker, self.on_job, args.log_dir / f"pool-{stamp}.jsonl",
                         tls_name=args.tls_name, idle_timeout=args.idle_timeout, on_lost=self.on_lost)
        self.latest = None
        self.lock = threading.RLock()
        self.stage = StageClock()
        self.started = time.time()
        self.stopped = False
        self.srv = None

    def _card(self, t: dict, lane: int) -> Card:
        if self.args.backend == "cpu-ref":
            return Card(t, lane)
        try:
            spec, lib = kernel_for(t.get("compute_cap"), self.args.kernels_dir)
        except NoKernel as e:
            log.error("gpu %d %s: %s -- not mining it", t["index"], t["name"], e)
            return Card(t, lane, error=str(e))
        return Card(t, lane, spec, lib)

    # ------------------------------------------------------------------ pool -> cards

    def on_job(self, job):
        self.latest = job
        self._broadcast({"job": job_to_wire(job)})

    def on_lost(self):
        self.latest = None
        self._broadcast({"job": None})

    def _broadcast(self, msg):
        for card in self.cards:
            proc = card.proc
            if proc is not None and proc.poll() is None:
                self._send(card, proc, msg)

    def _send(self, card: Card, proc, msg: dict):
        with card.stdin_lock:
            try:
                proc.stdin.write(encode(msg))
                proc.stdin.flush()
            except OSError as e:     # the process is exiting: step() sees its exit
                log.info("gpu %d: pipe closed (%s)", card.index, e)

    def _submit(self, card: Card, proc, msg: dict):
        try:
            reply = self.pool.submit(msg["job_id"], msg["plain_proof"])
            out = {"reply": msg["submit"], "result": reply.get("result"), "error": reply.get("error")}
        except PoolError as e:
            out = {"reply": msg["submit"], "lost": str(e)}
        self._send(card, proc, out)

    # ------------------------------------------------------------------ card processes

    def _command(self, card: Card) -> list[str]:
        a = self.args
        if card.spec is None:      # --backend cpu-ref
            shape = {k: getattr(a, k) if getattr(a, k) is not None else v for k, v in SINGLE_SHAPE.items()}
            backend = "cpu-ref"
        else:
            s = card.spec
            shape = {"tile": s.tile, "k": s.k, "m": a.m or s.m, "n": a.n or s.n,
                     "portion_rows": a.portion_rows or s.portion_rows}
            backend = f"so:{card.lib}"
        cmd = [sys.executable, "-u", "-m", "miner.main", "--relay", "--backend", backend,
               "--tile", shape["tile"], "--k", str(shape["k"]), "--m", str(shape["m"]), "--n", str(shape["n"]),
               "--portion-rows", str(shape["portion_rows"]), "--matrices", a.matrices,
               "--api-host", "127.0.0.1", "--api-port", "0",
               "--nonce-lane", str(card.lane), "--nonce-lanes", str(len(self.cards)),
               "--call-timeout", str(a.call_timeout), "--init-timeout", str(a.init_timeout)]
        if a.dry_run:
            cmd.append("--dry-run")
        if a.nbits_override is not None:
            cmd += ["--nbits-override", hex(a.nbits_override)]
        if a.verbose:
            cmd.append("-v")
        return cmd

    def _spawn(self, card: Card):
        env = {k: v for k, v in os.environ.items() if k not in ("PEARL_DEVICE", "PEARL_VARIANT")}
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(card.index))
        if card.variant is not None:
            env["PEARL_VARIANT"] = card.variant
        proc = subprocess.Popen(self._command(card), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, cwd=ROOT)
        card.proc, card.api_port, card.child, card.last_line = proc, None, None, None
        card.spawned_at, card.ready_at, card.slow_since = time.time(), None, None
        card.state = "run"
        threading.Thread(target=self._read_out, args=(card, proc), name=f"g{card.index}-out", daemon=True).start()
        card.err_thread = threading.Thread(target=self._read_err, args=(card, proc), name=f"g{card.index}-err",
                                           daemon=True)
        card.err_thread.start()
        if self.latest is not None:
            self._send(card, proc, {"job": job_to_wire(self.latest)})
        log.info("gpu %d: card process %d started (kernel %s variant %s%s)", card.index, proc.pid,
                 card.spec.name if card.spec else "cpu-ref", card.variant, ", tuning" if card.tuning else "")

    def _read_out(self, card: Card, proc):
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                log.error("gpu %d: not a relay line: %r", card.index, line[:200])
                continue
            if "ready" in msg:
                with self.lock:
                    if card.proc is proc:
                        card.api_port, card.ready_at = int(msg["ready"]["api_port"]), time.time()
            elif "submit" in msg:
                threading.Thread(target=self._submit, args=(card, proc, msg), daemon=True).start()
            else:
                log.error("gpu %d: unknown relay message %r", card.index, msg)

    def _read_err(self, card: Card, proc):
        for line in proc.stderr:
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                card.last_line = text
                sys.stderr.write(f"[g{card.index}] {text}\n")

    def _poll(self, card: Card) -> bool:
        try:
            raw = urllib.request.urlopen(f"http://127.0.0.1:{card.api_port}/summary", timeout=2).read()
        except OSError as e:
            log.warning("gpu %d: card /summary: %s", card.index, e)
            return False
        card.child = json.loads(raw)
        card.sm_count = card.child["sm_count"] or card.sm_count
        return True

    def _fold(self, card: Card):
        """The counters of the card process that ends go into the card's base."""
        if card.child is not None:
            for k in COUNTERS:
                card.base[k] += card.child[k]
        card.child = None

    def _stop_child(self, card: Card):
        proc, card.proc = card.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                log.error("gpu %d: card process %d ignored SIGTERM for %.0fs: killed", card.index, proc.pid,
                          STOP_TIMEOUT)
                proc.kill()
                proc.wait()
        try:
            proc.stdin.close()
        except OSError:
            pass

    def _restart(self, card: Card, variant: str | None):
        """A planned restart (tune): last counters, stop, the new variant at once."""
        self._poll(card)
        self._stop_child(card)
        self._fold(card)
        card.variant = variant
        self._spawn(card)

    def _off(self, card: Card, state: str, flag: str, error: str):
        self._stop_child(card)
        self._fold(card)
        card.state, card.error, card.tuning = state, error, False
        card.flags.add(flag)
        log.error("gpu %d: %s", card.index, error)

    # ------------------------------------------------------------------ variants and tune

    def _start_tune(self, card: Card):
        card.tuning, card.tune, card.k_tune = True, {}, {}
        card.tune_queue = card.good_variants()
        card.variant = card.tune_queue.pop(0)

    def _finish_tune(self, card: Card, now: float) -> str | None:
        card.tuning, card.tuned_at, card.next_tune = False, now, now + self.args.tune_every
        measured = {v: hs for v, hs in card.tune.items() if hs is not None and v not in card.bad}
        if not measured:
            good = card.good_variants()
            return good[0] if good else None
        best = max(measured, key=measured.get)
        if card.k_tune.get(best) is not None:
            card.k_expected = card.k_tune[best]
        log.info("gpu %d: tune %s -> %s", card.index, {v: f"{hs:.3g}" for v, hs in measured.items()}, best)
        return best

    def _next_variant(self, card: Card, now: float) -> str | None:
        if card.tuning:
            while card.tune_queue:
                v = card.tune_queue.pop(0)
                if v not in card.bad:
                    return v
            return self._finish_tune(card, now)
        good = sorted(card.good_variants(), key=lambda v: -(card.tune.get(v) or 0.0))
        return good[0] if good else None

    def _learn_variants(self, card: Card):
        if card.variants_learned:
            return
        card.variants_learned = True
        exported = card.child.get("variants")
        if exported:
            if card.variants is not None and set(card.variants) != set(exported):
                log.warning("gpu %d: the library exports variants %s, the registry lists %s: the library's are used",
                            card.index, exported, card.variants)
            card.variants = list(exported)
        elif card.variants is None:
            card.variants = []
        if self.args.tune_hold > 0 and not card.tuning and card.tuned_at is None and len(card.good_variants()) >= 2:
            self._start_tune(card)
            self._restart(card, card.variant)

    def _tune(self, card: Card, now: float, t: dict | None):
        if card.tuning:
            if now - card.ready_at < self.args.tune_hold:
                return
            hs = card.child["hashrate"]
            card.tune[card.variant] = hs
            card.k_tune[card.variant] = k_of(hs, card.sm_count, (t or {}).get("core_clock_mhz"))
            nxt = self._next_variant(card, now)
            if nxt != card.variant:
                self._restart(card, nxt)
        elif (self.args.tune_hold > 0 and card.next_tune is not None and now >= card.next_tune
              and len(card.good_variants()) >= 2):
            self._start_tune(card)
            self._restart(card, card.variant)

    def _slow(self, card: Card, now: float, t: dict | None):
        warm = now - card.ready_at >= min(WINDOW, self.args.slow_after)
        if card.tuning or card.child["stage"] != "mining" or not warm or not card.k_expected:
            card.slow_since = None
            return
        k = k_of(card.child["hashrate"], card.sm_count, (t or {}).get("core_clock_mhz"))
        if k is not None and k < SLOW_RATIO * card.k_expected:
            card.slow_since = card.slow_since or now
            if now - card.slow_since >= self.args.slow_after and "slow_card" not in card.flags:
                card.flags.add("slow_card")
                log.warning("gpu %d: k %.4g < %.1f x expected %.4g for %.0fs: slow_card", card.index, k, SLOW_RATIO,
                            card.k_expected, now - card.slow_since)
        else:
            card.slow_since = None
            card.flags.discard("slow_card")

    # ------------------------------------------------------------------ the loop

    def _crash(self, card: Card, now: float, reason: str):
        card.crashes = [t for t in card.crashes if now - t <= self.args.restart_window] + [now]
        card.restarts += 1
        card.error = reason
        v = card.variant
        card.consecutive[v] = card.consecutive.get(v, 0) + 1
        log.error("gpu %d: card process crashed (variant %s, %d in a row): %s", card.index, v, card.consecutive[v],
                  reason)
        if card.tuning:
            card.tune[v] = None
        if card.consecutive[v] >= SWITCH_AFTER and card.variants:
            card.bad.add(v)
            nxt = self._next_variant(card, now)
            if nxt is None:
                self._off(card, "kernel_crash", "kernel_crash",
                          f"every kernel variant crashed {SWITCH_AFTER} times in a row; last: {reason}")
                return
            log.warning("gpu %d: variant %s crashed %d times in a row -> %s", card.index, v, SWITCH_AFTER, nxt)
            card.variant = nxt
        if len(card.crashes) >= self.args.max_restarts:
            self._off(card, "kernel_crash", "kernel_crash",
                      f"{len(card.crashes)} crashes in {self.args.restart_window:.0f}s; last: {reason}")
            return
        card.state, card.restart_at = "restarting", now + self.args.restart_pause

    def _exited(self, card: Card, rc: int, now: float):
        if card.err_thread is not None:
            card.err_thread.join(1)
        self._stop_child(card)
        self._fold(card)
        reason = f"exit code {rc}" + (f": {card.last_line}" if card.last_line else "")
        if rc == EXIT_STALLED:
            self._off(card, "stalled", "stalled", f"a kernel call did not return ({reason}): card off")
        else:
            self._crash(card, now, ("kernel error, " if rc == EXIT_KERNEL else "") + reason)

    def begin(self, card: Card):
        if self.args.tune_hold > 0 and len(card.good_variants()) >= 2:
            self._start_tune(card)
        self._spawn(card)

    def step(self, now: float):
        tele, _ = self.tele.get()
        by_index = {t["index"]: t for t in tele}
        for card in self.cards:
            if card.state in TERMINAL:
                continue
            if card.proc is None:
                if card.state == "restarting" and now >= card.restart_at:
                    self._spawn(card)
                continue
            rc = card.proc.poll()
            if rc is not None:
                self._exited(card, rc, now)
                continue
            if card.ready_at is None:
                if now - card.spawned_at > READY_TIMEOUT:
                    self._stop_child(card)
                    self._crash(card, now, f"no ready in {READY_TIMEOUT:.0f}s")
                continue
            if not self._poll(card):
                continue
            if card.child["stage"] == "mining":
                card.consecutive.pop(card.variant, None)
            errors = card.base["compute_errors"] + card.child["compute_errors"]
            if errors:
                self._off(card, "errors", "compute_errors", f"own share check failed {errors} times: card off")
                continue
            t = by_index.get(card.index)
            self._learn_variants(card)
            if card.child is not None:       # None: restarted just now for a tune
                self._tune(card, now, t)
            if card.child is not None:
                self._slow(card, now, t)
        if all(c.state in TERMINAL for c in self.cards):
            raise SystemExit("no card left mining: " + "; ".join(f"gpu {c.index} {c.state}: {c.error}"
                                                                 for c in self.cards))

    def run(self, stop: threading.Event, duration: float = 0):
        self.srv = serve(self.args.api_port, self.args.api_host, self.summary)
        pool_thread = threading.Thread(target=self.pool.run, name="pool", daemon=True)
        pool_thread.start()
        log.info("supervisor %s: %d cards, api :%d, worker %s", VERSION, len(self.cards), self.srv.server_address[1],
                 self.pool.login)
        with self.lock:
            for card in self.cards:
                if card.state not in TERMINAL:
                    self.begin(card)
        deadline = time.time() + duration if duration else None
        failure = None
        try:
            while not stop.wait(self.tick):
                if self.pool.fatal:
                    failure = f"fatal: {self.pool.fatal}"
                    break
                if deadline and time.time() >= deadline:
                    break
                with self.lock:
                    self.step(time.time())
        except SystemExit as e:
            failure = str(e.code)
        finally:
            self.shutdown(pool_thread)
        if failure:
            log.error(failure)
            raise SystemExit(failure)

    def shutdown(self, pool_thread=None):
        with self.lock:
            self.stopped = True
            for card in self.cards:
                if card.proc is not None and card.proc.poll() is None:
                    card.proc.terminate()
            for card in self.cards:
                self._stop_child(card)
                self._fold(card)
        self.pool.stop()
        if self.srv is not None:
            self.srv.shutdown()
        if pool_thread is not None:
            pool_thread.join(5)

    # ------------------------------------------------------------------ /summary

    def _shown(self, card: Card) -> str:
        if card.state != "run":
            return card.state
        if card.tuning:
            return "tuning"
        if card.ready_at is None or card.child is None:
            return "starting"
        return card.child["stage"]

    def _gpu(self, card: Card, t: dict | None, has_job: bool) -> dict:
        ch = card.child if card.proc is not None else None
        hs = ch["hashrate"] if ch and has_job else 0.0
        return {"id": card.index, "name": card.name, "pci_bus_id": card.pci, "sm_count": card.sm_count,
                "hashrate": hs, **{k: card.base[k] + (ch[k] if ch else 0) for k in COUNTERS},
                **card_fields(t, hs, card.sm_count),
                "k_expected": card.k_expected,
                "kernel": card.spec.name if card.spec else (None if card.error else "cpu-ref"),
                "variant": card.variant, "variants": card.variants, "bad_variants": sorted(map(str, card.bad)),
                "tune": dict(card.tune) or None, "tuned_at": int(card.tuned_at) if card.tuned_at else None,
                "state": self._shown(card), "error": card.error, "flags": sorted(card.flags),
                "restarts": card.restarts}

    def _stage(self, shown: list[str]) -> str:
        if self.stopped or all(s in TERMINAL for s in shown):
            return "stopped"
        if self.pool.rejected:
            return "rejected"
        for s in ("tuning", "mining", "no_job", "kernel_init"):
            if s in shown:
                return s
        return "starting"

    def summary(self) -> dict:
        tele, error = self.tele.get()
        by_index = {t["index"]: t for t in tele}
        job = self.latest
        with self.lock:
            gpus = [self._gpu(c, by_index.get(c.index), job is not None) for c in self.cards]
            st, since = self.stage(self._stage([g["state"] for g in gpus]))
        target = None
        if job is not None:
            target = R.nbits_to_target(self.args.nbits_override) if self.args.nbits_override is not None else job.target
        return {
            "miner": "own",
            "version": VERSION,
            "uptime": int(time.time() - self.started),
            "stage": st,
            "stage_since": int(since),
            "backend": "cpu-ref" if self.args.backend == "cpu-ref" else "registry",
            "pool": {**self.pool.status(), "difficulty": DIFF1 / target if target else None},
            "hashrate": sum(g["hashrate"] for g in gpus),
            **{k: sum(g[k] for g in gpus) for k in ("accepted", "rejected", "stale", "dropped")},
            "gpus": gpus,
            "gpu_telemetry_error": error,
            "cpu": None,
        }
