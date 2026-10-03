"""Job loop: pool job -> operands -> portions of work to the kernel -> share -> own verify -> submit.

Per job (header): MiningConfiguration (r=128, tile 8x16 of s0 §6, k chosen), job_key, B and its root,
seed_B, E_B, B'^T — B' stays fixed for the whole job (s0 §5).
Per pass (job, nonce): A with the nonce written into its first row, root of A, seed_A, E_A, A'.
A pass is all tiles of C' = A'·B', split into portions of row tiles for the backend; between portions the loop
looks at the latest job: a new job means the pass is dropped; candidates of a job whose height is no longer
current are stale and are not submitted. A = B = 0 (plus the nonce) by default; values stay in [-64, 64].
The mining thread only queues candidates; the submit thread builds PlainProof (Merkle tree of B^T once per job,
of A once per pass), verifies it against the exact share target and submits.
"""
import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

import pearl_ref as R
from .kernel import Backend, Candidate, Operands
from .pool import Job, PoolError

log = logging.getLogger("jobs")

ROWS_8 = [0, 8, 16, 24, 32, 40, 48, 56]                      # s0 §6: warp tile 64x64 of mma.sync m16n8
COLS_16 = [c + d for c in range(0, 64, 8) for d in (0, 1)]   # [0, 1, 8, 9, ..., 56, 57]
NONCE_DIGITS = 8                                              # base-129 digits in A[0, :8] -> 129^8 passes per job


def mining_config(k: int, r: int = 128) -> R.Config:
    return R.Config(k, r, R.Pattern.from_list(ROWS_8), R.Pattern.from_list(COLS_16))


def nonce_digits(nonce: int) -> np.ndarray:
    d = []
    for _ in range(NONCE_DIGITS):
        nonce, x = divmod(nonce, 129)
        d.append(x - 64)
    if nonce:
        raise OverflowError("nonce space of the job exhausted")
    return np.array(d, dtype=np.int8)


@dataclass
class JobWork:
    job: Job
    cfg: R.Config
    jk: bytes
    B: np.ndarray        # int8 (k, n)
    Bt: np.ndarray       # int8 (n, k)
    hb: bytes
    seed_b: bytes
    bt_noised: np.ndarray
    target: int          # exact share target (a pool target need not have a compact form)
    bound: int
    bt_tree: R.Tree | None = None   # built by the submit thread at the job's first candidate


@dataclass
class PassWork:
    jw: JobWork
    A: np.ndarray
    ops: Operands
    a_tree: R.Tree | None = None    # built by the submit thread at the pass's first candidate


class Stats:
    def __init__(self, devices: list[dict], window: float = 60.0):
        self.started = time.time()
        self.window = window
        self.lock = threading.Lock()
        self.accepted = self.rejected = self.stale = 0
        self.found = self.dropped = 0
        self.dev = {d["id"]: {"accepted": 0, "rejected": 0, "compute_errors": 0} for d in devices}
        self.samples = collections.deque()   # (ts, device, macs)

    def add_macs(self, device: int, macs: int):
        now = time.time()
        with self.lock:
            self.samples.append((now, device, macs))
            while self.samples and self.samples[0][0] < now - self.window:
                self.samples.popleft()

    def hashrate(self, device: int | None = None) -> float:
        now = time.time()
        span = min(self.window, max(now - self.started, 1e-9))
        with self.lock:
            macs = sum(m for ts, d, m in self.samples if ts >= now - span and (device is None or d == device))
        return macs / span

    def count(self, what: str, device: int | None = None, n: int = 1):
        with self.lock:
            setattr(self, what, getattr(self, what) + n)
            if device is not None and what in ("accepted", "rejected"):
                self.dev[device][what] += 1

    def compute_error(self, device: int):
        with self.lock:
            self.dev[device]["compute_errors"] += 1


class Miner:
    def __init__(self, backend: Backend, pool, stats: Stats, k: int = 2048, m: int = 1024, n: int = 1024,
                 portion_rows: int = 256, matrices: str = "zero", nbits_override: int | None = None,
                 dry_run: bool = False, seed: int = 0):
        self.cfg = mining_config(k)
        self.cfg.sanity(m, n)
        for dim, name, per in ((m, "m", self.cfg.rows.period), (n, "n", self.cfg.cols.period)):
            if dim % per:
                raise ValueError(f"{name}={dim} must be a multiple of the pattern period {per}")
        if portion_rows <= 0 or portion_rows % self.cfg.rows.period:
            raise ValueError(f"portion_rows={portion_rows} must be a positive multiple of {self.cfg.rows.period}")
        if matrices not in ("zero", "random"):
            raise ValueError("matrices: zero | random")
        self.backend, self.pool, self.stats = backend, pool, stats
        self.m, self.n = m, n
        self.row_part, self.col_part = self.cfg.rows.partition(m), self.cfg.cols.partition(n)
        self.portion_tiles = portion_rows // self.cfg.h
        self.matrices = matrices
        self.rng = np.random.default_rng(seed)
        self.nbits_override = nbits_override
        self.dry_run = dry_run
        self._latest: Job | None = None
        self._job_cv = threading.Condition()
        self._shares: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._ids = 0

    # ------------------------------------------------------------------ jobs in

    def on_job(self, job: Job):
        with self._job_cv:
            prev = self._latest
            self._latest = job
            self._job_cv.notify_all()
        log.info("job %s height %d nbits %#010x diff %.0f%s", job.job_id, job.height, job.nbits, job.difficulty,
                 "  <- new block" if prev and prev.height != job.height else "")

    def latest(self) -> Job | None:
        with self._job_cv:
            return self._latest

    def is_stale(self, job: Job) -> bool:
        cur = self.latest()
        return cur is None or cur.height != job.height or cur.conn != job.conn

    def share_target(self, job: Job) -> int:
        return R.nbits_to_target(self.nbits_override) if self.nbits_override is not None else job.target

    def stop(self):
        self._stop.set()
        with self._job_cv:
            self._job_cv.notify_all()
        self._shares.put(None)

    # ------------------------------------------------------------------ operands

    def _next_id(self):
        self._ids += 1
        return self._ids

    def prepare_job(self, job: Job) -> JobWork:
        cfg, k = self.cfg, self.cfg.k
        jk = R.job_key(job.header, cfg)
        if self.matrices == "random":
            B = self.rng.integers(-64, 65, (k, self.n), dtype=np.int8)
        else:
            B = np.zeros((k, self.n), np.int8)
        Bt = np.ascontiguousarray(B.T)
        hb = R.matrix_root(Bt, jk)
        _, hb_salted = R.salt_roots(bytes(32), hb, self.m, self.n)
        seed_b = R.H(jk + hb_salted)
        pb, qb = R.sparse_factor(seed_b, R.LABEL_B, k, cfg.r)
        e_brt = R.uniform_factor(seed_b, R.LABEL_B, np.arange(self.n), cfg.r)
        bt_noised = (Bt.astype(np.int16) + e_brt[:, pb] - e_brt[:, qb]).astype(np.int8)
        target = self.share_target(job)
        bound = min(target * cfg.h * cfg.w * cfg.L, R.U256_MAX)
        return JobWork(job, cfg, jk, B, Bt, hb, seed_b, bt_noised, target, bound)

    def prepare_pass(self, jw: JobWork, nonce: int, b_id: int):
        cfg, k = self.cfg, self.cfg.k
        if self.matrices == "random":
            A = self.rng.integers(-64, 65, (self.m, k), dtype=np.int8)
        else:
            A = np.zeros((self.m, k), np.int8)
        A[0, :NONCE_DIGITS] = nonce_digits(nonce)
        ha = R.matrix_root(A, jw.jk)
        ha_salted, _ = R.salt_roots(ha, bytes(32), self.m, self.n)
        seed_a = R.H(jw.seed_b + ha_salted)
        pa, qa = R.sparse_factor(seed_a, R.LABEL_A, k, cfg.r)
        e_al = R.uniform_factor(seed_a, R.LABEL_A, np.arange(self.m), cfg.r)
        a_noised = (A.astype(np.int16) + e_al[:, pa] - e_al[:, qa]).astype(np.int8)
        ops = Operands(cfg, a_noised, jw.bt_noised, seed_a, jw.bound, self._next_id(), b_id,
                       self.row_part, self.col_part)
        return PassWork(jw, A, ops)

    # ------------------------------------------------------------------ mining thread

    def run(self):
        jw, b_id, nonce = None, 0, 0
        while not self._stop.is_set():
            with self._job_cv:
                while self._latest is None and not self._stop.is_set():
                    self._job_cv.wait()
                job = self._latest
            if job is None:
                return
            if jw is None or jw.job is not job:
                t0 = time.time()
                jw, b_id, nonce = self.prepare_job(job), self._next_id(), 0
                log.debug("job %s prepared in %.3fs", job.job_id, time.time() - t0)
            pw = self.prepare_pass(jw, nonce, b_id)
            nonce += 1
            self._run_pass(pw)

    def _run_pass(self, pw: PassWork):
        total = pw.ops.row_tiles
        for lo in range(0, total, self.portion_tiles):
            if self._stop.is_set() or self.latest() is not pw.jw.job:
                return
            res = self.backend.search(pw.ops, lo, min(lo + self.portion_tiles, total))
            self.stats.add_macs(res.device, res.macs)
            if res.dropped:
                self.stats.count("dropped", n=res.dropped)
                log.warning("%d candidates over the backend cap %d dropped (job %s)", res.dropped,
                            self.backend.cap, pw.jw.job.job_id)
            for cand in res.candidates:
                self._on_candidate(pw, cand, res.device)

    def _on_candidate(self, pw: PassWork, cand: Candidate, device: int):
        self.stats.count("found")
        if self.is_stale(pw.jw.job):
            self.stats.count("stale")
            log.info("share of job %s dropped: height changed", pw.jw.job.job_id)
            return
        self._shares.put((pw, cand, device))

    # ------------------------------------------------------------------ submit thread

    def run_submitter(self):
        while True:
            item = self._shares.get()
            if item is None or self._stop.is_set():
                return
            self.handle_share(*item)

    def build_proof(self, pw: PassWork, cand: Candidate) -> R.PlainProof:
        jw, k = pw.jw, self.cfg.k
        if jw.bt_tree is None:
            jw.bt_tree = R.matrix_tree(jw.Bt, jw.jk)
        if pw.a_tree is None:
            pw.a_tree = R.matrix_tree(pw.A, jw.jk)
        return R.PlainProof(self.m, self.n, k, self.cfg.r, R.tree_proof(pw.a_tree, self.row_part[cand.row_tile], k),
                            R.tree_proof(jw.bt_tree, self.col_part[cand.col_tile], k))

    def handle_share(self, pw: PassWork, cand: Candidate, device: int):
        job = pw.jw.job
        if self.is_stale(job):
            self.stats.count("stale")
            log.info("share of job %s dropped before proof: height changed", job.job_id)
            return
        proof = self.build_proof(pw, cand)
        ok, msg = R.verify(job.header, proof, target_override=pw.jw.target)
        if not ok:
            self.stats.compute_error(device)
            log.error("share of job %s failed own verify: %s", job.job_id, msg)
            return
        if self.is_stale(job):
            self.stats.count("stale")
            log.info("share of job %s dropped before submit: height changed", job.job_id)
            return
        b64 = proof.to_base64()
        if self.dry_run:
            log.info("DRY-RUN share ready: job %s height %d jackpot %s proof %d B base64 (not submitted)",
                     job.job_id, job.height, cand.jackpot[::-1].hex()[:16], len(b64))
            return
        try:
            reply = self.pool.submit(job.job_id, b64)
        except PoolError as e:  # lost on the way (no connection, no reply): not the pool's verdict
            self.stats.count("stale")
            log.error("submit of job %s failed: %s", job.job_id, e)
            return
        if reply.get("result") is True and not reply.get("error"):
            self.stats.count("accepted", device)
            log.info("share accepted: job %s", job.job_id)
        else:
            err = str(reply.get("error") or reply.get("result"))
            self.stats.count("stale" if "stale" in err.lower() or "job not found" in err.lower() else "rejected",
                             device)
            log.warning("share rejected: job %s: %s", job.job_id, err)
