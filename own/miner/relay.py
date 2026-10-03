"""Pipe between the supervisor (--all-gpus) and a card process (--relay), JSON lines:

    supervisor -> card   {"job": {...} | null}        every pool job; null -- the pool connection is lost
                         {"reply": id, "result", "error"} | {"reply": id, "lost": why}   to a submit
    card -> supervisor   {"ready": {"api_port": p}}    once its /summary listens
                         {"submit": id, "job_id", "plain_proof"}

The pool connection is the supervisor's, one per server (worker c<server_id>); a card process only mines. Its stdout
carries only this protocol: protocol_stdout() moves fd 1 to stderr, so a kernel library's prints cannot break it."""
import json
import os
import select
import threading
import time

from .pool import Job, PoolError

POLL = 0.2


def job_to_wire(job: Job | None) -> dict | None:
    if job is None:
        return None
    return {"job_id": job.job_id, "header": job.header.hex(), "target": f"{job.target:x}", "height": job.height,
            "cert_version": job.cert_version, "conn": job.conn, "received": job.received}


def job_from_wire(d: dict | None) -> Job | None:
    if d is None:
        return None
    return Job(str(d["job_id"]), bytes.fromhex(d["header"]), int(d["target"], 16), int(d["height"]),
               int(d["cert_version"]), int(d["conn"]), float(d["received"]))


def protocol_stdout():
    """The real stdout as a binary stream for the protocol; fd 1 from now on writes to stderr."""
    fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(fd, "wb", buffering=0)


def encode(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


class RelayPool:
    """The pool of a card process: jobs and submit replies from the supervisor on in_fd, submits to it on out.
    Same face as pool.Pool for Miner, App and /summary: on_job, submit, run, stop, connected, fatal, url."""
    url = "relay"
    rejected = False

    def __init__(self, on_job, on_lost, in_fd: int, out, reply_timeout: float = 60.0):
        self.on_job, self.on_lost = on_job, on_lost
        self.connected = True
        self.fatal: Exception | None = None
        self.reply_timeout = reply_timeout
        self._in, self._out = in_fd, out
        self._out_lock = threading.Lock()
        self._pending: dict[int, dict | None] = {}
        self._cv = threading.Condition()
        self._next = 1
        self._stop = threading.Event()

    def send(self, obj: dict):
        with self._out_lock:
            self._out.write(encode(obj))
            self._out.flush()

    def status(self) -> dict:
        return {"url": self.url, "connected": self.connected}

    def submit(self, job_id: str, plain_proof_b64: str) -> dict:
        with self._cv:
            rid = self._next
            self._next += 1
            self._pending[rid] = None
        try:
            self.send({"submit": rid, "job_id": job_id, "plain_proof": plain_proof_b64})
            deadline = time.monotonic() + self.reply_timeout
            with self._cv:
                while self._pending[rid] is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise PoolError(f"submit: no reply from the supervisor in {self.reply_timeout:.0f}s")
                    self._cv.wait(left)
                msg = self._pending[rid]
        finally:
            with self._cv:
                self._pending.pop(rid, None)
        if "lost" in msg:
            raise PoolError(msg["lost"])
        return {"result": msg.get("result"), "error": msg.get("error")}

    def stop(self):
        self._stop.set()

    def run(self):
        """Read the supervisor's lines until stop(); its end of the pipe closing is fatal (the card stops)."""
        buf = b""
        try:
            while not self._stop.is_set():
                if not select.select([self._in], [], [], POLL)[0]:
                    continue
                chunk = os.read(self._in, 1 << 16)
                if not chunk:
                    if not self._stop.is_set():
                        self.fatal = PoolError("the supervisor closed the pipe")
                    return
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    if line.strip():
                        self._dispatch(json.loads(line))
        except Exception as e:
            self.fatal = e
        finally:
            self.connected = False

    def _dispatch(self, msg: dict):
        if "job" in msg:
            job = job_from_wire(msg["job"])
            self.on_job(job) if job is not None else self.on_lost()
        elif "reply" in msg:
            with self._cv:
                if msg["reply"] in self._pending:
                    self._pending[msg["reply"]] = msg
                    self._cv.notify_all()
        else:
            raise PoolError(f"unknown message from the supervisor: {msg!r}")
