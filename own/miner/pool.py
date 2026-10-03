"""Pool client, HeroMiners Stratum `Pearl` (s1/README.md): JSON-RPC lines, params are objects.

    -> mining.subscribe {"agent"} ; mining.authorize {"wallet": "prl1….worker"}
    <- mining.notify {job_id, header (76 B hex), target (BE hex, share target), height, cert_version}
    -> mining.submit {"job_id", "plain_proof": base64(bincode PlainProof)}

TLS: the certificate is issued for pearl.herominers.com, the node host (de.…) may differ -> the name to check
is set apart from the host. Every line in both directions goes to a jsonl log {"ts","utc","dir","conn","raw"}.
No line from the pool for idle_timeout (HM sends a job every ~35 s) -> the connection is dropped and made again.
"""
import datetime
import json
import logging
import select
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger("pool")

AGENT = "own-pearl-miner"
DIFF1 = 0xFFFF * 2 ** 208  # pool «diff 1» = bitcoin pdiff: HM target = DIFF1 / 2^21 exactly (s1 §4)
TLS_NAME = "pearl.herominers.com"
POLL = 0.2            # s, reader holds the io lock at most this long
SEND_TIMEOUT = 30.0
IDLE_TIMEOUT = 3 * 35.0   # s without a line from the pool: three missed HM jobs


class PoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    job_id: str
    header: bytes      # 76 bytes, as is into job_key
    target: int        # share target (without the h*w*L factor)
    height: int
    cert_version: int
    conn: int          # connection number: a job_id counter is per connection
    received: float

    @property
    def nbits(self) -> int:
        return struct.unpack_from("<I", self.header, 72)[0]

    @property
    def difficulty(self) -> float:
        return DIFF1 / self.target


def parse_notify(params: dict, conn: int = 0, received: float = 0.0) -> Job:
    if not isinstance(params, dict):
        raise PoolError(f"notify params must be an object: {params!r}")
    header = bytes.fromhex(params["header"])
    if len(header) != 76:
        raise PoolError(f"header is {len(header)} bytes, expected 76")
    target = int(params["target"], 16)
    if target <= 0:
        raise PoolError(f"bad target {params['target']}")
    cert = int(params["cert_version"])
    if cert != 3:
        raise PoolError(f"cert_version {cert} is not supported (only 3)")
    return Job(str(params["job_id"]), header, target, int(params["height"]), cert, conn, received)


def parse_url(url: str):
    """stratum+ssl://host:port | stratum+tcp://host:port -> (host, port, tls)."""
    u = urlsplit(url)
    if u.scheme not in ("stratum+ssl", "stratum+tls", "stratum+tcp") or not u.hostname or not u.port:
        raise ValueError(f"pool url must be stratum+ssl://host:port or stratum+tcp://host:port, got {url!r}")
    return u.hostname, u.port, u.scheme != "stratum+tcp"


class Pool:
    """Connection thread: connect, authorize, read lines; notify -> on_job(Job); reconnect after a pause.

    submit() is called from other threads and waits for the reply with its id. run() returns on stop() or with
    self.fatal set: authorize rejected or an unexpected exception (a bug) in the connection or reader thread."""

    def __init__(self, url: str, wallet: str, worker: str, on_job, log_path: Path,
                 tls_name: str = TLS_NAME, reconnect_pause: float = 5.0, reply_timeout: float = 30.0,
                 idle_timeout: float = IDLE_TIMEOUT):
        if idle_timeout <= 0:
            raise ValueError(f"idle_timeout must be positive, got {idle_timeout}")
        self.url = url
        self.host, self.port, self.tls = parse_url(url)
        self.login = f"{wallet}.{worker}"
        self.on_job = on_job
        self.tls_name = tls_name
        self.reconnect_pause = reconnect_pause
        self.reply_timeout = reply_timeout
        self.idle_timeout = idle_timeout
        self.connected = False
        self.conn = 0
        self.fatal: Exception | None = None
        self._sock = None
        # one SSL object must not be read and written from two threads at once: concurrent sendall/recv gave
        # SSLV3_ALERT_BAD_RECORD_MAC on atlas 03.10 -> every socket call goes under this lock, recv polls
        self._io_lock = threading.Lock()
        self._next_id = 10
        self._pending: dict[int, dict] = {}
        self._pending_cv = threading.Condition()
        self._stop = threading.Event()
        self._reader_stop = threading.Event()
        self._reader_error: Exception | None = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("a", encoding="utf-8")
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------ io

    def _write(self, direction: str, raw: str):
        ts = time.time()
        utc = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="milliseconds")
        with self._log_lock:
            self._log.write(json.dumps({"ts": ts, "utc": utc, "dir": direction, "conn": self.conn, "raw": raw}) + "\n")
            self._log.flush()

    def _connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=15)
        if self.tls:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=self.tls_name)
        sock.settimeout(POLL)
        return sock

    def _send(self, obj: dict):
        line = json.dumps(obj, separators=(",", ":"))
        with self._io_lock:
            if self._sock is None:
                raise PoolError("not connected")
            self._sock.settimeout(SEND_TIMEOUT)
            try:
                self._sock.sendall(line.encode() + b"\n")
            finally:
                self._sock.settimeout(POLL)
        self._write(">", line)

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._pending_cv:
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = None
        try:
            self._send({"id": rid, "method": method, "params": params})
            deadline = time.time() + timeout
            with self._pending_cv:
                while self._pending[rid] is None:
                    left = deadline - time.time()
                    if left <= 0:
                        raise PoolError(f"{method}: no reply in {timeout:.0f}s")
                    self._pending_cv.wait(left)
                return self._pending[rid]
        finally:
            with self._pending_cv:
                self._pending.pop(rid, None)

    # ------------------------------------------------------------------ public

    def submit(self, job_id: str, plain_proof_b64: str) -> dict:
        """-> the pool reply {"id","result","error"}; raises PoolError when not connected or no reply."""
        if not self.connected:
            raise PoolError("not connected")
        return self._request("mining.submit", {"job_id": job_id, "plain_proof": plain_proof_b64}, self.reply_timeout)

    def stop(self):
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(self):
        """Connection loop; returns on stop() or with self.fatal set."""
        try:
            while not self._stop.is_set():
                try:
                    self._session()
                except PoolError as e:
                    if self.fatal:
                        log.error("pool: %s", e)
                        return
                    log.warning("pool: %s", e)
                except OSError as e:
                    log.warning("pool: connection error %s", e)
                except Exception as e:
                    self.fatal = e
                    log.exception("pool: unexpected error, giving up")
                    return
                finally:
                    self._close()
                if self._stop.wait(self.reconnect_pause):
                    break
        finally:
            self._log.close()

    def _close(self):
        self.connected = False
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if not self._stop.is_set():
            self._write("#", "disconnected")
        with self._pending_cv:  # wake waiters; they time out on their own
            self._pending_cv.notify_all()

    def _session(self):
        self.conn += 1
        t0 = time.time()
        self._sock = self._connect()
        note = f"connected {self.host}:{self.port} tls={self.tls} in {time.time() - t0:.3f}s"
        if self.tls:
            note += f" {self._sock.version()} name={self.tls_name}"
        self._write("#", note)
        log.info("pool: %s", note)
        self._reader_stop.clear()
        self._reader_error = None
        reader = threading.Thread(target=self._read_loop, args=(self._sock,), name="pool-reader", daemon=True)
        reader.start()
        try:
            # subscribe is optional at HM (s1 §1) and once went unanswered for 30 s (atlas 03.10 14:06): send it
            # without waiting, as stratum_hm.py did, and wait only for authorize
            self._send({"id": 1, "method": "mining.subscribe", "params": {"agent": AGENT}})
            reply = self._request("mining.authorize", {"wallet": self.login}, self.reply_timeout)
            if reply.get("error") or reply.get("result") is not True:
                self.fatal = PoolError(f"authorize rejected: {json.dumps(reply)}")
                raise self.fatal
            self.connected = True
            log.info("pool: authorized as %s", self.login)
            reader.join()
        finally:
            # the reader must be gone before _close() closes the socket under it (select on fd -1)
            self._reader_stop.set()
            reader.join()
            if self._reader_error is not None:
                raise self._reader_error
        if not self._stop.is_set():
            raise PoolError("connection closed")

    def _read_loop(self, sock):
        buf = b""
        last_line = time.monotonic()
        try:
            while not self._stop.is_set() and not self._reader_stop.is_set():
                if time.monotonic() - last_line > self.idle_timeout:
                    self._write("#", f"no line from the pool for {self.idle_timeout:.0f}s: reconnect")
                    log.warning("pool: no line for %.0fs, reconnecting", self.idle_timeout)
                    return
                # wait for data outside the lock so that sends are not starved; pending() = bytes already
                # decrypted inside the SSL object
                if not (self.tls and sock.pending()) and not select.select([sock], [], [], POLL)[0]:
                    continue
                with self._io_lock:
                    try:
                        chunk = sock.recv(1 << 16)
                    except (socket.timeout, ssl.SSLWantReadError):
                        continue
                if not chunk:
                    self._write("#", "pool closed the connection")
                    return
                buf += chunk
                while b"\n" in buf:
                    raw, _, buf = buf.partition(b"\n")
                    text = raw.decode(errors="replace").strip()
                    if text:
                        last_line = time.monotonic()
                        self._write("<", text)
                        self._dispatch(text)
        except OSError as e:
            if not self._stop.is_set():
                self._write("#", f"read error {e}")
        except Exception as e:
            self._reader_error = e

    def _dispatch(self, text: str):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            log.warning("pool: not json: %s", text[:200])
            return
        if not isinstance(msg, dict):
            return
        method = msg.get("method")
        if method == "mining.notify":
            try:
                job = parse_notify(msg.get("params"), self.conn, time.time())
            except (PoolError, KeyError, ValueError) as e:
                log.error("pool: bad notify (%s): %s", e, text[:300])
                return
            self.on_job(job)
        elif method == "mining.ping" and msg.get("id") is not None:
            self._send({"id": msg["id"], "result": "pong", "error": None})
        elif method is None and msg.get("id") is not None:
            with self._pending_cv:
                if msg["id"] in self._pending:
                    self._pending[msg["id"]] = msg
                    self._pending_cv.notify_all()
        else:
            log.info("pool: unhandled %s", text[:300])
