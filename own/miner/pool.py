"""Pool client, HeroMiners Stratum `Pearl` (s1/README.md): JSON-RPC lines, params are objects.

    -> mining.subscribe {"agent"} ; mining.authorize {"wallet": "prl1….worker"}
    <- mining.notify {job_id, header (76 B hex), target (BE hex, share target), height, cert_version}
    -> mining.submit {"job_id", "plain_proof": base64(bincode PlainProof)}

TLS: the certificate is issued for pearl.herominers.com, the node host (de.…) may differ -> the name to check
is set apart from the host. Every line in both directions goes to a jsonl log {"ts","utc","dir","conn","raw"}.
No line from the pool for idle_timeout (HM sends a job every ~35 s) -> the connection is dropped and made again.

Nodes: one or more official URLs of the same pool (`--pool url,url` or `hm` -- every HeroMiners node of FACTS.md).
Before each connection the nodes are probed by a TCP connect to the pool port and the fastest not demoted one is taken;
a node that answers fewer than ACK_FLOOR of the last ACK_WINDOW submits is demoted for DEMOTE_FOR and the connection
moves to the next one (RU -> de lost a third of the shares that a ping did not show, FACTS.md). No node reachable ->
`unreachable` and `last_error` "pool_unreachable: …"; the miner never looks for a relay or a proxy around it.
A failed connection or a rejected authorize is not fatal: the pause before the next try doubles up to MAX_PAUSE.
"""
import collections
import concurrent.futures
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
HM_NODES = ("de", "fi", "ru", "tr", "us", "us2", "ca", "br", "sg", "hk", "au")   # Pearl nodes on 02.10 (FACTS.md)
HM_PORT = 1200
PROBE_TIMEOUT = 3.0   # s, TCP connect of a node probe
ACK_WINDOW = 20       # last submits judged per node
ACK_MIN = 10          # judged only with this many
ACK_FLOOR = 0.8       # answered share below it -> next node
DEMOTE_FOR = 1800.0   # s a node stays at the back of the list
MAX_PAUSE = 600.0     # s, the longest pause between failed connections
REJECTS_KEPT = 10


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


def hm_urls() -> list[str]:
    return [f"stratum+ssl://{TLS_NAME}:{HM_PORT}"] + [f"stratum+ssl://{n}.{TLS_NAME}:{HM_PORT}" for n in HM_NODES]


def parse_pool_arg(s: str) -> list[str]:
    """`hm` -> every HeroMiners node; otherwise comma-separated stratum URLs (each checked by parse_url)."""
    urls = hm_urls() if s.strip() == "hm" else [u.strip() for u in s.split(",") if u.strip()]
    if not urls:
        raise ValueError("no pool url")
    for u in urls:
        parse_url(u)
    if len(set(urls)) != len(urls):
        raise ValueError(f"pool urls repeat: {s!r}")
    return urls


def probe(url: str, timeout: float = PROBE_TIMEOUT) -> float:
    """TCP connect time to the pool port, s; OSError when it does not connect."""
    host, port, _ = parse_url(url)
    t0 = time.monotonic()
    socket.create_connection((host, port), timeout=timeout).close()
    return time.monotonic() - t0


class Pool:
    """Connection thread: choose a node, connect, authorize, read lines; notify -> on_job(Job); a lost connection ->
    on_lost(); reconnect after a pause that doubles while connections fail.

    submit() is called from other threads and waits for the reply with its id. run() returns on stop() or with
    self.fatal set: an unexpected exception (a bug) in the connection or reader thread. status() -> /summary "pool"."""

    def __init__(self, urls, wallet: str, worker: str, on_job, log_path: Path,
                 tls_name: str = TLS_NAME, reconnect_pause: float = 5.0, reply_timeout: float = 30.0,
                 idle_timeout: float = IDLE_TIMEOUT, on_lost=None, max_pause: float = MAX_PAUSE):
        if idle_timeout <= 0:
            raise ValueError(f"idle_timeout must be positive, got {idle_timeout}")
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        for u in self.urls:
            parse_url(u)
        self.url = self.urls[0]
        self.host, self.port, self.tls = parse_url(self.url)
        self.login = f"{wallet}.{worker}"
        self.on_job = on_job
        self.on_lost = on_lost
        self.tls_name = tls_name
        self.reconnect_pause = reconnect_pause
        self.max_pause = max_pause
        self.reply_timeout = reply_timeout
        self.idle_timeout = idle_timeout
        self.connected = False
        self.conn = 0
        self.fatal: Exception | None = None
        # what /summary shows of the pool
        self.last_error: str | None = None
        self.authorize_error: str | None = None
        self.rejected = False          # the last authorize was rejected
        self.unreachable = False       # the last try reached no node
        self.rejects = 0
        self.last_rejects = collections.deque(maxlen=REJECTS_KEPT)
        self.submits = self.replies = self.switches = 0
        self.latency: dict[str, float | None] = {}
        self._acks = collections.deque(maxlen=ACK_WINDOW)
        self._demoted: dict[str, float] = {}
        self._fails = 0                # connections in a row that did not get authorized
        self._authorized = False
        self._switching = False        # the session ends because _ack moved off its node: keep that last_error
        self._state_lock = threading.Lock()
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

    # ------------------------------------------------------------------ nodes

    def _choose(self) -> str:
        """The node to connect to: the only one, or the fastest reachable not demoted (demoted ones last)."""
        if len(self.urls) == 1:
            return self.urls[0]
        errors = {}
        with concurrent.futures.ThreadPoolExecutor(len(self.urls)) as ex:
            futures = {u: ex.submit(probe, u) for u in self.urls}
            for u, f in futures.items():
                try:
                    self.latency[u] = f.result()
                except OSError as e:
                    self.latency[u], errors[u] = None, str(e)
        now = time.time()
        alive = [u for u in self.urls if self.latency[u] is not None]
        if not alive:
            raise PoolError("pool_unreachable: " + "; ".join(f"{parse_url(u)[0]}: {e}" for u, e in errors.items()))
        return min(alive, key=lambda u: (self._demoted.get(u, 0) > now, self.latency[u]))

    def _ack(self, node: str, answered: bool):
        with self._state_lock:
            if node != self.url:
                return
            self._acks.append(answered)
            ratio = sum(self._acks) / len(self._acks)
            if len(self.urls) < 2 or len(self._acks) < ACK_MIN or ratio >= ACK_FLOOR:
                return
            self._demoted[node] = time.time() + DEMOTE_FOR
            self.switches += 1
            self._acks.clear()
            self.last_error = f"node {parse_url(node)[0]} answered {ratio:.0%} of the last submits: next node"
            self._switching = True
        log.warning("pool: %s", self.last_error)
        self._write("#", self.last_error)
        self._drop()

    def _drop(self):
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # ------------------------------------------------------------------ public

    def status(self) -> dict:
        with self._state_lock:
            ratio = sum(self._acks) / len(self._acks) if self._acks else None
            return {"url": self.url, "node": parse_url(self.url)[0], "connected": bool(self.connected),
                    "last_error": self.last_error, "authorize_error": self.authorize_error,
                    "rejected": self.rejected, "unreachable": self.unreachable,
                    "rejects": {"count": self.rejects, "last": list(self.last_rejects)},
                    "submits": self.submits, "replies": self.replies, "submit_ack_ratio": ratio,
                    "switches": self.switches, "reconnects": max(self.conn - 1, 0)}

    def _reject(self, text: str):
        with self._state_lock:
            self.rejects += 1
            self.last_rejects.append(text)

    def submit(self, job_id: str, plain_proof_b64: str) -> dict:
        """-> the pool reply {"id","result","error"}; raises PoolError when not connected or no reply."""
        if not self.connected:
            raise PoolError("not connected")
        node = self.url
        with self._state_lock:
            self.submits += 1
        try:
            reply = self._request("mining.submit", {"job_id": job_id, "plain_proof": plain_proof_b64},
                                  self.reply_timeout)
        except PoolError:
            self._ack(node, False)
            raise
        with self._state_lock:
            self.replies += 1
        self._ack(node, True)
        if reply.get("error") or reply.get("result") is not True:
            self._reject(f"submit: {json.dumps(reply.get('error') or reply.get('result'))}")
        return reply

    def stop(self):
        self._stop.set()
        self._drop()

    def pause(self) -> float:
        """Before the next connection: the base pause, doubling with every connection in a row not authorized."""
        if self._fails == 0:
            return self.reconnect_pause
        return min(self.reconnect_pause * 2 ** (self._fails - 1), self.max_pause)

    def run(self):
        """Connection loop; returns on stop() or with self.fatal set."""
        try:
            while not self._stop.is_set():
                self._authorized = False
                try:
                    self._session()
                except PoolError as e:
                    if not self._switching:
                        self.last_error = str(e)
                    self._switching = False
                    log.warning("pool: %s", e)
                except OSError as e:
                    self.last_error = f"{self.host}:{self.port}: {e}"
                    log.warning("pool: connection error %s", self.last_error)
                except Exception as e:
                    self.fatal = e
                    log.exception("pool: unexpected error, giving up")
                    return
                finally:
                    self._close()
                self._fails = 0 if self._authorized else self._fails + 1
                pause = self.pause()
                if self._fails > 1:
                    log.warning("pool: %d failed connections in a row, next try in %.0fs", self._fails, pause)
                if self._stop.wait(pause):
                    break
        finally:
            self._log.close()

    def _close(self):
        was = self.connected
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
        if was and self.on_lost is not None:
            self.on_lost()

    def _session(self):
        self.conn += 1
        t0 = time.time()
        try:
            node = self._choose()
        except PoolError:
            self.unreachable = True
            raise
        with self._state_lock:
            if node != self.url:
                self._acks.clear()
            self.url = node
        self.host, self.port, self.tls = parse_url(node)
        try:
            self._sock = self._connect()
        except OSError:
            self.unreachable = True
            raise
        self.unreachable = False
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
                text = json.dumps(reply.get("error") or reply.get("result"))
                self.rejected, self.authorize_error = True, text
                self._reject(f"authorize: {text}")
                raise PoolError(f"authorize rejected: {text}")
            self.rejected, self.authorize_error = False, None
            self._authorized = True
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
                self.last_error = f"bad notify: {e}"
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
