"""Kernel interface: the GEMM + transcript + jackpot part of mining, behind one Backend contract.

What stays on the CPU (jobs.py, per job / per pass): job_key, BLAKE3 roots of A and B^T, V3 salt, seeds,
noise E_A, E_B and the noised operands A' = A + E_A, B'^T = B^T + E_B (values in [-127, 127], int8);
building PlainProof (Merkle multi-proof), own verification, submit.

What the kernel does (`search`): for row tiles [lo, hi) x all column tiles of C' = A'·B'
  - running int32 sums over k in chunks of r, after each chunk XOR of the h*w sums of every tile, mixed into
    the tile transcript T[16] (u32): T[c % 16] = rotl13(T[c % 16]) ^ xor   (L / r chunks, L = k - k % r)
  - jackpot = keyed BLAKE3(T as 64 bytes LE, key = seed_A), 32 bytes
  - candidate if LE-uint256(jackpot) <= bound (bound = target * h*w*L, saturated at 2^256-1)
and returns the candidates plus the MAC count done (= tiles * h*w*L; one pool hash = one MAC).

GPU backend contract (ctypes .so or a subprocess speaking the same buffers), see README.md:
    int pearl_search(const int8_t *a, uint32_t m,     // A'  row-major m x k
                     const int8_t *bt, uint32_t n,    // B'^T row-major n x k  (column j of B' = row j)
                     uint32_t k, uint32_t r,
                     const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],   // PeriodicPattern bytes
                     const uint8_t seed_a[32], const uint8_t bound_le[32],
                     uint32_t row_tile_lo, uint32_t row_tile_hi,
                     cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);
    typedef struct { uint32_t row_tile, col_tile; uint8_t jackpot[32]; } cand_t;   // 40 bytes, packed
Tile numbering = Pattern.partition order: tile t has offset = t-th valid offset, rows offset + pattern;
the partition is computed once by the miner and travels in Operands (row_part, col_part).
Operands may stay resident: a_id / b_id change only when A' / B'^T change (B' is fixed for a job).
"""
import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
import platform

import numpy as np

import pearl_ref as R


@dataclass(frozen=True)
class Operands:
    cfg: R.Config
    a_noised: np.ndarray   # int8 (m, k), C-contiguous
    bt_noised: np.ndarray  # int8 (n, k), C-contiguous
    seed_a: bytes          # jackpot key
    bound: int             # u256
    a_id: int              # changes when A' changes (every pass)
    b_id: int              # changes when B'^T changes (every job)
    row_part: np.ndarray   # cfg.rows.partition(m): row indices of every row tile
    col_part: np.ndarray   # cfg.cols.partition(n)

    @property
    def m(self):
        return self.a_noised.shape[0]

    @property
    def n(self):
        return self.bt_noised.shape[0]

    @property
    def row_tiles(self):
        return len(self.row_part)


@dataclass
class Candidate:
    row_tile: int
    col_tile: int
    jackpot: bytes


@dataclass
class SearchResult:
    candidates: list = field(default_factory=list)
    macs: int = 0
    device: int = 0
    dropped: int = 0       # candidates over cap, not returned


class Backend:
    name = "abstract"
    kernel = "none"
    cap = 16  # candidates per search call; more are counted in SearchResult.dropped (a hit is ~2^-34 per tile)
    card_note = None  # why a GPU backend's device is not tied to an nvidia-smi card (its work then goes to "cpu")
    variants = None   # kernel variants the library exports (pearl_variants), None when it does not
    variant = None    # the variant asked for by PEARL_VARIANT, None = the library's own choice
    resident = False  # operands built on the device from job_key and the nonce (job / pass_ / search_resident)

    def devices(self) -> list[dict]:
        """[{"id", "name", "pci_bus_id", "sm_count", "nvidia_index"}]; nvidia_index -> nvidia-smi telemetry."""
        raise NotImplementedError

    def search(self, ops: Operands, row_tile_lo: int, row_tile_hi: int) -> SearchResult:
        raise NotImplementedError


class CpuRefBackend(Backend):
    """numpy reference (pearl_ref.transcripts): exact, slow; tests and oracle for a GPU backend."""
    name = "cpu-ref"
    kernel = "numpy-f64-blas"

    def devices(self):
        return [{"id": 0, "name": f"cpu {platform.machine()}", "pci_bus_id": None, "sm_count": None,
                 "nvidia_index": None}]

    def transcripts(self, ops: Operands, lo: int, hi: int) -> np.ndarray:
        """(hi-lo, col_tiles, 16) u32 — the oracle to compare a GPU kernel's transcripts with."""
        tiles = ops.row_part[lo:hi]
        rows = np.unique(tiles)
        local = np.searchsorted(rows, tiles)
        return R.transcripts(ops.a_noised[rows], ops.bt_noised, ops.cfg, local, ops.col_part)

    def search(self, ops, lo, hi):
        T = self.transcripts(ops, lo, hi)
        J = R.jackpot_hashes(T, ops.seed_a)
        res = SearchResult(macs=(hi - lo) * T.shape[1] * ops.cfg.h * ops.cfg.w * ops.cfg.L)
        for i, j in np.ndindex(J.shape[:2]):
            jp = bytes(J[i, j])
            if R.le_int(jp) <= ops.bound:
                if len(res.candidates) < self.cap:
                    res.candidates.append(Candidate(lo + i, j, jp))
                else:
                    res.dropped += 1
        return res


class CandT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("row_tile", ctypes.c_uint32), ("col_tile", ctypes.c_uint32), ("jackpot", ctypes.c_uint8 * 32)]


assert ctypes.sizeof(CandT) == 40


class KernelError(RuntimeError):
    """pearl_search returned non-zero, or its output breaks the contract."""

    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


_I8P = ctypes.POINTER(ctypes.c_int8)
_U8P = ctypes.POINTER(ctypes.c_uint8)
_U32P = ctypes.POINTER(ctypes.c_uint32)
RESIDENT_EXPORTS = ("pearl_job", "pearl_pass", "pearl_tree_nodes", "pearl_search_resident")


def _u8(b: bytes):
    return (ctypes.c_uint8 * len(b)).from_buffer_copy(b)


def pinned_card(env=os.environ) -> tuple[int | None, str | None]:
    """The nvidia-smi index of the one card this process runs on: (index, None), or (None, why it is unknown).

    One card per process: CUDA_VISIBLE_DEVICES=i (the library then sees it as device 0) or PEARL_DEVICE=i. CUDA
    numbers cards fastest-first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, which is the nvidia-smi order -- without it
    the index points at an unknown card, so it is required (ValueError)."""
    cvd, dev = env.get("CUDA_VISIBLE_DEVICES"), env.get("PEARL_DEVICE")
    if cvd is not None:
        items = [x.strip() for x in cvd.split(",") if x.strip()]
        if len(items) != 1:
            return None, f"CUDA_VISIBLE_DEVICES={cvd!r} is not one card"
        if not items[0].isdigit():
            return None, f"CUDA_VISIBLE_DEVICES={cvd!r} is not an nvidia-smi index"
        if dev not in (None, "0"):
            raise ValueError(f"CUDA_VISIBLE_DEVICES={cvd!r} leaves one device, PEARL_DEVICE={dev!r} must be 0 or unset")
        index, by = int(items[0]), "CUDA_VISIBLE_DEVICES"
    elif dev is not None:
        if not dev.strip().isdigit():
            raise ValueError(f"PEARL_DEVICE={dev!r} is not a device index")
        index, by = int(dev), "PEARL_DEVICE"
    else:
        return None, "neither CUDA_VISIBLE_DEVICES (one index) nor PEARL_DEVICE is set"
    if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError(f"{by}={index} needs CUDA_DEVICE_ORDER=PCI_BUS_ID (the nvidia-smi order), got "
                         f"{env.get('CUDA_DEVICE_ORDER')!r}")
    return index, None


class SoBackend(Backend):
    """A kernel in a shared library (README, "Контракт GPU-бэкенда"): pearl_search over ctypes, one device.

    Optional exports: int pearl_device_info(uint32_t *sm_count, char *name, uint32_t name_len) -- 0 = ok;
    without it sm_count is None and the device is named after the file. const char *pearl_last_error(void) -- the
    reason of the last non-zero return, added to KernelError; without it the error carries the code only.
    int pearl_variants(char *names, uint32_t len) -- 0 = ok, comma-separated variant names (the supervisor tunes over
    them). int pearl_set_variant(const char *name) -- 0 = ok, called at load with PEARL_VARIANT; a library without it
    reads PEARL_VARIANT itself (kernels/v100) or has no variants.
    The device is tied to its nvidia-smi card by pinned_card() (one card per process); otherwise card_note says why
    not and /summary counts its work under "cpu"."""
    name = "so"

    def __init__(self, path, cap: int = Backend.cap, env=os.environ):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"kernel library {self.path} not found")
        if cap <= 0:
            raise ValueError(f"cap must be positive, got {cap}")
        self.lib = ctypes.CDLL(str(self.path))
        f = self.lib.pearl_search
        f.restype = ctypes.c_int
        f.argtypes = [_I8P, ctypes.c_uint32, _I8P, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                      _U8P, _U8P, _U8P, _U8P, ctypes.c_uint32, ctypes.c_uint32,
                      ctypes.POINTER(CandT), ctypes.c_uint32, _U32P, ctypes.POINTER(ctypes.c_uint64)]
        self._search = f
        self.cap = cap
        self._out = (CandT * cap)()
        self.kernel = f"so:{self.path.name}"
        index, self.card_note = pinned_card(env)
        self._device = {"id": 0, "name": self.path.name, "pci_bus_id": None, "sm_count": None, "nvidia_index": index}
        self._last_error = getattr(self.lib, "pearl_last_error", None)
        if self._last_error is not None:
            self._last_error.restype = ctypes.c_char_p
            self._last_error.argtypes = []
        info = getattr(self.lib, "pearl_device_info", None)
        if info is not None:
            info.restype = ctypes.c_int
            info.argtypes = [_U32P, ctypes.c_char_p, ctypes.c_uint32]
            sm, buf = ctypes.c_uint32(0), ctypes.create_string_buffer(256)
            rc = info(ctypes.byref(sm), buf, len(buf))
            if rc != 0:
                raise self._error(f"pearl_device_info returned {rc}", rc)
            self._device.update(name=buf.value.decode("utf-8", "replace"), sm_count=sm.value)
        names = getattr(self.lib, "pearl_variants", None)
        if names is not None:
            names.restype = ctypes.c_int
            names.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
            buf = ctypes.create_string_buffer(4096)
            rc = names(buf, len(buf))
            if rc != 0:
                raise self._error(f"pearl_variants returned {rc}", rc)
            self.variants = [v for v in buf.value.decode("utf-8", "replace").split(",") if v]
            if not self.variants:
                raise KernelError("pearl_variants returned no names")
        want = env.get("PEARL_VARIANT") or None
        if want is not None:
            if self.variants is not None and want not in self.variants:
                raise ValueError(f"PEARL_VARIANT={want!r} is not one of the library's {self.variants}")
            setter = getattr(self.lib, "pearl_set_variant", None)
            if setter is not None:
                setter.restype = ctypes.c_int
                setter.argtypes = [ctypes.c_char_p]
                rc = setter(want.encode())
                if rc != 0:
                    raise self._error(f"pearl_set_variant({want!r}) returned {rc}", rc)
        self.variant = want
        self.resident = all(hasattr(self.lib, f) for f in RESIDENT_EXPORTS)
        if self.resident:
            lib = self.lib
            lib.pearl_job.restype = lib.pearl_pass.restype = ctypes.c_int
            lib.pearl_tree_nodes.restype = lib.pearl_search_resident.restype = ctypes.c_int
            lib.pearl_job.argtypes = [_U8P, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, _U8P,
                                      _U8P, _U8P, _U8P]
            lib.pearl_pass.argtypes = [_I8P, _U8P, _U8P]
            lib.pearl_tree_nodes.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, _U8P,
                                             ctypes.c_uint32, _U32P, _U8P]
            lib.pearl_search_resident.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _U8P, _U8P, _U8P, _U8P,
                                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CandT),
                                                  ctypes.c_uint32, _U32P, ctypes.POINTER(ctypes.c_uint64)]

    # ---- resident path (kernels/common/README.md, TZ_operands_gpu.md): A = B = 0 except the nonce at A[0, 0:8]

    def job(self, jk: bytes, cfg, m: int, n: int) -> tuple[bytes, bytes]:
        """Trees of B^T = 0 and of A, B' on the device -> (hash_b, seed_b)."""
        hb, sb = (ctypes.c_uint8 * 32)(), (ctypes.c_uint8 * 32)()
        rc = self.lib.pearl_job(_u8(jk), m, n, cfg.k, cfg.r, _u8(cfg.rows.to_bytes()), _u8(cfg.cols.to_bytes()), hb, sb)
        if rc != 0:
            raise self._error(f"pearl_job returned {rc}", rc)
        return bytes(hb), bytes(sb)

    def pass_(self, nonce: np.ndarray) -> tuple[bytes, bytes]:
        """The nonce (8 int8 in [-64, 64]) -> A' on the device -> (hash_a, seed_a)."""
        nonce = np.ascontiguousarray(nonce, dtype=np.int8)
        ha, sa = (ctypes.c_uint8 * 32)(), (ctypes.c_uint8 * 32)()
        rc = self.lib.pearl_pass(nonce.ctypes.data_as(_I8P), ha, sa)
        if rc != 0:
            raise self._error(f"pearl_pass returned {rc}", rc)
        return bytes(ha), bytes(sa)

    def search_resident(self, cfg, seed_a: bytes, bound: int, lo: int, hi: int, col_tiles: int) -> SearchResult:
        count, macs = ctypes.c_uint32(0), ctypes.c_uint64(0)
        rc = self.lib.pearl_search_resident(cfg.k, cfg.r, _u8(cfg.rows.to_bytes()), _u8(cfg.cols.to_bytes()),
                                            _u8(seed_a), _u8(bound.to_bytes(32, "little")), lo, hi, self._out,
                                            self.cap, ctypes.byref(count), ctypes.byref(macs))
        if rc != 0:
            raise self._error(f"pearl_search_resident returned {rc} (row tiles [{lo}, {hi}))", rc)
        return self._result(cfg, lo, hi, col_tiles, count.value, macs.value)

    def tree_nodes(self, matrix: int, leaves: list[int]) -> tuple[list[bytes], bytes]:
        """Siblings (Tree.multiproof order) and root of A (matrix 0, the current pass) or B^T (1, the job)."""
        cap = 64 * (len(leaves) + 64)
        idx = (ctypes.c_uint64 * len(leaves))(*leaves)
        out, cnt, root = (ctypes.c_uint8 * (32 * cap))(), ctypes.c_uint32(0), (ctypes.c_uint8 * 32)()
        rc = self.lib.pearl_tree_nodes(matrix, idx, len(leaves), out, cap, ctypes.byref(cnt), root)
        if rc != 0:
            raise self._error(f"pearl_tree_nodes returned {rc}", rc)
        raw = bytes(out)
        return [raw[32 * i:32 * (i + 1)] for i in range(cnt.value)], bytes(root)

    def _error(self, msg, rc):
        if self._last_error is not None:
            text = self._last_error()
            if text:
                msg += f": {text.decode('utf-8', 'replace')}"
        return KernelError(msg, rc)

    def devices(self):
        return [dict(self._device)]

    def search(self, ops, lo, hi):
        if not 0 <= lo < hi <= ops.row_tiles:
            raise ValueError(f"row tiles [{lo}, {hi}) outside [0, {ops.row_tiles})")
        a, bt = ops.a_noised, ops.bt_noised
        for name, x in (("a_noised", a), ("bt_noised", bt)):
            if x.dtype != np.int8 or not x.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous int8")
        cfg = ops.cfg
        count, macs = ctypes.c_uint32(0), ctypes.c_uint64(0)
        rc = self._search(a.ctypes.data_as(_I8P), ops.m, bt.ctypes.data_as(_I8P), ops.n, cfg.k, cfg.r,
                          _u8(cfg.rows.to_bytes()), _u8(cfg.cols.to_bytes()), _u8(ops.seed_a),
                          _u8(ops.bound.to_bytes(32, "little")), lo, hi, self._out, self.cap,
                          ctypes.byref(count), ctypes.byref(macs))
        if rc != 0:
            raise self._error(f"pearl_search returned {rc} (row tiles [{lo}, {hi}))", rc)
        return self._result(cfg, lo, hi, len(ops.col_part), count.value, macs.value)

    def _result(self, cfg, lo, hi, col_tiles, count, macs) -> SearchResult:
        want_macs = (hi - lo) * col_tiles * cfg.h * cfg.w * cfg.L
        if macs != want_macs:
            raise KernelError(f"the kernel reported {macs} MAC, the contract says {want_macs}")
        got = min(count, self.cap)
        res = SearchResult(macs=macs, dropped=count - got)
        for c in self._out[:got]:
            if not (lo <= c.row_tile < hi and c.col_tile < col_tiles):
                raise KernelError(f"candidate tile ({c.row_tile}, {c.col_tile}) outside [{lo}, {hi}) x {col_tiles}")
            res.candidates.append(Candidate(c.row_tile, c.col_tile, bytes(c.jackpot)))
        return res


BACKENDS = {"cpu-ref": CpuRefBackend}
SO_PREFIX = "so:"      # so:/path/lib.so -> SoBackend


def is_backend_name(name: str) -> bool:
    return name in BACKENDS or (name.startswith(SO_PREFIX) and len(name) > len(SO_PREFIX))


def make_backend(name: str) -> Backend:
    if not is_backend_name(name):
        raise ValueError(f"unknown backend {name!r}; known: {', '.join(BACKENDS)}, {SO_PREFIX}<path>")
    if name.startswith(SO_PREFIX):
        return SoBackend(name[len(SO_PREFIX):])
    return BACKENDS[name]()
