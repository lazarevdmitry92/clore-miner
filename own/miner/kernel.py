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
from dataclasses import dataclass, field
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


BACKENDS = {"cpu-ref": CpuRefBackend}


def make_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; known: {', '.join(BACKENDS)}")
    return BACKENDS[name]()
