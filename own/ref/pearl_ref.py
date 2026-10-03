"""Reference Pearl miner (cert_version 3, dense), pure Python + numpy, independent of Pearl's code.

Follows s0_algorithm.md: job_key -> keyed BLAKE3 roots -> V3 salt -> seeds -> low-rank noise ->
C' = (A+E_A)(B+E_B) with XOR checkpoints every r -> keyed BLAKE3 jackpot -> PlainProof (bincode, base64).
"""
import base64
import struct
from dataclasses import dataclass

import blake3
import numpy as np

import blake3_np as b3

U256_MAX = (1 << 256) - 1
SALT_A = blake3.blake3(b"pearl/cert-v3/noise-seed/A").digest()
SALT_B = blake3.blake3(b"pearl/cert-v3/noise-seed/B").digest()
LABEL_A = b"A_tensor" + bytes(24)
LABEL_B = b"B_tensor" + bytes(24)
JACKPOT_SIZE, LROT = 16, 13
CHUNK = 1024


def H(data: bytes, key: bytes | None = None) -> bytes:
    return (blake3.blake3(data, key=key) if key else blake3.blake3(data)).digest()


# ---------------------------------------------------------------- header, config, target

def header_bytes(version: int, prev_block: bytes, merkle_root: bytes, timestamp: int, nbits: int) -> bytes:
    """76 bytes; prev_block and merkle_root are written reversed (as IncompleteBlockHeader.to_bytes)."""
    return (struct.pack("<I", version) + prev_block[::-1] + merkle_root[::-1]
            + struct.pack("<II", timestamp, nbits))


def header_nbits(header76: bytes) -> int:
    return struct.unpack_from("<I", header76, 72)[0]


class Pattern:
    """PeriodicPattern: shape = 3 x (stride, length)."""

    def __init__(self, shape):
        self.shape = list(shape)

    @classmethod
    def from_list(cls, lst):
        assert lst and lst[0] == 0 and all(a < b for a, b in zip(lst, lst[1:])), "bad pattern list"
        p, shape = list(lst), []
        while len(p) > 1:
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape.append((s, len(p) // period))
                        p = p[:period]
                        break
            else:
                raise ValueError("pattern is not periodic")
        shape.reverse()
        per = shape[-1][0] * shape[-1][1] if shape else 1
        while len(shape) < 3:
            shape.append((per, 1))
        pat = cls(shape)
        assert cls.from_bytes(pat.to_bytes()).shape == pat.shape, "non-canonical pattern"
        return pat

    def to_bytes(self) -> bytes:
        out, min_stride = bytearray(), 1
        for stride, length in self.shape:
            out += bytes([stride // min_stride - 1, length - 1])
            min_stride = stride * length
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes):
        shape, min_stride, done = [], 1, False
        for i in range(3):
            factor, length = data[2 * i] + 1, data[2 * i + 1] + 1
            if length == 1 or done:
                assert factor == 1 and length == 1, "non-canonical"
                done = True
            elif factor <= 1 and min_stride != 1:
                raise ValueError("a single stride must not be broken")
            assert min_stride <= (1 << 24) // (factor * length)
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        pat = cls(shape)
        assert pat.to_bytes() == bytes(data)
        return pat

    def to_list(self):
        res = [0]
        for stride, length in self.shape:
            res = [r + i * stride for i in range(length) for r in res]
        return res

    @property
    def period(self):
        s, l = self.shape[-1]
        return s * l

    @property
    def size(self):
        return int(np.prod([l for _, l in self.shape]))

    def offset_is_valid(self, off):
        for stride, length in reversed(self.shape):
            off %= stride * length
            if off >= stride:
                return False
        return True

    def partition(self, total):
        """All tiles (index lists) covering 0..total-1, in the CPU miner's order (threads_partition)."""
        assert total % self.period == 0, "dimension must be a multiple of the pattern period"
        base = np.array(self.to_list())
        offs = [o for o in range(total) if self.offset_is_valid(o)]
        return np.array([o + base for o in offs])


@dataclass
class Config:
    k: int
    r: int
    rows: Pattern
    cols: Pattern

    def to_bytes(self) -> bytes:
        return (struct.pack("<IHH", self.k, self.r, 0) + self.rows.to_bytes() + self.cols.to_bytes()
                + bytes(32))

    @property
    def L(self):
        return self.k - self.k % self.r

    @property
    def h(self):
        return self.rows.size

    @property
    def w(self):
        return self.cols.size

    def sanity(self, m, n, t_rows=0, t_cols=0):
        k, r, h, w, L = self.k, self.r, self.h, self.w, self.L
        assert r & (r - 1) == 0 and 32 <= r <= 1024 and r % 16 == 0
        assert k <= 1 << 16 and k % 64 == 0 and 16 * r <= k <= 4 * r * r and k >= 1024
        assert h % 2 == 0 and w % 2 == 0 and 32 <= h * w <= 256
        assert L % 8 == 0
        assert m <= 1 << 24 and n <= 1 << 24 and (h + w) * L <= 1 << 22
        assert t_rows + max(self.rows.to_list()) < m and t_cols + max(self.cols.to_list()) < n


def nbits_to_target(nbits: int) -> int:
    exp, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or exp == 0 or mant & 0x800000:
        return 0
    return mant >> (8 * (3 - exp)) if exp <= 3 else (mant << (8 * (exp - 3))) & U256_MAX


def jackpot_bound(nbits: int, cfg: Config) -> int:
    """Consensus bound: target * h*w*L, saturated at 2^256-1 (extract_difficulty_bound)."""
    return min(nbits_to_target(nbits) * cfg.h * cfg.w * cfg.L, U256_MAX)


# ---------------------------------------------------------------- commitments and seeds

def job_key(header76: bytes, cfg: Config) -> bytes:
    data = header76 + cfg.to_bytes()
    assert len(data) == 128
    return H(data)


def pad1024(b: bytes) -> bytes:
    return b + bytes(-len(b) % CHUNK)


def matrix_root(mat_i8: np.ndarray, key: bytes) -> bytes:
    """Merkle root of a row-major int8 matrix = keyed BLAKE3 of its zero-padded bytes."""
    return H(pad1024(mat_i8.astype(np.int8).tobytes()), key)


def salt_roots(hash_a: bytes, hash_b: bytes, m: int, n: int, salted: bool = True):
    if not salted:
        return hash_a, hash_b
    a = H(hash_a + struct.pack("<I", m) + bytes(28), SALT_A)
    b = H(hash_b + struct.pack("<I", n) + bytes(28), SALT_B)
    return a, b


def seeds(jk: bytes, hash_a: bytes, hash_b: bytes, m: int, n: int, salted: bool = True):
    """-> (seed_B, seed_A), each 32 bytes."""
    ha, hb = salt_roots(hash_a, hash_b, m, n, salted)
    seed_b = H(jk + hb)
    seed_a = H(seed_b + ha)
    return seed_b, seed_a


# ---------------------------------------------------------------- noise

def _prg_blocks(n_blocks: int, slot: int, label: bytes) -> np.ndarray:
    blk = np.zeros((16, n_blocks), dtype=np.uint32)
    blk[slot] = np.arange(1, n_blocks + 1, dtype=np.uint32)
    blk[8:16] = b3.words(label)[:, None]
    return blk


def uniform_factor(seed: bytes, label: bytes, rows, r: int) -> np.ndarray:
    """E_AL (or E_BR^T) rows `rows`, r columns, values (byte & 63) - 32 in [-32, 31]."""
    rows = np.asarray(rows)
    blocks_per_row = r // 32
    blk_ids = (rows[:, None] * blocks_per_row + np.arange(blocks_per_row)[None, :]).ravel()
    blk = np.zeros((16, len(blk_ids)), dtype=np.uint32)
    blk[0] = (blk_ids + 1).astype(np.uint32)
    blk[8:16] = b3.words(label)[:, None]
    out = b3.keyed_single_block(seed, blk)  # (8, N)
    bytes_ = out.T.astype("<u4", order="C").view(np.uint8).reshape(len(rows), r)
    return (bytes_ & 63).astype(np.int16) - 32


def sparse_factor(seed: bytes, label: bytes, k: int, r: int):
    """E_AR^T (or E_BL), k x r with +1 at p_l and -1 at q_l -> (p, q) arrays of length k."""
    nb = -(-k // 8)
    out = b3.keyed_single_block(seed, _prg_blocks(nb, 1, label))  # (8, nb)
    x = out.T.reshape(-1)[:k].astype(np.uint64)
    p = x & np.uint64(r - 1)
    q = p ^ (np.uint64(1) + ((np.uint64(r - 1) * x) >> np.uint64(32)))
    return p.astype(np.int64), q.astype(np.int64)


def noise(seed_b: bytes, seed_a: bytes, k: int, r: int, a_rows, b_cols):
    """-> (E_A rows a_rows: len(a_rows) x k, E_B^T rows b_cols: len(b_cols) x k), int16 in [-63, 63]."""
    e_al = uniform_factor(seed_a, LABEL_A, a_rows, r)
    pa, qa = sparse_factor(seed_a, LABEL_A, k, r)
    e_brt = uniform_factor(seed_b, LABEL_B, b_cols, r)
    pb, qb = sparse_factor(seed_b, LABEL_B, k, r)
    return e_al[:, pa] - e_al[:, qa], e_brt[:, pb] - e_brt[:, qb]


# ---------------------------------------------------------------- GEMM + transcript + jackpot

def rotl13(x):
    return (x << np.uint32(LROT)) | (x >> np.uint32(32 - LROT))


def transcripts(a_noised, bt_noised, cfg: Config, row_tiles, col_tiles) -> np.ndarray:
    """Running sums over chunks of depth r; XOR per tile after each chunk -> transcript (Tr, Tc, 16) u32.

    a_noised (m x k), bt_noised (n x k) ints in [-127, 127]. Chunk products are exact in float64
    (|partial| <= r*127^2 < 2^53); the running sum is kept in int64 and read as u32 (two's complement)."""
    m, n = a_noised.shape[0], bt_noised.shape[0]
    af = a_noised.astype(np.float64)
    bf = bt_noised.astype(np.float64)
    S = np.zeros((m, n), dtype=np.int64)
    T = np.zeros((len(row_tiles), len(col_tiles), JACKPOT_SIZE), dtype=np.uint32)
    for c in range(cfg.L // cfg.r):
        sl = slice(c * cfg.r, (c + 1) * cfg.r)
        S += np.rint(af[:, sl] @ bf[:, sl].T).astype(np.int64)
        u = (S & 0xFFFFFFFF).astype(np.uint32)
        g = u[row_tiles[:, :, None, None], col_tiles[None, None, :, :]]  # (Tr, h, Tc, w)
        x = np.bitwise_xor.reduce(np.bitwise_xor.reduce(g, axis=3), axis=1)
        t = c % JACKPOT_SIZE
        T[:, :, t] = rotl13(T[:, :, t]) ^ x
    return T


def jackpot_hashes(T: np.ndarray, seed_a: bytes) -> np.ndarray:
    """keyed BLAKE3(transcript 64 B, key=seed_A) per tile -> (Tr, Tc, 32) uint8."""
    shp = T.shape[:2]
    d = b3.keyed_single_block(seed_a, T.reshape(-1, 16).T.copy())
    return d.T.astype("<u4", order="C").view(np.uint8).reshape(*shp, 32)


def le_int(b) -> int:
    return int.from_bytes(bytes(b), "little")


# ---------------------------------------------------------------- Merkle multi-proof

class Tree:
    def __init__(self, padded: bytes, key: bytes):
        assert len(padded) % CHUNK == 0 and len(padded) > CHUNK, "tree of >= 2 chunks"
        self.data, self.key = padded, key
        self.layers = [b3.chunk_cvs(padded, key)]
        while len(self.layers[-1]) > 2:
            prev = self.layers[-1]
            even = len(prev) // 2 * 2
            nxt = b3.parent_cvs(prev[0:even:2], prev[1:even:2], key)
            if len(prev) % 2:
                nxt = np.concatenate([nxt, prev[-1:]])
            self.layers.append(nxt)
        last = self.layers[-1]
        self.root = bytes(b3.parent_cvs(last[0:1], last[1:2], key, root=True)[0])

    @property
    def n_leaves(self):
        return len(self.layers[0])

    def multiproof(self, leaf_indices):
        cur = sorted(set(leaf_indices))
        leaves = [self.data[i * CHUNK:(i + 1) * CHUNK] for i in cur]
        sib, level, level_len, s = [], 0, self.n_leaves, set(cur)
        while level_len > 1 and s:
            nodes = self.layers[level]
            for i in sorted(s):
                if i % 2 == 1:
                    if i - 1 not in s:
                        sib.append(bytes(nodes[i - 1]))
                elif i + 1 not in s and i + 1 < level_len:
                    sib.append(bytes(nodes[i + 1]))
            s = {i // 2 for i in s}
            level_len = -(-level_len // 2)
            level += 1
        return leaves, cur, sib


def compute_root(leaf_data, leaf_indices, total, siblings, key) -> bytes | None:
    """Root from leaves + siblings (MerkleProof::compute_root)."""
    cur = {}
    for idx, data in zip(leaf_indices, leaf_data):
        cur[idx] = _chunk_cv_at(data, idx, key)
    sib = iter(siblings)
    level_len = total
    if level_len == 1:
        return None
    try:
        while level_len > 2:
            nxt = {}
            for i in sorted(cur):
                if i % 2 == 0:
                    if i + 1 in cur:
                        right = cur[i + 1]
                    elif i + 1 < level_len:
                        right = next(sib)
                    else:
                        right = None
                    nxt[i // 2] = cur[i] if right is None else _parent(cur[i], right, key)
                elif i - 1 in cur:
                    continue
                else:
                    nxt[i // 2] = _parent(next(sib), cur[i], key)
            cur = nxt
            level_len = -(-level_len // 2)
        left = cur[0] if 0 in cur else next(sib)
        right = cur[1] if 1 in cur else next(sib)
    except StopIteration:
        return None
    if next(sib, None) is not None:
        return None
    return bytes(b3.parent_cvs(np.frombuffer(left, np.uint8)[None], np.frombuffer(right, np.uint8)[None],
                               key, root=True)[0])


def _chunk_cv_at(data: bytes, idx: int, key: bytes) -> bytes:
    w = np.frombuffer(data, dtype="<u4").reshape(16, 16)
    cv = b3.words(key)[:, None].copy()
    for b in range(16):
        flags = b3.KEYED_HASH | (b3.CHUNK_START if b == 0 else 0) | (b3.CHUNK_END if b == 15 else 0)
        cv = b3.compress(cv, w[b][:, None].astype(np.uint32), idx, 64, flags)[0:8]
    return b3.to_bytes(cv[:, 0])


def _parent(l: bytes, r: bytes, key: bytes) -> bytes:
    return bytes(b3.parent_cvs(np.frombuffer(l, np.uint8)[None], np.frombuffer(r, np.uint8)[None], key)[0])


def leaf_indices_from_rows(rows, row_len):
    out = set()
    for row in rows:
        out.update(range(row * row_len // CHUNK, ((row + 1) * row_len - 1) // CHUNK + 1))
    return sorted(out)


# ---------------------------------------------------------------- PlainProof (bincode 1.x fixint LE)

@dataclass
class MatrixProof:
    leaf_data: list
    leaf_indices: list
    total_leaves: int
    root: bytes
    siblings: list
    row_indices: list


@dataclass
class PlainProof:
    m: int
    n: int
    k: int
    noise_rank: int
    a: MatrixProof
    bt: MatrixProof

    def to_bytes(self) -> bytes:
        out = bytearray(struct.pack("<4Q", self.m, self.n, self.k, self.noise_rank))
        for mp in (self.a, self.bt):
            out += struct.pack("<Q", len(mp.leaf_data))
            for leaf in mp.leaf_data:
                assert len(leaf) == CHUNK
                out += struct.pack("<Q", CHUNK) + leaf
            out += struct.pack("<Q", len(mp.leaf_indices)) + struct.pack(f"<{len(mp.leaf_indices)}Q", *mp.leaf_indices)
            out += struct.pack("<Q", mp.total_leaves) + mp.root
            out += struct.pack("<Q", len(mp.siblings)) + b"".join(mp.siblings)
            out += struct.pack("<Q", len(mp.row_indices)) + struct.pack(f"<{len(mp.row_indices)}Q", *mp.row_indices)
        out += b"\x00"  # moe: None
        return bytes(out)

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode()

    @classmethod
    def from_bytes(cls, b: bytes):
        pos = 0

        def u64():
            nonlocal pos
            v = struct.unpack_from("<Q", b, pos)[0]
            pos += 8
            return v

        def raw(n):
            nonlocal pos
            v = b[pos:pos + n]
            assert len(v) == n, "truncated"
            pos += n
            return v

        m, n, k, r = u64(), u64(), u64(), u64()
        mps = []
        for _ in range(2):
            leaves = [raw(u64()) for _ in range(u64())]
            li = [u64() for _ in range(u64())]
            total, root = u64(), raw(32)
            sib = [raw(32) for _ in range(u64())]
            rows = [u64() for _ in range(u64())]
            mps.append(MatrixProof(leaves, li, total, root, sib, rows))
        tag = raw(1)
        assert tag == b"\x00", "MoE proofs are not supported by the reference"
        assert pos == len(b), "trailing bytes"
        return cls(m, n, k, r, mps[0], mps[1])


def matrix_tree(mat_i8: np.ndarray, key: bytes) -> Tree:
    return Tree(pad1024(mat_i8.astype(np.int8).tobytes()), key)


def tree_proof(tree: Tree, rows, row_len: int) -> MatrixProof:
    """Proof of whole rows (row_len bytes each) from an already built tree of the matrix."""
    leaves, li, sib = tree.multiproof(leaf_indices_from_rows(rows, row_len))
    return MatrixProof(leaves, li, tree.n_leaves, tree.root, sib, [int(x) for x in rows])


def matrix_proof(mat_i8: np.ndarray, key: bytes, rows) -> MatrixProof:
    return tree_proof(matrix_tree(mat_i8, key), rows, mat_i8.shape[1])


# ---------------------------------------------------------------- miner

@dataclass
class Share:
    proof: PlainProof
    tile: tuple
    jackpot: bytes
    attempts: int


def mine_once(header76: bytes, cfg: Config, A: np.ndarray, B: np.ndarray, nbits: int,
              salted: bool = True, first_only: bool = True):
    """One pass over all tiles of C' for fixed A (m x k), B (k x n). Returns list of Share."""
    m, k = A.shape
    n = B.shape[1]
    assert B.shape[0] == k == cfg.k
    assert A.min() >= -64 and A.max() <= 64 and B.min() >= -64 and B.max() <= 64
    Bt = np.ascontiguousarray(B.T)
    jk = job_key(header76, cfg)
    ha, hb = matrix_root(A, jk), matrix_root(Bt, jk)
    seed_b, seed_a = seeds(jk, ha, hb, m, n, salted)
    ea, ebt = noise(seed_b, seed_a, k, cfg.r, np.arange(m), np.arange(n))
    a_n = A.astype(np.int16) + ea
    bt_n = Bt.astype(np.int16) + ebt
    row_tiles, col_tiles = cfg.rows.partition(m), cfg.cols.partition(n)
    T = transcripts(a_n, bt_n, cfg, row_tiles, col_tiles)
    J = jackpot_hashes(T, seed_a)
    bound = jackpot_bound(nbits, cfg)
    shares = []
    for i in range(len(row_tiles)):
        for j in range(len(col_tiles)):
            if le_int(J[i, j]) <= bound:
                proof = PlainProof(m, n, k, cfg.r, matrix_proof(A, jk, row_tiles[i]),
                                   matrix_proof(Bt, jk, col_tiles[j]))
                shares.append(Share(proof, (i, j), bytes(J[i, j]), len(row_tiles) * len(col_tiles)))
                if first_only:
                    return shares
    return shares


# ---------------------------------------------------------------- independent verifier

def list_to_pattern(idx):
    assert all(a < b for a, b in zip(idx, idx[1:])), "indices not strictly increasing"
    off = idx[0]
    pat = Pattern.from_list([i - off for i in idx])
    assert pat.offset_is_valid(off), "invalid offset"
    return pat, off


def verify(header76: bytes, proof: PlainProof, nbits_override: int | None = None, salted: bool = True,
           target_override: int | None = None):
    """Mirror of verify_plain_proof for dense proofs. Returns (ok, message).

    target_override: the exact share target (a pool target need not have a compact form); excludes nbits_override."""
    if target_override is not None and (nbits_override is not None or target_override <= 0):
        raise ValueError("target_override must be positive and excludes nbits_override")
    try:
        m, n, k, r = proof.m, proof.n, proof.k, proof.noise_rank
        assert m < 1 << 32 and n < 1 << 32 and k < 1 << 32 and r < 1 << 16
        assert proof.a.total_leaves == -(-m * k // CHUNK), "A leaf count"
        assert proof.bt.total_leaves == -(-n * k // CHUNK), "B leaf count"
        assert all(x < m for x in proof.a.row_indices)
        rows, t_rows = list_to_pattern(proof.a.row_indices)
        cols, t_cols = list_to_pattern(proof.bt.row_indices)
        cfg = Config(k, r, rows, cols)
        cfg.sanity(m, n, t_rows, t_cols)
        jk = job_key(header76, cfg)
        strips = []
        for mp, dim in ((proof.a, m), (proof.bt, n)):
            assert mp.leaf_indices == sorted(set(mp.leaf_indices)) and len(mp.leaf_indices) == len(mp.leaf_data)
            assert all(len(x) == CHUNK for x in mp.leaf_data)
            assert compute_root(mp.leaf_data, mp.leaf_indices, mp.total_leaves, mp.siblings, jk) == mp.root, \
                "Merkle root mismatch"
            pos = {li: d for li, d in zip(mp.leaf_indices, mp.leaf_data)}
            s = []
            for row in mp.row_indices:
                start = row * k
                buf = bytearray()
                for off in range(start, start + cfg.L):
                    ch = off // CHUNK
                    assert ch in pos, "row bytes not covered by leaves"
                    buf.append(pos[ch][off % CHUNK])
                s.append(np.frombuffer(bytes(buf), dtype=np.int8))
            strips.append(np.array(s, dtype=np.int16))
        sa, sb = strips
        assert sa.min() >= -64 and sa.max() <= 64 and sb.min() >= -64 and sb.max() <= 64, "value out of [-64, 64]"
        seed_b, seed_a = seeds(jk, proof.a.root, proof.bt.root, m, n, salted)
        ea, ebt = noise(seed_b, seed_a, k, r, proof.a.row_indices, proof.bt.row_indices)
        cfg_L = Config(cfg.L, r, rows, cols)  # strips are L long
        a_n, bt_n = sa + ea[:, :cfg.L], sb + ebt[:, :cfg.L]
        T = transcripts(a_n, bt_n, cfg_L, np.arange(cfg.h)[None], np.arange(cfg.w)[None])
        J = bytes(jackpot_hashes(T, seed_a)[0, 0])
        if target_override is not None:
            bound = min(target_override * cfg.h * cfg.w * cfg.L, U256_MAX)
        else:
            bound = jackpot_bound(header_nbits(header76) if nbits_override is None else nbits_override, cfg)
        if le_int(J) > bound:
            return False, "jackpot above target"
        return True, "ok"
    except (AssertionError, ValueError, KeyError) as e:
        return False, f"rejected: {e}"
