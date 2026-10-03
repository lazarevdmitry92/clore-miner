"""BLAKE3 compression vectorized over many independent inputs (numpy uint32).

Needed where pip `blake3` gives no access: chunk/parent chaining values of the Merkle tree
(siblings of a multi-proof) and batches of single-block keyed hashes (noise PRG, tile jackpots).
Full-message hashes go through pip `blake3`; tests check both agree.
"""
import numpy as np

IV = np.array([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=np.uint32)
PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16
CHUNK_LEN, BLOCK_LEN = 1024, 64


def _rotr(x, n):
    return (x >> np.uint32(n)) | (x << np.uint32(32 - n))


def _g(s, a, b, c, d, mx, my):
    s[a] = s[a] + s[b] + mx
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = s[c] + s[d]
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = s[a] + s[b] + my
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = s[c] + s[d]
    s[b] = _rotr(s[b] ^ s[c], 7)


def compress(cv, block, counter, block_len, flags):
    """cv (8,N), block (16,N) uint32; counter, block_len, flags scalar or (N,). Returns (16,N)."""
    n = cv.shape[1]
    counter = np.broadcast_to(np.asarray(counter, dtype=np.uint64), (n,))
    s = np.empty((16, n), dtype=np.uint32)
    s[0:8] = cv
    s[8:12] = IV[0:4, None]
    s[12] = (counter & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    s[13] = (counter >> np.uint64(32)).astype(np.uint32)
    s[14] = np.uint32(block_len)
    s[15] = np.broadcast_to(np.asarray(flags, dtype=np.uint32), (n,))
    m = [block[i] for i in range(16)]
    with np.errstate(over="ignore"):
        for r in range(7):
            _g(s, 0, 4, 8, 12, m[0], m[1])
            _g(s, 1, 5, 9, 13, m[2], m[3])
            _g(s, 2, 6, 10, 14, m[4], m[5])
            _g(s, 3, 7, 11, 15, m[6], m[7])
            _g(s, 0, 5, 10, 15, m[8], m[9])
            _g(s, 1, 6, 11, 12, m[10], m[11])
            _g(s, 2, 7, 8, 13, m[12], m[13])
            _g(s, 3, 4, 9, 14, m[14], m[15])
            if r < 6:
                m = [m[PERM[i]] for i in range(16)]
    out = s.copy()
    out[0:8] ^= s[8:16]
    out[8:16] ^= cv
    return out


def words(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<u4").astype(np.uint32)


def to_bytes(w8: np.ndarray) -> bytes:
    """(8,) words -> 32 bytes LE."""
    return w8.astype("<u4").tobytes()


def keyed_single_block(key: bytes, blocks: np.ndarray) -> np.ndarray:
    """BLAKE3(64-byte msg, key) for each column of blocks (16,N) -> digests (8,N)."""
    n = blocks.shape[1]
    cv = np.repeat(words(key)[:, None], n, axis=1)
    return compress(cv, blocks, 0, BLOCK_LEN, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT)[0:8]


def chunk_cvs(data: bytes, key: bytes) -> np.ndarray:
    """Non-root chaining values of all 1024-byte chunks (len(data) % 1024 == 0) -> (N,32) bytes array."""
    assert len(data) % CHUNK_LEN == 0 and data
    n = len(data) // CHUNK_LEN
    w = np.frombuffer(data, dtype="<u4").reshape(n, 16, 16)
    cv = np.repeat(words(key)[:, None], n, axis=1)
    idx = np.arange(n, dtype=np.uint64)
    for b in range(16):
        flags = KEYED_HASH | (CHUNK_START if b == 0 else 0) | (CHUNK_END if b == 15 else 0)
        cv = compress(cv, w[:, b, :].T.astype(np.uint32), idx, BLOCK_LEN, flags)[0:8]
    return cv.T.astype("<u4", order="C").view(np.uint8).reshape(n, 32)


def parent_cvs(left: np.ndarray, right: np.ndarray, key: bytes, root: bool = False) -> np.ndarray:
    """left, right (N,32) uint8 -> parent CVs (N,32)."""
    n = left.shape[0]
    block = np.concatenate([left, right], axis=1).view("<u4").T.astype(np.uint32)
    cv = np.repeat(words(key)[:, None], n, axis=1)
    out = compress(cv, block, 0, BLOCK_LEN, KEYED_HASH | PARENT | (ROOT if root else 0))[0:8]
    return out.T.astype("<u4", order="C").view(np.uint8).reshape(n, 32)
