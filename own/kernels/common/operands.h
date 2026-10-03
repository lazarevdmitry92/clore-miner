// Operand preparation on the device (TZ_operands_gpu.md): everything from job_key and the nonce to resident A', B'.
// The miner mines with A = B = 0 except a nonce in A[0, 0:8] (miner/jobs.py), so per job and per pass:
//
//   job:  Merkle tree of B^T = 0 (keyed by job_key) -> hash_b -> salted -> seed_B = BLAKE3(job_key || hash_b')
//         E_BR^T (n x r) and the sparse indices p'_l, q'_l from seed_B;  B'^T[j, l] = E_BR^T[j, p'_l] - E_BR^T[j, q'_l]
//         Merkle tree of A with the nonce chunk left out (its other chunks are zero for every pass of the job)
//   pass: chunk 0 of A with the nonce -> its leaf -> the path to the root (about log2(m*k/1024) parents) -> hash_a ->
//         salted -> seed_A = BLAKE3(seed_B || hash_a');  E_AL (m x r), p_l, q_l from seed_A;
//         A'[i, l] = E_AL[i, p_l] - E_AL[i, q_l] (+ the nonce in row 0)
//
// The per-element generators below are __host__ __device__: the CUDA kernels (operands_cuda.cuh), the CPU emulators
// and the host self-check run the same code. Independent reference: ref/pearl_ref.py (matrix_root, seeds,
// uniform_factor, sparse_factor, noise), s0_algorithm.md §3.2-3.4; the tests compare bit for bit.
#pragma once
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include "pearl_core.h"

#define PEARL_NONCE_BYTES 8   // A[0, 0:8]: base-129 digits in [-64, 64] (miner/jobs.py NONCE_DIGITS)

// ---------------------------------------------------------------- per element (host and device)

// noise PRG block (pearl_noise.rs:19-57): BLAKE3(8 x i32 zeros except word `slot` = 1 + i || label32, key)
PH void prg_block(const uint32_t key[8], uint32_t slot, uint32_t i, const uint32_t label[8], uint32_t out[8]) {
    uint32_t msg[16];
    for (int w = 0; w < 8; ++w) msg[w] = 0;
    msg[slot] = i + 1;
    for (int w = 0; w < 8; ++w) msg[8 + w] = label[w];
    blake3_keyed_block(key, msg, out);
}

// dense factor element: (byte & 63) - 32 in [-32, 31]
PH int8_t uniform_value(uint32_t byte) { return (int8_t)((int)(byte & 63) - 32); }

// sparse factor row l from its PRG word x: +1 at p, -1 at q != p (pearl_noise.rs:90-116)
PH void sparse_pq(uint32_t x, uint32_t r, uint8_t &p, uint8_t &q) {
    p = (uint8_t)(x & (r - 1));
    q = (uint8_t)(p ^ (1u + (uint32_t)(((uint64_t)(r - 1) * x) >> 32)));
}

// chaining value of 1024-byte chunk `idx` (non-root); chunk == nullptr: all zeros
PH void chunk_cv(const uint32_t key[8], const uint8_t *chunk, uint64_t idx, uint32_t out[8]) {
    uint32_t cv[8], msg[16];
    for (int w = 0; w < 8; ++w) cv[w] = key[w];
    for (int b = 0; b < 16; ++b) {
        for (int w = 0; w < 16; ++w) {
            uint32_t x = 0;
            if (chunk)
                for (int y = 0; y < 4; ++y) x |= (uint32_t)chunk[64 * b + 4 * w + y] << (8 * y);
            msg[w] = x;
        }
        const uint32_t flags = B3_KEYED | (b == 0 ? B3_CHUNK_START : 0u) | (b == 15 ? B3_CHUNK_END : 0u);
        blake3_compress(cv, msg, idx, 64, flags, cv);
    }
    for (int w = 0; w < 8; ++w) out[w] = cv[w];
}

PH void parent_cv(const uint32_t key[8], const uint32_t l[8], const uint32_t r[8], bool root, uint32_t out[8]) {
    uint32_t msg[16];
    for (int w = 0; w < 8; ++w) { msg[w] = l[w]; msg[8 + w] = r[w]; }
    blake3_compress(key, msg, 0, 64, B3_KEYED | B3_PARENT | (root ? B3_ROOT : 0u), out);
}

namespace pearl {

typedef std::array<uint32_t, 8> CV;

inline void le_words(const uint8_t *b, uint32_t *w, int n) {
    for (int i = 0; i < n; ++i) w[i] = b[4 * i] | b[4 * i + 1] << 8 | b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
}
inline void le_bytes(const uint32_t *w, uint8_t *b, int n) {
    for (int i = 0; i < n; ++i)
        for (int y = 0; y < 4; ++y) b[4 * i + y] = (uint8_t)(w[i] >> (8 * y));
}

// unkeyed BLAKE3 of a message up to 64 bytes (one block)
inline CV blake3_small(const uint8_t *data, uint32_t len) {
    static const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                   0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    uint8_t buf[64] = {0};
    memcpy(buf, data, len);
    uint32_t msg[16];
    le_words(buf, msg, 16);
    CV out;
    blake3_compress(IV, msg, 0, len, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, out.data());
    return out;
}

inline CV keyed64(const CV &key, const uint8_t *data64) {
    uint32_t msg[16];
    le_words(data64, msg, 16);
    CV out;
    blake3_keyed_block(key.data(), msg, out.data());
    return out;
}

inline CV from_bytes(const uint8_t *b) { CV c; le_words(b, c.data(), 8); return c; }
inline void to_bytes(const CV &c, uint8_t *b) { le_bytes(c.data(), b, 8); }

// "A_tensor" / "B_tensor" + 24 zeros as words
inline CV label(char which) {
    uint8_t b[32] = {0};
    memcpy(b, which == 'A' ? "A_tensor" : "B_tensor", 8);
    return from_bytes(b);
}

// V3 salt (seed.rs:20-59): root' = BLAKE3(root || u32le(dim) || 0^28, key = BLAKE3("pearl/cert-v3/noise-seed/X"))
inline CV salted(const CV &root, uint32_t dim, char which) {
    const char *s = which == 'A' ? "pearl/cert-v3/noise-seed/A" : "pearl/cert-v3/noise-seed/B";
    const CV salt = blake3_small((const uint8_t *)s, (uint32_t)strlen(s));
    uint8_t blk[64] = {0};
    to_bytes(root, blk);
    for (int y = 0; y < 4; ++y) blk[32 + y] = (uint8_t)(dim >> (8 * y));
    return keyed64(salt, blk);
}

// BLAKE3(x || y) unkeyed, x and y 32 bytes: seed_B = H(job_key || hash_b'), seed_A = H(seed_B || hash_a')
inline CV hash_pair(const CV &x, const CV &y) {
    uint8_t blk[64];
    to_bytes(x, blk);
    to_bytes(y, blk + 32);
    return blake3_small(blk, 64);
}

// The Merkle tree of a matrix as ref/pearl_ref.py Tree builds it (= BLAKE3's tree): leaves are chunk CVs, pairs
// bottom-up with the odd last node pushed up, the root = parent of the last two with ROOT. >= 2 leaves.
struct Tree {
    CV key;
    std::vector<std::vector<CV>> layers;
    CV root;

    void build(const CV &k, std::vector<CV> leaves) {
        key = k;
        layers.assign(1, std::move(leaves));
        while (layers.back().size() > 2) {
            const std::vector<CV> &prev = layers.back();
            std::vector<CV> next(prev.size() / 2 + prev.size() % 2);
            for (size_t i = 0; i + 1 < prev.size(); i += 2) parent_cv(key.data(), prev[i].data(), prev[i + 1].data(), false, next[i / 2].data());
            if (prev.size() % 2) next.back() = prev.back();
            layers.push_back(std::move(next));
        }
        finish();
    }
    void finish() { parent_cv(key.data(), layers.back()[0].data(), layers.back()[1].data(), true, root.data()); }

    // new leaf 0 (the nonce chunk): only the leftmost path changes
    void set_leaf0(const CV &leaf) {
        layers[0][0] = leaf;
        for (size_t lv = 0; lv + 1 < layers.size(); ++lv)
            parent_cv(key.data(), layers[lv][0].data(), layers[lv][1].data(), false, layers[lv + 1][0].data());
        finish();
    }

    // siblings of a multi-proof, Tree.multiproof order: level by level from the leaves, ascending index
    std::vector<CV> siblings(std::vector<uint64_t> leaves) const {
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        std::vector<CV> sib;
        std::vector<uint64_t> s = leaves;
        uint64_t level_len = layers[0].size();
        for (size_t level = 0; level_len > 1 && !s.empty(); ++level) {
            const std::vector<CV> &nodes = layers[level];
            for (size_t x = 0; x < s.size(); ++x) {
                const uint64_t i = s[x];
                const bool has_prev = x > 0 && s[x - 1] == i - 1, has_next = x + 1 < s.size() && s[x + 1] == i + 1;
                if (i % 2 == 1) { if (!has_prev) sib.push_back(nodes[i - 1]); }
                else if (!has_next && i + 1 < level_len) sib.push_back(nodes[i + 1]);
            }
            std::vector<uint64_t> up;
            for (uint64_t i : s)
                if (up.empty() || up.back() != i / 2) up.push_back(i / 2);
            s = up;
            level_len = (level_len + 1) / 2;
        }
        return sib;
    }
};

// Bytes of chunk 0 of A: the nonce at A[0, 0:8], zeros elsewhere (k >= 2048 > 1024: chunk 0 lies in row 0)
inline void nonce_chunk(const int8_t nonce[PEARL_NONCE_BYTES], uint8_t chunk[1024]) {
    memset(chunk, 0, 1024);
    memcpy(chunk, nonce, PEARL_NONCE_BYTES);
}

// What a job leaves on the host: keys, seeds, both trees (A's with leaf 0 replaced every pass)
struct JobState {
    bool ready = false;
    CV job_key, hash_b, seed_b, hash_a, seed_a;
    uint32_t m = 0, n = 0, k = 0;
    Tree tree_a, tree_bt;
    bool pass_ready = false;
};

inline CV leaf0_of(const JobState &j, const int8_t nonce[PEARL_NONCE_BYTES]) {
    uint8_t chunk[1024];
    nonce_chunk(nonce, chunk);
    CV leaf;
    chunk_cv(j.job_key.data(), chunk, 0, leaf.data());
    return leaf;
}

// job: trees from the zero-chunk leaves (computed by the caller, on the device or on the host), hash_b, seed_B
inline void job_seeds(JobState &j, std::vector<CV> leaves_bt, std::vector<CV> leaves_a) {
    j.tree_bt.build(j.job_key, std::move(leaves_bt));
    j.tree_a.build(j.job_key, std::move(leaves_a));
    j.hash_b = j.tree_bt.root;
    j.seed_b = hash_pair(j.job_key, salted(j.hash_b, j.n, 'B'));
}

// pass: the nonce leaf, the path, hash_a, seed_A
inline void pass_seeds(JobState &j, const int8_t nonce[PEARL_NONCE_BYTES]) {
    j.tree_a.set_leaf0(leaf0_of(j, nonce));
    j.hash_a = j.tree_a.root;
    j.seed_a = hash_pair(j.seed_b, salted(j.hash_a, j.m, 'A'));
    j.pass_ready = true;
}

inline std::string check_job(uint32_t m, uint32_t n, uint32_t k, uint32_t r) {
    if (r != PEARL_R) return "r must be 128";
    if ((uint64_t)m * k % 1024 || (uint64_t)n * k % 1024 || (uint64_t)m * k < 2048 || (uint64_t)n * k < 2048)
        return "m*k and n*k must be multiples of 1024 with >= 2 chunks";
    return "";
}

// pearl_tree_nodes body: siblings and root of A (this pass) or B^T (this job); "" or the reason it failed
inline std::string tree_nodes(const JobState &j, int matrix, const uint64_t *idx, uint32_t n, uint8_t *out,
                              uint32_t cap, uint32_t *n_sib, uint8_t root_out[32]) {
    if (!j.ready || (matrix == 0 && !j.pass_ready)) return "pearl_job / pearl_pass first";
    if (matrix != 0 && matrix != 1) return "matrix: 0 = A, 1 = B^T";
    const Tree &t = matrix == 0 ? j.tree_a : j.tree_bt;
    std::vector<uint64_t> leaves(idx, idx + n);
    for (uint64_t x : leaves)
        if (x >= t.layers[0].size()) return "leaf index out of range";
    const std::vector<CV> sib = t.siblings(leaves);
    *n_sib = (uint32_t)sib.size();
    if (sib.size() > cap) return "siblings over cap";
    for (size_t i = 0; i < sib.size(); ++i) to_bytes(sib[i], out + 32 * i);
    to_bytes(t.root, root_out);
    return "";
}

// ---------------------------------------------------------------- the same steps on the CPU (emulators, self-check)

inline std::vector<CV> zero_leaves(const CV &key, uint64_t chunks) {
    std::vector<CV> out(chunks);
    for (uint64_t c = 0; c < chunks; ++c) chunk_cv(key.data(), nullptr, c, out[c].data());
    return out;
}

// dense factor rows x r and the sparse indices of k rows from a seed
inline void factors_cpu(const CV &seed, char which, uint32_t rows, uint32_t k, uint32_t r, std::vector<int8_t> &dense,
                        std::vector<uint8_t> &p, std::vector<uint8_t> &q) {
    const CV lab = label(which);
    dense.resize((size_t)rows * r);
    for (uint32_t b = 0; b < rows * (r / 32); ++b) {
        uint32_t out[8];
        prg_block(seed.data(), 0, b, lab.data(), out);
        for (int y = 0; y < 32; ++y) dense[(size_t)b * 32 + y] = uniform_value(out[y / 4] >> (8 * (y % 4)));
    }
    p.resize(k);
    q.resize(k);
    for (uint32_t i = 0; i < (k + 7) / 8; ++i) {
        uint32_t out[8];
        prg_block(seed.data(), 1, i, lab.data(), out);
        for (uint32_t w = 0; w < 8 && 8 * i + w < k; ++w) sparse_pq(out[w], r, p[8 * i + w], q[8 * i + w]);
    }
}

// X'[i, l] = D[i, p_l] - D[i, q_l] (+ nonce in row 0)
inline void build_cpu(const std::vector<int8_t> &dense, const std::vector<uint8_t> &p, const std::vector<uint8_t> &q,
                      uint32_t rows, uint32_t k, uint32_t r, const int8_t *nonce, std::vector<int8_t> &out) {
    out.resize((size_t)rows * k);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t l = 0; l < k; ++l)
            out[(size_t)i * k + l] = (int8_t)(dense[(size_t)i * r + p[l]] - dense[(size_t)i * r + q[l]] +
                                              (nonce && i == 0 && l < PEARL_NONCE_BYTES ? nonce[l] : 0));
}

// The whole resident path on the CPU (emulators): job -> B'^T, pass -> A'
inline void job_cpu(JobState &j, const uint8_t job_key[32], uint32_t m, uint32_t n, uint32_t k,
                    std::vector<int8_t> &bt) {
    j = JobState();
    j.job_key = from_bytes(job_key);
    j.m = m; j.n = n; j.k = k;
    job_seeds(j, zero_leaves(j.job_key, (uint64_t)n * k / 1024), zero_leaves(j.job_key, (uint64_t)m * k / 1024));
    std::vector<int8_t> dense;
    std::vector<uint8_t> p, q;
    factors_cpu(j.seed_b, 'B', n, k, PEARL_R, dense, p, q);
    build_cpu(dense, p, q, n, k, PEARL_R, nullptr, bt);
    j.ready = true;
}

inline void pass_cpu(JobState &j, const int8_t nonce[PEARL_NONCE_BYTES], std::vector<int8_t> &a) {
    pass_seeds(j, nonce);
    std::vector<int8_t> dense;
    std::vector<uint8_t> p, q;
    factors_cpu(j.seed_a, 'A', j.m, j.k, PEARL_R, dense, p, q);
    build_cpu(dense, p, q, j.m, j.k, PEARL_R, nonce, a);
}

}  // namespace pearl
