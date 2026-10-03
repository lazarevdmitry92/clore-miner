// Host-side helpers shared by the CUDA library and the emulator: argument checks, the exact reference transcript
// (int64, independent of any GPU trick), jackpot/bound handling, the Cauchy-Schwarz exactness test, synthetic
// Pearl-like operands for the self-check and the bench.
#pragma once
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "pearl_api.h"

namespace pearl {

struct Dims {
    uint32_t m = 0, n = 0, k = 0;
    uint32_t L() const { return k - k % PEARL_R; }
    uint32_t row_tiles() const { return m / PEARL_ROW_PERIOD * 4; }
    uint32_t col_tiles() const { return n / PEARL_COL_PERIOD * 4; }
};

inline std::string check_dims(uint32_t m, uint32_t n, uint32_t k) {
    if (m == 0 || m % PEARL_ROW_PERIOD) return "m must be a positive multiple of 32 (row period)";
    if (n == 0 || n % PEARL_COL_PERIOD) return "n must be a positive multiple of 64 (column period)";
    if (k % 64 || k < 2048 || k > 65536) return "k must be a multiple of 64 in [2048, 65536]";
    return "";
}

inline std::string check_search(const Dims &d, uint32_t k, uint32_t r, const uint8_t *rows, const uint8_t *cols,
                                uint32_t lo, uint32_t hi) {
    if (r != PEARL_R) return "r must be 128";
    if (k != d.k) return "k differs from the resident operands";
    if (rows && cols && (memcmp(rows, PEARL_ROWS_PATTERN, 6) || memcmp(cols, PEARL_COLS_PATTERN, 6)))
        return "pattern: the V100 kernel needs rows 01 01 01 03 00 00, cols 00 01 01 01 01 03";
    if (lo % 4 || hi % 4 || lo > hi || hi > d.row_tiles()) return "row tiles: lo, hi multiples of 4, lo <= hi <= m/8";
    return "";
}

// Exact transcripts (hi-lo, col_tiles, 16) of row tiles [lo, hi): int64 running sums, XOR as u32 every r.
inline void ref_transcripts(const int8_t *a, const int8_t *bt, const Dims &d, uint32_t lo, uint32_t hi,
                            uint32_t *out) {
    const uint32_t ct = d.col_tiles(), L = d.L();
    std::vector<int64_t> S(PEARL_H * PEARL_W);
    for (uint32_t rt = lo; rt < hi; ++rt)
        for (uint32_t c = 0; c < ct; ++c) {
            uint32_t ro = row_tile_offset(rt), co = col_tile_offset(c);
            uint32_t *T = out + ((size_t)(rt - lo) * ct + c) * 16;
            memset(T, 0, 64);
            std::fill(S.begin(), S.end(), 0);
            for (uint32_t ch = 0; ch < L / PEARL_R; ++ch) {
                uint32_t x = 0;
                for (int i = 0; i < PEARL_H; ++i) {
                    const int8_t *ar = a + (size_t)(ro + pat_row(i)) * d.k + ch * PEARL_R;
                    for (int j = 0; j < PEARL_W; ++j) {
                        const int8_t *br = bt + (size_t)(co + pat_col(j)) * d.k + ch * PEARL_R;
                        int64_t s = 0;
                        for (int l = 0; l < PEARL_R; ++l) s += (int32_t)ar[l] * br[l];
                        S[i * PEARL_W + j] += s;
                        x ^= (uint32_t)S[i * PEARL_W + j];
                    }
                }
                transcript_step(T, ch, x);
            }
        }
}

inline void words_le(const uint8_t *b, uint32_t *w, int n) {
    for (int i = 0; i < n; ++i) w[i] = b[4 * i] | b[4 * i + 1] << 8 | b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
}

inline void jackpot_bytes(const uint32_t h[8], uint8_t out[32]) {
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b) out[4 * i + b] = (uint8_t)(h[i] >> (8 * b));
}

// max over rows of the squared L2 norm (first L elements are all that is ever summed; k is conservative)
inline uint32_t max_row_sumsq(const int8_t *x, uint32_t rows, uint32_t k) {
    uint32_t best = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t s = 0;
        for (uint32_t l = 0; l < k; ++l) s += (uint32_t)((int32_t)x[(size_t)r * k + l] * x[(size_t)r * k + l]);
        best = std::max(best, s);
    }
    return best;
}

// Every partial sum of a.b (any subset of terms, any order) is bounded by |a||b| (Cauchy-Schwarz); below 2^22 the
// biased fp32 accumulator 1.5*2^23 + S never leaves [2^23, 2^24) and holds S exactly.
inline bool cs_exact(uint32_t max_a2, uint32_t max_b2) { return (uint64_t)max_a2 * max_b2 < (1ull << 44); }

inline void sort_cands(cand_t *c, uint32_t n) {
    std::sort(c, c + n, [](const cand_t &x, const cand_t &y) {
        return x.row_tile != y.row_tile ? x.row_tile < y.row_tile : x.col_tile < y.col_tile;
    });
}

// Pearl-like operands: dist 0 = noise E_L[p] - E_L[q], E_L uniform [-32, 31] (triangular [-63, 63], what the miner
// feeds with A = B = 0); 1 = uniform [-127, 127]; 2 = only +-127.
inline std::vector<int8_t> synth(size_t count, int dist, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<int8_t> v(count);
    for (auto &x : v) {
        if (dist == 0) x = (int8_t)(((int)(g() & 63) - 32) - ((int)(g() & 63) - 32));
        else if (dist == 1) x = (int8_t)((int)(g() % 255) - 127);
        else x = (g() & 1) ? 127 : -127;
    }
    return v;
}

}  // namespace pearl
