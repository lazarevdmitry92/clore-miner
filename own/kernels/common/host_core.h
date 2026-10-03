// Host-side helpers shared by every Pearl kernel library, emulator and bench: byte/word conversions of the jackpot,
// candidate ordering, synthetic Pearl-like operands, and the exact int64 reference transcript for any periodic
// hash-tile geometry.
#pragma once
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <vector>
#include "pearl_core.h"

namespace pearl {

inline void words_le(const uint8_t *b, uint32_t *w, int n) {
    for (int i = 0; i < n; ++i) w[i] = b[4 * i] | b[4 * i + 1] << 8 | b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
}

inline void jackpot_bytes(const uint32_t h[8], uint8_t out[32]) {
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b) out[4 * i + b] = (uint8_t)(h[i] >> (8 * b));
}

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

// Pearl noise for big operands, fast (synth is mt19937 and slow at 10^8): E_L[p] - E_L[q], E_L uniform [-32, 31].
inline void fill_noise(std::vector<int8_t> &v, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
    for (auto &b : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        b = (int8_t)(((int)(x & 63) - 32) - ((int)((x >> 8) & 63) - 32));
    }
}

// A periodic hash-tile geometry: pattern elements, period and the valid tile offsets inside one period (ascending,
// = Pattern.partition order). Tile t: offset (t / per_period) * period + offs[t % per_period].
struct Axis {
    std::vector<uint32_t> elems, offs;
    uint32_t period;
    uint32_t tile_offset(uint32_t t) const { return t / offs.size() * period + offs[t % offs.size()]; }
    uint32_t tiles(uint32_t dim) const { return dim / period * (uint32_t)offs.size(); }
};

// Exact transcripts (hi-lo, col_tiles, 16) of row tiles [lo, hi): int64 running sums, XOR as u32 every r.
inline void ref_transcripts_geom(const int8_t *a, const int8_t *bt, uint32_t n, uint32_t k, const Axis &rows,
                                 const Axis &cols, uint32_t lo, uint32_t hi, uint32_t *out) {
    const uint32_t ct = cols.tiles(n), L = k - k % PEARL_R, h = (uint32_t)rows.elems.size(), w = (uint32_t)cols.elems.size();
    std::vector<int64_t> S((size_t)h * w);
    for (uint32_t rt = lo; rt < hi; ++rt)
        for (uint32_t c = 0; c < ct; ++c) {
            const uint32_t ro = rows.tile_offset(rt), co = cols.tile_offset(c);
            uint32_t *T = out + ((size_t)(rt - lo) * ct + c) * 16;
            memset(T, 0, 64);
            std::fill(S.begin(), S.end(), 0);
            for (uint32_t ch = 0; ch < L / PEARL_R; ++ch) {
                uint32_t x = 0;
                for (uint32_t i = 0; i < h; ++i) {
                    const int8_t *ar = a + (size_t)(ro + rows.elems[i]) * k + ch * PEARL_R;
                    for (uint32_t j = 0; j < w; ++j) {
                        const int8_t *br = bt + (size_t)(co + cols.elems[j]) * k + ch * PEARL_R;
                        int64_t s = 0;
                        for (int l = 0; l < PEARL_R; ++l) s += (int32_t)ar[l] * br[l];
                        S[i * w + j] += s;
                        x ^= (uint32_t)S[i * w + j];
                    }
                }
                transcript_step(T, ch, x);
            }
        }
}

}  // namespace pearl
