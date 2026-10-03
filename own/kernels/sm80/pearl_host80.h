// Host-side checks and the exact reference for the sm_80+ kernel's hash tile (8x16, periods 64 x 64).
#pragma once
#include <string.h>
#include <string>
#include "../common/host_core.h"
#include "pearl_layout80.h"

namespace pearl80 {

struct Dims {
    uint32_t m = 0, n = 0, k = 0;
    uint32_t L() const { return k - k % PEARL_R; }
    uint32_t row_tiles() const { return m / PEARL_ROW_PERIOD * 8; }
    uint32_t col_tiles() const { return n / PEARL_COL_PERIOD * 4; }
};

inline pearl::Axis rows_axis() { return {{0, 8, 16, 24, 32, 40, 48, 56}, {0, 1, 2, 3, 4, 5, 6, 7}, 64}; }
inline pearl::Axis cols_axis() {
    return {{0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57}, {0, 2, 4, 6}, 64};
}

inline std::string check_dims(uint32_t m, uint32_t n, uint32_t k) {
    if (m == 0 || m % PEARL_ROW_PERIOD) return "m must be a positive multiple of 64 (row period)";
    if (n == 0 || n % PEARL_COL_PERIOD) return "n must be a positive multiple of 64 (column period)";
    if (k % 64 || k < 2048 || k > 65536) return "k must be a multiple of 64 in [2048, 65536]";
    return "";
}

inline std::string check_search(const Dims &d, uint32_t k, uint32_t r, const uint8_t *rows, const uint8_t *cols,
                                uint32_t lo, uint32_t hi) {
    if (r != PEARL_R) return "r must be 128";
    if (k != d.k) return "k differs from the resident operands";
    if (rows && cols && (memcmp(rows, PEARL_ROWS_PATTERN, 6) || memcmp(cols, PEARL_COLS_PATTERN, 6)))
        return "pattern: the sm80 kernel needs the 8x16 tile, rows 07 07 00 00 00 00, cols 00 01 03 07 00 00";
    if (lo % 8 || hi % 8 || lo > hi || hi > d.row_tiles()) return "row tiles: lo, hi multiples of 8, lo <= hi <= m/8";
    return "";
}

inline bool pattern_mismatch(const uint8_t *rows, const uint8_t *cols) {
    return rows && cols && (memcmp(rows, PEARL_ROWS_PATTERN, 6) || memcmp(cols, PEARL_COLS_PATTERN, 6));
}

inline void ref_transcripts(const int8_t *a, const int8_t *bt, const Dims &d, uint32_t lo, uint32_t hi,
                            uint32_t *out) {
    pearl::ref_transcripts_geom(a, bt, d.n, d.k, rows_axis(), cols_axis(), lo, hi, out);
}

}  // namespace pearl80
