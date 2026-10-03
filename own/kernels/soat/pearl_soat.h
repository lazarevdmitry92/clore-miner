/* GPU backend of the own Pearl miner over the SOAT kernels (blindrun/soat-miner 48defc8, MIT, see LICENSE.soat).
 *
 * Contract: miner/README.md, "Контракт GPU-бэкенда". What this backend can do:
 *   rows and cols pattern = contiguous 16 (bytes 00 0f 00 00 00 00), r = 128, k a multiple of 128 (SOAT itself mines
 *   only k = 2048; other k pass the start-up self-check at that k or are refused), m and n multiples of 16,
 *   compute capability >= 8.0. Anything else is refused with an error code, never computed approximately.
 * Fast path: rows of a call ((row_tile_hi - row_tile_lo) * 16) and n multiples of 256 — otherwise the call runs
 * padded to 256 and reads back every hit.
 * Tile numbering = Pattern.partition of the contiguous-16 pattern: row_tile t = rows [16t, 16t+16),
 * col_tile u = columns [16u, 16u+16).
 */
#ifndef PEARL_SOAT_H
#define PEARL_SOAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
typedef struct {
    uint32_t row_tile, col_tile;
    uint8_t jackpot[32];
} cand_t;  /* 40 bytes */
#pragma pack(pop)

enum {
    PEARL_OK = 0,
    PEARL_E_ARG = 1,          /* null pointer, empty or out-of-range tile range */
    PEARL_E_K_R = 2,          /* r != 128 or k not a multiple of 128 (or > 65536) */
    PEARL_E_PATTERN = 3,      /* pattern is not contiguous 16 */
    PEARL_E_SHAPE = 4,        /* m or n not a multiple of 16, or no SOAT tile configuration runs the shape */
    PEARL_E_CUDA = 5,         /* CUDA runtime error (message in pearl_last_error) */
    PEARL_E_SELFCHECK = 6,    /* device disagrees with the host reference: refuse to mine */
    PEARL_E_ARCH = 7          /* compute capability below 8.0 */
};

int pearl_search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                 const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                 const uint8_t seed_a[32], const uint8_t bound_le[32],
                 uint32_t row_tile_lo, uint32_t row_tile_hi,
                 cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);

/* The same, with operand ids (Operands.a_id / b_id, unique within the process; 0 = no cache, copy).
 * Same id + same host pointer + same dims as the previous call -> the device copy is reused. */
int pearl_search_ids(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                     const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                     const uint8_t seed_a[32], const uint8_t bound_le[32],
                     uint32_t row_tile_lo, uint32_t row_tile_hi,
                     cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs,
                     uint64_t a_id, uint64_t b_id);

/* The one card of this process (CUDA ordinal PEARL_SOAT_DEVICE, default 0) — the signature miner/kernel.py SoBackend
 * reads: SM count and the CUDA device name (truncated to name_len - 1). 0 = ok. */
int pearl_device_info(uint32_t *sm_count, char *name, uint32_t name_len);

/* Human-readable reason of the last non-zero return; valid until the next call into the library. */
const char *pearl_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
