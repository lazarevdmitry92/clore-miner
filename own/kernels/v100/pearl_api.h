// C ABI of the V100 backend (libpearl_v100.so) and of its CPU emulator (libpearl_emu.*): same functions, same
// semantics, so miner/kernel.py SoBackend and the tests run against either.
//
// One process = one device. pearl_search is the contract of miner/README.md "Контракт GPU-бэкенда" (loaded by
// miner/kernel.py SoBackend): it initialises the device on first use (env PEARL_DEVICE, default 0; with
// CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES per process for a multi-GPU server) and uploads B'^T and
// the rows of A' of its portion on every call. The resident API (pearl_set_a/b + pearl_search_resident) keeps
// B'^T for a job and A' for a pass on the card; the bench uses it, and so can a backend that tracks a_id / b_id.
//
// Return codes: 0 ok, <0 error (text in pearl_last_error()).
#pragma once
#include <stdint.h>
#include "pearl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PEARL_OK 0
#define PEARL_ERR_ARGS -1        // dimensions, tile range, r, k
#define PEARL_ERR_PATTERN -2     // a pattern other than PEARL_ROWS_PATTERN x PEARL_COLS_PATTERN
#define PEARL_ERR_CUDA -3
#define PEARL_ERR_SELFCHECK -4   // the startup self-check failed: the backend refuses to mine
#define PEARL_ERR_STATE -5       // not initialised / operands not set

// Select the device, run the self-check (fast and exact kernels against the host reference) and the autotune.
// Called implicitly by the first other call with PEARL_DEVICE; a thread that calls it uses that device.
int pearl_init(int device);
const char *pearl_last_error(void);
const char *pearl_kernel_name(void);   // chosen variant, e.g. "v100-hmma884-128x256" (for /summary "kernel")
// SoBackend's optional export: SM count and "<name> <pci bus id> <kernel>" into name (name_len bytes)
int pearl_device_info(uint32_t *sm_count, char *name, uint32_t name_len);

int pearl_set_a(const int8_t *a, uint32_t m, uint32_t k);    // A'  row-major m x k
int pearl_set_b(const int8_t *bt, uint32_t n, uint32_t k);   // B'^T row-major n x k

// Row tiles [row_tile_lo, row_tile_hi) x all column tiles of the resident operands. lo, hi multiples of 4
// (one row period). count = all candidates found, min(count, cap) written to out sorted by (row, col).
int pearl_search_resident(uint32_t k, uint32_t r, const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                          const uint8_t seed_a[32], const uint8_t bound_le[32],
                          uint32_t row_tile_lo, uint32_t row_tile_hi,
                          cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);

int pearl_search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                 const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                 const uint8_t seed_a[32], const uint8_t bound_le[32],
                 uint32_t row_tile_lo, uint32_t row_tile_hi,
                 cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);

// Transcripts (hi-lo, col_tiles, 16) u32 of the resident operands — the oracle comparison of acceptance test 1.
int pearl_transcripts(uint32_t k, uint32_t row_tile_lo, uint32_t row_tile_hi, uint32_t *out);

// Which path the next search takes: 1 = fast (HMMA, single fp32 running sum, proven exact by Cauchy-Schwarz:
// max|a_i|^2 * max|b_j|^2 < 2^44), 0 = exact (dp4a, int32). Env PEARL_PATH=fast|exact overrides (tests, bench).
int pearl_fast_path(void);

#ifdef __cplusplus
}
#endif
