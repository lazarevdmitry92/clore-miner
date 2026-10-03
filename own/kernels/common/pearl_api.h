// C ABI of every Pearl kernel library (kernels/v100 libpearl_v100.so, kernels/sm80 libpearl_sm80.so) and of their
// CPU emulators: same functions, same semantics, so miner/kernel.py SoBackend and the tests run against any of them.
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
#include "pearl_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PEARL_OK 0
#define PEARL_ERR_ARGS -1        // dimensions, tile range, r, k
#define PEARL_ERR_PATTERN -2     // a pattern other than the kernel's PEARL_ROWS_PATTERN x PEARL_COLS_PATTERN
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

// Row tiles [row_tile_lo, row_tile_hi) x all column tiles of the resident operands. lo, hi multiples of one row
// period in tiles (V100: 4 = 32 rows, sm80: 8 = 64 rows). count = all candidates found, min(count, cap) written
// to out sorted by (row, col).
int pearl_search_resident(uint32_t k, uint32_t r, const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                          const uint8_t seed_a[32], const uint8_t bound_le[32],
                          uint32_t row_tile_lo, uint32_t row_tile_hi,
                          cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);

int pearl_search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                 const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                 const uint8_t seed_a[32], const uint8_t bound_le[32],
                 uint32_t row_tile_lo, uint32_t row_tile_hi,
                 cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs);

// ---- Resident path (TZ_operands_gpu.md §3): A = B = 0 except the nonce at A[0, 0:8]; operands built on the device.
// Per job: job_key -> Merkle trees of B^T and of A (nonce chunk aside), hash_b and seed_B out, B' resident.
int pearl_job(const uint8_t job_key[32], uint32_t m, uint32_t n, uint32_t k, uint32_t r,
              const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
              uint8_t hash_b_out[32], uint8_t seed_b_out[32]);
// Per pass: the nonce (8 values in [-64, 64]) -> hash_a and seed_A out (the path from the nonce leaf to the root is
// recomputed, nothing else), A' resident. Then pearl_search_resident(k, r, patterns, seed_a_out, bound, lo, hi, ...)
// over portions of row tiles.
int pearl_pass(const int8_t nonce[8], uint8_t hash_a_out[32], uint8_t seed_a_out[32]);
// PlainProof (ref/pearl_ref.py Tree.multiproof): siblings of the leaves (sorted, unique) of matrix 0 = A (this
// pass) or 1 = B^T (this job), 32 bytes each, level by level from the leaves in ascending index; and the root.
int pearl_tree_nodes(int matrix, const uint64_t *leaf_idx, uint32_t n_leaves, uint8_t *siblings_out, uint32_t cap,
                     uint32_t *n_siblings, uint8_t root_out[32]);

// Transcripts (hi-lo, col_tiles, 16) u32 of the resident operands — the oracle comparison of acceptance test 1.
int pearl_transcripts(uint32_t k, uint32_t row_tile_lo, uint32_t row_tile_hi, uint32_t *out);

// Variants of the fast kernel (mining ones "v100-*" / "sm80-*", bench-only ablations "ablate-*").
int pearl_variant_count(void);
const char *pearl_variant_name(int i);
// miner/kernel.py SoBackend: the mining variants this card can run, comma-separated (the supervisor tunes over them)
int pearl_variants(char *names, uint32_t len);
// Pin a mining variant (switches the online autotune off); "" or NULL unpins.
int pearl_set_variant(const char *name);
// pearl_bench: run any variant (ablations too) over row tiles [lo, hi) of the resident operands, no candidates.
int pearl_bench_run(const char *name, uint32_t row_tile_lo, uint32_t row_tile_hi, uint64_t *macs);
// TZ v2 §5: {"kernel","path","k_expected","k_actual","mac_s","sm_mhz","power_w","throttle":[...],
// "online_autotune": "measuring"|"hold"|"off"} for /summary. Online autotune: env PEARL_ONLINE=0 turns it off,
// PEARL_ONLINE_SECS (90) per candidate after PEARL_ONLINE_SETTLE (15) s, re-run after PEARL_ONLINE_HOLD (3600) s.
int pearl_telemetry(char *json, uint32_t len);

// Which path the next search takes. V100: 1 = fast (HMMA, single fp32 running sum, proven exact by Cauchy-Schwarz:
// max|a_i|^2 * max|b_j|^2 < 2^44), 0 = exact (dp4a, int32); env PEARL_PATH=fast|exact overrides (tests, bench).
// sm80+: always 1 (int8 tensor cores accumulate int32 exactly).
int pearl_fast_path(void);

#ifdef __cplusplus
}
#endif
