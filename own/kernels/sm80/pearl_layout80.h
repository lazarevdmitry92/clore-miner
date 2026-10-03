// The hash tile and the lane -> data mapping of the sm_80+ kernel (pearl_sm80.cu), shared with its CPU emulator.
//
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (PTX ISA, "Matrix Fragments for mma.m16n8k32", integer);
// g = lane / 4 (groupID), t = lane % 4 (threadID_in_group):
//   A 16x32 .row : a0 row g, k 4t..4t+3 | a1 row g+8, k 4t.. | a2 row g, k 16+4t.. | a3 row g+8, k 16+4t..
//   B 32x8  .col : b0 col g, k 4t..4t+3 | b1 col g, k 16+4t..      (4 consecutive bytes of a row of B'^T)
//   C 16x8  s32  : c0, c1 row g, cols 2t, 2t+1 | c2, c3 row g+8, cols 2t, 2t+1
// ldmatrix.m8n8.x4.b16: lanes 8q..8q+7 give the 8 row addresses of matrix q; register q of lane T = row T/4,
// bytes 4(T%4)..+3 of matrix q — exactly the A and B fragments above when matrix q = (row half q&1, k half q>>1).
//
// A warp tile 64x64 = 4 m-reps (16 rows) x 8 n-reps (8 cols): a thread holds rows g + 8i' (i' < 8) and columns
// 2t + {0,1} + 8j (j < 8) = one hash tile 8x16 (s0 §6): rows_pattern [0, 8, ..., 56], cols_pattern
// [0, 1, 8, 9, ..., 56, 57], h*w = 128 int32 accumulators, XOR in registers. Host tile "8x16" (miner/jobs.py).
#pragma once
#include "../common/pearl_core.h"

#define PEARL_H 8
#define PEARL_W 16
#define PEARL_ROW_PERIOD 64
#define PEARL_COL_PERIOD 64
static const uint8_t PEARL_ROWS_PATTERN[6] = {0x07, 0x07, 0x00, 0x00, 0x00, 0x00};  // (8, 8)
static const uint8_t PEARL_COLS_PATTERN[6] = {0x00, 0x01, 0x03, 0x07, 0x00, 0x00};  // (1, 2), (8, 8)

PH uint32_t pat_row(int i) { return (uint32_t)(8 * i); }
PH uint32_t pat_col(int j) { return (uint32_t)((j & 1) + 8 * (j >> 1)); }

// Pattern.partition order: valid row offsets in a period of 64 are 0..7, column offsets 0, 2, 4, 6
PH uint32_t row_tile_offset(uint32_t t) { return (t >> 3) * 64 + (t & 7); }
PH uint32_t col_tile_offset(uint32_t t) { return (t >> 2) * 64 + 2 * (t & 3); }

#define K_TILE 64          // k bytes per shared-memory stage = two mma k-steps; 4 chunks of 16 bytes per row

// Byte offset of 16-byte chunk c (0..3) of shared row r (row = 64 bytes). XOR swizzle: the 8 consecutive rows one
// ldmatrix phase reads, and the 2 rows x 4 chunks 8 consecutive cp.async lanes write, land in 8 distinct bank groups.
PH int smem_off(int r, int c) { return r * 64 + 16 * (c ^ ((r >> 1) & 3)); }

// ldmatrix row (within the warp's 64 rows of A' or of B'^T) and chunk (k half of a k-step) of lane l
PH int ld_row(int l) { return (l & 7) + ((l >> 3) & 1) * 8; }
PH int ld_chunk(int l) { return l >> 4; }

// warp-local position of accumulator e (0..3) of m-rep i (0..3), n-rep j (0..7)
PH int acc_row(int l, int i, int e) { return 16 * i + (l >> 2) + 8 * (e >> 1); }
PH int acc_col(int l, int j, int e) { return 8 * j + 2 * (l & 3) + (e & 1); }

// the thread's hash tile: warp row band and warp column base are absolute multiples of 64
PH uint32_t lane_row_tile(uint32_t band, int l) { return band / 64 * 8 + (l >> 2); }
PH uint32_t lane_col_tile(uint32_t wcol, int l) { return wcol / 64 * 4 + (l & 3); }
