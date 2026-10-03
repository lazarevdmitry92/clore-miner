// Lane -> data mapping of the fast kernel, shared by pearl_v100.cu and the emulator (pearl_emu.cpp checks it
// against the PTX fragment layout of mma.m8n8k4 and the shared-memory bank rules).
//
// PTX ISA, "Matrix Fragments for mma.m8n8k4" (.f16 A/B, .f32 C/D); quad-pair q = lanes 4q..4q+3 and 16+4q..16+4q+3:
//   A .row : lane holds row (l%4) + 4*(l>=16), k = 0..3            (4 consecutive bytes of a row of A')
//   B .col : lane holds col (l%4) + 4*(l>=16), k = 0..3            (4 consecutive bytes of a row of B'^T)
//   C .f32 : c[e], row (l&1) + (e&2) + 4*(l>=16), col (e&4) + (l&2) + (e&1)
#pragma once
#include "pearl_common.h"

#define K_TILE 32          // k per shared-memory stage: 32 halves = 64 bytes per row = 4 slots of 16 B
#define SLOTS 4

PH int lane_qr(int l) { return (l >> 3) & 1; }   // quad-pair row in the warp (quad-pairs 0,1 -> 0; 2,3 -> 1)
PH int lane_qc(int l) { return (l >> 2) & 1; }   // quad-pair column (interleaved by 8)
PH int lane_hi(int l) { return l >> 4; }

// warp-local row of A' that lane l feeds into mma rep i (0..3), warp-local row of B'^T for rep j (0..3)
PH int lane_a_row(int l, int i) { return 32 * lane_qr(l) + 8 * i + (l & 3) + 4 * lane_hi(l); }
PH int lane_b_row(int l, int j) { return 16 * j + 8 * lane_qc(l) + (l & 3) + 4 * lane_hi(l); }

// warp-local position of accumulator e (0..7) of rep (i, j)
PH int acc_row(int l, int i, int e) { return 32 * lane_qr(l) + 8 * i + (l & 1) + (e & 2) + 4 * lane_hi(l); }
PH int acc_col(int l, int j, int e) { return 16 * j + 8 * lane_qc(l) + (e & 4) + (l & 2) + (e & 1); }

// hash tile of the thread: row band (multiple of 32) and warp column base (multiple of 64) are absolute
PH uint32_t lane_row_tile(uint32_t band, int l) { return band / 32 * 4 + (l & 1) + 2 * lane_hi(l); }
PH uint32_t lane_col_tile(uint32_t wcol, int l) { return wcol / 64 * 4 + 2 * lane_qc(l) + ((l >> 1) & 1); }

// Byte offset of 16-byte slot s (k 8s..8s+7, fp16) of shared row r. XOR swizzle: the 8 lanes of one LDS.128 phase
// read rows {0..3} and {8..11} (+4 for lanes 16..31) of B and duplicate rows of A — all in distinct bank groups.
PH int smem_off(int r, int s) { return r * 64 + 16 * (s ^ (((r >> 1) & 1) | (((r >> 3) & 1) << 1))); }
