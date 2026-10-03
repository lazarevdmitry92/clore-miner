// Shared by the CUDA kernels (pearl_v100.cu), the CPU emulator (pearl_emu.cpp) and the host self-check:
// the hash-tile pattern of the V100 kernel and tile <-> row/col mapping. Must stay bit-identical host/device.
#pragma once
#include "../common/pearl_core.h"

// ---------------------------------------------------------------- the hash tile (s0 §2.3, §6; README §Шаблон)
// mma.m8n8k4 f32 accumulators: lane l of a quad-pair holds rows (l&1) + 4*(l>=16) + {0,2} and columns
// (l&2) + {0,1,4,5} of the 8x8 output. A warp tile 64x64 = 2x2 quad-pairs; each quad-pair repeats the 8x8 mma
// 4x4 times: rows step 8 (quad-pair band of 32 rows), columns step 16 (the two quad-pairs of a band interleave by 8).
// So a thread owns rows {0,2,8,10,16,18,24,26} x cols {0,1,4,5,16,17,20,21,32,33,36,37,48,49,52,53} = h*w 128.
#define PEARL_H 8
#define PEARL_W 16
#define PEARL_ROW_PERIOD 32
#define PEARL_COL_PERIOD 64
static const uint8_t PEARL_ROWS_PATTERN[6] = {0x01, 0x01, 0x01, 0x03, 0x00, 0x00};  // (2,2),(8,4)
static const uint8_t PEARL_COLS_PATTERN[6] = {0x00, 0x01, 0x01, 0x01, 0x01, 0x03};  // (1,2),(4,2),(16,4)

// element i < 8 of the row pattern, j < 16 of the column pattern
PH uint32_t pat_row(int i) { return (uint32_t)((i & 1) * 2 + (i >> 1) * 8); }
PH uint32_t pat_col(int j) { return (uint32_t)((j & 1) + ((j >> 1) & 1) * 4 + (j >> 2) * 16); }

// Tile numbering = Pattern.partition order (t-th valid offset). Row offsets valid in a period of 32: {0,1,4,5};
// column offsets in a period of 64: {0,2,8,10}.
PH uint32_t row_tile_offset(uint32_t t) { return (t >> 2) * 32 + (t & 1) + ((t >> 1) & 1) * 4; }
PH uint32_t col_tile_offset(uint32_t t) { return (t >> 2) * 64 + (t & 1) * 2 + ((t >> 1) & 1) * 8; }

// Running sums kept in fp32 as bits = 0x4B400000 + S (|S| < 2^22, exponent fixed, ulp 1). For an even number of
// such values, XOR of the int32 sums = XOR of the raw bits sign-extended from bit 22: the low 22 bits are the
// same, the 10 high bits are all ones iff an odd number of sums is negative (bit 22 clear), and since the count
// is even that parity equals the parity of the set bits 22 (s0 §9a).
PH uint32_t magic_xor_to_int(uint32_t x) { return (uint32_t)(((int32_t)(x << 9)) >> 9); }

