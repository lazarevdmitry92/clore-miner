// Shared by the CUDA kernels (pearl_v100.cu), the CPU emulator (pearl_emu.cpp) and the host self-check:
// the hash-tile pattern of the V100 kernel, tile <-> row/col mapping, the transcript step, keyed BLAKE3 of one
// block and the jackpot comparison. Everything here must stay bit-identical between host and device.
//
// BLAKE3 single-block compression follows SOAT `noisy_gemm.cuh` blake3KeyedBlock
// (github.com/blindrun/soat-miner, MIT, (c) the SOAT authors) and Pearl `csrc/blake3/blake3.cuh` (ISC).
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define PH __host__ __device__ __forceinline__
#else
#define PH static inline
#endif

// ---------------------------------------------------------------- the hash tile (s0 §2.3, §6; README §Шаблон)
// mma.m8n8k4 f32 accumulators: lane l of a quad-pair holds rows (l&1) + 4*(l>=16) + {0,2} and columns
// (l&2) + {0,1,4,5} of the 8x8 output. A warp tile 64x64 = 2x2 quad-pairs; each quad-pair repeats the 8x8 mma
// 4x4 times: rows step 8 (quad-pair band of 32 rows), columns step 16 (the two quad-pairs of a band interleave by 8).
// So a thread owns rows {0,2,8,10,16,18,24,26} x cols {0,1,4,5,16,17,20,21,32,33,36,37,48,49,52,53} = h*w 128.
#define PEARL_R 128
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

PH uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }

// Running sums kept in fp32 as bits = 0x4B400000 + S (|S| < 2^22, exponent fixed, ulp 1). For an even number of
// such values, XOR of the int32 sums = XOR of the raw bits sign-extended from bit 22: the low 22 bits are the
// same, the 10 high bits are all ones iff an odd number of sums is negative (bit 22 clear), and since the count
// is even that parity equals the parity of the set bits 22 (s0 §9a).
PH uint32_t magic_xor_to_int(uint32_t x) { return (uint32_t)(((int32_t)(x << 9)) >> 9); }

// ---------------------------------------------------------------- BLAKE3, keyed, one 64-byte block, ROOT
PH uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

PH void b3g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    a += b + mx; d = rotr32(d ^ a, 16);
    c += d;      b = rotr32(b ^ c, 12);
    a += b + my; d = rotr32(d ^ a, 8);
    c += d;      b = rotr32(b ^ c, 7);
}

// out = BLAKE3(msg as 64 bytes LE, key) — flags KEYED_HASH | CHUNK_START | CHUNK_END | ROOT, counter 0
PH void blake3_keyed_block(const uint32_t key[8], const uint32_t msg[16], uint32_t out[8]) {
    uint32_t s[16], m[16];
    for (int i = 0; i < 8; ++i) s[i] = key[i];
    s[8] = 0x6A09E667u; s[9] = 0xBB67AE85u; s[10] = 0x3C6EF372u; s[11] = 0xA54FF53Au;
    s[12] = 0; s[13] = 0; s[14] = 64; s[15] = 16u | 1u | 2u | 8u;
    for (int i = 0; i < 16; ++i) m[i] = msg[i];
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int r = 0; r < 7; ++r) {
        b3g(s[0], s[4], s[8], s[12], m[0], m[1]);
        b3g(s[1], s[5], s[9], s[13], m[2], m[3]);
        b3g(s[2], s[6], s[10], s[14], m[4], m[5]);
        b3g(s[3], s[7], s[11], s[15], m[6], m[7]);
        b3g(s[0], s[5], s[10], s[15], m[8], m[9]);
        b3g(s[1], s[6], s[11], s[12], m[10], m[11]);
        b3g(s[2], s[7], s[8], s[13], m[12], m[13]);
        b3g(s[3], s[4], s[9], s[14], m[14], m[15]);
        if (r < 6) {  // MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
            uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                              m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
            for (int i = 0; i < 16; ++i) m[i] = t[i];
        }
    }
    for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

// LE-uint256(h) <= LE-uint256(bound), most significant word last
PH bool le256_leq(const uint32_t h[8], const uint32_t bound[8]) {
    for (int i = 7; i >= 0; --i) {
        if (h[i] < bound[i]) return true;
        if (h[i] > bound[i]) return false;
    }
    return true;
}

// T[16] where slot s holds chunk c with c % 16 == s; T = words of the 64-byte transcript
PH void transcript_step(uint32_t T[16], uint32_t c, uint32_t x) { T[c & 15] = rotl13(T[c & 15]) ^ x; }

#pragma pack(push, 1)
typedef struct { uint32_t row_tile, col_tile; uint8_t jackpot[32]; } cand_t;  // 40 bytes
#pragma pack(pop)
