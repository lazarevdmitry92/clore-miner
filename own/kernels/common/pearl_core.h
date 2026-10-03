// Shared by every Pearl kernel (kernels/v100, kernels/sm80), their CPU emulators and host self-checks: the transcript
// step, keyed BLAKE3 of one block, the jackpot comparison and the candidate record of the contract. Everything here
// must stay bit-identical between host and device.
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

#define PEARL_R 128   // r: chunk of k between XOR checkpoints (consensus: exactly 128)

PH uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }

// ---------------------------------------------------------------- BLAKE3
PH uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

PH void b3g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    a += b + mx; d = rotr32(d ^ a, 16);
    c += d;      b = rotr32(b ^ c, 12);
    a += b + my; d = rotr32(d ^ a, 8);
    c += d;      b = rotr32(b ^ c, 7);
}

#define B3_CHUNK_START 1u
#define B3_CHUNK_END 2u
#define B3_PARENT 4u
#define B3_ROOT 8u
#define B3_KEYED 16u

// BLAKE3 compression: the first 8 output words (the chaining value / the digest of a root block)
PH void blake3_compress(const uint32_t cv[8], const uint32_t msg[16], uint64_t counter, uint32_t len, uint32_t flags,
                        uint32_t out[8]) {
    uint32_t s[16], m[16];
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = 0x6A09E667u; s[9] = 0xBB67AE85u; s[10] = 0x3C6EF372u; s[11] = 0xA54FF53Au;
    s[12] = (uint32_t)counter; s[13] = (uint32_t)(counter >> 32); s[14] = len; s[15] = flags;
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

// out = BLAKE3(msg as 64 bytes LE, key): one block that is the whole message (jackpot, noise PRG)
PH void blake3_keyed_block(const uint32_t key[8], const uint32_t msg[16], uint32_t out[8]) {
    blake3_compress(key, msg, 0, 64, B3_KEYED | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, out);
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
