// Operand preparation on the device (TZ_operands_gpu.md §2, step 1): zero-chunk leaves of the job's trees, the noise
// factors from a seed, and X' = D[:, p] - D[:, q] (+ nonce) built straight into the kernel's resident buffer together
// with the max row sum of squares (the V100 Cauchy-Schwarz test). Per pass only seed_A crosses PCIe.
// Included by pearl_v100.cu and pearl_sm80.cu; the per-element code is operands.h, shared with the emulators.
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "operands.h"

namespace pearl_ops {

struct W8 { uint32_t w[8]; };
struct Nonce { int8_t b[PEARL_NONCE_BYTES]; };

inline W8 w8(const pearl::CV &c) { W8 x; for (int i = 0; i < 8; ++i) x.w[i] = c[i]; return x; }

// CV of every all-zero chunk c < chunks (chunk index = BLAKE3 counter)
__global__ void zero_leaves_kernel(W8 key, uint64_t chunks, uint32_t *out) {
    const uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    uint32_t cv[8];
    chunk_cv(key.w, nullptr, c, cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[c * 8 + i] = cv[i];
}

// dense factor: PRG block b -> 32 elements b*32 .. b*32+31 of the row-major rows x r matrix
__global__ void dense_kernel(W8 seed, W8 lab, uint32_t blocks, int8_t *dense) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    uint32_t o[8];
    prg_block(seed.w, 0, b, lab.w, o);
    uint32_t packed[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t x = 0;
#pragma unroll
        for (int y = 0; y < 4; ++y) x |= (uint32_t)(uint8_t)uniform_value(o[w] >> (8 * y)) << (8 * y);
        packed[w] = x;
    }
    uint4 *dst = (uint4 *)(dense + (size_t)b * 32);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

// sparse factor: PRG block i -> rows l = 8i .. 8i+7, one (p, q) each
__global__ void sparse_kernel(W8 seed, W8 lab, uint32_t k, uint32_t r, uint8_t *p, uint8_t *q) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (k + 7) / 8) return;
    uint32_t o[8];
    prg_block(seed.w, 1, i, lab.w, o);
#pragma unroll
    for (int w = 0; w < 8; ++w)
        if (8 * i + w < k) sparse_pq(o[w], r, p[8 * i + w], q[8 * i + w]);
}

// one block per row: X'[row, l] = D[row, p_l] - D[row, q_l] (+ nonce in row 0), 4 columns per thread step;
// the row's sum of squares -> atomicMax(best)
__global__ void __launch_bounds__(256) build_kernel(const int8_t *__restrict__ dense, const uint8_t *__restrict__ p,
                                                    const uint8_t *__restrict__ q, uint32_t k, uint32_t r, Nonce nonce,
                                                    int has_nonce, int8_t *__restrict__ out, uint32_t *best) {
    __shared__ int8_t row[1024];
    __shared__ uint32_t part[8];
    const uint32_t i = blockIdx.x;
    for (uint32_t c = threadIdx.x; c < r; c += blockDim.x) row[c] = dense[(size_t)i * r + c];
    __syncthreads();
    uint32_t ss = 0;
    uint32_t *dst = (uint32_t *)(out + (size_t)i * k);
    for (uint32_t l = 4 * threadIdx.x; l < k; l += 4 * blockDim.x) {
        const uint32_t pp = *(const uint32_t *)(p + l), qq = *(const uint32_t *)(q + l);
        uint32_t w = 0;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            int v = row[(pp >> (8 * y)) & 0xFF] - row[(qq >> (8 * y)) & 0xFF];
            if (has_nonce && i == 0 && l + y < PEARL_NONCE_BYTES) v += nonce.b[l + y];
            ss += (uint32_t)(v * v);
            w |= (uint32_t)(uint8_t)(int8_t)v << (8 * y);
        }
        dst[l / 4] = w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t t = 0;
        for (uint32_t w = 0; w < blockDim.x / 32; ++w) t += part[w];
        atomicMax(best, t);
    }
}

// device scratch for the generators; the destination buffers belong to the kernel library
struct Gpu {
    int8_t *dense = nullptr;
    uint8_t *p = nullptr, *q = nullptr;
    uint32_t *leaves = nullptr, *best = nullptr;
    size_t cap_dense = 0, cap_p = 0, cap_q = 0, cap_leaves = 0, cap_best = 0;
    pearl::JobState job;

    template <class T>
    static cudaError_t grow(T *&ptr, size_t &cap, size_t bytes) {
        if (bytes <= cap) return cudaSuccess;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        cap = 0;
        cudaError_t e = cudaMalloc(&ptr, bytes);
        if (e == cudaSuccess) cap = bytes;
        return e;
    }

    cudaError_t leaves_of(const pearl::CV &key, uint64_t chunks, std::vector<pearl::CV> &out, cudaStream_t st) {
        cudaError_t e = grow(leaves, cap_leaves, chunks * 32);
        if (e != cudaSuccess) return e;
        zero_leaves_kernel<<<(unsigned)((chunks + 127) / 128), 128, 0, st>>>(w8(key), chunks, leaves);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        out.resize(chunks);
        if ((e = cudaMemcpyAsync(out.data(), leaves, chunks * 32, cudaMemcpyDeviceToHost, st)) != cudaSuccess) return e;
        return cudaStreamSynchronize(st);
    }

    // X' (rows x k) from `seed` into dst (device), max row sum of squares into *max2
    cudaError_t operand(const pearl::CV &seed, char which, uint32_t rows, uint32_t k, uint32_t r, const int8_t *nonce,
                        int8_t *dst, uint32_t *max2, cudaStream_t st) {
        cudaError_t e;
        if ((e = grow(dense, cap_dense, (size_t)rows * r)) != cudaSuccess) return e;
        if ((e = grow(p, cap_p, (size_t)k)) != cudaSuccess) return e;   // k % 64 == 0: whole u32 loads
        if ((e = grow(q, cap_q, (size_t)k)) != cudaSuccess) return e;
        if ((e = grow(best, cap_best, 4)) != cudaSuccess) return e;
        const W8 s = w8(seed), lab = w8(pearl::label(which));
        const uint32_t blocks = rows * (r / 32);
        dense_kernel<<<(blocks + 127) / 128, 128, 0, st>>>(s, lab, blocks, dense);
        sparse_kernel<<<((k + 7) / 8 + 127) / 128, 128, 0, st>>>(s, lab, k, r, p, q);
        Nonce nn = {};
        if (nonce) memcpy(nn.b, nonce, PEARL_NONCE_BYTES);
        if ((e = cudaMemsetAsync(best, 0, 4, st)) != cudaSuccess) return e;
        build_kernel<<<rows, 256, 0, st>>>(dense, p, q, k, r, nn, nonce != nullptr, dst, best);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(max2, best, 4, cudaMemcpyDeviceToHost, st)) != cudaSuccess) return e;
        return cudaStreamSynchronize(st);
    }

    // TZ_operands_gpu §4: the device generators against the same steps on the host (small job, one pass).
    // why = "" on success; leaves the job state empty.
    cudaError_t selfcheck(cudaStream_t st, std::string &why) {
        const uint32_t m = 128, n = 64, k = 2048, r = PEARL_R;
        uint8_t kb[32];
        for (int i = 0; i < 32; ++i) kb[i] = (uint8_t)(i * 29 + 7);
        const pearl::CV key = pearl::from_bytes(kb);
        std::vector<pearl::CV> leaves;
        cudaError_t e = leaves_of(key, (uint64_t)m * k / 1024, leaves, st);
        if (e != cudaSuccess) return e;
        if (leaves != pearl::zero_leaves(key, (uint64_t)m * k / 1024)) { why = "zero-chunk leaves"; return e; }
        const int8_t nonce[PEARL_NONCE_BYTES] = {-64, 64, 0, 1, -1, 17, -33, 5};
        int8_t *dst = nullptr;
        if ((e = cudaMalloc(&dst, (size_t)m * k)) != cudaSuccess) return e;
        for (int which = 0; which < 2 && why.empty(); ++which) {
            const uint32_t rows = which ? n : m;
            uint32_t max2 = 0;
            if ((e = operand(key, which ? 'B' : 'A', rows, k, r, which ? nullptr : nonce, dst, &max2, st)) != cudaSuccess)
                break;
            std::vector<int8_t> got((size_t)rows * k), want, dense;
            std::vector<uint8_t> p_, q_;
            if ((e = cudaMemcpy(got.data(), dst, got.size(), cudaMemcpyDeviceToHost)) != cudaSuccess) break;
            pearl::factors_cpu(key, which ? 'B' : 'A', rows, k, r, dense, p_, q_);
            pearl::build_cpu(dense, p_, q_, rows, k, r, which ? nullptr : nonce, want);
            uint32_t want2 = 0;
            for (uint32_t i = 0; i < rows; ++i) {
                uint32_t s2 = 0;
                for (uint32_t l = 0; l < k; ++l) s2 += (uint32_t)(want[(size_t)i * k + l] * want[(size_t)i * k + l]);
                want2 = std::max(want2, s2);
            }
            if (got != want || max2 != want2) why = which ? "B' generation" : "A' generation";
        }
        cudaFree(dst);
        job = pearl::JobState();
        return e;
    }
};

}  // namespace pearl_ops
