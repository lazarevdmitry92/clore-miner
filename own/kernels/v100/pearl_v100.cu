// Pearl (PRL) mining kernel for Tesla V100 (sm_70): C' = A'·B' on fp16 tensor cores (mma.m8n8k4, fp32 accumulate),
// XOR of every hash tile after each r = 128 of k, transcript, keyed BLAKE3 jackpot, comparison with the bound.
// Design and numbers: README.md. Contract: pearl_api.h, miner/README.md "Контракт GPU-бэкенда".
//
// Exactness. int8 operands in [-127, 127] are exact in fp16, products (<= 16129) exact in fp32. The running sum
// of a tile element lives in ONE fp32 accumulator started at 1.5*2^23: while |S| < 2^22 its exponent is fixed and
// ulp = 1, so every partial sum is an exact integer and bits - 0x4B400000 = S (s0 §9a). The host proves this per
// operand pair before choosing the fast path: any partial sum (any subset of terms, so any HMMA-internal order) is
// bounded by |a_i|·|b_j| <= sqrt(max|a|^2 · max|b|^2) < 2^22 (Cauchy-Schwarz). Pearl noise (A = B = 0) passes at
// k = 2048 and 4096; anything else (random +-127, k = 8192) takes the exact dp4a kernel.
//
// v2 (TZ_v100_kernel_v2.md): variants of the fast kernel (fp16 operands in global = no conversion in the loop,
// fragment prefetch), ablations for the bench (§3.1), online autotune under the real power limit and NVML
// throttle telemetry (§5). v1 on 8x V100: k 331, 7.94 J/TH; bare HMMA k 484 (probe21920/REPORT-2026-10-03-*).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "pearl_layout.h"
#include "pearl_host.h"
#include "../common/operands_cuda.cuh"
#include "../common/runtime.h"

using pearl::Dims;

#define MAGIC_F 12582912.0f   // 1.5 * 2^23
#define MAX_CANDS 4096

struct KeyBound { uint32_t key[8], bound[8]; };
struct DevOut { uint32_t count; cand_t c[MAX_CANDS]; };

// fast-kernel options: mining variants combine F16 / FRAGPF; ABL_* exist only for pearl_bench ablate (results are
// not the mining result and never reach pearl_search)
enum {
    OPT_F16 = 1,      // operands read as fp16 from a copy converted once at upload: no int8 -> fp16 in the loop
    OPT_FRAGPF = 2,   // fragments of k-step pair p+1 loaded from shared while the HMMAs of p run
    ABL_NOEPI = 4,    // no XOR/transcript every 128 of k (the sums are folded once at the end, so HMMA stays live)
    ABL_SMEM = 8,     // operands loaded into shared once; the k loop re-reads the same two stages (no global traffic)
    ABL_REGS = 16,    // fragments loaded once; the k loop is HMMA + epilogue only (~ bare peak)
};

// ---------------------------------------------------------------- device helpers

__device__ __forceinline__ void mma884(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// 4 int8 (two's complement) -> 2 x f16x2, exact: byte b -> u = b ^ 0x80 = b + 128; f16 0x6400 | u = 1024 + u;
// minus 1152 (0x6480) = b.
__device__ __forceinline__ uint2 i8x4_to_f16x4(uint32_t w) {
    uint32_t u = w ^ 0x80808080u, lo, hi;
    lo = __byte_perm(u, 0x64646464u, 0x4140);
    hi = __byte_perm(u, 0x64646464u, 0x4342);
    asm("sub.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0x64806480u));
    asm("sub.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0x64806480u));
    return make_uint2(lo, hi);
}

// transcript t (slot s = chunks c with c % 16 == s) -> jackpot, candidate, optional transcript output
__device__ __forceinline__ void finish_tile(const uint32_t (&t)[16], uint32_t rt, uint32_t ct, const KeyBound &kb,
                                            DevOut *out, uint32_t cap, uint32_t *transcripts, uint32_t lo,
                                            uint32_t col_tiles) {
    if (transcripts) {
        uint32_t *dst = transcripts + ((size_t)(rt - lo) * col_tiles + ct) * 16;
#pragma unroll
        for (int s = 0; s < 16; ++s) dst[s] = t[s];
    }
    uint32_t h[8];
    blake3_keyed_block(kb.key, t, h);
    if (le256_leq(h, kb.bound)) {
        uint32_t idx = atomicAdd(&out->count, 1u);
        if (idx < cap && idx < MAX_CANDS) {
            cand_t &c = out->c[idx];
            c.row_tile = rt;
            c.col_tile = ct;
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int b = 0; b < 4; ++b) c.jackpot[4 * i + b] = (uint8_t)(h[i] >> (8 * b));
        }
    }
}

// ---------------------------------------------------------------- fast kernel: HMMA, one fp32 running sum
// CTA = WM x WN warps, warp tile 64x64, CTA tile 64WM x 64WN. Shared memory: 2 stages x (64WM rows of A' +
// 64WN rows of B'^T) x K_TILE fp16, swizzled (pearl_layout.h). Global -> registers -> (convert) -> shared; one
// __syncthreads per stage; the chunk epilogue (XOR, transcript) every 4 stages = 128 of k.
// a, bt: int8 (or fp16 with OPT_F16) row-major with k elements per row; row index = absolute row.
template <int WM, int WN, int OPT>
__global__ void __launch_bounds__(WM * WN * 32, (WM * WN <= 4) ? 2 : 1)
fast_kernel(const void *__restrict__ a, const void *__restrict__ bt, uint32_t R0, uint32_t R1, uint32_t n,
            uint32_t k, uint32_t L, uint32_t ncb, KeyBound kb, DevOut *out, uint32_t cap, uint32_t *transcripts,
            uint32_t lo, uint32_t col_tiles) {
    constexpr bool F16 = OPT & OPT_F16;
    constexpr int NT = WM * WN * 32;
    constexpr int AR = 64 * WM, ROWS = 64 * (WM + WN);
    constexpr int ES = F16 ? 2 : 1;                 // bytes per element in global
    constexpr int PARTS = K_TILE * ES / 16;         // 16-byte global loads per row per stage (2 int8, 4 fp16)
    constexpr int NP = ROWS * PARTS / NT;           // per thread per stage
    static_assert(ROWS * PARTS % NT == 0, "staging split");
    extern __shared__ __align__(16) uint8_t sm[];

    const int tid = threadIdx.x, l = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const uint32_t cb = blockIdx.x % ncb, rb = blockIdx.x / ncb;
    const uint32_t cta_r0 = R0 + rb * AR, cta_c0 = cb * 64 * WN;
    const size_t row_bytes = (size_t)k * ES;

    // staging: item q -> shared row r, 16-byte part of the stage row
    const uint8_t *src[NP];
    int dst[NP];
#pragma unroll
    for (int q = 0; q < NP; ++q) {
        int idx = tid + q * NT, r = idx / PARTS, part = idx % PARTS;
        // rows past R1 / n read the zero padding of the device buffers (PAD_ROWS); their results are discarded
        const uint8_t *p = r < AR ? (const uint8_t *)a + (size_t)(cta_r0 + r) * row_bytes
                                  : (const uint8_t *)bt + (size_t)(cta_c0 + (uint32_t)(r - AR)) * row_bytes;
        src[q] = p + 16 * part;
        dst[q] = F16 ? smem_off(r, part) : smem_off(r, 2 * part);   // int8: slots 2part, 2part+1 (= dst ^ 16)
    }
    uint4 pre[NP];
    auto load_stage = [&](uint32_t k0) {
#pragma unroll
        for (int q = 0; q < NP; ++q) pre[q] = __ldg((const uint4 *)(src[q] + (size_t)k0 * ES));
    };
    auto store_stage = [&](uint8_t *buf) {
#pragma unroll
        for (int q = 0; q < NP; ++q) {
            if (F16) {
                *(uint4 *)(buf + dst[q]) = pre[q];
            } else {
                uint2 x0 = i8x4_to_f16x4(pre[q].x), x1 = i8x4_to_f16x4(pre[q].y);
                uint2 x2 = i8x4_to_f16x4(pre[q].z), x3 = i8x4_to_f16x4(pre[q].w);
                *(uint4 *)(buf + dst[q]) = make_uint4(x0.x, x0.y, x1.x, x1.y);
                *(uint4 *)(buf + (dst[q] ^ 16)) = make_uint4(x2.x, x2.y, x3.x, x3.y);
            }
        }
    };

    // fragment addresses: slot p of a row is (base ^ 16p) (rows are 64-byte aligned)
    int abase[4], bbase[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) abase[i] = smem_off(wm * 64 + lane_a_row(l, i), 0);
#pragma unroll
    for (int j = 0; j < 4; ++j) bbase[j] = smem_off(AR + wn * 64 + lane_b_row(l, j), 0);
    auto lds = [&](const uint8_t *cur, int p, uint4 (&fa)[4], uint4 (&fb)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) fa[i] = *(const uint4 *)(cur + (abase[i] ^ (16 * p)));
#pragma unroll
        for (int j = 0; j < 4; ++j) fb[j] = *(const uint4 *)(cur + (bbase[j] ^ (16 * p)));
    };

    float acc[4][4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = MAGIC_F;
    // k-step pair of one slot: the k-steps 2p (halves 0..3 = .x .y) and 2p+1 (halves 4..7 = .z .w)
    auto mmas = [&](const uint4 (&fa)[4], const uint4 (&fb)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) mma884(acc[i][j], fa[i].x, fa[i].y, fb[j].x, fb[j].y);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) mma884(acc[i][j], fa[i].z, fa[i].w, fb[j].z, fb[j].w);
    };
    uint32_t t[16];   // indexed by chunk % 16 at run time -> local memory, touched once per 128 of k
    for (int s = 0; s < 16; ++s) t[s] = 0;
    auto epilogue = [&](uint32_t chunk) {   // XOR of the 128 running sums of this thread's tile
        uint32_t x[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 8; e += 2)
                    x[e >> 1] = xor3(x[e >> 1], __float_as_uint(acc[i][j][e]), __float_as_uint(acc[i][j][e + 1]));
        transcript_step(t, chunk, magic_xor_to_int(xor3(x[0], x[1], x[2] ^ x[3])));
    };

    const uint32_t nst = L / K_TILE;
    uint8_t *buf0 = sm, *buf1 = sm + ROWS * 64;
    load_stage(0);
    store_stage(buf0);
    if (OPT & ABL_SMEM) { load_stage(K_TILE); store_stage(buf1); }
    __syncthreads();

    if (OPT & ABL_REGS) {
        uint4 fa[4], fb[4];
        lds(buf0, 0, fa, fb);
        for (uint32_t st = 0; st < nst; ++st) {
#pragma unroll
            for (int p = 0; p < SLOTS; ++p) mmas(fa, fb);
            if ((st & 3) == 3) epilogue(st >> 2);
        }
    } else {
        for (uint32_t st = 0; st < nst; ++st) {
            const uint8_t *cur = (st & 1) ? buf1 : buf0;
            uint8_t *nxt = (st & 1) ? buf0 : buf1;
            const bool stage_io = !(OPT & ABL_SMEM) && st + 1 < nst;
            if (stage_io) load_stage((st + 1) * K_TILE);
            if (OPT & OPT_FRAGPF) {
                uint4 fa[2][4], fb[2][4];
                lds(cur, 0, fa[0], fb[0]);
#pragma unroll
                for (int p = 0; p < SLOTS; ++p) {
                    if (p + 1 < SLOTS) lds(cur, p + 1, fa[(p + 1) & 1], fb[(p + 1) & 1]);
                    mmas(fa[p & 1], fb[p & 1]);
                }
            } else {
#pragma unroll
                for (int p = 0; p < SLOTS; ++p) {
                    uint4 fa[4], fb[4];
                    lds(cur, p, fa, fb);
                    mmas(fa, fb);
                }
            }
            if (stage_io) store_stage(nxt);
            __syncthreads();
            if (!(OPT & ABL_NOEPI) && (st & 3) == 3) epilogue(st >> 2);
        }
    }
    if (OPT & ABL_NOEPI) epilogue(0);   // keep the HMMAs live: one fold at the end

    const uint32_t band = cta_r0 + wm * 64 + 32 * lane_qr(l), wcol = cta_c0 + wn * 64;
    if (band >= R1 || wcol >= n) return;
    finish_tile(t, lane_row_tile(band, l), lane_col_tile(wcol, l), kb, out, cap, transcripts, lo, col_tiles);
}

// ---------------------------------------------------------------- exact kernel: dp4a, int32 running sums
// One thread per hash tile, operands straight from global/L1. Exact for any int8 operands; used when the
// Cauchy-Schwarz test of the fast path fails, and as the independent second implementation in the self-check.
__global__ void __launch_bounds__(128)
exact_kernel(const int8_t *__restrict__ a, const int8_t *__restrict__ bt, uint32_t k, uint32_t L, uint32_t lo,
             uint32_t hi, uint32_t col_tiles, KeyBound kb, DevOut *out, uint32_t cap, uint32_t *transcripts) {
    const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (uint64_t)(hi - lo) * col_tiles) return;
    const uint32_t rt = lo + (uint32_t)(id / col_tiles), ct = (uint32_t)(id % col_tiles);
    const uint32_t ro = row_tile_offset(rt), co = col_tile_offset(ct);
    const uint32_t *ap[PEARL_H], *bp[PEARL_W];
#pragma unroll
    for (int i = 0; i < PEARL_H; ++i) ap[i] = (const uint32_t *)(a + (size_t)(ro + pat_row(i)) * k);
#pragma unroll
    for (int j = 0; j < PEARL_W; ++j) bp[j] = (const uint32_t *)(bt + (size_t)(co + pat_col(j)) * k);
    int32_t S[PEARL_H][PEARL_W];
#pragma unroll
    for (int i = 0; i < PEARL_H; ++i)
#pragma unroll
        for (int j = 0; j < PEARL_W; ++j) S[i][j] = 0;
    uint32_t t[16];
#pragma unroll
    for (int s = 0; s < 16; ++s) t[s] = 0;
    for (uint32_t w = 0; w < L / 4; ++w) {
        uint32_t av[PEARL_H], bv[PEARL_W];
#pragma unroll
        for (int i = 0; i < PEARL_H; ++i) av[i] = __ldg(ap[i] + w);
#pragma unroll
        for (int j = 0; j < PEARL_W; ++j) bv[j] = __ldg(bp[j] + w);
#pragma unroll
        for (int i = 0; i < PEARL_H; ++i)
#pragma unroll
            for (int j = 0; j < PEARL_W; ++j) S[i][j] = __dp4a((int)av[i], (int)bv[j], S[i][j]);
        if ((w & 31) == 31) {
            uint32_t x = 0;
#pragma unroll
            for (int i = 0; i < PEARL_H; ++i)
#pragma unroll
                for (int j = 0; j < PEARL_W; j += 2) x = xor3(x, (uint32_t)S[i][j], (uint32_t)S[i][j + 1]);
            transcript_step(t, w >> 5, x);
        }
    }
    finish_tile(t, rt, ct, kb, out, cap, transcripts, lo, col_tiles);
}

// max over rows of sum of squares, one warp per row
__global__ void row_sumsq_max(const int8_t *__restrict__ x, uint32_t rows, uint32_t k, uint32_t *best) {
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, l = threadIdx.x & 31;
    if (row >= rows) return;
    const uint32_t *p = (const uint32_t *)(x + (size_t)row * k);
    int32_t s = 0;
    for (uint32_t w = l; w < k / 4; w += 32) s = __dp4a((int)p[w], (int)p[w], s);
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (l == 0) atomicMax(best, (uint32_t)s);
}

// int8 -> fp16 copy (exact), 16 elements per thread
__global__ void to_f16(const int8_t *__restrict__ x, uint16_t *__restrict__ y, size_t n16) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n16) return;
    const uint4 v = ((const uint4 *)x)[i];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint4 o[2];
    uint32_t *op = (uint32_t *)o;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint2 h = i8x4_to_f16x4(w[q]);
        op[2 * q] = h.x;
        op[2 * q + 1] = h.y;
    }
    ((uint4 *)y)[2 * i] = o[0];
    ((uint4 *)y)[2 * i + 1] = o[1];
}

// ---------------------------------------------------------------- host side

namespace {

struct Ctx;
typedef void (*LaunchFn)(const Ctx &, uint32_t, uint32_t, const KeyBound &, uint32_t, uint32_t *, cudaStream_t);

struct Variant {
    const char *name;
    int wm, wn, opt;
    bool mining;      // ablations run only through pearl_bench_run
    LaunchFn launch;
};

struct Ctx {
    int dev = -1, sm = 0, max_mhz = 0;
    bool ready = false;
    cudaStream_t st = nullptr;
    int8_t *da = nullptr, *db = nullptr;     // da holds rows [a_row0, a_row0 + a_rows) of A'
    uint16_t *da16 = nullptr, *db16 = nullptr; // fp16 bits, copies for OPT_F16
    size_t cap_a = 0, cap_b = 0;
    uint32_t m = 0, n = 0, ka = 0, kb = 0, a_row0 = 0, a_rows = 0;
    uint32_t max_a2 = 0, max_b2 = 0;
    DevOut *dout = nullptr;
    uint32_t *dmax = nullptr, *dtr = nullptr;
    size_t cap_tr = 0;
    int variant = 0;
    std::string kernel_name = "none";
    void *nvdev = nullptr;
    // startup autotune: MAC per second of every mining variant (0 = not measured) and the SM clock meanwhile
    std::vector<double> rate;
    double tune_mhz = 0;
    pearl::OnlineTune tune;          // TZ v2 §5
    pearl_ops::Gpu ops;              // resident path (TZ_operands_gpu): generators, job state, trees
    bool a16_ok = false, b16_ok = false;   // fp16 copies current (made lazily, only for -f16 variants)
    Dims dims() const { Dims d; d.m = m; d.n = n; d.k = ka; return d; }
};

const int MAX_DEV = 16;
Ctx g_ctx[MAX_DEV];
int g_dev = -1;                       // process default (pearl_init); a thread that called pearl_init uses its own
thread_local int t_dev = -1;
thread_local std::string t_err;

int cur_index() { return t_dev >= 0 ? t_dev : g_dev; }

Ctx *cur() { return cur_index() >= 0 ? &g_ctx[cur_index()] : nullptr; }

int fail(int code, const std::string &msg) { t_err = msg; return code; }

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    return fail(PEARL_ERR_CUDA, std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

template <int WM, int WN, int OPT>
void launch_fast(const Ctx &c, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr,
                 cudaStream_t st) {
    const uint32_t R0 = lo / 4 * 32, R1 = hi / 4 * 32, L = c.ka - c.ka % PEARL_R;
    const uint32_t nrb = (R1 - R0 + 64 * WM - 1) / (64 * WM), ncb = (c.n + 64 * WN - 1) / (64 * WN);
    const size_t smem = 2 * 64 * (WM + WN) * 64;
    const void *a = (OPT & OPT_F16) ? (const void *)(c.da16 - (size_t)c.a_row0 * c.ka)
                                    : (const void *)(c.da - (size_t)c.a_row0 * c.ka);
    const void *b = (OPT & OPT_F16) ? (const void *)c.db16 : (const void *)c.db;
    fast_kernel<WM, WN, OPT><<<nrb * ncb, WM * WN * 32, smem, st>>>(a, b, R0, R1, c.n, c.ka, L, ncb, kb, c.dout, cap,
                                                                     tr, lo, c.n / 64 * 4);
}

template <int WM, int WN, int OPT>
cudaError_t prepare_fast() {
    const int smem = 2 * 64 * (WM + WN) * 64;
    cudaError_t e = cudaFuncSetAttribute(fast_kernel<WM, WN, OPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e == cudaSuccess)
        e = cudaFuncSetAttribute(fast_kernel<WM, WN, OPT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    return e;
}

void launch_exact(const Ctx &c, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr,
                  cudaStream_t st) {
    const uint32_t ct = c.n / 64 * 4, L = c.ka - c.ka % PEARL_R;
    const uint64_t tiles = (uint64_t)(hi - lo) * ct;
    exact_kernel<<<(unsigned)((tiles + 127) / 128), 128, 0, st>>>(c.da - (size_t)c.a_row0 * c.ka, c.db, c.ka, L, lo,
                                                                   hi, ct, kb, c.dout, cap, tr);
}

#define FAST(WM, WN, OPT, NAME, MINING) {NAME, WM, WN, OPT, MINING, launch_fast<WM, WN, OPT>}
const Variant VARIANTS[] = {
    FAST(2, 2, 0, "v100-hmma884-128x128", true),
    FAST(2, 2, OPT_F16, "v100-hmma884-128x128-f16", true),
    FAST(2, 2, OPT_FRAGPF, "v100-hmma884-128x128-pf", true),
    FAST(2, 2, OPT_F16 | OPT_FRAGPF, "v100-hmma884-128x128-f16-pf", true),
    FAST(2, 4, 0, "v100-hmma884-128x256", true),
    FAST(2, 4, OPT_F16, "v100-hmma884-128x256-f16", true),
    FAST(2, 4, OPT_FRAGPF, "v100-hmma884-128x256-pf", true),
    FAST(4, 2, 0, "v100-hmma884-256x128", true),
    FAST(4, 2, OPT_F16, "v100-hmma884-256x128-f16", true),
    // TZ v2 §3.1 ablations (shape of the v1 autotune winner); "no-convert" = the -f16 mining variant
    FAST(2, 2, ABL_NOEPI, "ablate-no-epilogue", false),
    FAST(2, 2, ABL_SMEM, "ablate-smem-only", false),
    FAST(2, 2, ABL_REGS, "ablate-regs-only", false),
};
const int NVAR = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

cudaError_t prepare_all() {
    cudaError_t e = cudaSuccess;
#define PREP(WM, WN, OPT) if (e == cudaSuccess) e = prepare_fast<WM, WN, OPT>();
    PREP(2, 2, 0) PREP(2, 2, OPT_F16) PREP(2, 2, OPT_FRAGPF) PREP(2, 2, OPT_F16 | OPT_FRAGPF)
    PREP(2, 4, 0) PREP(2, 4, OPT_F16) PREP(2, 4, OPT_FRAGPF)
    PREP(4, 2, 0) PREP(4, 2, OPT_F16)
    PREP(2, 2, ABL_NOEPI) PREP(2, 2, ABL_SMEM) PREP(2, 2, ABL_REGS)
#undef PREP
    return e;
}

int find_variant(const char *name) {
    for (int v = 0; v < NVAR; ++v)
        if (!strcmp(name, VARIANTS[v].name)) return v;
    return -1;
}

int path_fast(const Ctx &c) {
    if (const char *p = getenv("PEARL_PATH")) {
        if (!strcmp(p, "fast")) return 1;
        if (!strcmp(p, "exact")) return 0;
    }
    return pearl::cs_exact(c.max_a2, c.max_b2) ? 1 : 0;
}

// Device copies carry PAD_ROWS zero rows past the data: a CTA reaching past R1 / n reads them instead of
// clamping its row (no per-load min, fewer registers in the staging loop).
const uint32_t PAD_ROWS = 256;

// device buffers for rows x k of A' or B'^T (+ zero padding rows), int8 and its fp16 twin
int ensure(Ctx &c, uint32_t rows, uint32_t k, bool is_a) {
    const size_t bytes = (size_t)rows * k, alloc = bytes + (size_t)PAD_ROWS * k;
    int8_t *&dp = is_a ? c.da : c.db;
    uint16_t *&dp16 = is_a ? c.da16 : c.db16;
    size_t &cap = is_a ? c.cap_a : c.cap_b;
    if (alloc > cap) {
        if (dp) CU(cudaFree(dp));
        if (dp16) CU(cudaFree(dp16));
        dp = nullptr;
        dp16 = nullptr;
        CU(cudaMalloc(&dp, alloc));
        CU(cudaMalloc(&dp16, alloc * 2));
        cap = alloc;
    }
    CU(cudaMemsetAsync(dp + bytes, 0, alloc - bytes, c.st));
    CU(cudaMemsetAsync(dp16 + bytes, 0, (alloc - bytes) * 2, c.st));
    (is_a ? c.a16_ok : c.b16_ok) = false;
    return PEARL_OK;
}

int upload(Ctx &c, const int8_t *h, uint32_t rows, uint32_t k, bool is_a) {
    int rc = ensure(c, rows, k, is_a);
    if (rc) return rc;
    int8_t *dp = is_a ? c.da : c.db;
    CU(cudaMemcpyAsync(dp, h, (size_t)rows * k, cudaMemcpyHostToDevice, c.st));
    CU(cudaMemsetAsync(c.dmax, 0, 4, c.st));
    row_sumsq_max<<<(rows + 7) / 8, 256, 0, c.st>>>(dp, rows, k, c.dmax);
    CU(cudaGetLastError());
    uint32_t mx = 0;
    CU(cudaMemcpyAsync(&mx, c.dmax, 4, cudaMemcpyDeviceToHost, c.st));
    CU(cudaStreamSynchronize(c.st));
    (is_a ? c.max_a2 : c.max_b2) = mx;
    return PEARL_OK;
}

// the fp16 twins of the resident operands, converted on first use by an -f16 variant
int ensure_f16(Ctx &c) {
    struct Side { bool &ok; const int8_t *src; uint16_t *dst; size_t bytes; };
    const Side sides[2] = {{c.a16_ok, c.da, c.da16, (size_t)c.a_rows * c.ka}, {c.b16_ok, c.db, c.db16, (size_t)c.n * c.kb}};
    for (const Side &x : sides) {
        if (x.ok || !x.bytes) continue;
        to_f16<<<(unsigned)((x.bytes / 16 + 255) / 256), 256, 0, c.st>>>(x.src, x.dst, x.bytes / 16);   // k % 64 == 0
        CU(cudaGetLastError());
        x.ok = true;
    }
    return PEARL_OK;
}

// fast (variant v) or exact (v < 0) over [lo, hi); transcripts to the device buffer when tr
int run(Ctx &c, int v, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr) {
    if (v >= 0 && (VARIANTS[v].opt & OPT_F16)) {
        int rc = ensure_f16(c);
        if (rc) return rc;
    }
    CU(cudaMemsetAsync(&c.dout->count, 0, 4, c.st));
    if (hi > lo) {
        if (v >= 0) VARIANTS[v].launch(c, lo, hi, kb, cap, tr, c.st);
        else launch_exact(c, lo, hi, kb, cap, tr, c.st);
        CU(cudaGetLastError());
    }
    return PEARL_OK;
}

int ensure_tr(Ctx &c, size_t words) {
    if (words * 4 > c.cap_tr) {
        if (c.dtr) CU(cudaFree(c.dtr));
        c.dtr = nullptr;
        CU(cudaMalloc(&c.dtr, words * 4));
        c.cap_tr = words * 4;
    }
    return PEARL_OK;
}

KeyBound make_kb(const uint8_t *seed_a, const uint8_t *bound_le) {
    KeyBound kb;
    pearl::words_le(seed_a, kb.key, 8);
    pearl::words_le(bound_le, kb.bound, 8);
    return kb;
}

// transcripts of the resident operands by variant v (or exact) -> host
int transcripts_by(Ctx &c, int v, uint32_t lo, uint32_t hi, std::vector<uint32_t> &host) {
    const size_t words = (size_t)(hi - lo) * (c.n / 64 * 4) * 16;
    int rc = ensure_tr(c, words);
    if (rc) return rc;
    KeyBound kb = {};
    rc = run(c, v, lo, hi, kb, 0, c.dtr);
    if (rc) return rc;
    host.resize(words);
    CU(cudaMemcpyAsync(host.data(), c.dtr, words * 4, cudaMemcpyDeviceToHost, c.st));
    CU(cudaStreamSynchronize(c.st));
    return PEARL_OK;
}

// Self-check (TZ §8a.4): every mining kernel against the host int64 reference, and the device BLAKE3 + comparison
// against the host one. Fails -> pearl_init fails -> the miner does not start.
int selfcheck(Ctx &c) {
    std::string why;
    CU(c.ops.selfcheck(c.st, why));
    if (!why.empty()) return fail(PEARL_ERR_SELFCHECK, "self-check: device operands differ from the host: " + why);
    struct Case { uint32_t m, n, k; int dist; bool fast; };
    const Case cases[] = {
        {256, 512, 2048, 0, true},     // Pearl noise: fast path and exact path
        {128, 256, 4096, 0, true},
        {64, 128, 2048, 1, false},     // full range and +-127: exact path only (fast is not provably exact here)
        {64, 128, 2048, 2, false},
    };
    for (const Case &cs : cases) {
        auto A = pearl::synth((size_t)cs.m * cs.k, cs.dist, 11 + cs.m + cs.dist);
        auto B = pearl::synth((size_t)cs.n * cs.k, cs.dist, 23 + cs.n + cs.dist);
        int rc = upload(c, A.data(), cs.m, cs.k, true);
        if (!rc) rc = upload(c, B.data(), cs.n, cs.k, false);
        if (rc) return rc;
        c.m = c.a_rows = cs.m; c.a_row0 = 0; c.n = cs.n; c.ka = c.kb = cs.k;
        if (cs.fast && !pearl::cs_exact(c.max_a2, c.max_b2))
            return fail(PEARL_ERR_SELFCHECK, "self-check: noise operands failed the Cauchy-Schwarz test");
        const Dims d = c.dims();
        const uint32_t hi = d.row_tiles();
        std::vector<uint32_t> ref((size_t)hi * d.col_tiles() * 16), got;
        pearl::ref_transcripts(A.data(), B.data(), d, 0, hi, ref.data());
        for (int v = -1; v < NVAR; ++v) {
            if (v >= 0 && (!cs.fast || !VARIANTS[v].mining)) continue;
            if ((rc = transcripts_by(c, v, 0, hi, got))) return rc;
            if (got != ref) {
                size_t bad = 0;
                for (size_t i = 0; i < ref.size(); ++i) bad += got[i] != ref[i];
                char msg[200];
                snprintf(msg, sizeof msg, "self-check: %s transcripts differ from the reference (%zu of %zu words, "
                         "m=%u n=%u k=%u dist=%d)", v >= 0 ? VARIANTS[v].name : "exact", bad, ref.size(),
                         cs.m, cs.n, cs.k, cs.dist);
                return fail(PEARL_ERR_SELFCHECK, msg);
            }
        }
        // jackpots: bound = all ones -> every tile is a candidate (tiles <= MAX_CANDS)
        const uint32_t tiles = hi * d.col_tiles();
        if (tiles <= MAX_CANDS) {
            uint8_t seed[32], ones[32];
            for (int i = 0; i < 32; ++i) seed[i] = (uint8_t)(i * 7 + cs.k);
            memset(ones, 0xFF, 32);
            KeyBound kb = make_kb(seed, ones);
            if ((rc = run(c, cs.fast ? 0 : -1, 0, hi, kb, MAX_CANDS, nullptr))) return rc;
            std::vector<cand_t> cand(tiles);
            uint32_t cnt = 0;
            CU(cudaMemcpyAsync(&cnt, &c.dout->count, 4, cudaMemcpyDeviceToHost, c.st));
            CU(cudaMemcpyAsync(cand.data(), c.dout->c, tiles * sizeof(cand_t), cudaMemcpyDeviceToHost, c.st));
            CU(cudaStreamSynchronize(c.st));
            if (cnt != tiles) return fail(PEARL_ERR_SELFCHECK, "self-check: candidate count with bound 2^256-1");
            pearl::sort_cands(cand.data(), tiles);
            for (uint32_t i = 0; i < tiles; ++i) {
                uint32_t h[8];
                uint8_t jb[32];
                blake3_keyed_block(kb.key, &ref[(size_t)i * 16], h);
                pearl::jackpot_bytes(h, jb);
                if (cand[i].row_tile != i / d.col_tiles() || cand[i].col_tile != i % d.col_tiles() ||
                    memcmp(cand[i].jackpot, jb, 32))
                    return fail(PEARL_ERR_SELFCHECK, "self-check: device jackpot differs from host BLAKE3");
            }
        }
    }
    c.m = c.n = c.ka = c.kb = 0;
    return PEARL_OK;
}

// Startup autotune: time every mining variant on a mining-shaped problem (seconds, before the card heats up);
// the online autotune then re-measures the best few under the real power limit. PEARL_VARIANT=<name> pins one.
int autotune(Ctx &c) {
    c.rate.assign(NVAR, 0.0);
    if (const char *want = getenv("PEARL_VARIANT")) {
        int v = find_variant(want);
        if (v < 0 || !VARIANTS[v].mining) return fail(PEARL_ERR_ARGS, std::string("PEARL_VARIANT: unknown ") + want);
        c.variant = v;
        c.kernel_name = want;
        c.tune.pinned = true;
        return PEARL_OK;
    }
    const uint32_t m = 8192, n = 2048, k = 2048;   // the mining shape: B'^T 4 MB stays in the 6 MB L2
    auto A = pearl::synth((size_t)m * k, 0, 5), B = pearl::synth((size_t)n * k, 0, 6);
    int rc = upload(c, A.data(), m, k, true);
    if (!rc) rc = upload(c, B.data(), n, k, false);
    if (rc) return rc;
    c.m = c.a_rows = m; c.a_row0 = 0; c.n = n; c.ka = c.kb = k;
    const double macs = (double)m * n * k;
    KeyBound kb = {};
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    double best = 0, mhz = 0;
    int nmhz = 0;
    for (int v = 0; v < NVAR; ++v) {
        if (!VARIANTS[v].mining) continue;
        if ((rc = run(c, v, 0, m / 8, kb, 0, nullptr))) return rc;   // warm-up
        CU(cudaEventRecord(e0, c.st));
        for (int rep = 0; rep < 5; ++rep)
            if ((rc = run(c, v, 0, m / 8, kb, 0, nullptr))) return rc;
        CU(cudaEventRecord(e1, c.st));
        CU(cudaEventSynchronize(e1));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, e0, e1));
        pearl::Telemetry t = pearl::read_telemetry(c.nvdev);
        if (t.ok) { mhz += t.mhz; ++nmhz; }
        c.rate[v] = 5 * macs / (ms / 1e3);
        if (c.rate[v] > best) { best = c.rate[v]; c.variant = v; }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    c.tune_mhz = nmhz ? mhz / nmhz : c.max_mhz;
    c.kernel_name = VARIANTS[c.variant].name;
    // online candidates: the best three of the startup ranking
    std::vector<int> order;
    for (int v = 0; v < NVAR; ++v)
        if (VARIANTS[v].mining) order.push_back(v);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return c.rate[x] > c.rate[y]; });
    c.tune.start(std::vector<int>(order.begin(), order.begin() + std::min<size_t>(3, order.size())));
    c.m = c.n = c.ka = c.kb = 0;
    return PEARL_OK;
}

int init_device(int device) {
    if (device < 0 || device >= MAX_DEV) return fail(PEARL_ERR_ARGS, "device index");
    Ctx &c = g_ctx[device];
    t_dev = device;
    if (g_dev < 0) g_dev = device;
    if (c.ready) return PEARL_OK;
    CU(cudaSetDevice(device));
    cudaDeviceProp p;
    CU(cudaGetDeviceProperties(&p, device));
    if (p.major != 7 || p.minor != 0)
        return fail(PEARL_ERR_ARGS, std::string("not a Volta sm_70 device: ") + p.name);
    c.dev = device;
    c.sm = p.multiProcessorCount;
    c.max_mhz = p.clockRate / 1000;
    if (!c.st) {
        CU(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));
        CU(cudaMalloc(&c.dout, sizeof(DevOut)));
        CU(cudaMalloc(&c.dmax, 4));
        CU(prepare_all());
        char bus[32];
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, device) == cudaSuccess) c.nvdev = pearl::nvml().device(bus);
    }
    int rc = selfcheck(c);
    if (!rc) rc = autotune(c);
    if (rc) return rc;
    c.ready = true;
    return PEARL_OK;
}

// the calling thread's device, initialised on first use (PEARL_DEVICE, default 0)
Ctx *ready_ctx(int &rc) {
    Ctx *c = cur();
    if (!c || !c->ready) {
        const char *d = getenv("PEARL_DEVICE");
        rc = init_device(cur_index() >= 0 ? cur_index() : (d ? atoi(d) : 0));
        if (rc) return nullptr;
        c = cur();
    }
    rc = PEARL_OK;
    if (cudaSetDevice(c->dev) != cudaSuccess) { rc = fail(PEARL_ERR_CUDA, "cudaSetDevice"); return nullptr; }
    return c;
}

int set_a_rows(Ctx &c, const int8_t *a_rows_ptr, uint32_t m, uint32_t row0, uint32_t rows, uint32_t k) {
    std::string e = pearl::check_dims(m, PEARL_COL_PERIOD, k);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    c.m = 0;
    int rc = rows ? upload(c, a_rows_ptr, rows, k, true) : PEARL_OK;
    if (rc) return rc;
    c.m = m; c.ka = k; c.a_row0 = row0; c.a_rows = rows;
    return PEARL_OK;
}

int check_resident(Ctx &c, uint32_t k, uint32_t r, const uint8_t *rows, const uint8_t *cols, uint32_t lo,
                   uint32_t hi) {
    if (!c.m || !c.n) return fail(PEARL_ERR_STATE, "pearl_set_a/b first");
    if (c.ka != c.kb) return fail(PEARL_ERR_ARGS, "A' and B'^T have different k");
    std::string e = pearl::check_search(c.dims(), k, r, rows, cols, lo, hi);
    if (!e.empty())
        return fail(rows && cols && (memcmp(rows, PEARL_ROWS_PATTERN, 6) || memcmp(cols, PEARL_COLS_PATTERN, 6))
                    ? PEARL_ERR_PATTERN : PEARL_ERR_ARGS, e);
    if (hi > lo && (lo / 4 * 32 < c.a_row0 || hi / 4 * 32 > c.a_row0 + c.a_rows))
        return fail(PEARL_ERR_ARGS, "row tiles outside the rows of A' on the device");
    return PEARL_OK;
}

}  // namespace

// ---------------------------------------------------------------- C API

extern "C" {

const char *pearl_last_error(void) { return t_err.c_str(); }

const char *pearl_kernel_name(void) {
    Ctx *c = cur();
    return c ? c->kernel_name.c_str() : "none";
}

int pearl_init(int device) { return init_device(device); }

int pearl_device_info(uint32_t *sm_count, char *name, uint32_t name_len) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (name_len == 0) return fail(PEARL_ERR_ARGS, "name_len");
    cudaDeviceProp p;
    CU(cudaGetDeviceProperties(&p, c->dev));
    snprintf(name, name_len, "%s %08X:%02X:%02X.0 %s", p.name, p.pciDomainID, p.pciBusID, p.pciDeviceID,
             c->kernel_name.c_str());
    *sm_count = (uint32_t)p.multiProcessorCount;
    return PEARL_OK;
}

int pearl_set_a(const int8_t *a, uint32_t m, uint32_t k) {
    int rc;
    Ctx *c = ready_ctx(rc);
    return c ? set_a_rows(*c, a, m, 0, m, k) : rc;
}

int pearl_set_b(const int8_t *bt, uint32_t n, uint32_t k) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    std::string e = pearl::check_dims(PEARL_ROW_PERIOD, n, k);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    c->n = 0;
    if ((rc = upload(*c, bt, n, k, false))) return rc;
    c->n = n; c->kb = k;
    return PEARL_OK;
}

int pearl_fast_path(void) {
    Ctx *c = cur();
    return c ? path_fast(*c) : 0;
}

int pearl_search_resident(uint32_t k, uint32_t r, const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                          const uint8_t seed_a[32], const uint8_t bound_le[32], uint32_t lo, uint32_t hi,
                          cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if ((rc = check_resident(*c, k, r, rows_pattern, cols_pattern, lo, hi))) return rc;
    const KeyBound kb = make_kb(seed_a, bound_le);
    const bool fast = path_fast(*c);
    const int v = fast ? c->tune.pick(c->variant) : -1;
    const double t0 = pearl::now_s();
    if ((rc = run(*c, v, lo, hi, kb, cap, nullptr))) return rc;
    uint32_t cnt = 0;
    CU(cudaMemcpyAsync(&cnt, &c->dout->count, 4, cudaMemcpyDeviceToHost, c->st));
    CU(cudaStreamSynchronize(c->st));
    const uint64_t done = (uint64_t)(hi - lo) * c->dims().col_tiles() * PEARL_H * PEARL_W * c->dims().L();
    if (fast) {
        const int win = c->tune.account(v, (double)done, pearl::now_s() - t0);
        if (win >= 0) { c->variant = win; c->kernel_name = VARIANTS[win].name; }
    }
    const uint32_t w = std::min(std::min(cnt, cap), (uint32_t)MAX_CANDS);
    if (w) {
        CU(cudaMemcpyAsync(out, c->dout->c, w * sizeof(cand_t), cudaMemcpyDeviceToHost, c->st));
        CU(cudaStreamSynchronize(c->st));
        pearl::sort_cands(out, w);
    }
    *count = cnt;
    *macs = done;
    return PEARL_OK;
}

int pearl_search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                 const uint8_t rows_pattern[6], const uint8_t cols_pattern[6], const uint8_t seed_a[32],
                 const uint8_t bound_le[32], uint32_t lo, uint32_t hi, cand_t *out, uint32_t cap, uint32_t *count,
                 uint64_t *macs) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (lo % 4 || hi % 4 || lo > hi || hi > m / 8) return fail(PEARL_ERR_ARGS, "row tiles: lo, hi multiples of 4");
    const uint32_t R0 = lo / 4 * 32, R1 = hi / 4 * 32;
    if ((rc = set_a_rows(*c, a + (size_t)R0 * k, m, R0, R1 - R0, k))) return rc;
    if ((rc = pearl_set_b(bt, n, k))) return rc;
    return pearl_search_resident(k, r, rows_pattern, cols_pattern, seed_a, bound_le, lo, hi, out, cap, count, macs);
}

int pearl_job(const uint8_t job_key[32], uint32_t m, uint32_t n, uint32_t k, uint32_t r,
              const uint8_t rows_pattern[6], const uint8_t cols_pattern[6], uint8_t hash_b_out[32],
              uint8_t seed_b_out[32]) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    std::string e = pearl::check_dims(m, n, k);
    if (e.empty()) e = pearl::check_job(m, n, k, r);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    if (memcmp(rows_pattern, PEARL_ROWS_PATTERN, 6) || memcmp(cols_pattern, PEARL_COLS_PATTERN, 6))
        return fail(PEARL_ERR_PATTERN, "pattern: the V100 kernel needs rows 01 01 01 03 00 00, cols 00 01 01 01 01 03");
    pearl::JobState &j = c->ops.job;
    j = pearl::JobState();
    j.job_key = pearl::from_bytes(job_key);
    j.m = m; j.n = n; j.k = k;
    std::vector<pearl::CV> lb, la;
    CU(c->ops.leaves_of(j.job_key, (uint64_t)n * k / 1024, lb, c->st));
    CU(c->ops.leaves_of(j.job_key, (uint64_t)m * k / 1024, la, c->st));
    pearl::job_seeds(j, std::move(lb), std::move(la));
    c->n = 0;
    c->m = 0;
    if ((rc = ensure(*c, n, k, false))) return rc;
    CU(c->ops.operand(j.seed_b, 'B', n, k, r, nullptr, c->db, &c->max_b2, c->st));
    c->n = n; c->kb = k;
    j.ready = true;
    pearl::to_bytes(j.hash_b, hash_b_out);
    pearl::to_bytes(j.seed_b, seed_b_out);
    return PEARL_OK;
}

int pearl_pass(const int8_t nonce[8], uint8_t hash_a_out[32], uint8_t seed_a_out[32]) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    pearl::JobState &j = c->ops.job;
    if (!j.ready) return fail(PEARL_ERR_STATE, "pearl_job first");
    j.pass_ready = false;
    pearl::pass_seeds(j, nonce);
    c->m = 0;
    if ((rc = ensure(*c, j.m, j.k, true))) return rc;
    CU(c->ops.operand(j.seed_a, 'A', j.m, j.k, PEARL_R, nonce, c->da, &c->max_a2, c->st));
    c->m = c->a_rows = j.m; c->ka = j.k; c->a_row0 = 0;
    pearl::to_bytes(j.hash_a, hash_a_out);
    pearl::to_bytes(j.seed_a, seed_a_out);
    return PEARL_OK;
}

int pearl_tree_nodes(int matrix, const uint64_t *leaf_idx, uint32_t n_leaves, uint8_t *siblings_out, uint32_t cap,
                     uint32_t *n_siblings, uint8_t root_out[32]) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    const std::string e = pearl::tree_nodes(c->ops.job, matrix, leaf_idx, n_leaves, siblings_out, cap, n_siblings,
                                            root_out);
    return e.empty() ? PEARL_OK : fail(PEARL_ERR_ARGS, e);
}

int pearl_transcripts(uint32_t k, uint32_t lo, uint32_t hi, uint32_t *out) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if ((rc = check_resident(*c, k, PEARL_R, nullptr, nullptr, lo, hi))) return rc;
    std::vector<uint32_t> host;
    if ((rc = transcripts_by(*c, path_fast(*c) ? c->variant : -1, lo, hi, host))) return rc;
    memcpy(out, host.data(), host.size() * 4);
    return PEARL_OK;
}

int pearl_variant_count(void) { return NVAR; }

const char *pearl_variant_name(int i) { return i >= 0 && i < NVAR ? VARIANTS[i].name : nullptr; }

int pearl_variants(char *names, uint32_t len) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    std::string out;
    for (int v = 0; v < NVAR; ++v)
        if (VARIANTS[v].mining) out += (out.empty() ? "" : ",") + std::string(VARIANTS[v].name);
    if (out.size() + 1 > len) return fail(PEARL_ERR_ARGS, "variants buffer too small");
    memcpy(names, out.c_str(), out.size() + 1);
    return PEARL_OK;
}

int pearl_set_variant(const char *name) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (!name || !*name) { c->tune.pinned = false; return PEARL_OK; }
    const int v = find_variant(name);
    if (v < 0 || !VARIANTS[v].mining) return fail(PEARL_ERR_ARGS, std::string("not a mining variant: ") + name);
    c->variant = v;
    c->kernel_name = name;
    c->tune.pinned = true;
    return PEARL_OK;
}

int pearl_bench_run(const char *name, uint32_t lo, uint32_t hi, uint64_t *macs) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if ((rc = check_resident(*c, c->ka, PEARL_R, nullptr, nullptr, lo, hi))) return rc;
    const int v = find_variant(name);
    if (v < 0) return fail(PEARL_ERR_ARGS, std::string("unknown variant ") + name);
    KeyBound kb = {};
    if ((rc = run(*c, v, lo, hi, kb, 0, nullptr))) return rc;
    CU(cudaStreamSynchronize(c->st));
    *macs = (uint64_t)(hi - lo) * c->dims().col_tiles() * PEARL_H * PEARL_W * c->dims().L();
    return PEARL_OK;
}

int pearl_telemetry(char *json, uint32_t len) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (len == 0) return fail(PEARL_ERR_ARGS, "len");
    pearl::telemetry_json(json, len, c->kernel_name, path_fast(*c) ? "fast" : "exact", c->sm,
                          c->rate.empty() ? 0 : c->rate[c->variant], c->tune_mhz, c->tune, c->nvdev);
    return PEARL_OK;
}

}  // extern "C"
