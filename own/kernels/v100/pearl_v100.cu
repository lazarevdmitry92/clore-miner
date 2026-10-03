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
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "pearl_layout.h"
#include "pearl_host.h"

using pearl::Dims;

#define MAGIC_F 12582912.0f   // 1.5 * 2^23
#define MAX_CANDS 4096

struct KeyBound { uint32_t key[8], bound[8]; };
struct DevOut { uint32_t count; cand_t c[MAX_CANDS]; };

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
// 64WN rows of B'^T) x K_TILE fp16, swizzled (pearl_layout.h). Global -> registers -> convert -> shared; one
// __syncthreads per stage; the chunk epilogue (XOR, transcript) every 4 stages = 128 of k.
template <int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32, (WM * WN <= 4) ? 2 : 1)
fast_kernel(const int8_t *__restrict__ a, const int8_t *__restrict__ bt, uint32_t R0, uint32_t R1, uint32_t n,
            uint32_t k, uint32_t L, uint32_t ncb, KeyBound kb, DevOut *out, uint32_t cap, uint32_t *transcripts,
            uint32_t lo, uint32_t col_tiles) {
    constexpr int NT = WM * WN * 32;
    constexpr int AR = 64 * WM, ROWS = 64 * (WM + WN);
    constexpr int NP = ROWS * 2 / NT;              // 16-byte global loads per thread per stage
    static_assert(ROWS * 2 % NT == 0, "staging split");
    extern __shared__ __align__(16) uint8_t sm[];

    const int tid = threadIdx.x, l = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const uint32_t cb = blockIdx.x % ncb, rb = blockIdx.x / ncb;
    const uint32_t cta_r0 = R0 + rb * AR, cta_c0 = cb * 64 * WN;

    // staging: item q -> shared row r = idx / 2, 16-byte half sp = idx % 2 of the 32-byte stage row
    const int8_t *src[NP];
    int dst[NP];
#pragma unroll
    for (int q = 0; q < NP; ++q) {
        int idx = tid + q * NT, r = idx >> 1, sp = idx & 1;
        const int8_t *p;
        if (r < AR) p = a + (size_t)min(cta_r0 + r, R1 - 1) * k;      // rows past R1 / n: clamped, results unused
        else p = bt + (size_t)min(cta_c0 + (uint32_t)(r - AR), n - 1) * k;
        src[q] = p + 16 * sp;
        dst[q] = smem_off(r, 2 * sp);                                    // slot 2sp; slot 2sp+1 = dst ^ 16
    }
    uint4 pre[NP];
    auto load_stage = [&](uint32_t k0) {
#pragma unroll
        for (int q = 0; q < NP; ++q) pre[q] = __ldg((const uint4 *)(src[q] + k0));
    };
    auto store_stage = [&](uint8_t *buf) {
#pragma unroll
        for (int q = 0; q < NP; ++q) {
            uint2 x0 = i8x4_to_f16x4(pre[q].x), x1 = i8x4_to_f16x4(pre[q].y);
            uint2 x2 = i8x4_to_f16x4(pre[q].z), x3 = i8x4_to_f16x4(pre[q].w);
            *(uint4 *)(buf + dst[q]) = make_uint4(x0.x, x0.y, x1.x, x1.y);
            *(uint4 *)(buf + (dst[q] ^ 16)) = make_uint4(x2.x, x2.y, x3.x, x3.y);
        }
    };

    // fragment addresses: row base | 16 * (bit 1 of the row) — slot p is then (base ^ 16p) (rows are 64-aligned)
    int abase[4], bbase[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = wm * 64 + lane_a_row(l, i);
        abase[i] = smem_off(r, 0);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int r = AR + wn * 64 + lane_b_row(l, j);
        bbase[j] = smem_off(r, 0);
    }

    float acc[4][4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = MAGIC_F;
    uint32_t t[16];   // indexed by chunk % 16 at run time -> local memory, touched once per 128 of k
    for (int s = 0; s < 16; ++s) t[s] = 0;

    const uint32_t nst = L / K_TILE;
    uint8_t *buf0 = sm, *buf1 = sm + ROWS * 64;
    load_stage(0);
    store_stage(buf0);
    __syncthreads();

    for (uint32_t st = 0; st < nst; ++st) {
        uint8_t *cur = (st & 1) ? buf1 : buf0, *nxt = (st & 1) ? buf0 : buf1;
        if (st + 1 < nst) load_stage((st + 1) * K_TILE);
#pragma unroll
        for (int p = 0; p < SLOTS; ++p) {
            uint4 fa[4], fb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) fa[i] = *(const uint4 *)(cur + (abase[i] ^ (16 * p)));
#pragma unroll
            for (int j = 0; j < 4; ++j) fb[j] = *(const uint4 *)(cur + (bbase[j] ^ (16 * p)));
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma884(acc[i][j], fa[i].x, fa[i].y, fb[j].x, fb[j].y);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma884(acc[i][j], fa[i].z, fa[i].w, fb[j].z, fb[j].w);
        }
        if (st + 1 < nst) store_stage(nxt);
        __syncthreads();
        if ((st & 3) == 3) {   // end of chunk c = st / 4: XOR of the 128 running sums of this thread's tile
            uint32_t x[4] = {0, 0, 0, 0};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 8; e += 2)
                        x[e >> 1] = xor3(x[e >> 1], __float_as_uint(acc[i][j][e]), __float_as_uint(acc[i][j][e + 1]));
            transcript_step(t, st >> 2, magic_xor_to_int(xor3(x[0], x[1], x[2] ^ x[3])));
        }
    }

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

// ---------------------------------------------------------------- host side

namespace {

struct Ctx {
    int dev = -1;
    bool ready = false;
    cudaStream_t st = nullptr;
    int8_t *da = nullptr, *db = nullptr;     // da holds rows [a_row0, a_row0 + a_rows) of A'
    size_t cap_a = 0, cap_b = 0;
    uint32_t m = 0, n = 0, ka = 0, kb = 0, a_row0 = 0, a_rows = 0;
    uint32_t max_a2 = 0, max_b2 = 0;
    DevOut *dout = nullptr;
    uint32_t *dmax = nullptr, *dtr = nullptr;
    size_t cap_tr = 0;
    int variant = 0;
    std::string kernel_name = "none";
    Dims dims() const { Dims d; d.m = m; d.n = n; d.k = ka; return d; }
    const int8_t *a_base() const { return da - (size_t)a_row0 * ka; }   // row index = absolute row of A'
};

struct Variant {
    const char *name;
    int wm, wn;
    void (*launch)(const Ctx &, uint32_t, uint32_t, const KeyBound &, uint32_t, uint32_t *, cudaStream_t);
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

template <int WM, int WN>
void launch_fast(const Ctx &c, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr,
                 cudaStream_t st) {
    const uint32_t R0 = lo / 4 * 32, R1 = hi / 4 * 32, L = c.ka - c.ka % PEARL_R;
    const uint32_t nrb = (R1 - R0 + 64 * WM - 1) / (64 * WM), ncb = (c.n + 64 * WN - 1) / (64 * WN);
    const size_t smem = 2 * 64 * (WM + WN) * 64;
    fast_kernel<WM, WN><<<nrb * ncb, WM * WN * 32, smem, st>>>(c.a_base(), c.db, R0, R1, c.n, c.ka, L, ncb, kb, c.dout,
                                                                cap, tr, lo, c.n / 64 * 4);
}

void launch_exact(const Ctx &c, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr,
                  cudaStream_t st) {
    const uint32_t ct = c.n / 64 * 4, L = c.ka - c.ka % PEARL_R;
    const uint64_t tiles = (uint64_t)(hi - lo) * ct;
    exact_kernel<<<(unsigned)((tiles + 127) / 128), 128, 0, st>>>(c.a_base(), c.db, c.ka, L, lo, hi, ct, kb, c.dout, cap, tr);
}

const Variant VARIANTS[] = {
    {"v100-hmma884-128x256", 2, 4, launch_fast<2, 4>},
    {"v100-hmma884-256x128", 4, 2, launch_fast<4, 2>},
    {"v100-hmma884-128x128", 2, 2, launch_fast<2, 2>},
};
const int NVAR = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

int path_fast(const Ctx &c) {
    if (const char *p = getenv("PEARL_PATH")) {
        if (!strcmp(p, "fast")) return 1;
        if (!strcmp(p, "exact")) return 0;
    }
    return pearl::cs_exact(c.max_a2, c.max_b2) ? 1 : 0;
}

int upload(Ctx &c, const int8_t *h, uint32_t rows, uint32_t k, bool is_a) {
    size_t bytes = (size_t)rows * k;
    int8_t *&dp = is_a ? c.da : c.db;
    size_t &cap = is_a ? c.cap_a : c.cap_b;
    if (bytes > cap) {
        if (dp) CU(cudaFree(dp));
        dp = nullptr;
        CU(cudaMalloc(&dp, bytes));
        cap = bytes;
    }
    CU(cudaMemcpyAsync(dp, h, bytes, cudaMemcpyHostToDevice, c.st));
    CU(cudaMemsetAsync(c.dmax, 0, 4, c.st));
    row_sumsq_max<<<(rows + 7) / 8, 256, 0, c.st>>>(dp, rows, k, c.dmax);
    CU(cudaGetLastError());
    uint32_t mx = 0;
    CU(cudaMemcpyAsync(&mx, c.dmax, 4, cudaMemcpyDeviceToHost, c.st));
    CU(cudaStreamSynchronize(c.st));
    (is_a ? c.max_a2 : c.max_b2) = mx;
    return PEARL_OK;
}

// fast (variant v) or exact (v < 0) over [lo, hi); transcripts to the device buffer when tr
int run(Ctx &c, int v, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr) {
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

// Self-check (TZ §8a.4): every kernel against the host int64 reference, and the device BLAKE3 + comparison against
// the host one. Fails -> pearl_init fails -> the miner does not start.
int selfcheck(Ctx &c) {
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
        for (int v = cs.fast ? 0 : -1; v < (cs.fast ? NVAR : 0); ++v) {
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
        if (cs.fast) {   // the exact kernel on the same noise: the second, independent implementation
            if ((rc = transcripts_by(c, -1, 0, hi, got))) return rc;
            if (got != ref) return fail(PEARL_ERR_SELFCHECK, "self-check: exact kernel differs on noise operands");
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

// Autotune (TZ §8a.8): time each fast variant on a mining-sized problem, keep the fastest. PEARL_VARIANT=<name>
// pins one.
int autotune(Ctx &c) {
    if (const char *want = getenv("PEARL_VARIANT")) {
        for (int v = 0; v < NVAR; ++v)
            if (!strcmp(want, VARIANTS[v].name)) { c.variant = v; c.kernel_name = want; return PEARL_OK; }
        return fail(PEARL_ERR_ARGS, std::string("PEARL_VARIANT: unknown ") + want);
    }
    const uint32_t m = 8192, n = 2048, k = 2048;   // the mining shape: B'^T 4 MB stays in the 6 MB L2
    auto A = pearl::synth((size_t)m * k, 0, 5), B = pearl::synth((size_t)n * k, 0, 6);
    int rc = upload(c, A.data(), m, k, true);
    if (!rc) rc = upload(c, B.data(), n, k, false);
    if (rc) return rc;
    c.m = c.a_rows = m; c.a_row0 = 0; c.n = n; c.ka = c.kb = k;
    KeyBound kb = {};
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int v = 0; v < NVAR; ++v) {
        if ((rc = run(c, v, 0, m / 8, kb, 0, nullptr))) return rc;   // warm-up
        CU(cudaEventRecord(e0, c.st));
        for (int rep = 0; rep < 5; ++rep)
            if ((rc = run(c, v, 0, m / 8, kb, 0, nullptr))) return rc;
        CU(cudaEventRecord(e1, c.st));
        CU(cudaEventSynchronize(e1));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) { best = ms; c.variant = v; }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    c.kernel_name = VARIANTS[c.variant].name;
    c.m = c.n = c.ka = c.kb = 0;
    return PEARL_OK;
}

}  // namespace

// ---------------------------------------------------------------- C API

namespace {

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
    if (!c.st) {
        CU(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));
        CU(cudaMalloc(&c.dout, sizeof(DevOut)));
        CU(cudaMalloc(&c.dmax, 4));
        CU(cudaFuncSetAttribute(fast_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 64 * 6 * 64));
        CU(cudaFuncSetAttribute(fast_kernel<4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 64 * 6 * 64));
        CU(cudaFuncSetAttribute(fast_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 64 * 4 * 64));
        CU(cudaFuncSetAttribute(fast_kernel<2, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CU(cudaFuncSetAttribute(fast_kernel<4, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CU(cudaFuncSetAttribute(fast_kernel<2, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
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

}  // namespace

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
    if (!c->m || !c->n) return fail(PEARL_ERR_STATE, "pearl_set_a/b first");
    if (c->ka != c->kb) return fail(PEARL_ERR_ARGS, "A' and B'^T have different k");
    std::string e = pearl::check_search(c->dims(), k, r, rows_pattern, cols_pattern, lo, hi);
    if (!e.empty()) return fail(memcmp(rows_pattern, PEARL_ROWS_PATTERN, 6) || memcmp(cols_pattern,
                                PEARL_COLS_PATTERN, 6) ? PEARL_ERR_PATTERN : PEARL_ERR_ARGS, e);
    if (hi > lo && (lo / 4 * 32 < c->a_row0 || hi / 4 * 32 > c->a_row0 + c->a_rows))
        return fail(PEARL_ERR_ARGS, "row tiles outside the rows of A' on the device");
    const KeyBound kb = make_kb(seed_a, bound_le);
    if ((rc = run(*c, path_fast(*c) ? c->variant : -1, lo, hi, kb, cap, nullptr))) return rc;
    uint32_t cnt = 0;
    CU(cudaMemcpyAsync(&cnt, &c->dout->count, 4, cudaMemcpyDeviceToHost, c->st));
    CU(cudaStreamSynchronize(c->st));
    const uint32_t w = std::min(std::min(cnt, cap), (uint32_t)MAX_CANDS);
    if (w) {
        CU(cudaMemcpyAsync(out, c->dout->c, w * sizeof(cand_t), cudaMemcpyDeviceToHost, c->st));
        CU(cudaStreamSynchronize(c->st));
        pearl::sort_cands(out, w);
    }
    *count = cnt;
    *macs = (uint64_t)(hi - lo) * c->dims().col_tiles() * PEARL_H * PEARL_W * c->dims().L();
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

int pearl_transcripts(uint32_t k, uint32_t lo, uint32_t hi, uint32_t *out) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (!c->m || !c->n) return fail(PEARL_ERR_STATE, "pearl_set_a/b first");
    std::string e = pearl::check_search(c->dims(), k, PEARL_R, nullptr, nullptr, lo, hi);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    if (hi > lo && (lo / 4 * 32 < c->a_row0 || hi / 4 * 32 > c->a_row0 + c->a_rows))
        return fail(PEARL_ERR_ARGS, "row tiles outside the rows of A' on the device");
    std::vector<uint32_t> host;
    if ((rc = transcripts_by(*c, path_fast(*c) ? c->variant : -1, lo, hi, host))) return rc;
    memcpy(out, host.data(), host.size() * 4);
    return PEARL_OK;
}

}  // extern "C"
