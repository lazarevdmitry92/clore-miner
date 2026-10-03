// Pearl (PRL) mining kernel for Ampere, Ada, Blackwell (sm_80/86/89/120): C' = A'·B' on int8 tensor cores
// (mma.m16n8k32, s32 accumulate), XOR of every hash tile after each r = 128 of k, transcript, keyed BLAKE3 jackpot,
// comparison with the bound. TZ_sm80_kernel.md; design and numbers: README.md; contract: ../common/pearl_api.h.
//
// One thread owns one 8x16 hash tile (s0 §6): warp tile 64x64, 128 int32 accumulators per thread, XOR and the
// transcript in registers — no warp reduction (SOAT's 16x16 contiguous tile needs one, noisy_gemm.cuh:76).
// Global -> shared by a cp.async ring of STAGES x 64 bytes of k (two mma k-steps), fragments by ldmatrix.x4.
// int8 x int8 -> int32 is exact for any operands: |sum| <= 65536 * 127^2 < 2^31.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "../common/pearl_api.h"
#include "../common/operands_cuda.cuh"
#include "../common/runtime.h"
#include "pearl_host80.h"

using pearl80::Dims;

#define MAX_CANDS 4096

struct KeyBound { uint32_t key[8], bound[8]; };
struct DevOut { uint32_t count; cand_t c[MAX_CANDS]; };

// options: ABL_* exist only for pearl_bench ablate (TZ_v100_kernel_v2 §3.1 / TZ_sm80 §3) and never reach pearl_search
enum {
    ABL_NOEPI = 4,    // no XOR/transcript every 128 of k (the sums are folded once at the end, so the MMAs stay live)
    ABL_SMEM = 8,     // the ring is filled once; the k loop re-reads it (no global/L2 traffic)
    ABL_REGS = 16,    // fragments loaded once; the k loop is mma + epilogue only (~ bare peak)
};

// ---------------------------------------------------------------- device helpers

__device__ __forceinline__ void mma_s8(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(saddr));
}

__device__ __forceinline__ void cp_async16(uint32_t saddr, const void *g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
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

// ---------------------------------------------------------------- the kernel
// CTA = WM x WN warps, CTA tile 64WM x 64WN. Shared ring: STAGES x (64WM rows of A' + 64WN rows of B'^T) x 64 bytes,
// swizzled (pearl_layout80.h). One __syncthreads per stage (64 of k); the chunk epilogue every 2 stages.
// a, bt: int8 row-major, k bytes per row, row index = absolute row; the device buffers have PAD_ROWS zero rows past
// the data, so a CTA reaching past R1 / n reads zeros there and discards those results.
template <int WM, int WN, int STAGES, int OPT>
__global__ void __launch_bounds__(WM * WN * 32, (WM * WN <= 4) ? 2 : 1)
imma_kernel(const int8_t *__restrict__ a, const int8_t *__restrict__ bt, uint32_t R0, uint32_t R1, uint32_t n,
            uint32_t k, uint32_t L, uint32_t ncb, KeyBound kb, DevOut *out, uint32_t cap, uint32_t *transcripts,
            uint32_t lo, uint32_t col_tiles) {
    constexpr int NT = WM * WN * 32;
    constexpr int AR = 64 * WM, ROWS = 64 * (WM + WN);
    constexpr int NP = ROWS * 4 / NT;              // 16-byte cp.async per thread per stage
    constexpr int STAGE = ROWS * 64;               // bytes
    static_assert(ROWS * 4 % NT == 0, "staging split");
    static_assert(STAGES >= 2, "ring");
    extern __shared__ __align__(128) uint8_t sm[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);

    const int tid = threadIdx.x, l = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const uint32_t cb = blockIdx.x % ncb, rb = blockIdx.x / ncb;
    const uint32_t cta_r0 = R0 + rb * AR, cta_c0 = cb * 64 * WN;
    const uint32_t nst = L / K_TILE;

    // staging: item q -> shared row r = idx / 4, chunk idx % 4
    const int8_t *src[NP];
    uint32_t dst[NP];
#pragma unroll
    for (int q = 0; q < NP; ++q) {
        const int idx = tid + q * NT, r = idx >> 2, c = idx & 3;
        src[q] = (r < AR ? a + (size_t)(cta_r0 + r) * k : bt + (size_t)(cta_c0 + (uint32_t)(r - AR)) * k) + 16 * c;
        dst[q] = smem_off(r, c);
    }
    auto issue = [&](uint32_t st) {   // stage st of k into its ring slot; always one commit group per call
        if (st < nst) {
            const uint32_t base = sbase + (st % STAGES) * STAGE;
#pragma unroll
            for (int q = 0; q < NP; ++q) cp_async16(base + dst[q], src[q] + (size_t)st * K_TILE);
        }
        cp_async_commit();
    };

    // ldmatrix addresses for k-step 0 of a stage; k-step s is (off ^ 32s): chunk = 2s + ld_chunk, rows 64-aligned
    uint32_t aoff[4], boff[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) aoff[i] = smem_off(wm * 64 + 16 * i + ld_row(l), ld_chunk(l));
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) boff[jp] = smem_off(AR + wn * 64 + 16 * jp + ld_row(l), ld_chunk(l));

    int32_t acc[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

    // one mma k-step (32 of k) from fragments
    auto kstep = [&](const uint32_t (&af)[4][4], const uint32_t (&bf)[8][2]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], af[i], bf[j]);
    };
    auto load_frags = [&](uint32_t stage_base, int s, uint32_t (&af)[4][4], uint32_t (&bf)[8][2]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) ldsm_x4(af[i], stage_base + (aoff[i] ^ (32 * s)));
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {   // matrices: (cols 0-7, k lo), (cols 8-15, k lo), (0-7, k hi), (8-15, k hi)
            uint32_t r4[4];
            ldsm_x4(r4, stage_base + (boff[jp] ^ (32 * s)));
            bf[2 * jp][0] = r4[0];
            bf[2 * jp + 1][0] = r4[1];
            bf[2 * jp][1] = r4[2];
            bf[2 * jp + 1][1] = r4[3];
        }
    };
    uint32_t t[16];   // indexed by chunk % 16 at run time -> local memory, touched once per 128 of k
    for (int s = 0; s < 16; ++s) t[s] = 0;
    auto epilogue = [&](uint32_t chunk) {   // XOR of the 128 running sums of this thread's tile
        uint32_t x[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                x[j & 3] = xor3(x[j & 3], (uint32_t)acc[i][j][0] ^ (uint32_t)acc[i][j][1],
                                (uint32_t)acc[i][j][2] ^ (uint32_t)acc[i][j][3]);
        transcript_step(t, chunk, xor3(x[0], x[1], x[2] ^ x[3]));
    };

    if (OPT & ABL_REGS) {
        issue(0);
        cp_async_wait<0>();
        __syncthreads();
        uint32_t af[4][4], bf[8][2];
        load_frags(sbase, 0, af, bf);
        for (uint32_t st = 0; st < nst; ++st) {
            kstep(af, bf);
            kstep(af, bf);
            if (st & 1) epilogue(st >> 1);
        }
    } else {
        const uint32_t prologue = (OPT & ABL_SMEM) ? STAGES : STAGES - 1;
        for (uint32_t s = 0; s < prologue; ++s) issue(s);
        if (OPT & ABL_SMEM) { cp_async_wait<0>(); __syncthreads(); }
        for (uint32_t st = 0; st < nst; ++st) {
            if (!(OPT & ABL_SMEM)) {
                cp_async_wait<STAGES - 2>();   // stage st has landed (this thread's copies) ...
                __syncthreads();               // ... everyone's; and slot (st - 1) % STAGES is free again
                issue(st + STAGES - 1);
            }
            const uint32_t cur = sbase + (st % STAGES) * STAGE;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                uint32_t af[4][4], bf[8][2];
                load_frags(cur, s, af, bf);
                kstep(af, bf);
            }
            if (!(OPT & ABL_NOEPI) && (st & 1)) epilogue(st >> 1);
        }
    }
    if (OPT & ABL_NOEPI) epilogue(0);   // keep the MMAs live: one fold at the end
    cp_async_wait<0>();

    const uint32_t band = cta_r0 + wm * 64, wcol = cta_c0 + wn * 64;
    if (band >= R1 || wcol >= n) return;
    finish_tile(t, lane_row_tile(band, l), lane_col_tile(wcol, l), kb, out, cap, transcripts, lo, col_tiles);
}

// ---------------------------------------------------------------- host side

namespace {

struct Ctx;
typedef void (*LaunchFn)(const Ctx &, uint32_t, uint32_t, const KeyBound &, uint32_t, uint32_t *, cudaStream_t);

struct Variant {
    const char *name;
    int wm, wn, stages, opt;
    bool mining;      // ablations run only through pearl_bench_run
    LaunchFn launch;
    int smem() const { return stages * 64 * (wm + wn) * 64; }
};

struct Ctx {
    int dev = -1, sm = 0, max_mhz = 0, smem_optin = 0;
    bool ready = false;
    cudaStream_t st = nullptr;
    int8_t *da = nullptr, *db = nullptr;     // da holds rows [a_row0, a_row0 + a_rows) of A'
    size_t cap_a = 0, cap_b = 0;
    uint32_t m = 0, n = 0, ka = 0, kb = 0, a_row0 = 0, a_rows = 0;
    DevOut *dout = nullptr;
    uint32_t *dtr = nullptr;
    size_t cap_tr = 0;
    int variant = 0;
    std::string kernel_name = "none";
    void *nvdev = nullptr;
    std::vector<double> rate;   // startup MAC/s per variant (0 = not measured / does not fit this card)
    double tune_mhz = 0;
    pearl::OnlineTune tune;
    pearl_ops::Gpu ops;         // resident path: generators, job state, trees
    Dims dims() const { Dims d; d.m = m; d.n = n; d.k = ka; return d; }
};

const int MAX_DEV = 16;
Ctx g_ctx[MAX_DEV];
int g_dev = -1;
thread_local int t_dev = -1;
thread_local std::string t_err;

int cur_index() { return t_dev >= 0 ? t_dev : g_dev; }
Ctx *cur() { return cur_index() >= 0 ? &g_ctx[cur_index()] : nullptr; }
int fail(int code, const std::string &msg) { t_err = msg; return code; }

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    return fail(PEARL_ERR_CUDA, std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

template <int WM, int WN, int STAGES, int OPT>
void launch(const Ctx &c, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr, cudaStream_t st) {
    const uint32_t R0 = lo / 8 * 64, R1 = hi / 8 * 64, L = c.ka - c.ka % PEARL_R;
    const uint32_t nrb = (R1 - R0 + 64 * WM - 1) / (64 * WM), ncb = (c.n + 64 * WN - 1) / (64 * WN);
    const size_t smem = (size_t)STAGES * 64 * (WM + WN) * 64;
    imma_kernel<WM, WN, STAGES, OPT><<<nrb * ncb, WM * WN * 32, smem, st>>>(
        c.da - (size_t)c.a_row0 * c.ka, c.db, R0, R1, c.n, c.ka, L, ncb, kb, c.dout, cap, tr, lo, c.n / 64 * 4);
}

template <int WM, int WN, int STAGES, int OPT>
cudaError_t prepare(int smem_optin) {
    const int smem = STAGES * 64 * (WM + WN) * 64;
    if (smem > smem_optin) return cudaSuccess;   // does not fit this card; the variant is skipped
    cudaError_t e = cudaFuncSetAttribute(imma_kernel<WM, WN, STAGES, OPT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem);
    if (e == cudaSuccess)
        e = cudaFuncSetAttribute(imma_kernel<WM, WN, STAGES, OPT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    return e;
}

#define V(WM, WN, S, OPT, NAME, MINING) {NAME, WM, WN, S, OPT, MINING, launch<WM, WN, S, OPT>}
const Variant VARIANTS[] = {
    V(2, 2, 3, 0, "sm80-imma-128x128-s3", true),
    V(2, 2, 4, 0, "sm80-imma-128x128-s4", true),
    V(2, 4, 3, 0, "sm80-imma-128x256-s3", true),
    V(2, 4, 4, 0, "sm80-imma-128x256-s4", true),
    V(4, 2, 3, 0, "sm80-imma-256x128-s3", true),
    V(4, 2, 4, 0, "sm80-imma-256x128-s4", true),
    // ablations on the likely winner's shape (128x256, 3 stages)
    V(2, 4, 3, ABL_NOEPI, "ablate-no-epilogue", false),
    V(2, 4, 3, ABL_SMEM, "ablate-smem-only", false),
    V(2, 4, 3, ABL_REGS, "ablate-regs-only", false),
};
const int NVAR = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

cudaError_t prepare_all(int optin) {
    cudaError_t e = cudaSuccess;
#define PREP(WM, WN, S, OPT) if (e == cudaSuccess) e = prepare<WM, WN, S, OPT>(optin);
    PREP(2, 2, 3, 0) PREP(2, 2, 4, 0) PREP(2, 4, 3, 0) PREP(2, 4, 4, 0) PREP(4, 2, 3, 0) PREP(4, 2, 4, 0)
    PREP(2, 4, 3, ABL_NOEPI) PREP(2, 4, 3, ABL_SMEM) PREP(2, 4, 3, ABL_REGS)
#undef PREP
    return e;
}

bool fits(const Ctx &c, int v) { return VARIANTS[v].smem() <= c.smem_optin; }

int find_variant(const char *name) {
    for (int v = 0; v < NVAR; ++v)
        if (!strcmp(name, VARIANTS[v].name)) return v;
    return -1;
}

// Device copies carry PAD_ROWS zero rows past the data (see imma_kernel).
const uint32_t PAD_ROWS = 256;

// device buffer for rows x k of A' or B'^T (+ zero padding rows)
int ensure(Ctx &c, uint32_t rows, uint32_t k, bool is_a) {
    const size_t bytes = (size_t)rows * k, alloc = bytes + (size_t)PAD_ROWS * k;
    int8_t *&dp = is_a ? c.da : c.db;
    size_t &cap = is_a ? c.cap_a : c.cap_b;
    if (alloc > cap) {
        if (dp) CU(cudaFree(dp));
        dp = nullptr;
        CU(cudaMalloc(&dp, alloc));
        cap = alloc;
    }
    CU(cudaMemsetAsync(dp + bytes, 0, alloc - bytes, c.st));
    return PEARL_OK;
}

int upload(Ctx &c, const int8_t *h, uint32_t rows, uint32_t k, bool is_a) {
    int rc = ensure(c, rows, k, is_a);
    if (rc) return rc;
    CU(cudaMemcpyAsync(is_a ? c.da : c.db, h, (size_t)rows * k, cudaMemcpyHostToDevice, c.st));
    CU(cudaStreamSynchronize(c.st));
    return PEARL_OK;
}

int run(Ctx &c, int v, uint32_t lo, uint32_t hi, const KeyBound &kb, uint32_t cap, uint32_t *tr) {
    CU(cudaMemsetAsync(&c.dout->count, 0, 4, c.st));
    if (hi > lo) {
        VARIANTS[v].launch(c, lo, hi, kb, cap, tr, c.st);
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

int transcripts_by(Ctx &c, int v, uint32_t lo, uint32_t hi, std::vector<uint32_t> &host) {
    const size_t words = (size_t)(hi - lo) * (c.n / 64 * 4) * 16;
    int rc = ensure_tr(c, words);
    if (rc) return rc;
    KeyBound kb = {};
    if ((rc = run(c, v, lo, hi, kb, 0, c.dtr))) return rc;
    host.resize(words);
    CU(cudaMemcpyAsync(host.data(), c.dtr, words * 4, cudaMemcpyDeviceToHost, c.st));
    CU(cudaStreamSynchronize(c.st));
    return PEARL_OK;
}

void set_dims(Ctx &c, uint32_t m, uint32_t n, uint32_t k) { c.m = c.a_rows = m; c.a_row0 = 0; c.n = n; c.ka = c.kb = k; }

// Self-check: every variant that fits against the host int64 reference (noise, uniform +-127, only +-127; partial
// CTAs), and the device BLAKE3 + comparison against the host one. Fails -> every call returns PEARL_ERR_SELFCHECK.
int selfcheck(Ctx &c) {
    std::string why;
    CU(c.ops.selfcheck(c.st, why));
    if (!why.empty()) return fail(PEARL_ERR_SELFCHECK, "self-check: device operands differ from the host: " + why);
    struct Case { uint32_t m, n, k; int dist; };
    const Case cases[] = {{320, 448, 2048, 0}, {128, 192, 4096, 1}, {64, 128, 2048, 2}};
    for (const Case &cs : cases) {
        auto A = pearl::synth((size_t)cs.m * cs.k, cs.dist, 11 + cs.m + cs.dist);
        auto B = pearl::synth((size_t)cs.n * cs.k, cs.dist, 23 + cs.n + cs.dist);
        int rc = upload(c, A.data(), cs.m, cs.k, true);
        if (!rc) rc = upload(c, B.data(), cs.n, cs.k, false);
        if (rc) return rc;
        set_dims(c, cs.m, cs.n, cs.k);
        const Dims d = c.dims();
        const uint32_t hi = d.row_tiles();
        std::vector<uint32_t> ref((size_t)hi * d.col_tiles() * 16), got;
        pearl80::ref_transcripts(A.data(), B.data(), d, 0, hi, ref.data());
        for (int v = 0; v < NVAR; ++v) {
            if (!VARIANTS[v].mining || !fits(c, v)) continue;
            if ((rc = transcripts_by(c, v, 0, hi, got))) return rc;
            if (got != ref) {
                size_t bad = 0;
                for (size_t i = 0; i < ref.size(); ++i) bad += got[i] != ref[i];
                char msg[200];
                snprintf(msg, sizeof msg, "self-check: %s transcripts differ from the reference (%zu of %zu words, "
                         "m=%u n=%u k=%u dist=%d)", VARIANTS[v].name, bad, ref.size(), cs.m, cs.n, cs.k, cs.dist);
                return fail(PEARL_ERR_SELFCHECK, msg);
            }
        }
        const uint32_t tiles = hi * d.col_tiles();
        if (tiles <= MAX_CANDS) {
            uint8_t seed[32], ones[32];
            for (int i = 0; i < 32; ++i) seed[i] = (uint8_t)(i * 7 + cs.k);
            memset(ones, 0xFF, 32);
            KeyBound kb = make_kb(seed, ones);
            if ((rc = run(c, 0, 0, hi, kb, MAX_CANDS, nullptr))) return rc;
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
    set_dims(c, 0, 0, 0);
    return PEARL_OK;
}

// Startup autotune on a mining-shaped problem; the online autotune re-measures the best three under the real
// power limit. PEARL_VARIANT=<name> pins one.
int autotune(Ctx &c) {
    c.rate.assign(NVAR, 0.0);
    if (const char *want = getenv("PEARL_VARIANT")) {
        const int v = find_variant(want);
        if (v < 0 || !VARIANTS[v].mining || !fits(c, v))
            return fail(PEARL_ERR_ARGS, std::string("PEARL_VARIANT: unknown or does not fit: ") + want);
        c.variant = v;
        c.kernel_name = want;
        c.tune.pinned = true;
        return PEARL_OK;
    }
    const uint32_t m = 16384, n = 4096, k = 2048;
    auto A = pearl::synth((size_t)m * k, 0, 5), B = pearl::synth((size_t)n * k, 0, 6);
    int rc = upload(c, A.data(), m, k, true);
    if (!rc) rc = upload(c, B.data(), n, k, false);
    if (rc) return rc;
    set_dims(c, m, n, k);
    const double macs = (double)m * n * k;
    KeyBound kb = {};
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    double best = 0, mhz = 0;
    int nmhz = 0;
    for (int v = 0; v < NVAR; ++v) {
        if (!VARIANTS[v].mining || !fits(c, v)) continue;
        if ((rc = run(c, v, 0, m / 8, kb, 0, nullptr))) return rc;
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
    if (best == 0) return fail(PEARL_ERR_ARGS, "no variant fits the shared memory of this card");
    c.tune_mhz = nmhz ? mhz / nmhz : c.max_mhz;
    c.kernel_name = VARIANTS[c.variant].name;
    std::vector<int> order;
    for (int v = 0; v < NVAR; ++v)
        if (c.rate[v] > 0) order.push_back(v);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return c.rate[x] > c.rate[y]; });
    c.tune.start(std::vector<int>(order.begin(), order.begin() + std::min<size_t>(3, order.size())));
    set_dims(c, 0, 0, 0);
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
    if (p.major < 8) return fail(PEARL_ERR_ARGS, std::string("needs sm_80 or newer: ") + p.name);
    c.dev = device;
    c.sm = p.multiProcessorCount;
    c.max_mhz = p.clockRate / 1000;
    c.smem_optin = (int)p.sharedMemPerBlockOptin;
    if (!c.st) {
        CU(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));
        CU(cudaMalloc(&c.dout, sizeof(DevOut)));
        CU(prepare_all(c.smem_optin));
        char bus[32];
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, device) == cudaSuccess) c.nvdev = pearl::nvml().device(bus);
    }
    int rc = selfcheck(c);
    if (!rc) rc = autotune(c);
    if (rc) return rc;
    c.ready = true;
    return PEARL_OK;
}

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
    std::string e = pearl80::check_dims(m, PEARL_COL_PERIOD, k);
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
    std::string e = pearl80::check_search(c.dims(), k, r, rows, cols, lo, hi);
    if (!e.empty()) return fail(pearl80::pattern_mismatch(rows, cols) ? PEARL_ERR_PATTERN : PEARL_ERR_ARGS, e);
    if (hi > lo && (lo / 8 * 64 < c.a_row0 || hi / 8 * 64 > c.a_row0 + c.a_rows))
        return fail(PEARL_ERR_ARGS, "row tiles outside the rows of A' on the device");
    return PEARL_OK;
}

uint64_t macs_of(const Ctx &c, uint32_t lo, uint32_t hi) {
    return (uint64_t)(hi - lo) * c.dims().col_tiles() * PEARL_H * PEARL_W * c.dims().L();
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
    std::string e = pearl80::check_dims(PEARL_ROW_PERIOD, n, k);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    c->n = 0;
    if ((rc = upload(*c, bt, n, k, false))) return rc;
    c->n = n; c->kb = k;
    return PEARL_OK;
}

int pearl_fast_path(void) { return 1; }

int pearl_search_resident(uint32_t k, uint32_t r, const uint8_t rows_pattern[6], const uint8_t cols_pattern[6],
                          const uint8_t seed_a[32], const uint8_t bound_le[32], uint32_t lo, uint32_t hi,
                          cand_t *out, uint32_t cap, uint32_t *count, uint64_t *macs) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if ((rc = check_resident(*c, k, r, rows_pattern, cols_pattern, lo, hi))) return rc;
    const KeyBound kb = make_kb(seed_a, bound_le);
    const int v = c->tune.pick(c->variant);
    const double t0 = pearl::now_s();
    if ((rc = run(*c, v, lo, hi, kb, cap, nullptr))) return rc;
    uint32_t cnt = 0;
    CU(cudaMemcpyAsync(&cnt, &c->dout->count, 4, cudaMemcpyDeviceToHost, c->st));
    CU(cudaStreamSynchronize(c->st));
    const uint64_t done = macs_of(*c, lo, hi);
    const int win = c->tune.account(v, (double)done, pearl::now_s() - t0);
    if (win >= 0) { c->variant = win; c->kernel_name = VARIANTS[win].name; }
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
    if (lo % 8 || hi % 8 || lo > hi || hi > m / 8) return fail(PEARL_ERR_ARGS, "row tiles: lo, hi multiples of 8");
    const uint32_t R0 = lo / 8 * 64, R1 = hi / 8 * 64;
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
    std::string e = pearl80::check_dims(m, n, k);
    if (e.empty()) e = pearl::check_job(m, n, k, r);
    if (!e.empty()) return fail(PEARL_ERR_ARGS, e);
    if (pearl80::pattern_mismatch(rows_pattern, cols_pattern))
        return fail(PEARL_ERR_PATTERN, "pattern: the sm80 kernel needs the 8x16 tile");
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
    uint32_t max2 = 0;
    CU(c->ops.operand(j.seed_b, 'B', n, k, r, nullptr, c->db, &max2, c->st));
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
    uint32_t max2 = 0;
    CU(c->ops.operand(j.seed_a, 'A', j.m, j.k, PEARL_R, nonce, c->da, &max2, c->st));
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
    if ((rc = transcripts_by(*c, c->variant, lo, hi, host))) return rc;
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
        if (VARIANTS[v].mining && fits(*c, v)) out += (out.empty() ? "" : ",") + std::string(VARIANTS[v].name);
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
    if (v < 0 || !VARIANTS[v].mining || !fits(*c, v))
        return fail(PEARL_ERR_ARGS, std::string("not a mining variant for this card: ") + name);
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
    if (v < 0 || !fits(*c, v)) return fail(PEARL_ERR_ARGS, std::string("unknown or does not fit: ") + name);
    KeyBound kb = {};
    if ((rc = run(*c, v, lo, hi, kb, 0, nullptr))) return rc;
    CU(cudaStreamSynchronize(c->st));
    *macs = macs_of(*c, lo, hi);
    return PEARL_OK;
}

int pearl_telemetry(char *json, uint32_t len) {
    int rc;
    Ctx *c = ready_ctx(rc);
    if (!c) return rc;
    if (len == 0) return fail(PEARL_ERR_ARGS, "len");
    pearl::telemetry_json(json, len, c->kernel_name, "fast", c->sm, c->rate.empty() ? 0 : c->rate[c->variant],
                          c->tune_mhz, c->tune, c->nvdev);
    return PEARL_OK;
}

}  // extern "C"
