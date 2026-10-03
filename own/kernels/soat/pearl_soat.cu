// GPU backend of the own Pearl miner (contract: ../miner/README.md, "Контракт GPU-бэкенда") over the SOAT kernels.
//
// The GEMM + transcript kernels (noisyGemmPtx), powScan, transcriptFingerprint and the host tileTranscript /
// powDigest come unchanged from SOAT through #include of ../src/soat/src/algos/pearl-pow/{job.h, noisy_gemm.cuh,
// prepare.cuh}. Copied, because in SOAT they sit in an anonymous namespace / private class members of algo.cu:
// launchPtx and the "ptx" rows of kTileConfigs (algo.cu:115-139, 170-185, 222-241). The tuner and the self-check
// below are written after PearlPow::tuneShape / selfCheck (algo.cu:434-679).
//
// Derived from blindrun/soat-miner 48defc8 — Copyright (c) 2026 Son of a Tech, MIT License (LICENSE.soat).
//
// SOAT hash tile: contiguous 16x16, one warp per tile, r = 128 (SOAT mines only k = 2048; the kernel takes k as a
// parameter, and the self-check here runs at every k it is called with). Its transcript buffer is laid out
// block-major over r x r blocks, then (hi, wi) inside a block (noisy_gemm.cuh:1340-1349); decode() inverts it to the
// tile's first row / column, and row_tile = first_row / 16, col_tile = first_col / 16 is exactly the
// Pattern.partition order of the contiguous-16 pattern.
//
// Shapes: the kernel needs the rows of a call and n divisible by its block (64..256). A call is run on rows and n
// rounded up to 256; the extra rows / columns are whatever lies in the device buffers past the operands, and their
// tiles are dropped on the host (then every hit of the call is read back, so *count stays exact).

#include <cuda_runtime.h>
#include <cuda_pipeline.h>   // at file scope: noisy_gemm.cuh includes it inside namespace om::pearl

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "job.h"
#include "noisy_gemm.cuh"
#include "prepare.cuh"

#include "pearl_soat.h"

namespace {

constexpr uint32_t kR = 128, kSide = 16, kPad = 256;
const uint8_t kPattern16[6] = {0x00, 0x0f, 0x00, 0x00, 0x00, 0x00};

// ------------------------------------------------- copied from SOAT algo.cu (MIT, LICENSE.soat)

struct TileConfig {
    const char *name;
    int blockM, blockN, threads;
    int blockK;
    void (*launch)(dim3, int, cudaStream_t, const int8_t *, const int8_t *, uint32_t *, int, int, int, int);
};

template <int WM, int WN, int TM, int TN, int KKB, int ST = 3, int GM = 8>
void launchPtx(dim3 grid, int threads, cudaStream_t s, const int8_t *a, const int8_t *b, uint32_t *t, int m, int n,
               int k, int rank) {
    constexpr int kStages = ST;
    constexpr int aStride = KKB + 16, bStride = KKB + 16;
    constexpr int blockM = WM * TM * 16, blockN = WN * TN * 16;
    constexpr int smem = kStages * (blockM * aStride + blockN * bStride);
    static bool optedIn = false;
    if (!optedIn) {
        cudaFuncSetAttribute(om::pearl::noisyGemmPtx<WM, WN, TM, TN, KKB, ST, GM>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        optedIn = true;
    }
    om::pearl::noisyGemmPtx<WM, WN, TM, TN, KKB, ST, GM>
        <<<grid, threads, smem, s>>>(a, b, nullptr, t, m, n, k, rank, false);
}

// Only the raw-mma family: it reads B n-major, which is our B'^T (n x k row-major) as is. The WMMA families
// (single/dbuf/async) want B k-major and are slower on sm_80+ (soat.md §1), so they are not offered.
const TileConfig kTileConfigs[] = {
    {"ptx k64 2x4/2x2", 64, 128, 256, 64, &launchPtx<2, 4, 2, 2, 64>},
    {"ptx k64 2x4/4x2", 128, 128, 256, 64, &launchPtx<2, 4, 4, 2, 64>},
    {"ptx k64 2x4/2x4", 64, 256, 256, 64, &launchPtx<2, 4, 2, 4, 64>},
    {"ptx k64 2x4/4x4", 128, 256, 256, 64, &launchPtx<2, 4, 4, 4, 64>},
    {"ptx k64 4x2/2x2", 128, 64, 256, 64, &launchPtx<4, 2, 2, 2, 64>},
    {"ptx k64 4x4/2x2", 128, 128, 512, 64, &launchPtx<4, 4, 2, 2, 64>},
    {"ptx k64 4x4/4x2", 256, 128, 512, 64, &launchPtx<4, 4, 4, 2, 64>},
    {"ptx k64 4x4/2x4", 128, 256, 512, 64, &launchPtx<4, 4, 2, 4, 64>},
    {"ptx k64 8x2/2x2", 256, 64, 512, 64, &launchPtx<8, 2, 2, 2, 64>},
    {"ptx k64 4x8/2x2", 128, 256, 1024, 64, &launchPtx<4, 8, 2, 2, 64>},
    {"ptx k32 2x4/2x2", 64, 128, 256, 32, &launchPtx<2, 4, 2, 2, 32>},
    {"ptx k32 2x4/4x2", 128, 128, 256, 32, &launchPtx<2, 4, 4, 2, 32>},
    {"ptx k32 2x4/2x4", 64, 256, 256, 32, &launchPtx<2, 4, 2, 4, 32>},
    {"ptx k32 2x4/4x4", 128, 256, 256, 32, &launchPtx<2, 4, 4, 4, 32>},
    {"ptx k32 4x2/2x2", 128, 64, 256, 32, &launchPtx<4, 2, 2, 2, 32>},
    {"ptx k32 4x4/2x2", 128, 128, 512, 32, &launchPtx<4, 4, 2, 2, 32>},
    {"ptx k32 4x4/4x2", 256, 128, 512, 32, &launchPtx<4, 4, 4, 2, 32>},
    {"ptx k32 4x4/2x4", 128, 256, 512, 32, &launchPtx<4, 4, 2, 4, 32>},
    {"ptx k32 8x2/2x2", 256, 64, 512, 32, &launchPtx<8, 2, 2, 2, 32>},
    {"ptx k32 4x8/2x2", 128, 256, 1024, 32, &launchPtx<4, 8, 2, 2, 32>},
};
constexpr int kConfigs = sizeof(kTileConfigs) / sizeof(kTileConfigs[0]);

// ------------------------------------------------------------------------- state

struct State {
    bool init = false;
    int dev = 0;
    cudaDeviceProp prop{};
    cudaStream_t s = nullptr;
    cudaEvent_t evA = nullptr, evB = nullptr;
    int8_t *dA = nullptr, *dB = nullptr;
    size_t aCap = 0, bCap = 0;
    uint32_t *dT = nullptr, *dHitIndex = nullptr, *dHitDigest = nullptr;
    size_t tCap = 0, hiCap = 0, hdCap = 0;
    uint8_t *dKey = nullptr;
    uint32_t *dTarget = nullptr, *dHitCount = nullptr, *dFp = nullptr;
    // device copy of A' / B'^T: valid for (id, host pointer, dims) when id != 0
    uint64_t aId = 0, bId = 0;
    const void *aPtr = nullptr, *bPtr = nullptr;
    uint32_t aM = 0, aK = 0, bN = 0, bK = 0;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, int> tuned;   // (rows, n, k) after padding -> config
    std::set<std::pair<int, uint32_t>> checked;                        // (config, k) passed the self-check
    bool refused = false;                                              // a self-check failed: never mine here
};

State g;
std::mutex g_mu;
std::string g_err;

int fail(int code, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_err = buf;
    return code;
}

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) return fail(PEARL_E_CUDA, "%s: %s", #x, cudaGetErrorString(e_)); \
    } while (0)

#define TRY(x)                \
    do {                      \
        int rc_ = (x);        \
        if (rc_) return rc_;  \
    } while (0)

template <typename T>
int ensure(T **p, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 0;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *cap = 0;
    CU(cudaMalloc(p, bytes));
    *cap = bytes;
    return 0;
}

uint32_t roundUp(uint32_t x, uint32_t to) { return (x + to - 1) / to * to; }

int initDevice() {
    if (g.init) return cudaSetDevice(g.dev) == cudaSuccess ? 0 : fail(PEARL_E_CUDA, "cudaSetDevice(%d)", g.dev);
    int dev = 0;
    if (const char *e = getenv("PEARL_SOAT_DEVICE")) {
        char *end = nullptr;
        const long v = strtol(e, &end, 10);
        if (!*e || *end || v < 0) return fail(PEARL_E_ARG, "PEARL_SOAT_DEVICE=%s is not a device index", e);
        dev = (int)v;
    }
    CU(cudaSetDevice(dev));
    CU(cudaGetDeviceProperties(&g.prop, dev));
    if (g.prop.major < 8)
        return fail(PEARL_E_ARCH, "%s is sm_%d%d: the SOAT ptx kernels need sm_80+", g.prop.name, g.prop.major,
                    g.prop.minor);
    CU(cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking));
    CU(cudaEventCreate(&g.evA));
    CU(cudaEventCreate(&g.evB));
    CU(cudaMalloc(&g.dKey, 32));
    CU(cudaMalloc(&g.dTarget, 32));
    CU(cudaMalloc(&g.dHitCount, 4));
    CU(cudaMalloc(&g.dFp, 4));
    g.dev = dev;
    g.init = true;
    return 0;
}

// Inverse of the transcript index of noisy_gemm.cuh (as SOAT openWin, algo.cu:1095-1104); n = the kernel's n.
void decode(uint32_t flat, uint32_t n, uint32_t *tRow, uint32_t *tCol) {
    const uint32_t per = kR / kSide, blocksPerRow = n / kR;
    const uint32_t wi = flat % per, hi = (flat / per) % per, block = flat / (per * per);
    *tRow = (block / blocksPerRow) * kR + hi * kSide;
    *tCol = (block % blocksPerRow) * kR + wi * kSide;
}

// GEMM + transcripts (config cfg) and powScan of all tiles of rows x n under g.dKey / g.dTarget; the first
// maxHits hits (in the device's order) come back in index / digest, *total = all hits.
int scan(int cfg, const int8_t *dA, uint32_t rows, const int8_t *dB, uint32_t n, uint32_t k, uint32_t maxHits,
         uint32_t *total, std::vector<uint32_t> *index, std::vector<uint32_t> *digest) {
    const TileConfig &tc = kTileConfigs[cfg];
    const uint32_t tiles = (rows / kSide) * (n / kSide);
    TRY(ensure(&g.dHitIndex, &g.hiCap, (size_t)std::max(maxHits, 1u) * 4));
    TRY(ensure(&g.dHitDigest, &g.hdCap, (size_t)std::max(maxHits, 1u) * 32));
    CU(cudaMemsetAsync(g.dHitCount, 0, 4, g.s));
    tc.launch(dim3(n / tc.blockN, rows / tc.blockM), tc.threads, g.s, dA, dB, g.dT, (int)rows, (int)n, (int)k,
              (int)kR);
    CU(cudaGetLastError());
    om::pearl::powScan<<<(tiles + 255) / 256, 256, 0, g.s>>>(g.dT, tiles, g.dKey, g.dTarget, g.dHitIndex,
                                                             g.dHitDigest, g.dHitCount, maxHits);
    CU(cudaGetLastError());
    CU(cudaMemcpyAsync(total, g.dHitCount, 4, cudaMemcpyDeviceToHost, g.s));
    CU(cudaStreamSynchronize(g.s));
    const uint32_t got = std::min(*total, maxHits);
    index->assign(got, 0);
    digest->assign((size_t)got * 8, 0);
    if (got) {
        CU(cudaMemcpyAsync(index->data(), g.dHitIndex, (size_t)got * 4, cudaMemcpyDeviceToHost, g.s));
        CU(cudaMemcpyAsync(digest->data(), g.dHitDigest, (size_t)got * 32, cudaMemcpyDeviceToHost, g.s));
        CU(cudaStreamSynchronize(g.s));
    }
    return 0;
}

// As PearlPow::tuneShape: time every eligible config on these operands (best of 3 after a warmup), accept a number
// only if the transcript fingerprint agrees with the first config that ran.
int tune(uint32_t rows, uint32_t n, uint32_t k, const int8_t *dA, const int8_t *dB, int *best) {
    const uint64_t tiles = (uint64_t)(rows / kSide) * (n / kSide);
    const char *filter = getenv("SOAT_PEARL_TILE");
    double bestRate = 0;
    *best = -1;
    uint32_t refFp = 0;
    bool haveRef = false;
    std::string lastErr = "none";
    for (int i = 0; i < kConfigs; i++) {
        const TileConfig &tc = kTileConfigs[i];
        if (rows % tc.blockM || n % tc.blockN || (k / 32) % (tc.blockK / 32)) continue;
        if (filter && strncmp(tc.name, filter, strlen(filter))) continue;
        CU(cudaMemsetAsync(g.dT, 0, tiles * 16 * 4, g.s));
        float ms = 0.0f;
        bool ok = true;
        for (int rep = 0; rep < 4; rep++) {
            cudaEventRecord(g.evA, g.s);
            tc.launch(dim3(n / tc.blockN, rows / tc.blockM), tc.threads, g.s, dA, dB, g.dT, (int)rows, (int)n,
                      (int)k, (int)kR);
            const cudaError_t le = cudaGetLastError();
            cudaEventRecord(g.evB, g.s);
            const cudaError_t se = cudaStreamSynchronize(g.s);
            if (le != cudaSuccess || se != cudaSuccess) {
                lastErr = std::string(tc.name) + ": " + cudaGetErrorString(le != cudaSuccess ? le : se);
                ok = false;
                break;
            }
            float d = 0.0f;
            cudaEventElapsedTime(&d, g.evA, g.evB);
            if (rep && (ms == 0.0f || d < ms)) ms = d;
        }
        if (!ok || ms <= 0.0f) continue;
        uint32_t fp = 0;
        CU(cudaMemsetAsync(g.dFp, 0, 4, g.s));
        om::pearl::transcriptFingerprint<<<256, 256, 0, g.s>>>(g.dT, tiles * 16, g.dFp);
        CU(cudaMemcpyAsync(&fp, g.dFp, 4, cudaMemcpyDeviceToHost, g.s));
        CU(cudaStreamSynchronize(g.s));
        if (fp == 0) continue;   // wrote nothing
        if (!haveRef) {
            refFp = fp;
            haveRef = true;
        } else if (fp != refFp) {
            fprintf(stderr, "[pearl-soat] %s disagrees with the other configs at %ux%u k=%u - not selected\n",
                    tc.name, rows, n, k);
            continue;
        }
        const double rate = (double)tiles / (ms * 1e-3) / 1e6;
        if (rate > bestRate) {
            bestRate = rate;
            *best = i;
        }
    }
    if (*best < 0)
        return fail(PEARL_E_SHAPE, "no SOAT tile config runs %ux%u k=%u on %s (last error: %s)", rows, n, k,
                    g.prop.name, lastErr.c_str());
    fprintf(stderr, "[pearl-soat] %ux%u k=%u on %s: %s, %.1f M tiles/s (GEMM only)\n", rows, n, k, g.prop.name,
            kTileConfigs[*best].name, bestRate);
    return 0;
}

// As PearlPow::selfCheck, but stronger: all 256 tiles of a 256x256 product at this k with operands over the full
// [-127, 127] against the host tileTranscript, and every jackpot (bound = 2^256-1: all tiles hit) with its decoded
// index against the host powDigest of that tile.
int selfCheck(int cfg, uint32_t k) {
    constexpr uint32_t m = 256, n = 256, tiles = (m / kSide) * (n / kSide);
    std::vector<int8_t> A((size_t)m * k), Bt((size_t)n * k), Bk((size_t)k * n);
    uint64_t x = 0x5e1f0123456789ull;
    auto next = [&x]() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return (int8_t)((int)((x >> 33) % 255) - 127);
    };
    for (auto &v : A) v = next();
    for (auto &v : Bt) v = next();
    for (uint32_t q = 0; q < k; q++) A[q] = Bt[q] = 127;   // the extremes reach the kernel: row 0 x col 0 at +127
    for (uint32_t q = 0; q < k; q++) A[k + q] = -127;      // and row 1 at -127
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t q = 0; q < k; q++) Bk[(size_t)q * n + j] = Bt[(size_t)j * k + q];
    uint8_t key[32], target[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(7 * i + 1);
    memset(target, 0xff, sizeof(target));

    int8_t *dA = nullptr, *dB = nullptr;
    CU(cudaMalloc(&dA, A.size()));
    if (cudaMalloc(&dB, Bt.size()) != cudaSuccess) {
        cudaFree(dA);
        return fail(PEARL_E_CUDA, "self-check: cudaMalloc");
    }
    int rc = 0;
    uint32_t total = 0;
    std::vector<uint32_t> index, digest, got((size_t)tiles * 16);
    do {
        if ((rc = ensure(&g.dT, &g.tCap, (size_t)tiles * 16 * 4))) break;
        if (cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dB, Bt.data(), Bt.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g.dKey, key, 32, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g.dTarget, target, 32, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = fail(PEARL_E_CUDA, "self-check: upload");
            break;
        }
        if ((rc = scan(cfg, dA, m, dB, n, k, tiles, &total, &index, &digest))) break;
        if (cudaMemcpy(got.data(), g.dT, got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            rc = fail(PEARL_E_CUDA, "self-check: download");
    } while (0);
    cudaFree(dA);
    cudaFree(dB);
    if (rc) return rc;

    const char *name = kTileConfigs[cfg].name;
    std::vector<uint32_t> want((size_t)tiles * 16);
    for (uint32_t flat = 0; flat < tiles; flat++) {
        uint32_t tRow, tCol;
        decode(flat, n, &tRow, &tCol);
        om::pearl::tileTranscript(A.data(), Bk.data(), k, n, (int)kR, tRow, tCol, (int)kSide, (int)kSide,
                                  &want[(size_t)flat * 16]);
        if (memcmp(&want[(size_t)flat * 16], &got[(size_t)flat * 16], 64))
            return fail(PEARL_E_SELFCHECK, "self-check FAILED (%s, k=%u, %s): transcript of tile (%u,%u) differs",
                        name, k, g.prop.name, tRow, tCol);
    }
    if (total != tiles || index.size() != tiles)
        return fail(PEARL_E_SELFCHECK, "self-check FAILED (%s, k=%u): %u hits of %u tiles at bound 2^256-1", name, k,
                    total, tiles);
    std::vector<bool> seen(tiles, false);
    for (size_t h = 0; h < index.size(); h++) {
        const uint32_t flat = index[h];
        if (flat >= tiles || seen[flat])
            return fail(PEARL_E_SELFCHECK, "self-check FAILED (%s, k=%u): bad or repeated hit index %u", name, k,
                        flat);
        seen[flat] = true;
        uint8_t d[32];
        om::pearl::powDigest(&want[(size_t)flat * 16], key, d);
        if (memcmp(d, &digest[h * 8], 32))   // device words are little-endian, as the digest bytes
            return fail(PEARL_E_SELFCHECK, "self-check FAILED (%s, k=%u): jackpot of tile %u differs", name, k,
                        flat);
    }
    fprintf(stderr, "[pearl-soat] self-check ok: %s, k=%u, 256 tiles of 256x256, transcripts and jackpots\n", name,
            k);
    return 0;
}

int upload(const int8_t *host, size_t bytes, size_t capBytes, int8_t **dBuf, size_t *cap) {
    TRY(ensure(dBuf, cap, capBytes));
    CU(cudaMemcpyAsync(*dBuf, host, bytes, cudaMemcpyHostToDevice, g.s));
    return 0;
}

int search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
           const uint8_t rows_pattern[6], const uint8_t cols_pattern[6], const uint8_t seed_a[32],
           const uint8_t bound_le[32], uint32_t lo, uint32_t hi, cand_t *out, uint32_t cap, uint32_t *count,
           uint64_t *macs, uint64_t a_id, uint64_t b_id) {
    if (!a || !bt || !rows_pattern || !cols_pattern || !seed_a || !bound_le || !count || !macs || (cap && !out))
        return fail(PEARL_E_ARG, "null pointer argument");
    if (r != kR || k < kR || k % kR || k > (1u << 16))
        return fail(PEARL_E_K_R, "k=%u r=%u: the SOAT kernel does r=128 and k a multiple of 128 (SOAT: 2048)", k, r);
    if (memcmp(rows_pattern, kPattern16, 6) || memcmp(cols_pattern, kPattern16, 6))
        return fail(PEARL_E_PATTERN,
                    "patterns %02x%02x%02x%02x%02x%02x / %02x%02x%02x%02x%02x%02x: the SOAT kernel does only the "
                    "contiguous 16x16 tile (000f00000000 / 000f00000000)",
                    rows_pattern[0], rows_pattern[1], rows_pattern[2], rows_pattern[3], rows_pattern[4],
                    rows_pattern[5], cols_pattern[0], cols_pattern[1], cols_pattern[2], cols_pattern[3],
                    cols_pattern[4], cols_pattern[5]);
    if (!m || m % kSide || !n || n % kSide)
        return fail(PEARL_E_SHAPE, "m=%u n=%u: need multiples of 16", m, n);
    if (lo >= hi || hi > m / kSide) return fail(PEARL_E_ARG, "row tiles [%u, %u) of %u", lo, hi, m / kSide);
    const uint32_t rows = (hi - lo) * kSide, rowsPad = roundUp(rows, kPad), nPad = roundUp(n, kPad);
    const uint64_t tilesPad = (uint64_t)(rowsPad / kSide) * (nPad / kSide);
    if (tilesPad * 16 > 0x7fffffffull)
        return fail(PEARL_E_SHAPE, "%llu tiles in one call: over the kernel's int index", (unsigned long long)tilesPad);
    const bool padded = rowsPad != rows || nPad != n;

    TRY(initDevice());
    if (g.refused) return fail(PEARL_E_SELFCHECK, "this card failed the self-check earlier: refusing to mine");

    // A: rows [lo*16, lo*16 + rowsPad) must lie in the buffer; past the operand they are junk, filtered below.
    const int8_t *dA;
    if (a_id && a_id == g.aId && a == g.aPtr && m == g.aM && k == g.aK) {
        dA = g.dA + (size_t)lo * kSide * k;
    } else if (a_id) {
        g.aId = 0;
        TRY(upload(a, (size_t)m * k, (size_t)(m + kPad) * k, &g.dA, &g.aCap));
        g.aId = a_id, g.aPtr = a, g.aM = m, g.aK = k;
        dA = g.dA + (size_t)lo * kSide * k;
    } else {
        g.aId = 0;
        TRY(upload(a + (size_t)lo * kSide * k, (size_t)rows * k, (size_t)rowsPad * k, &g.dA, &g.aCap));
        dA = g.dA;
    }
    if (!(b_id && b_id == g.bId && bt == g.bPtr && n == g.bN && k == g.bK)) {
        g.bId = 0;
        TRY(upload(bt, (size_t)n * k, (size_t)nPad * k, &g.dB, &g.bCap));
        g.bId = b_id, g.bPtr = bt, g.bN = n, g.bK = k;
    }
    TRY(ensure(&g.dT, &g.tCap, (size_t)tilesPad * 16 * 4));
    CU(cudaStreamSynchronize(g.s));

    int cfg;
    const auto shape = std::make_tuple(rowsPad, nPad, k);
    auto it = g.tuned.find(shape);
    if (it != g.tuned.end()) {
        cfg = it->second;
    } else {
        TRY(tune(rowsPad, nPad, k, dA, g.dB, &cfg));
        g.tuned[shape] = cfg;
    }
    if (!g.checked.count({cfg, k})) {
        const int rc = selfCheck(cfg, k);
        if (rc == PEARL_E_SELFCHECK) g.refused = true;
        if (rc) return rc;
        g.checked.insert({cfg, k});
    }

    CU(cudaMemcpyAsync(g.dKey, seed_a, 32, cudaMemcpyHostToDevice, g.s));
    CU(cudaMemcpyAsync(g.dTarget, bound_le, 32, cudaMemcpyHostToDevice, g.s));   // LE bytes = LE u32 words (x86)
    // Padded: read back every hit and drop the junk tiles, so the count is exact. Otherwise: min(cap, tiles).
    const uint32_t maxHits = padded ? (uint32_t)tilesPad : (uint32_t)std::min<uint64_t>(cap, tilesPad);
    uint32_t total = 0;
    std::vector<uint32_t> index, digest;
    TRY(scan(cfg, dA, rowsPad, g.dB, nPad, k, maxHits, &total, &index, &digest));

    std::vector<cand_t> cands;
    cands.reserve(index.size());
    for (size_t h = 0; h < index.size(); h++) {
        uint32_t tRow, tCol;
        decode(index[h], nPad, &tRow, &tCol);
        if (tRow >= rows || tCol >= n) continue;
        cand_t c;
        c.row_tile = lo + tRow / kSide;
        c.col_tile = tCol / kSide;
        memcpy(c.jackpot, &digest[h * 8], 32);
        cands.push_back(c);
    }
    if (padded) total = (uint32_t)cands.size();
    std::sort(cands.begin(), cands.end(), [](const cand_t &x, const cand_t &y) {
        return x.row_tile != y.row_tile ? x.row_tile < y.row_tile : x.col_tile < y.col_tile;
    });
    const size_t written = std::min<size_t>(cands.size(), cap);
    if (written) memcpy(out, cands.data(), written * sizeof(cand_t));
    *count = total;
    *macs = (uint64_t)(hi - lo) * (n / kSide) * kSide * kSide * (k - k % kR);
    return 0;
}

}  // namespace

extern "C" {

int pearl_search(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                 const uint8_t rows_pattern[6], const uint8_t cols_pattern[6], const uint8_t seed_a[32],
                 const uint8_t bound_le[32], uint32_t row_tile_lo, uint32_t row_tile_hi, cand_t *out, uint32_t cap,
                 uint32_t *count, uint64_t *macs) {
    std::lock_guard<std::mutex> lock(g_mu);
    return search(a, m, bt, n, k, r, rows_pattern, cols_pattern, seed_a, bound_le, row_tile_lo, row_tile_hi, out, cap,
                  count, macs, 0, 0);
}

int pearl_search_ids(const int8_t *a, uint32_t m, const int8_t *bt, uint32_t n, uint32_t k, uint32_t r,
                     const uint8_t rows_pattern[6], const uint8_t cols_pattern[6], const uint8_t seed_a[32],
                     const uint8_t bound_le[32], uint32_t row_tile_lo, uint32_t row_tile_hi, cand_t *out,
                     uint32_t cap, uint32_t *count, uint64_t *macs, uint64_t a_id, uint64_t b_id) {
    std::lock_guard<std::mutex> lock(g_mu);
    return search(a, m, bt, n, k, r, rows_pattern, cols_pattern, seed_a, bound_le, row_tile_lo, row_tile_hi, out, cap,
                  count, macs, a_id, b_id);
}

int pearl_device_info(uint32_t *sm_count, char *name, uint32_t name_len) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!sm_count || !name || !name_len) return fail(PEARL_E_ARG, "pearl_device_info needs sm_count, name, name_len");
    TRY(initDevice());
    *sm_count = (uint32_t)g.prop.multiProcessorCount;
    snprintf(name, name_len, "%s", g.prop.name);
    return 0;
}

const char *pearl_last_error(void) { return g_err.c_str(); }

}  // extern "C"
