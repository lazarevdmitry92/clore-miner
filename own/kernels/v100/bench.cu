// pearl_bench — the V100 kernel without a pool (TZ §3, §7, §8):
//
//   pearl_bench exact  [--dev 0] [--tiles 1048576]   test 1 on the card: fast (HMMA) vs exact (dp4a) transcripts on
//                                                     >= --tiles tiles per case, both vs the CPU int64 reference on
//                                                     a slice; noise k = 2048/4096/8192, uniform +-127, only +-127
//   pearl_bench peak   [--dev 0] [--secs 10] [--dp4a-warps 2]
//                                                     stage 1: MAC/(SM*clock) and W of bare HMMA.884, bare dp4a and
//                                                     both in one block — decides whether dp4a beside HMMA pays
//   pearl_bench search [--devs all|0,1] [--secs 900] [--warmup 300] [--m 65536] [--n 2048] [--k 2048]
//                                                     test 3: the mining kernel on every card, synthetic noise
//                                                     operands; per card MAC/s, SM clock, W, k = MAC/(80*clock),
//                                                     J/TH; JSON line per card + the median
//
// Watts and clocks come from NVML (dlopen, no link stub). Clocks and power limits are never touched (rented cards).
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include "pearl_api.h"
#include "pearl_host.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define PK(x) do { int rc_ = (x); if (rc_) { fprintf(stderr, "%s -> %d: %s\n", #x, rc_, pearl_last_error()); \
    exit(1); } } while (0)

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- args
static const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; ++i)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static double argf(int argc, char **argv, const char *name, double def) {
    const char *v = arg(argc, argv, name, nullptr);
    return v ? atof(v) : def;
}

// ---------------------------------------------------------------- NVML
struct Nvml {
    void *h = nullptr;
    int (*init)() = nullptr;
    int (*byIndex)(unsigned, void **) = nullptr;
    int (*power)(void *, unsigned *) = nullptr;
    int (*clock)(void *, int, unsigned *) = nullptr;
    bool ok = false;
    Nvml() {
        h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) { fprintf(stderr, "no NVML: %s (watts and clocks unknown)\n", dlerror()); return; }
        init = (int (*)())dlsym(h, "nvmlInit_v2");
        byIndex = (int (*)(unsigned, void **))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        power = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
        clock = (int (*)(void *, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
        ok = init && byIndex && power && clock && init() == 0;
    }
};
static Nvml g_nvml;

struct Sample { double watts = 0, mhz = 0; int n = 0; };

// NVML index = CUDA index because main() sets CUDA_DEVICE_ORDER=PCI_BUS_ID before any CUDA call
static void sample(int dev, std::atomic<bool> &measuring, std::atomic<bool> &run, Sample &s) {
    void *h = nullptr;
    if (!g_nvml.ok || g_nvml.byIndex(dev, &h)) return;
    while (run.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (!measuring.load()) continue;
        unsigned mw = 0, mhz = 0;
        if (g_nvml.power(h, &mw) || g_nvml.clock(h, 1 /* SM */, &mhz)) continue;
        s.watts += mw / 1000.0; s.mhz += mhz; s.n++;
    }
    if (s.n) { s.watts /= s.n; s.mhz /= s.n; }
}

// fast deterministic Pearl-like noise for big operands (pearl::synth is mt19937 and slow at 10^8)
static void fill_noise(std::vector<int8_t> &v, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
    for (auto &b : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        b = (int8_t)(((int)(x & 63) - 32) - ((int)((x >> 8) & 63) - 32));
    }
}

// ---------------------------------------------------------------- exact: acceptance test 1 on the card
static int set_ops(const std::vector<int8_t> &a, uint32_t m, const std::vector<int8_t> &b, uint32_t n, uint32_t k) {
    int rc = pearl_set_a(a.data(), m, k);
    return rc ? rc : pearl_set_b(b.data(), n, k);
}

static void transcripts(const char *path, uint32_t k, uint32_t lo, uint32_t hi, uint32_t ct, std::vector<uint32_t> &t) {
    if (path) setenv("PEARL_PATH", path, 1); else unsetenv("PEARL_PATH");
    t.resize((size_t)(hi - lo) * ct * 16);
    PK(pearl_transcripts(k, lo, hi, t.data()));
}

static int mode_exact(int argc, char **argv) {
    const int dev = (int)argf(argc, argv, "--dev", 0);
    const double want = argf(argc, argv, "--tiles", 1 << 20);
    PK(pearl_init(dev));
    printf("device %d, kernel %s\n", dev, pearl_kernel_name());
    struct Case { const char *name; uint32_t k; int dist; };
    const Case cases[] = {{"noise", 2048, 0}, {"noise", 4096, 0}, {"noise", 8192, 0},
                          {"uniform+-127", 2048, 1}, {"only+-127", 4096, 2}};
    int failures = 0;
    for (const Case &cs : cases) {
        // one case = repeats of a 4096 x 4096 problem (512 x 256 tiles) until --tiles are covered
        const uint32_t m = 4096, n = 4096, k = cs.k;
        pearl::Dims d; d.m = m; d.n = n; d.k = k;
        const uint32_t rt = d.row_tiles(), ct = d.col_tiles();
        uint64_t tiles = 0, fast_bad = 0, exact_bad = 0, ref_bad = 0, ref_tiles = 0;
        int fast_path = 0;
        for (uint32_t rep = 0; tiles < want; ++rep) {
            std::vector<int8_t> a((size_t)m * k), b((size_t)n * k);
            if (cs.dist == 0) { fill_noise(a, 1000 + rep); fill_noise(b, 2000 + rep); }
            else { a = pearl::synth(a.size(), cs.dist, 1000 + rep); b = pearl::synth(b.size(), cs.dist, 2000 + rep); }
            PK(set_ops(a, m, b, n, k));
            unsetenv("PEARL_PATH");
            fast_path = pearl_fast_path();
            std::vector<uint32_t> tf, te;
            transcripts("fast", k, 0, rt, ct, tf);       // forced: on non-provable cases this measures the margin
            transcripts("exact", k, 0, rt, ct, te);
            // CPU reference on 4 row tiles (all columns) per repeat
            std::vector<uint32_t> tr((size_t)4 * ct * 16);
            const uint32_t r0 = (rep * 4) % rt;
            pearl::ref_transcripts(a.data(), b.data(), d, r0, r0 + 4, tr.data());
            for (size_t i = 0; i < (size_t)rt * ct; ++i) {
                bool f = memcmp(&tf[i * 16], &te[i * 16], 64) != 0;
                fast_bad += f;
            }
            for (size_t i = 0; i < (size_t)4 * ct; ++i) {
                const size_t g = (size_t)r0 * ct + i;
                exact_bad += memcmp(&te[g * 16], &tr[i * 16], 64) != 0;
                if (fast_path) ref_bad += memcmp(&tf[g * 16], &tr[i * 16], 64) != 0;
            }
            tiles += (uint64_t)rt * ct;
            ref_tiles += 4 * ct;
        }
        const bool ok = exact_bad == 0 && (!fast_path || (fast_bad == 0 && ref_bad == 0));
        failures += !ok;
        printf("%-13s k=%-5u tiles %-8llu path %-5s | fast vs exact differ %llu | exact vs CPU differ %llu of %llu"
               " | %s\n", cs.name, cs.k, (unsigned long long)tiles, fast_path ? "fast" : "exact",
               (unsigned long long)fast_bad, (unsigned long long)exact_bad, (unsigned long long)ref_tiles,
               ok ? "PASS" : "FAIL");
    }
    printf("%s\n", failures ? "TEST 1: FAIL" : "TEST 1: PASS (on the path the miner takes; forced-fast "
                                               "mismatches on non-provable cases are informational)");
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------- peak: stage 1 micro-bench
__device__ __forceinline__ void mma884b(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// warps < hmma_warps: 16 independent HMMA chains (1024 MAC each per instruction); others: 32 dp4a chains
__global__ void peak_kernel(int hmma_warps, int iters, uint32_t seed, float *sink) {
    const int warp = threadIdx.x / 32;
    uint32_t a0 = 0x3C003C00u ^ (seed & 1), a1 = a0, b0 = a0, b1 = a0;
    if (warp < hmma_warps) {
        float acc[16][8];
        for (int i = 0; i < 16; ++i)
            for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int i = 0; i < 16; ++i) mma884b(acc[i], a0, a1, b0, b1);
        float s = 0;
        for (int i = 0; i < 16; ++i)
            for (int e = 0; e < 8; ++e) s += acc[i][e];
        if (s == 1.2345f) sink[0] = s;
    } else {
        int acc[32];
        for (int i = 0; i < 32; ++i) acc[i] = i;
        const int x = (int)(seed | 0x01010101u), y = (int)(seed ^ 0x02020202u);
        // 16 dp4a per HMMA-iteration slot-equivalent: iters * 16 * 1024 / 4 / 32 per thread keeps the same MAC goal
        for (int it = 0; it < iters * 8; ++it)
#pragma unroll
            for (int i = 0; i < 32; ++i) acc[i] = __dp4a(x, y + i, acc[i]);
        int s = 0;
        for (int i = 0; i < 32; ++i) s ^= acc[i];
        if (s == 12345) sink[1] = (float)s;
    }
}

static int mode_peak(int argc, char **argv) {
    const int dev = (int)argf(argc, argv, "--dev", 0);
    const double secs = argf(argc, argv, "--secs", 10);
    const int dpw = (int)argf(argc, argv, "--dp4a-warps", 2);
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    float *sink;
    CK(cudaMalloc(&sink, 8));
    struct Cfg { const char *name; int warps, hmma; };
    const Cfg cfgs[] = {{"hmma", 8, 8}, {"dp4a", 8, 0}, {"mixed", 8 - dpw, 8 - dpw}};
    const int iters = 4096;
    for (int ci = 0; ci < 3; ++ci) {
        Cfg cf = cfgs[ci];
        const int warps = ci == 2 ? 8 : cf.warps;
        const int blocks = p.multiProcessorCount * 2;
        std::atomic<bool> run(true), measuring(true);
        Sample s;
        std::thread smp(sample, dev, std::ref(measuring), std::ref(run), std::ref(s));
        uint64_t macs = 0;
        auto t0 = Clock::now();
        double el = 0;
        while (el < secs) {
            peak_kernel<<<blocks, warps * 32>>>(cf.hmma, iters, 1, sink);
            CK(cudaDeviceSynchronize());
            const uint64_t hm = (uint64_t)cf.hmma * iters * 16 * 1024;
            const uint64_t dp = (uint64_t)(warps - cf.hmma) * 32 * iters * 8 * 32 * 4;
            macs += (hm + dp) * blocks;
            el = std::chrono::duration<double>(Clock::now() - t0).count();
        }
        run = false;
        smp.join();
        const double rate = macs / el, mhz = s.n ? s.mhz : p.clockRate / 1000.0;
        printf("%-6s %.1f TMAC/s  SM clock %.0f MHz  %.0f W  k = %.0f MAC/(SM*clock)  %.2f J/TMAC\n", cf.name,
               rate / 1e12, mhz, s.watts, rate / (p.multiProcessorCount * mhz * 1e6),
               s.watts ? s.watts / (rate / 1e12) : 0.0);
    }
    return 0;
}

// ---------------------------------------------------------------- search: acceptance test 3
struct CardResult { int dev, sm; double macs_s, mhz, watts; std::string name, kernel; int fast; };

static void card(int dev, uint32_t m, uint32_t n, uint32_t k, double warmup, double secs, CardResult &out) {
    PK(pearl_init(dev));
    std::vector<int8_t> a((size_t)m * k), b((size_t)n * k);
    fill_noise(a, 7 + dev);
    fill_noise(b, 99 + dev);
    PK(set_ops(a, m, b, n, k));
    out.fast = pearl_fast_path();
    out.kernel = pearl_kernel_name();
    uint8_t seed[32] = {1}, bound[32] = {0};
    cand_t cands[16];
    uint32_t count;
    uint64_t macs;
    const uint32_t tiles = m / 8;
    std::atomic<bool> run(true), measuring(false);
    Sample s;
    std::thread smp(sample, dev, std::ref(measuring), std::ref(run), std::ref(s));
    auto t0 = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - t0).count() < warmup)
        PK(pearl_search_resident(k, 128, PEARL_ROWS_PATTERN, PEARL_COLS_PATTERN, seed, bound, 0, tiles, cands, 16,
                                 &count, &macs));
    measuring = true;
    uint64_t total = 0;
    auto t1 = Clock::now();
    double el = 0;
    while (el < secs) {
        PK(pearl_search_resident(k, 128, PEARL_ROWS_PATTERN, PEARL_COLS_PATTERN, seed, bound, 0, tiles, cands, 16,
                                 &count, &macs));
        total += macs;
        el = std::chrono::duration<double>(Clock::now() - t1).count();
    }
    run = false;
    smp.join();
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    out.dev = dev;
    out.sm = p.multiProcessorCount;
    out.name = p.name;
    out.macs_s = total / el;
    out.mhz = s.n ? s.mhz : 0;
    out.watts = s.watts;
}

static int mode_search(int argc, char **argv) {
    const uint32_t m = (uint32_t)argf(argc, argv, "--m", 65536), n = (uint32_t)argf(argc, argv, "--n", 2048);
    const uint32_t k = (uint32_t)argf(argc, argv, "--k", 2048);
    const double secs = argf(argc, argv, "--secs", 900), warmup = argf(argc, argv, "--warmup", 300);
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs;
    const char *ds = arg(argc, argv, "--devs", "all");
    if (!strcmp(ds, "all")) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    else for (const char *p = ds; *p; ) { devs.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    printf("search: %zu card(s), m=%u n=%u k=%u, warm-up %.0f s, measure %.0f s (B'^T %.1f MB)\n", devs.size(), m, n,
           k, warmup, secs, (double)n * k / 1e6);
    std::vector<CardResult> res(devs.size());
    std::vector<std::thread> th;
    for (size_t i = 0; i < devs.size(); ++i) th.emplace_back(card, devs[i], m, n, k, warmup, secs, std::ref(res[i]));
    for (auto &t : th) t.join();
    std::vector<double> ks, jt;
    for (auto &r : res) {
        const double kk = r.mhz ? r.macs_s / (r.sm * r.mhz * 1e6) : 0, j = r.watts ? r.watts / (r.macs_s / 1e12) : 0;
        ks.push_back(kk);
        jt.push_back(j);
        printf("{\"dev\": %d, \"name\": \"%s\", \"kernel\": \"%s\", \"path\": \"%s\", \"th_s\": %.2f, \"mhz\": %.0f, "
               "\"watts\": %.1f, \"k\": %.1f, \"j_per_th\": %.2f}\n", r.dev, r.name.c_str(), r.kernel.c_str(),
               r.fast ? "fast" : "exact", r.macs_s / 1e12, r.mhz, r.watts, kk, j);
    }
    std::sort(ks.begin(), ks.end());
    std::sort(jt.begin(), jt.end());
    printf("median k = %.1f (acceptance >= 430, goal >= 500; SRBMiner 334-344), median J/TH = %.2f (<= 6.0, goal <= 5.0)\n",
           ks[ks.size() / 2], jt[jt.size() / 2]);
    return 0;
}

int main(int argc, char **argv) {
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
    const char *mode = argc > 1 ? argv[1] : "";
    if (!strcmp(mode, "exact")) return mode_exact(argc, argv);
    if (!strcmp(mode, "peak")) return mode_peak(argc, argv);
    if (!strcmp(mode, "search")) return mode_search(argc, argv);
    fprintf(stderr, "usage: pearl_bench exact|peak|search [options] (see the header of bench.cu)\n");
    return 2;
}
