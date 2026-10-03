// The part of pearl_bench shared by every kernel (kernels/v100, kernels/sm80): argument parsing, NVML sampling of
// watts and SM clock while a mode runs, `ablate` (TZ_v100_kernel_v2 §3.1) and `search` (acceptance test 3), main().
// A kernel's bench.cu includes its tile header (PEARL_ROWS/COLS_PATTERN), then this, then defines ABLATE_DEFAULT[],
// mode_exact() and mode_peak().
// Clocks and power limits are never touched (rented cards).
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include "host_core.h"
#include "pearl_api.h"
#include "runtime.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define PK(x) do { int rc_ = (x); if (rc_) { fprintf(stderr, "%s -> %d: %s\n", #x, rc_, pearl_last_error()); \
    exit(1); } } while (0)

using Clock = std::chrono::steady_clock;

static int mode_exact(int argc, char **argv);
static int mode_peak(int argc, char **argv);
extern const char *const ABLATE_DEFAULT[5];

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

// ---------------------------------------------------------------- NVML sampling
struct Sample { double watts = 0, mhz = 0; int n = 0; };

// average watts and SM clock every 200 ms while `measuring`, until `run` drops
static void sample(int dev, std::atomic<bool> &measuring, std::atomic<bool> &run, Sample &s) {
    char bus[32];
    void *h = cudaDeviceGetPCIBusId(bus, sizeof bus, dev) == cudaSuccess ? pearl::nvml().device(bus) : nullptr;
    while (run.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (!measuring.load() || !h) continue;
        const pearl::Telemetry t = pearl::read_telemetry(h);
        if (!t.ok) continue;
        s.watts += t.mw / 1000.0; s.mhz += t.mhz; s.n++;
    }
    if (s.n) { s.watts /= s.n; s.mhz /= s.n; }
}

static int set_ops(const std::vector<int8_t> &a, uint32_t m, const std::vector<int8_t> &b, uint32_t n, uint32_t k) {
    int rc = pearl_set_a(a.data(), m, k);
    return rc ? rc : pearl_set_b(b.data(), n, k);
}

// ---------------------------------------------------------------- ablate: TZ_v100_kernel_v2 §3.1, TZ_sm80 §3
static std::vector<std::string> split(const char *s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char *p = s; ; ++p) {
        if (*p == ',' || !*p) { if (!cur.empty()) out.push_back(cur); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return out;
}

static int mode_ablate(int argc, char **argv) {
    const int dev = (int)argf(argc, argv, "--dev", 0);
    const double secs = argf(argc, argv, "--secs", 60), settle = 15;
    const uint32_t m = (uint32_t)argf(argc, argv, "--m", 65536), n = (uint32_t)argf(argc, argv, "--n", 2048);
    const uint32_t k = (uint32_t)argf(argc, argv, "--k", 2048);
    PK(pearl_init(dev));
    std::vector<std::string> vs;
    const char *want = arg(argc, argv, "--variants", nullptr);
    if (want && !strcmp(want, "all")) {
        for (int i = 0; i < pearl_variant_count(); ++i) vs.push_back(pearl_variant_name(i));
    } else if (want) {
        vs = split(want);
    } else {
        vs.assign(ABLATE_DEFAULT, ABLATE_DEFAULT + sizeof(ABLATE_DEFAULT) / sizeof(ABLATE_DEFAULT[0]));
    }
    std::vector<int8_t> a((size_t)m * k), b((size_t)n * k);
    pearl::fill_noise(a, 7 + dev);
    pearl::fill_noise(b, 99 + dev);
    PK(set_ops(a, m, b, n, k));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    printf("ablate: device %d, m=%u n=%u k=%u, %.0f s per variant after %.0f s to settle (startup choice %s)\n",
           dev, m, n, k, secs, settle, pearl_kernel_name());
    printf("%-30s %9s %7s %7s %7s %8s\n", "variant", "TMAC/s", "MHz", "W", "k", "J/TMAC");
    double base = 0;
    for (const auto &v : vs) {
        std::atomic<bool> run(true), measuring(false);
        Sample s;
        std::thread smp(sample, dev, std::ref(measuring), std::ref(run), std::ref(s));
        uint64_t macs = 0, total = 0;
        auto t0 = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - t0).count() < settle)
            PK(pearl_bench_run(v.c_str(), 0, m / 8, &macs));
        measuring = true;
        auto t1 = Clock::now();
        double el = 0;
        while (el < secs) {
            PK(pearl_bench_run(v.c_str(), 0, m / 8, &macs));
            total += macs;
            el = std::chrono::duration<double>(Clock::now() - t1).count();
        }
        run = false;
        smp.join();
        const double rate = total / el, kk = s.mhz ? rate / (p.multiProcessorCount * s.mhz * 1e6) : 0;
        if (base == 0) base = rate;
        printf("%-30s %9.2f %7.0f %7.1f %7.1f %8.2f   x%.2f\n", v.c_str(), rate / 1e12, s.mhz, s.watts, kk,
               s.watts ? s.watts / (rate / 1e12) : 0.0, rate / base);
        fflush(stdout);
    }
    return 0;
}

// ---------------------------------------------------------------- search: acceptance test 3
struct CardResult {
    int dev, sm;
    double macs_s, mhz, watts, prep_share;
    std::string name, kernel, telemetry;
    int fast;
};

// live = the miner's resident path (TZ_operands_gpu §6.3): pearl_job once, then per iteration a new pass
// (pearl_pass with the next nonce: tree path, seeds, A' built on the card) and the search over all row tiles —
// MAC/s then include the operand preparation, as in the miner. Otherwise fixed synthetic operands.
static void card(int dev, uint32_t m, uint32_t n, uint32_t k, double warmup, double secs, const char *variant,
                 bool live, CardResult &out) {
    PK(pearl_init(dev));
    if (variant) PK(pearl_set_variant(variant));
    uint8_t seed[32] = {1}, bound[32] = {0};
    if (live) {
        uint8_t jk[32], hb[32];
        for (int i = 0; i < 32; ++i) jk[i] = (uint8_t)(i * 13 + dev);
        PK(pearl_job(jk, m, n, k, PEARL_R, PEARL_ROWS_PATTERN, PEARL_COLS_PATTERN, hb, seed));
    } else {
        std::vector<int8_t> a((size_t)m * k), b((size_t)n * k);
        pearl::fill_noise(a, 7 + dev);
        pearl::fill_noise(b, 99 + dev);
        PK(set_ops(a, m, b, n, k));
    }
    cand_t cands[16];
    uint32_t count;
    uint64_t macs, nonce = 0;
    const uint32_t tiles = m / 8;
    double prep = 0;
    auto step = [&]() {
        if (live) {
            int8_t nd[8];
            uint64_t x = nonce++;
            for (int i = 0; i < 8; ++i) { nd[i] = (int8_t)((int)(x % 129) - 64); x /= 129; }
            uint8_t ha[32];
            auto p0 = Clock::now();
            PK(pearl_pass(nd, ha, seed));
            prep += std::chrono::duration<double>(Clock::now() - p0).count();
        }
        PK(pearl_search_resident(k, PEARL_R, PEARL_ROWS_PATTERN, PEARL_COLS_PATTERN, seed, bound, 0, tiles, cands, 16,
                                 &count, &macs));
    };
    std::atomic<bool> run(true), measuring(false);
    Sample s;
    std::thread smp(sample, dev, std::ref(measuring), std::ref(run), std::ref(s));
    auto t0 = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - t0).count() < warmup) step();
    out.fast = pearl_fast_path();
    measuring = true;
    prep = 0;
    uint64_t total = 0;
    auto t1 = Clock::now();
    double el = 0;
    while (el < secs) {
        step();
        total += macs;
        el = std::chrono::duration<double>(Clock::now() - t1).count();
    }
    run = false;
    smp.join();
    char tj[512];
    if (pearl_telemetry(tj, sizeof tj) == 0) out.telemetry = tj;
    out.kernel = pearl_kernel_name();   // the online autotune may have switched during the run
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    out.dev = dev;
    out.sm = p.multiProcessorCount;
    out.name = p.name;
    out.macs_s = total / el;
    out.mhz = s.n ? s.mhz : 0;
    out.watts = s.watts;
    out.prep_share = prep / el;
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
    bool live = false;
    for (int i = 2; i < argc; ++i) live |= !strcmp(argv[i], "--live-operands");
    printf("search: %zu card(s), m=%u n=%u k=%u, warm-up %.0f s, measure %.0f s (B'^T %.1f MB), operands %s\n",
           devs.size(), m, n, k, warmup, secs, (double)n * k / 1e6, live ? "live (pearl_job/pearl_pass)" : "fixed");
    std::vector<CardResult> res(devs.size());
    std::vector<std::thread> th;
    const char *variant = arg(argc, argv, "--variant", nullptr);
    for (size_t i = 0; i < devs.size(); ++i)
        th.emplace_back(card, devs[i], m, n, k, warmup, secs, variant, live, std::ref(res[i]));
    for (auto &t : th) t.join();
    std::vector<double> ks, jt;
    for (auto &r : res) {
        const double kk = r.mhz ? r.macs_s / (r.sm * r.mhz * 1e6) : 0, j = r.watts ? r.watts / (r.macs_s / 1e12) : 0;
        ks.push_back(kk);
        jt.push_back(j);
        printf("{\"dev\": %d, \"name\": \"%s\", \"kernel\": \"%s\", \"path\": \"%s\", \"th_s\": %.2f, \"mhz\": %.0f, "
               "\"watts\": %.1f, \"k\": %.1f, \"j_per_th\": %.2f, \"prep_share\": %.4f, \"telemetry\": %s}\n", r.dev,
               r.name.c_str(), r.kernel.c_str(), r.fast ? "fast" : "exact", r.macs_s / 1e12, r.mhz, r.watts, kk, j,
               r.prep_share, r.telemetry.empty() ? "null" : r.telemetry.c_str());
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
    if (!strcmp(mode, "ablate")) return mode_ablate(argc, argv);
    fprintf(stderr, "usage: pearl_bench exact|peak|ablate|search [options] (see the header of bench.cu)\n");
    return 2;
}
