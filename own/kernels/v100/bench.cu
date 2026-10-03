// pearl_bench — the V100 kernel without a pool (TZ §3, §7, §8):
//
//   pearl_bench exact  [--dev 0] [--tiles 1048576]   test 1 on the card: fast (HMMA) vs exact (dp4a) transcripts on
//                                                     >= --tiles tiles per case, both vs the CPU int64 reference on
//                                                     a slice; noise k = 2048/4096/8192, uniform +-127, only +-127
//   pearl_bench peak   [--dev 0] [--secs 10] [--dp4a-warps 2]
//                                                     stage 1: MAC/(SM*clock) and W of bare HMMA.884, bare dp4a and
//                                                     both in one block — decides whether dp4a beside HMMA pays
//   pearl_bench ablate [--dev 0] [--secs 60] [--m 65536] [--n 2048] [--k 2048] [--variants a,b,...]
//                                                     TZ v2 §3.1: one card, each variant --secs (after 15 s to settle):
//                                                     full, no-epilogue, no-convert (-f16), smem-only, regs-only;
//                                                     --variants all = every variant (mining ones under real power)
//   pearl_bench search [--devs all|0,1] [--secs 900] [--warmup 300] [--m 65536] [--n 2048] [--k 2048]
//                      [--variant <mining variant>] [--live-operands]
//                                                     (default: startup + online autotune, as in the miner;
//                                                     --live-operands: a new pass per iteration as in the miner)
//                                                     test 3: the mining kernel on every card, synthetic noise
//                                                     operands; per card MAC/s, SM clock, W, k = MAC/(80*clock),
//                                                     J/TH; JSON line per card + the median
//
// Watts and clocks come from NVML (dlopen, no link stub). Clocks and power limits are never touched (rented cards).
#include "pearl_host.h"   // the V100 tile first: bench_main.h searches with PEARL_ROWS/COLS_PATTERN
#include "../common/bench_main.h"

const char *const ABLATE_DEFAULT[5] = {"v100-hmma884-128x128", "ablate-no-epilogue", "v100-hmma884-128x128-f16",
                                       "ablate-smem-only", "ablate-regs-only"};

// ---------------------------------------------------------------- exact: acceptance test 1 on the card
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
            if (cs.dist == 0) { pearl::fill_noise(a, 1000 + rep); pearl::fill_noise(b, 2000 + rep); }
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

