// pearl_bench — the sm_80+ kernel without a pool (TZ_sm80_kernel.md §3, §6):
//
//   pearl_bench exact  [--dev 0] [--tiles 1048576]   test 1 on the card: every variant against the first one on
//                                                     >= --tiles tiles per case and against the CPU int64 reference
//                                                     on a slice; noise k = 2048/4096/8192, uniform +-127, only +-127
//   pearl_bench peak   [--dev 0] [--secs 10]          bare mma.m16n8k32 s8: MAC/(SM*clock) and W, the ceiling
//   pearl_bench ablate [--dev 0] [--secs 60] [--m 65536] [--n 4096] [--k 2048] [--variants a,b,...|all]
//                                                     full, no-epilogue, smem-only, regs-only (+ 3 stages vs 4)
//   pearl_bench search [--devs all|0,1] [--secs 900] [--warmup 300] [--m 65536] [--n 4096] [--k 2048]
//                      [--variant <name>] [--live-operands]
//                                                     acceptance: MAC/s, clock, W, k, J/TH per card + medians;
//                                                     --live-operands: pearl_job + a new pearl_pass per iteration
//
// Planks (TZ_sm80 §1-2): TH/s >= 0.98 x SRBMiner on the same server; k >= 974 where clocks, not watts, limit.
#include "pearl_host80.h"   // the sm80 tile first: bench_main.h searches with PEARL_ROWS/COLS_PATTERN
#include "../common/bench_main.h"

const char *const ABLATE_DEFAULT[5] = {"sm80-imma-128x256-s3", "ablate-no-epilogue", "ablate-smem-only",
                                       "ablate-regs-only", "sm80-imma-128x256-s4"};

// ---------------------------------------------------------------- exact: acceptance test 1 on the card
static int mode_exact(int argc, char **argv) {
    const int dev = (int)argf(argc, argv, "--dev", 0);
    const double want = argf(argc, argv, "--tiles", 1 << 20);
    PK(pearl_init(dev));
    printf("device %d, kernel %s\n", dev, pearl_kernel_name());
    std::vector<std::string> variants;
    for (int i = 0; i < pearl_variant_count(); ++i)
        if (!strncmp(pearl_variant_name(i), "sm80-", 5) && pearl_set_variant(pearl_variant_name(i)) == 0)
            variants.push_back(pearl_variant_name(i));   // the ones that fit this card's shared memory
    struct Case { const char *name; uint32_t k; int dist; };
    const Case cases[] = {{"noise", 2048, 0}, {"noise", 4096, 0}, {"noise", 8192, 0},
                          {"uniform+-127", 2048, 1}, {"only+-127", 4096, 2}};
    int failures = 0;
    for (const Case &cs : cases) {
        const uint32_t m = 4096, n = 4096, k = cs.k;
        pearl80::Dims d; d.m = m; d.n = n; d.k = k;
        const uint32_t rt = d.row_tiles(), ct = d.col_tiles();
        uint64_t tiles = 0, cross_bad = 0, ref_bad = 0, ref_tiles = 0;
        for (uint32_t rep = 0; tiles < want; ++rep) {
            std::vector<int8_t> a((size_t)m * k), b((size_t)n * k);
            if (cs.dist == 0) { pearl::fill_noise(a, 1000 + rep); pearl::fill_noise(b, 2000 + rep); }
            else { a = pearl::synth(a.size(), cs.dist, 1000 + rep); b = pearl::synth(b.size(), cs.dist, 2000 + rep); }
            PK(set_ops(a, m, b, n, k));
            std::vector<uint32_t> first, t((size_t)rt * ct * 16);
            for (size_t v = 0; v < variants.size(); ++v) {
                PK(pearl_set_variant(variants[v].c_str()));
                PK(pearl_transcripts(k, 0, rt, t.data()));
                if (v == 0) { first = t; continue; }
                for (size_t i = 0; i < (size_t)rt * ct; ++i) cross_bad += memcmp(&t[i * 16], &first[i * 16], 64) != 0;
            }
            std::vector<uint32_t> tr((size_t)8 * ct * 16);       // CPU reference on 8 row tiles per repeat
            const uint32_t r0 = (rep * 8) % rt;
            pearl80::ref_transcripts(a.data(), b.data(), d, r0, r0 + 8, tr.data());
            for (size_t i = 0; i < (size_t)8 * ct; ++i)
                ref_bad += memcmp(&first[((size_t)r0 * ct + i) * 16], &tr[i * 16], 64) != 0;
            tiles += (uint64_t)rt * ct;
            ref_tiles += 8 * ct;
        }
        const bool ok = cross_bad == 0 && ref_bad == 0;
        failures += !ok;
        printf("%-13s k=%-5u tiles %-8llu variants %zu | differ between variants %llu | vs CPU %llu of %llu | %s\n",
               cs.name, cs.k, (unsigned long long)tiles, variants.size(), (unsigned long long)cross_bad,
               (unsigned long long)ref_bad, (unsigned long long)ref_tiles, ok ? "PASS" : "FAIL");
    }
    PK(pearl_set_variant(""));
    printf("%s\n", failures ? "TEST 1: FAIL" : "TEST 1: PASS");
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------- peak: the tensor-core ceiling
__device__ __forceinline__ void mma_s8b(int32_t (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a0), "r"(a1), "r"(b0), "r"(b0));
}

// 16 independent mma chains per warp (4096 MAC each per instruction)
__global__ void peak_kernel(int iters, uint32_t seed, int *sink) {
    int32_t acc[16][4];
    for (int i = 0; i < 16; ++i)
        for (int e = 0; e < 4; ++e) acc[i][e] = 0;
    const uint32_t a0 = 0x01010101u ^ seed, a1 = 0x02020202u, b0 = 0x03030303u ^ (seed << 1);
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int i = 0; i < 16; ++i) mma_s8b(acc[i], a0, a1, b0);
    int s = 0;
    for (int i = 0; i < 16; ++i)
        for (int e = 0; e < 4; ++e) s ^= acc[i][e];
    if (s == 0x12345) sink[0] = s;
}

static int mode_peak(int argc, char **argv) {
    const int dev = (int)argf(argc, argv, "--dev", 0);
    const double secs = argf(argc, argv, "--secs", 10);
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    int *sink;
    CK(cudaMalloc(&sink, 4));
    const int iters = 4096, warps = 8, blocks = p.multiProcessorCount * 2;
    std::atomic<bool> run(true), measuring(true);
    Sample s;
    std::thread smp(sample, dev, std::ref(measuring), std::ref(run), std::ref(s));
    uint64_t macs = 0;
    auto t0 = Clock::now();
    double el = 0;
    while (el < secs) {
        peak_kernel<<<blocks, warps * 32>>>(iters, 1, sink);
        CK(cudaDeviceSynchronize());
        macs += (uint64_t)blocks * warps * iters * 16 * 4096;
        el = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    run = false;
    smp.join();
    const double rate = macs / el, mhz = s.n ? s.mhz : p.clockRate / 1000.0;
    printf("imma %.1f TMAC/s  SM clock %.0f MHz  %.0f W  k = %.0f MAC/(SM*clock)  %.2f J/TMAC  (%s, %d SM)\n",
           rate / 1e12, mhz, s.watts, rate / (p.multiProcessorCount * mhz * 1e6),
           s.watts ? s.watts / (rate / 1e12) : 0.0, p.name, p.multiProcessorCount);
    return 0;
}
