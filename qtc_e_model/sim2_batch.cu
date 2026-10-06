// CUDA batch scorer for miner.qtc_e_model.hybrid_accel binary batches.
// One thread owns one candidate/seed simulation; CPU supplies exact Python
// Random offsets, so the integer state machine matches sim2.py.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct Header {
  char magic[8];
  uint32_t version, candidates, length, seeds, nwarps, reps, fields;
};
static_assert(sizeof(Header) == 36, "batch header must match Python <8s7I");

constexpr uint32_t kMaxCandidates = 1000000;
constexpr uint32_t kMaxTraceLength = 65535;
constexpr uint32_t kMaxSeeds = 4096;
constexpr uint64_t kMaxJobs = 10000000;
constexpr uint64_t kMaxBatchBytes = uint64_t{1} << 30;
constexpr uint32_t kMaxReps = 64;

static bool checked_mul(uint64_t left, uint64_t right, uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
  *result = left * right;
  return true;
}

__global__ void score_kernel(const uint8_t* traces, const uint16_t* offsets,
                             double* output, uint32_t candidates,
                             uint32_t length, uint32_t seeds,
                             uint32_t nwarps, uint32_t reps) {
  const uint64_t job64 = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_jobs = static_cast<uint64_t>(candidates) * seeds;
  if (job64 >= total_jobs) return;
  const uint32_t job = static_cast<uint32_t>(job64);
  const uint32_t candidate = job / seeds;
  const uint32_t seed = job % seeds;
  const uint8_t* base = traces + candidate * 5 * length;
  const uint8_t* pipe = base;
  const uint8_t* stall = base + length;
  const uint8_t* wait = base + 2 * length;
  const uint8_t* wb = base + 3 * length;
  const uint8_t* bank = base + 4 * length;
  int pc[8], iteration[8], next_issue[8], barrier[8][6];
  for (uint32_t w = 0; w < nwarps; ++w) {
    pc[w] = offsets[seed * nwarps + w];
    iteration[w] = next_issue[w] = 0;
    for (int b = 0; b < 6; ++b) barrier[w][b] = 0;
  }
  int fma_free = 0, alu_free = 0, mio_free = 0;
  int cycle = 0, last = -1, issued = 0;
  const int target = reps * length;
  while (true) {
    int progress = iteration[0] * length + pc[0];
    for (uint32_t w = 1; w < nwarps; ++w)
      progress = progress < iteration[w] * static_cast<int>(length) + pc[w]
                     ? progress
                     : iteration[w] * static_cast<int>(length) + pc[w];
    if (progress >= target + static_cast<int>(length)) break;
    int chosen = -1;
    for (uint32_t offset = 1; offset <= nwarps; ++offset) {
      const int w = (last + offset) % nwarps;
      if (next_issue[w] > cycle) continue;
      const int index = pc[w];
      bool blocked = false;
      for (int b = 0; b < 6; ++b)
        if (((wait[index] >> b) & 1) && barrier[w][b] > cycle) blocked = true;
      if (blocked) continue;
      if (pipe[index] == 0 && (fma_free > cycle || alu_free > cycle)) continue;
      if (pipe[index] == 1 && fma_free > cycle) continue;
      if (pipe[index] == 2 && alu_free > cycle) continue;
      if (pipe[index] == 3 && mio_free > cycle) continue;
      chosen = w;
      break;
    }
    if (chosen >= 0) {
      const int index = pc[chosen];
      const int extra = bank[index];
      if (pipe[index] == 0) { fma_free = cycle + 4 + extra; alu_free = cycle + 2 + extra; }
      else if (pipe[index] == 1) fma_free = cycle + 2 + extra;
      else if (pipe[index] == 2) alu_free = cycle + 2 + extra;
      else if (pipe[index] == 3) mio_free = cycle + 2;
      if (wb[index] != 7) barrier[chosen][wb[index]] = cycle + 26;
      next_issue[chosen] = cycle + (stall[index] > 1 ? stall[index] : 1) + extra;
      if (++pc[chosen] == static_cast<int>(length)) { pc[chosen] = 0; ++iteration[chosen]; }
      ++issued;
      last = chosen;
    }
    if (++cycle > 50000000) { output[job] = NAN; return; }
  }
  output[job] = static_cast<double>(cycle) * length / issued;
}

static void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) { std::cerr << what << ": " << cudaGetErrorString(status) << "\n"; std::exit(2); }
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) { std::cerr << "usage: sim2_batch BATCH.bin [SCORES.bin]\n"; return 2; }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) { std::cerr << "cannot open batch\n"; return 2; }
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  input.seekg(0, std::ios::beg);
  Header header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input || std::string(header.magic, 8) != "QTCE4GPU" || header.version != 1 || header.fields != 5) {
    std::cerr << "unsupported batch\n"; return 2;
  }
  if (header.candidates == 0 || header.candidates > kMaxCandidates ||
      header.length == 0 || header.length > kMaxTraceLength ||
      header.seeds == 0 || header.seeds > kMaxSeeds ||
      header.nwarps == 0 || header.nwarps > 8 ||
      header.reps == 0 || header.reps > kMaxReps) {
    std::cerr << "batch dimensions outside safe bounds\n"; return 2;
  }
  uint64_t jobs64, trace_bytes64, offset_count64;
  if (!checked_mul(header.candidates, header.seeds, &jobs64) || jobs64 > kMaxJobs ||
      !checked_mul(header.candidates, header.length, &trace_bytes64) ||
      !checked_mul(trace_bytes64, 5, &trace_bytes64) ||
      !checked_mul(header.seeds, header.nwarps, &offset_count64)) {
    std::cerr << "batch dimensions overflow\n"; return 2;
  }
  uint64_t offset_bytes64, result_bytes64;
  if (!checked_mul(offset_count64, sizeof(uint16_t), &offset_bytes64) ||
      !checked_mul(jobs64, sizeof(double), &result_bytes64) ||
      trace_bytes64 > kMaxBatchBytes || offset_bytes64 > kMaxBatchBytes || result_bytes64 > kMaxBatchBytes) {
    std::cerr << "batch byte count overflow\n"; return 2;
  }
  const uint64_t expected_file_size = sizeof(Header) + trace_bytes64 + offset_bytes64 + result_bytes64;
  if (expected_file_size > kMaxBatchBytes || file_size < 0 || static_cast<uint64_t>(file_size) != expected_file_size) {
    std::cerr << "batch has wrong exact size\n"; return 2;
  }
  const size_t trace_bytes = static_cast<size_t>(trace_bytes64);
  const size_t offset_count = static_cast<size_t>(offset_count64);
  const size_t jobs = static_cast<size_t>(jobs64);
  std::vector<uint8_t> traces;
  std::vector<uint16_t> offsets;
  std::vector<double> expected, actual;
  try {
    traces.resize(trace_bytes);
    offsets.resize(offset_count);
    expected.resize(jobs);
    actual.resize(jobs);
  } catch (const std::bad_alloc&) {
    std::cerr << "batch allocations failed\n"; return 2;
  }
  input.read(reinterpret_cast<char*>(traces.data()), traces.size());
  input.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(uint16_t));
  input.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(double));
  if (!input) { std::cerr << "truncated batch payload\n"; return 2; }
  for (uint32_t candidate = 0; candidate < header.candidates; ++candidate) {
    const uint8_t* base = traces.data() + static_cast<size_t>(candidate) * 5 * header.length;
    for (uint32_t index = 0; index < header.length; ++index) {
      if (base[index] > 4 || base[header.length + index] > 15 ||
          base[2 * header.length + index] > 0x3f ||
          !(base[3 * header.length + index] <= 5 || base[3 * header.length + index] == 7) ||
          base[4 * header.length + index] > 1) {
        std::cerr << "trace field outside its domain\n"; return 2;
      }
    }
  }
  for (uint16_t offset : offsets) {
    if (offset >= header.length) { std::cerr << "warp offset outside trace\n"; return 2; }
  }
  bool has_finite_reference = false, has_nan_reference = false;
  for (double value : expected) {
    if (std::isfinite(value)) has_finite_reference = true;
    else if (std::isnan(value)) has_nan_reference = true;
    else { std::cerr << "infinite reference score\n"; return 2; }
  }
  if (has_finite_reference && has_nan_reference) {
    std::cerr << "mixed scored/unscored references\n"; return 2;
  }
  uint8_t* d_traces = nullptr; uint16_t* d_offsets = nullptr; double* d_output = nullptr;
  check(cudaMalloc(&d_traces, trace_bytes), "cudaMalloc traces");
  check(cudaMalloc(&d_offsets, offset_count * sizeof(uint16_t)), "cudaMalloc offsets");
  check(cudaMalloc(&d_output, jobs * sizeof(double)), "cudaMalloc output");
  check(cudaMemcpy(d_traces, traces.data(), trace_bytes, cudaMemcpyHostToDevice), "copy traces");
  check(cudaMemcpy(d_offsets, offsets.data(), offset_count * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy offsets");
  score_kernel<<<(jobs + 255) / 256, 256>>>(d_traces, d_offsets, d_output, header.candidates,
                                            header.length, header.seeds, header.nwarps, header.reps);
  check(cudaGetLastError(), "score_kernel");
  check(cudaMemcpy(actual.data(), d_output, jobs * sizeof(double), cudaMemcpyDeviceToHost), "copy output");
  double maximum = 0.0; size_t references = 0;
  for (size_t i = 0; i < jobs; ++i) {
    if (!std::isfinite(actual[i])) { std::cerr << "non-finite CUDA score\n"; return 2; }
    if (!std::isfinite(expected[i])) continue;
    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    ++references;
  }
  if (argc == 3) {
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(actual.data()), actual.size() * sizeof(double));
    if (!output) { std::cerr << "cannot write scores\n"; return 2; }
  }
  std::cout << "jobs=" << jobs << " references=" << references
            << " max_abs_difference=" << maximum
            << " parity=" << (references == 0 ? "SKIP" : maximum <= 1e-12 ? "PASS" : "FAIL") << "\n";
  cudaFree(d_output); cudaFree(d_offsets); cudaFree(d_traces);
  return references == 0 || maximum <= 1e-12 ? 0 : 1;
}
