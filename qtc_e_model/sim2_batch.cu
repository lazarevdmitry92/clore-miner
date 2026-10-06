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
#include <string>
#include <vector>

struct Header {
  char magic[8];
  uint32_t version, candidates, length, seeds, nwarps, reps, fields;
};

__global__ void score_kernel(const uint8_t* traces, const uint16_t* offsets,
                             double* output, uint32_t candidates,
                             uint32_t length, uint32_t seeds,
                             uint32_t nwarps, uint32_t reps) {
  const uint32_t job = blockIdx.x * blockDim.x + threadIdx.x;
  if (job >= candidates * seeds || nwarps > 8) return;
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
    ++cycle;
  }
  output[job] = static_cast<double>(cycle) * length / issued;
}

static void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) { std::cerr << what << ": " << cudaGetErrorString(status) << "\n"; std::exit(2); }
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) { std::cerr << "usage: sim2_batch BATCH.bin [SCORES.bin]\n"; return 2; }
  std::ifstream input(argv[1], std::ios::binary);
  Header header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (std::string(header.magic, 8) != "QTCE4GPU" || header.version != 1 || header.fields != 5) {
    std::cerr << "unsupported batch\n"; return 2;
  }
  const size_t trace_bytes = static_cast<size_t>(header.candidates) * header.length * 5;
  const size_t offset_count = static_cast<size_t>(header.seeds) * header.nwarps;
  const size_t jobs = static_cast<size_t>(header.candidates) * header.seeds;
  std::vector<uint8_t> traces(trace_bytes);
  std::vector<uint16_t> offsets(offset_count);
  std::vector<double> expected(jobs), actual(jobs);
  input.read(reinterpret_cast<char*>(traces.data()), traces.size());
  input.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(uint16_t));
  input.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(double));
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
