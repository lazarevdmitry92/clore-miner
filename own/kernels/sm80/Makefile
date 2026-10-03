# Pearl kernel for Ampere, Ada, Blackwell: one fat binary for sm_80/86/89/120 (CUDA >= 12.8), Linux x86-64, cudart
# static (TZ_sm80_kernel.md §3).
#   make            -> build/libpearl_sm80.so (ctypes, SoBackend) and build/pearl_bench
#   make emu        -> build/libpearl_emu80.so (CPU emulator, any OS with a C++17 compiler)
#   make ptxas      -> register / spill / shared-memory report per architecture
#   ARCHS="80 86" make   -> a subset (faster builds while iterating)
NVCC ?= nvcc
CXX ?= c++
ARCHS ?= 80 86 89 120
GENCODE = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
NVFLAGS = -O3 -std=c++17 $(GENCODE) -cudart static -Xcompiler -fPIC,-Wall -lineinfo
HDRS = pearl_layout80.h pearl_host80.h ../common/pearl_core.h ../common/host_core.h ../common/runtime.h \
       ../common/pearl_api.h ../common/operands.h ../common/operands_cuda.cuh

all: build/libpearl_sm80.so build/pearl_bench

build:
	mkdir -p build

build/libpearl_sm80.so: pearl_sm80.cu $(HDRS) | build
	$(NVCC) $(NVFLAGS) -shared -o $@ pearl_sm80.cu

build/pearl_bench: bench.cu pearl_sm80.cu $(HDRS) ../common/bench_main.h | build
	$(NVCC) $(NVFLAGS) -o $@ bench.cu pearl_sm80.cu -ldl -lpthread

emu: build/libpearl_emu80.so

build/libpearl_emu80.so: pearl_emu80.cpp $(HDRS) | build
	$(CXX) -O2 -std=c++17 -Wall -Wextra -shared -fPIC -o $@ pearl_emu80.cpp

ptxas: | build
	$(NVCC) $(NVFLAGS) -Xptxas -v -c -o build/ptxas.o pearl_sm80.cu

clean:
	rm -rf build

.PHONY: all emu ptxas clean
