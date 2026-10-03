# V100 Pearl kernel. CUDA 12.x, Linux x86-64; cudart static (TZ §3).
#   make            -> build/libpearl_v100.so (ctypes, SoBackend) and build/pearl_bench
#   make emu        -> build/libpearl_emu.so  (CPU emulator, any OS with a C++17 compiler)
#   make ptxas      -> register / spill / shared-memory report of every kernel
NVCC ?= nvcc
CXX ?= c++
ARCH = -gencode arch=compute_70,code=sm_70
NVFLAGS = -O3 -std=c++17 $(ARCH) -cudart static -Xcompiler -fPIC,-Wall -lineinfo
HDRS = pearl_common.h pearl_layout.h pearl_host.h pearl_api.h ../common/pearl_core.h ../common/host_core.h ../common/runtime.h ../common/pearl_api.h ../common/operands.h ../common/operands_cuda.cuh

all: build/libpearl_v100.so build/pearl_bench

build:
	mkdir -p build

build/libpearl_v100.so: pearl_v100.cu $(HDRS) | build
	$(NVCC) $(NVFLAGS) -shared -o $@ pearl_v100.cu

build/pearl_bench: bench.cu pearl_v100.cu $(HDRS) ../common/bench_main.h | build
	$(NVCC) $(NVFLAGS) -o $@ bench.cu pearl_v100.cu -ldl -lpthread

emu: build/libpearl_emu.so

build/libpearl_emu.so: pearl_emu.cpp $(HDRS) | build
	$(CXX) -O2 -std=c++17 -Wall -Wextra -shared -fPIC -o $@ pearl_emu.cpp

ptxas: | build
	$(NVCC) $(NVFLAGS) -Xptxas -v -c -o build/ptxas.o pearl_v100.cu

clean:
	rm -rf build

.PHONY: all emu ptxas clean
