# libpearl_soat.so — GPU backend of the own miner over the SOAT kernels (sm_80/86/89/120).
#   make ARCH=sm_86            one architecture (any CUDA 12.x; sm_120 needs CUDA >= 12.8)
#   make fat                   sm_80;86;89;120 SASS + compute_120 PTX in one .so (CUDA >= 12.8)
#   make NVCC=/usr/local/cuda-12.8/bin/nvcc fat
NVCC  ?= nvcc
ARCH  ?= sm_86
SOAT  ?= ../src/soat/src/algos/pearl-pow
OUT   ?= libpearl_soat.so

GENCODE ?= -gencode arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH)
FAT_GENCODE = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 \
              -gencode arch=compute_120,code=compute_120

NVFLAGS = -O3 -std=c++17 -shared -Xcompiler -fPIC --cudart static -lineinfo -I$(SOAT) -I.

SRC = pearl_soat.cu pearl_soat.h $(SOAT)/job.h $(SOAT)/noisy_gemm.cuh $(SOAT)/prepare.cuh $(SOAT)/blake3.cuh

$(OUT): $(SRC)
	$(NVCC) $(NVFLAGS) $(GENCODE) pearl_soat.cu -o $@
	@nm -D --defined-only $@ | grep -E ' pearl_(search|search_ids|device_info|last_error)$$'

.PHONY: fat clean
fat:
	$(MAKE) -B GENCODE="$(FAT_GENCODE)"

clean:
	rm -f $(OUT)
