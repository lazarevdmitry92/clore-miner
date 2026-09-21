# Two images from one recipe: each carries exactly one miner, so nothing can start the wrong one and neither image
# pays for the other's binary. Built with `--target srb` and `--target wildrig` (see .github/workflows/build.yml).
FROM ubuntu:24.04 AS base

# The host's NVIDIA runtime injects the driver; the miner needs only compute + NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


FROM base AS srb
ARG SRBMINER_VERSION=3.6.7
RUN srb_dir="SRBMiner-Multi-$(echo "$SRBMINER_VERSION" | tr . -)" \
 && curl -fsSL "https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRBMINER_VERSION}/${srb_dir}-Linux.tar.gz" | tar xz -C /opt \
 && mv "/opt/${srb_dir}" /opt/srbminer
EXPOSE 21550


# WildRig asks pearlhash for no dev fee where SRBMiner takes 2%. It reaches NVIDIA through OpenCL, so this image
# carries the ICD loader and names the driver's library: the container runtime injects libnvidia-opencl with
# `compute`, but nothing inside says where to look for it.
# The last line proves the binary loads its libraries, and checks what it prints rather than what it returns:
# --version leaves a non-zero code behind on a machine with no GPU. A missing library would print the loader's
# complaint instead of the version, and the build would stop here rather than on a paid rental.
FROM base AS wildrig
ARG WILDRIG_VERSION=0.51.2
RUN mkdir -p /opt/wildrig \
 && curl -fsSL "https://github.com/andru-kun/wildrig-multi/releases/download/${WILDRIG_VERSION}/wildrig-multi-linux-${WILDRIG_VERSION}.tar.gz" \
    | tar xz -C /opt/wildrig \
 && chmod +x /opt/wildrig/wildrig-multi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ocl-icd-libopencl1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /etc/OpenCL/vendors \
 && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
 && /opt/wildrig/wildrig-multi --version 2>&1 | grep -q "WildRig Multi"
EXPOSE 21551
