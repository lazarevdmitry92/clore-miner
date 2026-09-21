FROM ubuntu:24.04

ARG SRBMINER_VERSION=3.6.7
ARG WILDRIG_VERSION=0.51.2

# The host's NVIDIA runtime injects the driver; the miner needs only compute + NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN srb_dir="SRBMiner-Multi-$(echo "$SRBMINER_VERSION" | tr . -)" \
 && curl -fsSL "https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRBMINER_VERSION}/${srb_dir}-Linux.tar.gz" | tar xz -C /opt \
 && mv "/opt/${srb_dir}" /opt/srbminer

# WildRig asks pearlhash for no dev fee where SRBMiner takes 2%. It reaches NVIDIA through OpenCL, so the image carries
# the ICD loader and names the driver's library: the container runtime injects libnvidia-opencl with `compute`, but
# nothing there says where to look for it.
RUN mkdir -p /opt/wildrig \
 && curl -fsSL "https://github.com/andru-kun/wildrig-multi/releases/download/${WILDRIG_VERSION}/wildrig-multi-linux-${WILDRIG_VERSION}.tar.gz" \
    | tar xz -C /opt/wildrig --strip-components=1 \
 && chmod +x /opt/wildrig/wildrig

RUN apt-get update \
 && apt-get install -y --no-install-recommends ocl-icd-libopencl1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /etc/OpenCL/vendors \
 && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

COPY entrypoint.sh /entrypoint.sh
EXPOSE 21550 21551
ENTRYPOINT ["/entrypoint.sh"]
