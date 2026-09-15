FROM ubuntu:24.04

ARG SRBMINER_VERSION=3.6.7
ARG PEAKMINER_VERSION=2.16.2

# The host's NVIDIA runtime injects the driver; the miners need only compute + NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN srb_dir="SRBMiner-Multi-$(echo "$SRBMINER_VERSION" | tr . -)" \
 && curl -fsSL "https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRBMINER_VERSION}/${srb_dir}-Linux.tar.gz" | tar xz -C /opt \
 && mv "/opt/${srb_dir}" /opt/srbminer \
 && mkdir /opt/peakminer \
 && curl -fsSL "https://github.com/peakminer/peakminer/releases/download/v${PEAKMINER_VERSION}/peakminer-${PEAKMINER_VERSION}-linux-x86_64" -o /opt/peakminer/peakminer \
 && chmod +x /opt/peakminer/peakminer

COPY entrypoint.sh /entrypoint.sh
EXPOSE 21550 4068
ENTRYPOINT ["/entrypoint.sh"]
