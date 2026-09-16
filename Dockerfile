FROM ubuntu:24.04

ARG SRBMINER_VERSION=3.6.7
ARG PEAKMINER_VERSION=2.16.2
ARG KRIG_VERSION=1.5.1
ARG KRIG_SHA256=dbd6c69488e33777d839394b8f59b785b03fd44748d57f45da431e5d46f48663

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

# Kryptex's own miner, 0% dev fee. Pinned by checksum: a young project, NVIDIA support still in beta.
RUN curl -fsSL "https://github.com/kryptex/krig-miner/releases/download/v${KRIG_VERSION}/krig-miner-${KRIG_VERSION}-linux-x64.tar.gz" -o /tmp/krig.tar.gz \
 && echo "${KRIG_SHA256}  /tmp/krig.tar.gz" | sha256sum -c - \
 && mkdir /opt/krig \
 && tar xzf /tmp/krig.tar.gz -C /opt/krig \
 && rm /tmp/krig.tar.gz

COPY entrypoint.sh /entrypoint.sh
EXPOSE 21550 4068 4070
ENTRYPOINT ["/entrypoint.sh"]
