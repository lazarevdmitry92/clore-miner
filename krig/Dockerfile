FROM ubuntu:24.04

ARG KRIG_VERSION=1.5.1
ARG KRIG_SHA256=dbd6c69488e33777d839394b8f59b785b03fd44748d57f45da431e5d46f48663

# The host's NVIDIA runtime injects the driver; the miner needs only compute + NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# curl pulls in libssl3, which KRig loads at runtime for the TLS stratum connection.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Kryptex's own miner, 0% dev fee. Pinned by checksum: a young project, NVIDIA support still in beta.
RUN curl -fsSL "https://github.com/kryptex/krig-miner/releases/download/v${KRIG_VERSION}/krig-miner-${KRIG_VERSION}-linux-x64.tar.gz" -o /tmp/krig.tar.gz \
 && echo "${KRIG_SHA256}  /tmp/krig.tar.gz" | sha256sum -c - \
 && mkdir /opt/krig \
 && tar xzf /tmp/krig.tar.gz -C /opt/krig \
 && rm /tmp/krig.tar.gz

COPY entrypoint.sh /entrypoint.sh
EXPOSE 4070
ENTRYPOINT ["/entrypoint.sh"]
