# One image per miner, so nothing can start the wrong one and neither image pays for the other's binary.
# Built with `--target srb`, `--target peak` and `--target krig` (see .github/workflows/build.yml). WildRig had a
# fourth target until 21.09: Kryptex does not speak Stratum v1 at all, so a miner written for it has nothing
# to assemble here (FACTS.md, "Пул и майнер").
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
ARG SRBMINER_VERSION=3.6.9
RUN srb_dir="SRBMiner-Multi-$(echo "$SRBMINER_VERSION" | tr . -)" \
 && curl -fsSL "https://github.com/doktor83/SRBMiner-Multi/releases/download/${SRBMINER_VERSION}/${srb_dir}-Linux.tar.gz" | tar xz -C /opt \
 && mv "/opt/${srb_dir}" /opt/srbminer
EXPOSE 21550


# Second by hashrate on the pool (24% against SRBMiner's 36%, census 21.09) and half its stale share: 0.32% vs 0.66%.
# Same 2% dev fee, so it must win on speed alone -- but it is the only one of the three with published per-card
# overclock profiles (--oc-profile, oc.peakminer.org). A single binary, pinned by the checksum GitHub publishes.
FROM base AS peak
ARG PEAKMINER_VERSION=2.16.4
ARG PEAKMINER_SHA256=587e3c3be3c75eb6586d69b25adb4c4208c1cf7d39157946da543dd823117adc
RUN curl -fsSL "https://github.com/peakminer/peakminer/releases/download/v${PEAKMINER_VERSION}/peakminer-${PEAKMINER_VERSION}-linux-x86_64" -o /usr/local/bin/peakminer \
 && echo "${PEAKMINER_SHA256}  /usr/local/bin/peakminer" | sha256sum -c - \
 && chmod +x /usr/local/bin/peakminer \
 && peakminer --version 2>&1 | grep -qi peakminer
EXPOSE 4068


# Kryptex's own miner: 0% dev fee and the only free one the pool itself lists. Pinned by checksum -- a young project,
# and the binary is what we pay with. Its stratum is TLS-only, so POOL must name the SSL port (8048).
FROM base AS krig
ARG KRIG_VERSION=1.5.2
ARG KRIG_SHA256=53863c153c7fddf711482de21414392f856ed3472692757887e65b1c7583005e
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
RUN curl -fsSL "https://github.com/kryptex/krig-miner/releases/download/v${KRIG_VERSION}/krig-miner-${KRIG_VERSION}-linux-x64.tar.gz" -o /tmp/krig.tar.gz \
 && echo "${KRIG_SHA256}  /tmp/krig.tar.gz" | sha256sum -c - \
 && mkdir /opt/krig \
 && tar xzf /tmp/krig.tar.gz -C /opt/krig \
 && rm /tmp/krig.tar.gz \
 && /opt/krig/krig-miner --help 2>&1 | grep -qi -- "--url"
EXPOSE 4070
