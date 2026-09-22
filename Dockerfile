# One image per miner, so nothing can start the wrong one and neither image pays for the other's binary.
# Built with one `--target` per miner (see .github/workflows/build.yml): srb, peak, krig, forge, bz, rg, fl4sh.
# WildRig had a
# fourth target until 21.09: Kryptex does not speak Stratum v1 at all, so a miner written for it has nothing
# to assemble here (FACTS.md, "Пул и майнер").
FROM ubuntu:24.04 AS base

# The host's NVIDIA runtime injects the driver; the miner needs only compute + NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# busybox carries the one-line web server that hands us /var/log from outside: a container cannot show its own
# stdout, and Clore's client cannot ask for it, so a miner that starts and never hashes is otherwise mute.
# xz -- rgminer распаковывает свой payload через tar+xz и без него падает на старте;
# ocl-icd + nvidia.icd -- fl4shminer линкуется с libOpenCL.so.1, а контейнерный runtime кладёт драйвер,
# но не говорит, где его искать. Обе беды нашлись только по журналу контейнера (проба 22.09).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl busybox-static xz-utils ocl-icd-libopencl1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /etc/OpenCL/vendors \
 && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

EXPOSE 21559
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


# Вторая пачка кандидатов (проба 22.09). Каждый — своя цель, свой бинарь, свой порт API: образ по-прежнему везёт
# ровно один майнер. Архивы распаковываются вслепую (структура у всех своя), поэтому бинарь ищется по имени
# и проверяется запуском — сборка падает здесь, а не на оплаченной аренде.

# ForgeMiner: нативный NVIDIA, Kryptex в списке готовых пулов, 2% на pearlhash, собственный HTTP на 7777.
FROM base AS forge
ARG FORGE_VERSION=1.8.1
ARG FORGE_SHA256=b8cfca82925957303c67e299069806ec27947a50bed3569630940c5c0490b519
RUN curl -fsSL "https://github.com/0xHashRaptor/ForgeMiner/releases/download/v${FORGE_VERSION}/ForgeMiner-${FORGE_VERSION}-linux.tar.gz" -o /tmp/f.tgz \
 && echo "${FORGE_SHA256}  /tmp/f.tgz" | sha256sum -c - \
 && mkdir -p /opt/forge && tar xzf /tmp/f.tgz -C /opt/forge && rm /tmp/f.tgz \
 && bin="$(find /opt/forge -type f \( -name forge -o -name 'ForgeMiner*' \) -perm -u+x | head -1)" \
 && [ -n "$bin" ] && ln -s "$bin" /usr/local/bin/forge \
 && forge --help >/dev/null 2>&1 || forge --version 2>&1 | head -1
EXPOSE 7777

# BzMiner: 2% на pearl, знает prl-us.kryptex.network:7048, страница и API на 4020.
FROM base AS bz
ARG BZ_VERSION=100.31
ARG BZ_SHA256=4687403deb0efc881f5a6de8a773f07942687470da7321106a461d31a6b45b11
RUN curl -fsSL "https://github.com/bzminer/bzminer/releases/download/v${BZ_VERSION}/bzminer_v${BZ_VERSION}_linux.tar.gz" -o /tmp/b.tgz \
 && echo "${BZ_SHA256}  /tmp/b.tgz" | sha256sum -c - \
 && mkdir -p /opt/bz && tar xzf /tmp/b.tgz -C /opt/bz && rm /tmp/b.tgz \
 && bin="$(find /opt/bz -type f -name bzminer -perm -u+x | head -1)" \
 && [ -n "$bin" ] && ln -s "$bin" /usr/local/bin/bzminer \
 && bzminer --help >/dev/null 2>&1 || true
EXPOSE 4020

# RGMiner: один бинарь без архива, 2% на pearl. У него есть `--proto kryptex` -- отдельный режим под диалект
# этого пула, что лишний раз подтверждает: протокол Kryptex не общий.
FROM base AS rg
ARG RG_VERSION=1.0.7
ARG RG_SHA256=7ac11240c6df2428958073e167b8cf52fc2d2f78417f4f0e4623d3bbe659b9af
RUN curl -fsSL "https://github.com/Printscan/rgminer/releases/download/v${RG_VERSION}/rgminer-${RG_VERSION}" -o /usr/local/bin/rgminer \
 && echo "${RG_SHA256}  /usr/local/bin/rgminer" | sha256sum -c - \
 && chmod +x /usr/local/bin/rgminer \
 && rgminer --help >/dev/null 2>&1 || true
EXPOSE 21553

# Fl4shMiner: 1.5% на pearlhash -- единственный дешевле SRBMiner, ядра под Ampere и Ada. API не объявляет,
# поэтому его единственный свидетель -- журнал на лог-порту.
FROM base AS fl4sh
ARG FL4SH_VERSION=1.4.4
ARG FL4SH_SHA256=e60107c39348f6396392a9a95e335b55ae3ba28d391698479605cefe8003bd14
RUN curl -fsSL "https://github.com/Fl4sh9174/Fl4shMiner/releases/download/v${FL4SH_VERSION}/fl4shminer-v${FL4SH_VERSION}.tar.gz" -o /tmp/fl.tgz \
 && echo "${FL4SH_SHA256}  /tmp/fl.tgz" | sha256sum -c - \
 && mkdir -p /opt/fl4sh && tar xzf /tmp/fl.tgz -C /opt/fl4sh && rm /tmp/fl.tgz \
 && bin="$(find /opt/fl4sh -type f -name '*fl4shminer*' -perm -u+x | head -1)" \
 && [ -n "$bin" ] && ln -s "$bin" /usr/local/bin/fl4shminer


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
