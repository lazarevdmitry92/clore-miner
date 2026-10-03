#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb-peak, srb, peak, krig, forge, bz, rg, fl4sh, or the -diag form of one}"

if [ "$MINER" = "peak-diag" ]; then
  echo "=== --version ==="
  /usr/local/bin/peakminer --version || echo "exit $?"
  echo "=== --help ==="
  /usr/local/bin/peakminer --help || echo "exit $?"
  exit 0
fi

if [ "$MINER" = "krig-diag" ]; then
  echo "=== --help ==="
  /opt/krig/krig-miner --help || echo "exit $?"
  echo "=== ldd ==="
  ldd /opt/krig/krig-miner || echo "exit $?"
  exit 0
fi

if [ "$MINER" = "srb-diag" ]; then
  # Журнал наружу поднимается первым: диагностика стоит раньше обычного запуска, и без этого её вывод
  # прочитать нечем (проба 22.09 -- заказ отработал вхолостую именно поэтому).
  busybox httpd -p 21559 -h /var/log || echo "log server did not start: $?"
  # Диагностика на живой карте: у --gpu-extra-config нет ни одного примера ни в Parameters, ни в релизах,
  # ни в examples.md, поэтому формат значения спрашиваем у самого бинаря. Вывод ложится в журнал, контейнер
  # остаётся жив -- иначе лог-порт умрёт вместе с ним и читать будет нечего.
  {
    echo "=== --help ==="
    /opt/srbminer/SRBMiner-MULTI --help 2>&1 || echo "exit $?"
    echo "=== --list-algorithms ==="
    /opt/srbminer/SRBMiner-MULTI --list-algorithms 2>&1 || echo "exit $?"
    echo "=== --list-devices ==="
    /opt/srbminer/SRBMiner-MULTI --list-devices 2>&1 || echo "exit $?"
    for value in help ? list 0 1; do
      echo "=== --gpu-extra-config $value (15 с) ==="
      timeout 15 /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" \
        --pool "${POOL:-prl.kryptex.network:7048}" --wallet "${WALLET:-x}.${WORKER:-diag}" \
        --gpu-extra-config "$value" 2>&1 | head -60 || echo "exit $?"
    done
  } > /var/log/miner.log 2>&1
  echo "диагностика записана в /var/log/miner.log; контейнер остаётся жив для чтения"
  while true; do sleep 300; done
fi

: "${POOL:?POOL is required, e.g. prl-eu.kryptex.network:7048 (krig needs the SSL port, 8048)}"
: "${WALLET:?WALLET is required: the Kryptex account login or a Pearl address (prl1...) to be paid at}"
: "${WORKER:?WORKER is required, e.g. c110598}"

# Everything we will ever be able to ask this container once it is running, written before the miner starts:
# which driver and cards the host actually gave us, what the power limits are, and whether the binary loads.
# The rest of the diagnosis is the miner's own log, which the loop below keeps on disk.
{
  echo "=== $(date -u +%FT%TZ) miner=$MINER pool=$POOL worker=$WORKER ==="
  uname -a
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,power.limit,power.default_limit,power.max_limit \
    --format=csv 2>&1 || echo "nvidia-smi failed: $?"
  echo "--- persistence and accounting (can we set anything at all) ---"
  nvidia-smi -q -d PERFORMANCE 2>&1 | head -40 || true
} > /var/log/startup.log 2>&1

# The log server: busybox serves /var/log, so `curl http://<host>:<port>/miner.log` reads the miner's own words and
# `/startup.log` the state of the machine. The port is published only when the order asks for it.
busybox httpd -p 21559 -h /var/log || echo "log server did not start: $?" >> /var/log/startup.log

# Can the host reach the pool at all: a host that firewalls the pool port (`Socket error 113`, 112640) answers here in
# the first minute -- ok, timeout, or the system's own words (refused, no route to host).
pool_host="${POOL%:*}"
pool_port="${POOL##*:}"
{
  printf -- "--- pool %s:%s tcp ---\n" "$pool_host" "$pool_port"
  if out=$(timeout 5 bash -c "exec 3<>/dev/tcp/$pool_host/$pool_port" 2>&1); then
    echo ok
  elif [ $? -eq 124 ]; then
    echo timeout
  else
    echo "failed: $out"
  fi
} >> /var/log/startup.log 2>&1

# srb-peak: SRBMiner first; when its dev fee is blocked (RU/UA hosts: `Devfee related operations were blocked`, then it
# exits) twice in a row, PeakMiner on the same pool, worker and API port, for good. PeakMiner failing with CUDA_700 three
# times in a row (RTX 5070 Ti on driver 615) leaves the container idle: the monitor sees no miner and drops the order.
# The choice is in /var/log/miner.choice, read through the log port; the API answer itself names the miner too.
if [ "$MINER" = "srb-peak" ]; then
  choose() { echo "$1 $(date -u +%FT%TZ) $2" > /var/log/miner.choice; }
  {
    echo "--- srb-peak ---"
    ldd /opt/srbminer/SRBMiner-MULTI 2>&1 || echo "ldd failed: $?"
    /usr/local/bin/peakminer --version 2>&1 || echo "peakminer --version failed: $?"
  } >> /var/log/startup.log 2>&1

  choose srb start
  blocked=0
  while [ "$blocked" -lt 2 ]; do
    # each run gets a fresh srb.log, the run before it stays as srb.prev.log
    mv -f /var/log/srb.log /var/log/srb.prev.log 2>/dev/null || true
    /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" --pool "$POOL" \
      --wallet "$WALLET.$WORKER" --api-enable --api-port 21550 --log-file /var/log/srb.log ${MINER_FLAGS:-} \
      >> /var/log/miner.log 2>&1 || true
    if grep -q "Devfee related operations were blocked" /var/log/srb.log 2>/dev/null; then
      blocked=$((blocked + 1))
    else
      blocked=0
    fi
    echo "$(date -u +%FT%TZ) SRBMiner exited, dev fee blocked $blocked run(s) in a row, next start in 10 s" >> /var/log/miner.log
    sleep 10
  done

  coin="${ALGO:-pearlhash}"
  if [ "$coin" = "pearlhash" ]; then coin=pearl; fi
  choose peak "SRBMiner dev fee blocked twice"
  touch /var/log/peak.log
  cuda=0
  while [ "$cuda" -lt 3 ]; do
    before=$(wc -l < /var/log/peak.log)
    /usr/local/bin/peakminer --coin "$coin" --url "$POOL" --user "$WALLET.$WORKER" --api-port 0.0.0.0:21550 \
      >> /var/log/peak.log 2>&1 || true
    if tail -n "+$((before + 1))" /var/log/peak.log | grep -qi -e CUDA_700 -e "illegal address"; then
      cuda=$((cuda + 1))
    else
      cuda=0
    fi
    echo "$(date -u +%FT%TZ) PeakMiner exited, CUDA_700 $cuda run(s) in a row, next start in 10 s" >> /var/log/peak.log
    sleep 10
  done
  choose none "PeakMiner CUDA_700 three times in a row"
  while true; do sleep 300; done
fi

case "$MINER" in
  srb)
    # MINER_FLAGS -- дополнительные ключи майнера для проб настроек (--pearl-k2, --gpu-intensity и подобные).
    # Намеренно без кавычек: строка разбивается на слова, иначе майнер получит один склеенный аргумент.
    # Ключи, меняющие настройки чужой карты (--gpu-cclock*, --gpu-plimit*, --gpu-fan*), сюда не передаются:
    # железо арендованное, и его режим -- не наш ресурс (решение пользователя 22.09).
    set -- /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" \
      --pool "$POOL" --wallet "$WALLET.$WORKER" --api-enable --api-port 21550 ${MINER_FLAGS:-}
    ;;
  peak)
    # -u goes to the pool verbatim, so it carries the worker the way Kryptex wants it -- after a dot, like SRBMiner
    # sends it. The API binds 0.0.0.0 or nothing outside the container could reach it; 4068 is its own default and
    # is not in sources/fleet.MINER_BY_PORT, so the collector leaves it alone.
    # PeakMiner names the coin, not the algorithm: `pearl` where everyone else says `pearlhash`. The fleet speaks
    # one ALGO, so the translation happens here rather than in every caller.
    coin="${ALGO:-pearlhash}"
    if [ "$coin" = "pearlhash" ]; then coin=pearl; fi
    set -- /usr/local/bin/peakminer --coin "$coin" --url "$POOL" --user "$WALLET.$WORKER" \
      --api-port 0.0.0.0:4068
    ;;
  forge)
    # Kryptex стоит в его собственном списке пулов, так что адрес идёт как есть; воркер -- отдельным полем.
    set -- /usr/local/bin/forge --algorithm "${ALGO:-pearlhash}" --wallet "$WALLET" --pool "$POOL" \
      --worker "$WORKER" --api-bind 0.0.0.0:7777
    ;;
  bz)
    # -p это адрес пула (не пароль), схему требует явно; 4020 -- его собственная страница и API.
    # --llm_port этой сборкой не распознан (проба 22.09), поэтому API наружу нет: свидетель -- журнал.
    set -- /usr/local/bin/bzminer -a pearl -p "stratum+tcp://$POOL" -w "$WALLET" --worker "$WORKER"
    ;;
  rg)
    # --proto kryptex: у него отдельный режим под диалект этого пула, по умолчанию он говорит на AkoyaV2.
    # --no-cmp-unlock: без него он лезет модифицировать драйвер ради разблокировки CMP-карт, падает с кодом 126
    # и уходит в вечный перезапуск — прав на это в контейнере площадки нет (проба 22.09).
    set -- /usr/local/bin/rgminer --algo pearl --stratum "$POOL" --wallet "$WALLET.$WORKER" --proto kryptex \
      --no-cmp-unlock --api-host 0.0.0.0 --api-port 21553
    ;;
  fl4sh)
    # API не объявляет вовсе: всё, что он скажет, окажется в /var/log/miner.log и уйдёт наружу лог-портом.
    set -- /usr/local/bin/fl4shminer -a "${ALGO:-pearlhash}" -pool "stratum+tcp://$POOL" \
      -w "$WALLET.$WORKER" -pass x
    ;;
  krig)
    # Kryptex's own miner: no dev fee, TLS-only stratum (POOL must be the SSL port), and the worker goes after a
    # slash, not a dot. --no-rocm skips the AMD probe on a fleet that is all NVIDIA.
    # 4070 is deliberately not in sources/fleet.MINER_BY_PORT: the collector parses the SRBMiner format only and
    # must not poll this one. The pilot (pilots/miner/miner_bench) reads it itself.
    set -- /opt/krig/krig-miner --url "stratum+ssl://$POOL" --user "$WALLET/$WORKER" --no-rocm \
      --api-host 0.0.0.0 --api-port 4070
    ;;
  *)
    echo "unknown MINER '$MINER': expected srb-peak, srb, peak, krig, forge, bz, rg or fl4sh" >&2
    exit 64
    ;;
esac

# Each image carries one miner only, so a MINER that does not match it must stop here -- otherwise the loop below
# would spin forever on a binary that is not there.
[ -x "$1" ] || { echo "this image has no $1: MINER=$MINER belongs to the other image" >&2; exit 65; }

# A miner that enumerates the cards and hashes on none (KRig, 21.09) says nothing about why. Its libraries do.
{
  echo "--- command ---"
  echo "$@"
  echo "--- ldd $1 ---"
  ldd "$1" 2>&1 || echo "ldd failed: $?"
} >> /var/log/startup.log 2>&1

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
# The copy on disk is the only diagnosis when a miner starts but never hashes: a container's own stdout cannot be
# read from inside it, while `ssh` into the order can read a file (the order needs ssh_password for that).
while true; do
  { "$@" 2>&1; echo "miner exited with code $?, restarting in 10 s"; } | tee -a /var/log/miner.log
  sleep 10
done
