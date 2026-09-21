#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb, peak, krig, or the -diag form of one of them}"

if [ "$MINER" = "srb-diag" ]; then
  echo "=== --help ==="
  /opt/srbminer/SRBMiner-MULTI --help || echo "exit $?"
  echo "=== --list-algorithms ==="
  /opt/srbminer/SRBMiner-MULTI --list-algorithms || echo "exit $?"
  exit 0
fi

: "${POOL:?POOL is required, e.g. prl-eu.kryptex.network:7048 (krig needs the SSL port, 8048)}"
: "${WALLET:?WALLET is required: the Kryptex account login or a Pearl address (prl1...) to be paid at}"
: "${WORKER:?WORKER is required, e.g. c110598}"

case "$MINER" in
  srb)
    set -- /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" \
      --pool "$POOL" --wallet "$WALLET.$WORKER" --api-enable --api-port 21550
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
  krig)
    # Kryptex's own miner: no dev fee, TLS-only stratum (POOL must be the SSL port), and the worker goes after a
    # slash, not a dot. --no-rocm skips the AMD probe on a fleet that is all NVIDIA.
    # 4070 is deliberately not in sources/fleet.MINER_BY_PORT: the collector parses the SRBMiner format only and
    # must not poll this one. The pilot (pilots/miner_bench) reads it itself.
    set -- /opt/krig/krig-miner --url "stratum+ssl://$POOL" --user "$WALLET/$WORKER" --no-rocm \
      --api-host 0.0.0.0 --api-port 4070
    ;;
  *)
    echo "unknown MINER '$MINER': expected srb, peak or krig" >&2
    exit 64
    ;;
esac

# Each image carries one miner only, so a MINER that does not match it must stop here -- otherwise the loop below
# would spin forever on a binary that is not there.
[ -x "$1" ] || { echo "this image has no $1: MINER=$MINER belongs to the other image" >&2; exit 65; }

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
# The copy on disk is the only diagnosis when a miner starts but never hashes: a container's own stdout cannot be
# read from inside it, while `ssh` into the order can read a file (the order needs ssh_password for that).
while true; do
  { "$@" 2>&1; echo "miner exited with code $?, restarting in 10 s"; } | tee -a /var/log/miner.log
  sleep 10
done
