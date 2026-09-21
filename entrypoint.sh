#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb, wildrig, krig, or the -diag form of one of them}"

if [ "$MINER" = "srb-diag" ]; then
  echo "=== --help ==="
  /opt/srbminer/SRBMiner-MULTI --help || echo "exit $?"
  echo "=== --list-algorithms ==="
  /opt/srbminer/SRBMiner-MULTI --list-algorithms || echo "exit $?"
  exit 0
fi

# Whether OpenCL found the cards at all is the one thing that can fail silently in a container: the runtime injects
# the driver, but nothing inside says where to look for it. WildRig has no option that prints its fees -- they are in
# its README, and our own shares are what settles the matter.
if [ "$MINER" = "wildrig-diag" ]; then
  echo "=== --print-platforms ==="
  /opt/wildrig/wildrig-multi --print-platforms || echo "exit $?"
  echo "=== --print-devices ==="
  /opt/wildrig/wildrig-multi --print-devices || echo "exit $?"
  exit 0
fi

: "${POOL:?POOL is required, e.g. prl-eu.kryptex.network:7048 (krig needs the SSL port, 8048)}"
: "${WALLET:?WALLET is required: Kryptex mining username}"
: "${WORKER:?WORKER is required, e.g. c110598}"

case "$MINER" in
  srb)
    set -- /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" \
      --pool "$POOL" --wallet "$WALLET.$WORKER" --api-enable --api-port 21550
    ;;
  wildrig)
    # No dev fee on pearlhash, and its own API port. 21551 is deliberately not in sources/fleet.MINER_BY_PORT: the
    # collector parses the SRBMiner format only, so it must not poll this one. The pilot reads 21551 itself, and that
    # is how we learn what WildRig's API even answers.
    # The worker goes into --user after a dot, the way SRBMiner sends it: that is how the pool splits shares by
    # server, and the pool's own per-worker figures are this pilot's main witness. WildRig's own --worker would
    # name it elsewhere.
    set -- /opt/wildrig/wildrig-multi --algo "${ALGO:-pearlhash}" --url "$POOL" \
      --user "$WALLET.$WORKER" --api-port 21551 --opencl-platforms nvidia
    ;;
  krig)
    # Kryptex's own miner: no dev fee, TLS-only stratum (POOL must be the SSL port), and the worker goes after a
    # slash, not a dot. --no-rocm skips the AMD probe on a fleet that is all NVIDIA.
    set -- /opt/krig/krig-miner --url "stratum+ssl://$POOL" --user "$WALLET/$WORKER" --no-rocm \
      --api-host 0.0.0.0 --api-port 4070
    ;;
  *)
    echo "unknown MINER '$MINER': expected srb, wildrig or krig" >&2
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
