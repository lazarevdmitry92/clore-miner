#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb, wildrig, or the -diag form of either}"

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

: "${POOL:?POOL is required, e.g. prl-eu.kryptex.network:7048}"
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
  *)
    echo "unknown MINER '$MINER': expected srb or wildrig" >&2
    exit 64
    ;;
esac

# Each image carries one miner only, so a MINER that does not match it must stop here -- otherwise the loop below
# would spin forever on a binary that is not there.
[ -x "$1" ] || { echo "this image has no $1: MINER=$MINER belongs to the other image" >&2; exit 65; }

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
while true; do
  "$@" || echo "miner exited with code $?, restarting in 10 s" >&2
  sleep 10
done
