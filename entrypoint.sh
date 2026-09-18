#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb or srb-diag}"

if [ "$MINER" = "srb-diag" ]; then
  echo "=== --help ==="
  /opt/srbminer/SRBMiner-MULTI --help || echo "exit $?"
  echo "=== --list-algorithms ==="
  /opt/srbminer/SRBMiner-MULTI --list-algorithms || echo "exit $?"
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
  *)
    echo "unknown MINER '$MINER': expected srb" >&2
    exit 64
    ;;
esac

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
while true; do
  "$@" || echo "miner exited with code $?, restarting in 10 s" >&2
  sleep 10
done
