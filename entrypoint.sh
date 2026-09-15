#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb or peak}"
: "${POOL:?POOL is required, e.g. prl-eu.kryptex.network:7048}"
: "${WALLET:?WALLET is required: Kryptex mining username}"
: "${WORKER:?WORKER is required, e.g. c110598}"

case "$MINER" in
  srb)
    set -- /opt/srbminer/SRBMiner-MULTI --disable-cpu --algorithm "${ALGO:-pearlhash}" \
      --pool "$POOL" --wallet "$WALLET.$WORKER" --api-enable --api-port 21550
    if [ "${PEARL_K2:-}" = "1" ]; then
      set -- "$@" --pearl-k2
    fi
    ;;
  peak)
    set -- /opt/peakminer/peakminer --coin "${COIN:-pearl}" -o "$POOL" \
      -u "$WALLET/$WORKER" --api-port 0.0.0.0:4068
    ;;
  *)
    echo "unknown MINER '$MINER': expected srb or peak" >&2
    exit 64
    ;;
esac

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
while true; do
  "$@" || echo "miner exited with code $?, restarting in 10 s" >&2
  sleep 10
done
