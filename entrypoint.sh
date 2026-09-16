#!/bin/sh
set -eu

: "${MINER:?MINER is required: srb, peak, krig or srb-diag}"

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
    if [ "${PEARL_K2:-}" = "1" ]; then
      set -- "$@" --pearl-k2
    fi
    ;;
  peak)
    set -- /opt/peakminer/peakminer --coin "${COIN:-pearl}" -o "$POOL" \
      -u "$WALLET/$WORKER" --api-port 0.0.0.0:4068
    ;;
  krig)
    # TLS-only stratum: POOL must be the SSL port, e.g. prl.kryptex.network:8048
    export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1  # NativeAOT .NET binary; the image has no libicu
    set -- /opt/krig/krig-miner --url "stratum+ssl://$POOL" --user "$WALLET/$WORKER" --no-rocm \
      --api-host 0.0.0.0 --api-port 4070
    ;;
  *)
    echo "unknown MINER '$MINER': expected srb, peak or krig" >&2
    exit 64
    ;;
esac

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
while true; do
  "$@" || echo "miner exited with code $?, restarting in 10 s" >&2
  sleep 10
done
