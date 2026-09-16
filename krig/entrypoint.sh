#!/bin/sh
set -eu

# TLS-only stratum: POOL must be the SSL port, e.g. prl.kryptex.network:8048
: "${POOL:?POOL is required, e.g. prl.kryptex.network:8048}"
: "${WALLET:?WALLET is required: Kryptex mining username}"
: "${WORKER:?WORKER is required, e.g. k111237}"

export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1  # NativeAOT .NET binary; the image has no libicu

# Spot orders and host hiccups kill the miner; restart it instead of leaving a paid GPU idle.
while true; do
  /opt/krig/krig-miner --url "stratum+ssl://$POOL" --user "$WALLET/$WORKER" --no-rocm \
    --api-host 0.0.0.0 --api-port 4070 || echo "miner exited with code $?, restarting in 10 s" >&2
  sleep 10
done
