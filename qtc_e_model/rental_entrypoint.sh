#!/usr/bin/env bash
# Self-contained QTC E rental computation. This image never calls a marketplace
# or any project production service; atlas owns rental, watchdog, and teardown.
set -Eeuo pipefail

readonly PAYLOAD=/opt/qtc-e
readonly WORK=/work/qtc-e
readonly LOG_ROOT=/var/log/qtc-e
readonly SOURCE_ARCHIVE="$PAYLOAD/source.tar.gz"
readonly PILOT_ARCHIVE="$PAYLOAD/pilot-s10.tar.gz"
readonly CHECKSUMS="$PAYLOAD/SHA256SUMS"
readonly PACKAGE_CHECKSUM="$PAYLOAD/PACKAGE.sha256"
readonly RUN_DIR="$LOG_ROOT/run"
readonly RUN_LOG="$LOG_ROOT/run.log"
readonly FAILED="$LOG_ROOT/failed"
readonly DONE="$LOG_ROOT/done"
HTTP_PID=""

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

cleanup() {
  local status=$?
  if [[ -n "$HTTP_PID" ]]; then
    kill "$HTTP_PID" 2>/dev/null || true
    wait "$HTTP_PID" 2>/dev/null || true
  fi
  if (( status != 0 )) && [[ ! -e "$FAILED" ]]; then
    printf 'status=%d\nfinished_unix=%s\n' "$status" "$(date +%s)" >"$FAILED"
  fi
}
trap cleanup EXIT INT TERM HUP

start_http() {
  "$PYTHON_BIN" -m http.server 21559 --bind 0.0.0.0 --directory "$LOG_ROOT" >>"$LOG_ROOT/http.log" 2>&1 &
  HTTP_PID=$!
  sleep 1
  kill -0 "$HTTP_PID"
}

prepare() {
  rm -f "$DONE" "$DONE.tmp" "$FAILED"
  PYTHON_BIN="$(command -v python3 || true)"
  if [[ -z "$PYTHON_BIN" && -x /opt/python/bin/python3 ]]; then
    PYTHON_BIN=/opt/python/bin/python3
  fi
  [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]
  command -v nvcc >/dev/null
  command -v sha256sum >/dev/null
  [[ -s "$SOURCE_ARCHIVE" && -s "$PILOT_ARCHIVE" && -s "$CHECKSUMS" && -s "$PACKAGE_CHECKSUM" ]]
  (cd "$PAYLOAD" && sha256sum --check --strict PACKAGE.sha256 && sha256sum --check --strict SHA256SUMS)
  local actual_package
  actual_package="$(sha256sum "$CHECKSUMS" | awk '{print $1}')"
  printf '%s\n' "$actual_package" >"$LOG_ROOT/package.sha256"
  mkdir -p "$WORK"
  tar -xzf "$SOURCE_ARCHIVE" -C "$WORK"
  tar -xzf "$PILOT_ARCHIVE" -C "$WORK"
  [[ -f "$WORK/miner/qtc_e_model/hybrid_accel.py" ]]
  [[ -f "$WORK/pilots/miner/qtc_own_miner/s10/frag/sim2.py" ]]
  "$PYTHON_BIN" -m http.server --help >/dev/null
  start_http
}

compute() {
  cd "$WORK"
  export PYTHONPATH="$WORK"
  "$PYTHON_BIN" -m miner.qtc_e_model.hybrid_accel self-check \
    --pilot-root pilots/miner/qtc_own_miner
  nvcc -O3 -std=c++17 miner/qtc_e_model/sim2_batch.cu -o "$LOG_ROOT/qtc_sim2_batch"
  sha256sum "$LOG_ROOT/qtc_sim2_batch" >"$LOG_ROOT/qtc_sim2_batch.sha256"

  "$PYTHON_BIN" -m miner.qtc_e_model.hybrid_accel export \
    --pilot-root pilots/miner/qtc_own_miner \
    --batch "$LOG_ROOT/parity-reps3.bin" --candidates 8 --seeds 8 --reps 3
  "$PYTHON_BIN" -m miner.qtc_e_model.hybrid_accel export \
    --pilot-root pilots/miner/qtc_own_miner \
    --batch "$LOG_ROOT/parity-reps6.bin" --candidates 8 --seeds 8 --reps 6
  "$PYTHON_BIN" -m miner.qtc_e_model.hybrid_accel gpu-check \
    --pilot-root pilots/miner/qtc_own_miner \
    --coarse-parity-batch "$LOG_ROOT/parity-reps3.bin" \
    --confirmation-parity-batch "$LOG_ROOT/parity-reps6.bin" \
    --gpu-binary "$LOG_ROOT/qtc_sim2_batch" \
    --parity-artifact "$LOG_ROOT/parity.json"

  "$PYTHON_BIN" -m miner.qtc_e_model.hybrid_accel hybrid-search \
    --pilot-root pilots/miner/qtc_own_miner \
    --gpu-binary "$LOG_ROOT/qtc_sim2_batch" \
    --coarse-parity-batch "$LOG_ROOT/parity-reps3.bin" \
    --confirmation-parity-batch "$LOG_ROOT/parity-reps6.bin" \
    --parity-artifact "$LOG_ROOT/parity.json" \
    --archive-dir "$RUN_DIR" \
    --result "$LOG_ROOT/result.json" \
    --depth 4 --width 8 --branch-budget 48 --seeds 1 \
    --timeout-seconds 1800

  "$PYTHON_BIN" - "$RUN_DIR/manifest.json" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != "qtc-e-rental-run-v1" or manifest.get("status") != "complete":
    raise SystemExit("run manifest is not complete")
if not manifest.get("batches"):
    raise SystemExit("run manifest contains no scored batches")
PY
  find "$LOG_ROOT" -type f ! -name ARTIFACTS.sha256 ! -name done ! -name 'done.tmp' ! -name failed \
    ! -name run.log ! -name http.log ! -name compute-timeout.log \
    -print0 | sort -z | xargs -0 sha256sum >"$LOG_ROOT/ARTIFACTS.sha256"
  printf 'status=complete\nfinished_unix=%s\nmanifest_sha256=%s\n' \
    "$(date +%s)" "$(sha256sum "$RUN_DIR/manifest.json" | awk '{print $1}')" >"$DONE.tmp"
  mv "$DONE.tmp" "$DONE"
}

prepare
export WORK LOG_ROOT RUN_DIR PYTHON_BIN
export -f compute
if ! timeout --signal=TERM --kill-after=10s 1800 bash -c 'set -Eeuo pipefail; compute' \
  2>"$LOG_ROOT/compute-timeout.log"; then
  # Retry is forbidden: mark the single run failed and retain logs.
  printf 'status=compute_failed\nfinished_unix=%s\n' "$(date +%s)" >"$FAILED"
else
  :
fi

# Keep the log endpoint available until atlas' independent watchdog collects
# artifacts and drops the pilot. /done exists only after a complete manifest.
while :; do sleep 3600; done
