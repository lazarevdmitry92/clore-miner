# QTC E rental compute package

Decision 2026-10-06: the current schedule/frame family is formally closed.
Further calculation and analysis of a new QTC kernel runs on rented powerful
machines. The local machine is limited to package preparation, unit/parity
smoke tests, and result analysis. Before each rental the coordinator records
price and duration; the pilot has a 15-second watchdog. This package does not
rent a machine or access the network.

## Split and measured baseline

The CPU owns the dependency graph, legal relocate branching, bank-aware
recolor, control generation, and pruning. CUDA receives a flat batch of
independent `(candidate, seed)` sim2 traces. The compact trace is 689
instructions and 3,445 bytes per candidate; 10,000 candidates need about 35
MB plus outputs.

Mac profile, 8 candidates x 8 seeds:

| stage | time | share |
|---|---:|---:|
| CPU frontier preparation | 2.015 s | 17.7% |
| seed scoring (GPU eligible) | 9.349 s | 82.3% |

The encoded CPU reference is regressed directly against the original S10
`sim2.py` on real traces and seeds, for both optimizer modes: coarse `reps=3`
and confirmation `reps=6`. A 4-candidate x 4-seed binary export/check round
trip is exact. CUDA parity is a mandatory runtime gate (`<=1e-12`) before any
full search; a failure exits non-zero. The resulting JSON attestation binds
the exact CUDA binary, authoritative scored batch, and all model sources by
SHA-256. `hybrid-search` refuses a missing, stale, or mismatched attestation.

The conservative scorer model is 5,000--15,000 candidate-seed jobs/s. At the
profiled 8-seed mix that is 5.6x end-to-end; confirmation batches at 64 seeds
are expected to gain roughly 25--35x, while branch-heavy one-seed layers stay
CPU-bound. For the complete mixed optimizer, budget on 2--5x until measured.
The latency model breaks even at one job under a 20 ms transfer/launch envelope,
but use at least 256 jobs per batch for stable GPU occupancy.

## Rental specification and budget gate

- Executable hardware gate: exactly one RTX 4090 24 GB. The immutable image
  supplies CUDA 12.x. No target V100 is required: this GPU only evaluates the
  V100 simulator state machine. There is no L40S fallback in the reviewed
  runner.
- Host sizing preference/estimate: at least 24 vCPU, 32 GB RAM, and 10 GB
  disk. The current collector snapshot and public placement runner cannot
  prove those fields, so they are not claimed as pre-rent acceptance gates.
  If the platform later exposes them, promote them to executable checks;
  otherwise an insufficient host fails the run and the watchdog tears it down.
- Price ceiling: `$2.00/hour`; planned reservation 30 minutes; hard compute
  budget `$1.00`. The coordinator must record live price and planned duration
  before launch and must not take a profitable current-fleet server.
- Expected: self-check+build+two-mode parity 5--10 minutes; the packaged
  single deterministic depth-4 accelerator validation/current-search
  reproduction 5--15 minutes. It is not the three independent starts already
  used for the schedule-family saturation decision. Hard timeout 30 minutes;
  watchdog interval 15 seconds.

## Build, parity, and search on the rental

From the repository root on the rental host:

```sh
python3.11 -m miner.qtc_e_model.hybrid_accel self-check \
  --pilot-root pilots/miner/qtc_own_miner
nvcc -O3 -std=c++17 miner/qtc_e_model/sim2_batch.cu -o /tmp/qtc_sim2_batch
python3.11 -m miner.qtc_e_model.hybrid_accel export \
  --pilot-root pilots/miner/qtc_own_miner \
  --batch /tmp/qtc_e_parity_reps3.bin --candidates 8 --seeds 8 --reps 3
python3.11 -m miner.qtc_e_model.hybrid_accel export \
  --pilot-root pilots/miner/qtc_own_miner \
  --batch /tmp/qtc_e_parity_reps6.bin --candidates 8 --seeds 8 --reps 6
python3.11 -m miner.qtc_e_model.hybrid_accel gpu-check \
  --pilot-root pilots/miner/qtc_own_miner \
  --coarse-parity-batch /tmp/qtc_e_parity_reps3.bin \
  --confirmation-parity-batch /tmp/qtc_e_parity_reps6.bin \
  --gpu-binary /tmp/qtc_sim2_batch \
  --parity-artifact /tmp/qtc_e_parity.json
```

There is no public unscored-export command. Only the end-to-end hybrid
optimizer can create a production batch, after verifying both parity modes;
it batches every beam level and the 64-seed confirmation while retaining
branching/recolor/pruning on CPU:

```sh
python3.11 -m miner.qtc_e_model.hybrid_accel hybrid-search \
  --pilot-root pilots/miner/qtc_own_miner \
  --gpu-binary /tmp/qtc_sim2_batch \
  --coarse-parity-batch /tmp/qtc_e_parity_reps3.bin \
  --confirmation-parity-batch /tmp/qtc_e_parity_reps6.bin \
  --parity-artifact /tmp/qtc_e_parity.json \
  --archive-dir /tmp/qtc-e-run-001 \
  --result /tmp/qtc-e-result.json \
  --depth 4 --width 8 --branch-budget 48 --seeds 1
```

Coarse layers use the optimizer's authoritative `reps=3`; the 64-seed leaders
and baseline use `reps=6`. The whole search has one non-extendable 30-minute
deadline, including CPU branching and every GPU call; an individual command
also has a 15-minute ceiling. Subprocesses run in a new process group and the
entire group receives TERM, then KILL, on timeout, interrupt, or parent error.
The command emits real progress after every completed batch and a liveness
heartbeat every 15 seconds while a child is running.

Every input batch and output score file stays under `--archive-dir`; a
`manifest.json` records their SHA-256 hashes, dimensions, `reps`, the exact
binary/model and authoritative reps=3+reps=6 parity bindings, and terminal
`complete` or `failed` status.
Non-finite CUDA output, malformed dimensions, trailing/truncated bytes,
invalid trace fields, or stored offsets inconsistent with the seeds all fail
closed before a result can enter beam selection.

## Self-contained image

[`rental_entrypoint.sh`](rental_entrypoint.sh) is the only container
entrypoint. Its base is `nvidia/cuda:12.6.3-devel-ubuntu24.04`; the equivalent
build recipe is [`rental_image.Dockerfile`](rental_image.Dockerfile). Python 3
is part of the immutable image: the Docker route installs it at build time;
the crane route appends a pinned Linux x86_64 standalone CPython under
`/opt/python`. The entrypoint makes no package-manager or network call and
fails before computation if Python or nvcc is absent. The payload is:

- `source.tar.gz`: `git archive` of `miner/qtc_e_model` at one exact commit;
- `pilot-model.tar.gz`: the ignored authoritative `qtc_own_miner/s10` tree
  plus `qtc_own_miner/s6/s7/mining_n1n3s3e.cu`. Preserve their repository
  paths under `pilots/miner/qtc_own_miner/...`; S10's `ref.py` loads that
  kernel source while `gen -> ana -> frame -> ref` is imported, so an S10-only
  archive is invalid. The complete S10 tree is also required because `gen.py`
  loads `tools/guard.py`, `tools/sass_ir.py`, and `tools/lat_sm70.json` as
  active schedule/control inputs;
- the entrypoint from the same commit and `SHA256SUMS` covering all three;
- `PACKAGE.sha256`, the SHA-256 of `SHA256SUMS` itself.

Prepare the model archive from the pilot repository root; do not replace the
two explicit inputs with a glob or an S10-only archive:

```sh
tar -czf pilot-model.tar.gz \
  pilots/miner/qtc_own_miner/s10 \
  pilots/miner/qtc_own_miner/s6/s7/mining_n1n3s3e.cu
sha256sum source.tar.gz pilot-model.tar.gz rental_entrypoint.sh >SHA256SUMS
sha256sum SHA256SUMS >PACKAGE.sha256
```

The entrypoint verifies `PACKAGE.sha256` and then every entry in `SHA256SUMS`
before extraction, requires both the S10 simulator and its S6/S7 kernel source,
and publishes the package digest with the results. The
reviewed image digest binds that package layer externally; no runtime env
value can substitute another package or create brittle runner coupling.

At startup the entrypoint verifies checksums, extracts into a new work area,
starts a read-only HTTP endpoint on 21559, builds the CUDA scorer, produces the
authoritative parity attestation, and runs the hybrid search. Logs and all
artifacts are under `/var/log/qtc-e`; HTTP paths are `/run.log`, `/failed`,
`/result.json`, `/run/manifest.json`, and `/done`. It writes `/done` atomically
only after `manifest.json` says `complete` and the artifact checksum inventory
has been written. That inventory deliberately excludes the live HTTP access
log, combined run log, timeout log, `/done`, and `/failed`; all included model,
parity, batch, result, and manifest files are closed and stable. Failure is
never retried; `/failed` is written and the
container idles so atlas can collect evidence and terminate the rental. The
container contains no platform client and makes no project/API request.

## Atlas-only Clore handoff

The coordinator performs this section on `atlas`; the package itself never
calls Clore. Do not borrow a current-fleet order and do not select a
server with projected fleet margin above `$1/day`. Exclude banned servers and
Clore owner `570`. A candidate must be a new pilot rental selected from the
collector's current market data.

1. Freeze `<COMMIT>` and prepare the three payload files above from that exact
   commit. Record their SHA-256 values; never package an uncommitted checkout.
2. Require an immutable project-owned CUDA-devel qtcbench image in the form
   `ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<40-hex>@sha256:<64-hex>`.
   Build with the checked Docker recipe, or append both the prepared payload
   and pinned standalone-CPython layers to the pinned CUDA base with crane.
   Set this entrypoint, then inspect and record the resulting digest. A mutable
   tag, plain CUDA image, runtime Python install, or digest mismatch is a no-go.
3. The current primary candidate is Clore `69333` (snapshot: owner 4727, one
   RTX 4090 410 W, reliability 1.0, `below_roi`, about `$8.84/day`). Immediately
   before placement, re-read the collector snapshot and recheck: still free,
   same owner/card, no ban, no current order, fleet margin not above `$1/day`,
   and projected 35-minute spend within the pilot budget. Any mismatch aborts.
4. Use the versioned runner at repository path
   `miner/qtc_e_model/rental_runner/`, which is
   `~/mining/miner/qtc_e_model/rental_runner/` on atlas. Require
   `RUNNER_SHA256SUMS` digest
   `28319af18dee3c1c2f477f7e90f1e6d7dcf57500ae9f7543eedeb84582d9842c`;
   the runner verifies that inventory and records it in `plan.json` and
   `state.json`. From `~/mining`, first produce the no-action plan, review it,
   then use a new output directory and add `--yes`:

   ```sh
   python3.11 miner/qtc_e_model/rental_runner/qtc_rental.py \
     --project-root "$PWD" \
     --server 69333 --expected-owner 4727 \
     --runner-inventory-sha256 28319af18dee3c1c2f477f7e90f1e6d7dcf57500ae9f7543eedeb84582d9842c \
     --image 'ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<40hex>@sha256:<64hex>' \
     --output /tmp/qtc-e-atlas-plan
   ```

   The exact reviewed contract is in
   [`rental_runner/README.md`](rental_runner/README.md). Do not use a claimed
   `ops.test_order --image` CLI that does not exist. The runner's only
   placement call is public `ops.market_actions.place`: pilot
   `qtc_own_miner`, `until=35m`, `exempt_from=all`, the exact immutable image,
   `coin="qtc"`, and port 21559. It validates the image tag/digest and plan
   before the explicit `--yes` path. It does not import a platform client,
   request SSH, or use an existing rental.
5. Wait for the collector to bind the returned order to pilot `qtc_own_miner`
   and discover the mapped 21559 endpoint from that order snapshot. Never
   guess or reuse an endpoint. Poll `/run.log`, `/failed`, and `/done`; collect
   `/result.json`, `/parity.json`, `/ARTIFACTS.sha256`, and the complete `/run/`
   tree before teardown.
6. A coordinator-side watchdog checks every 15 seconds: pilot still owns the
   exact order, elapsed/cost remain under 30 minutes/`$1.00`, and either the
   HTTP log size or a search heartbeat/progress event advances. Any
   violation enters the same `finally` cleanup path.
7. Cleanup is mandatory on success, error, timeout, or interrupt: first
   `python3.11 -m ops.drop <server> --platform clore --order <id>
   --by qtc-e-rental --reason "rental calculation complete" --no-ban`, then
   `python3.11 -m ops.pilots end qtc_own_miner <server> --by qtc-e-rental`.
   Verify
   through `ops.pilots list` and the collector snapshot that neither an open
   pilot row nor a billable order remains.

Vast remains unsupported for this run: its public order action cannot yet
select this custom immutable compute image. This is fail-closed and is not a
reason to bypass the action layer. Clore is the primary supported path.
