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

The encoded CPU reference matches the original S10 `sim2.py` on eight real
seeds within `2.28e-13`. A 4-candidate x 4-seed binary export/check round trip
is exact. CUDA parity is a mandatory runtime gate (`<=1e-12`) before any full
search; a failure exits non-zero.

The conservative scorer model is 5,000--15,000 candidate-seed jobs/s. At the
profiled 8-seed mix that is 5.6x end-to-end; confirmation batches at 64 seeds
are expected to gain roughly 25--35x, while branch-heavy one-seed layers stay
CPU-bound. For the complete mixed optimizer, budget on 2--5x until measured.
The latency model breaks even at one job under a 20 ms transfer/launch envelope,
but use at least 256 jobs per batch for stable GPU occupancy.

## Rental specification and budget gate

- Preferred: one RTX 4090 24 GB, CUDA 12.x, at least 24 vCPU and 32 GB RAM.
- Acceptable fallback: L40S 48 GB with the same CPU/RAM floor.
- Disk: 10 GB. No target V100 is required: this GPU only evaluates the V100
  simulator state machine.
- Price ceiling: `$2.00/hour`; planned reservation 30 minutes; hard compute
  budget `$1.00`. The coordinator must record live price and planned duration
  before launch and must not take a profitable current-fleet server.
- Expected: build+parity 5--10 minutes; full three-start search 5--15 minutes;
  hard timeout 30 minutes. Watchdog interval: 15 seconds.

## Reproducible launch

From the repository root on the rental host:

```sh
python3.11 -m pytest -q miner/qtc_e_model
nvcc -O3 -std=c++17 miner/qtc_e_model/sim2_batch.cu -o /tmp/qtc_sim2_batch
python3.11 -m miner.qtc_e_model.hybrid_accel export \
  --pilot-root pilots/miner/qtc_own_miner \
  --batch /tmp/qtc_e_parity.bin --candidates 16 --seeds 64
python3.11 -m miner.qtc_e_model.hybrid_accel gpu-check \
  --batch /tmp/qtc_e_parity.bin --gpu-binary /tmp/qtc_sim2_batch
```

Only after `parity=PASS`, export an unscored production frontier and score it:

```sh
python3.11 -m miner.qtc_e_model.hybrid_accel export \
  --pilot-root pilots/miner/qtc_own_miner \
  --batch /tmp/qtc_e_frontier.bin --candidates 256 --seeds 64 --unscored
/tmp/qtc_sim2_batch /tmp/qtc_e_frontier.bin /tmp/qtc_e_scores.bin
```

The end-to-end hybrid optimizer batches every beam level and the 64-seed
confirmation while retaining branching/recolor/pruning on CPU:

```sh
python3.11 -m miner.qtc_e_model.hybrid_accel hybrid-search \
  --pilot-root pilots/miner/qtc_own_miner \
  --gpu-binary /tmp/qtc_sim2_batch \
  --depth 4 --width 8 --branch-budget 48 --seeds 1
```

The Python GPU calls emit a liveness heartbeat every 15 seconds and enforce a
15-minute command timeout. The coordinator additionally applies the cost
watchdog. A missing heartbeat, CUDA error, parity failure, 30-minute pilot
timeout, or `$1.00` projected spend stops the run. Production score files are
returned for CPU-side beam selection and archived with the batch hash.
