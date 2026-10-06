# QTC E/V100: fused linear reduction

This is the model-only successor to ignored pilot `s15`. It neither inspects
QP nor touches a GPU, network, image, miner wiring, or production.

## Objective and decision

Owner: `codex-01a10d3e › qtc-e-carry`, branch `pilots/qtc-e-carry`.

The paired objective is accepted work `E/D` at the V100 power cap. The
existing model gate is central `>=1.015` and one-sided `LCB95>=1.010`; the new
product goal is an independently confirmed `>=1.04`. QP is outside this
optimizer: E must first beat D in a randomized paired test.

The best feasible model point, **E2**, is central `1.04266 E/D` and LCB95
`1.03085`. It passes the existing model gate and clears +4% centrally, but it
does **not** establish +4% with 95% confidence. The decision is therefore
MODEL-GO for continued offline/SASS validation, not a performance claim and
not yet authorization for a GPU test.

## Arithmetic change

D materializes `c=r3+k` with `IMAD.X` before subtracting it. E carries the
wide-product predicate through the linear output fold:

```text
lo - r3 - k = lo + ~r3 + !k              (mod 2^32)
hi' = hi + k - borrow
```

The first `IADD3.X` consumes inverted `k` and emits no-borrow; the second
consumes no-borrow and `k`. No replacement ALU instruction is introduced.
The identity requires the reachable bound `r3+k <= 0xffffffff`; the pure
model rejects the wider wrapping case.

E2 applies this identity to all 12 partial outputs and full-round outputs
`0,4,5,6,7,8,9,10,11`. The full mask was selected by best-first one-bit
acquisition, not by permuting the old 60 forms or by a Cartesian grid.

## D -> E by process and resource

Partial round, reconstructed from the real S10 IR:

| process | D instr | E instr | D F/A | E F/A | effect |
|---|---:|---:|---:|---:|---|
| control/head | 6 | 6 | 2/4 | 2/4 | unchanged |
| row sum | 15 | 15 | 0/30 | 0/30 | unchanged A chain |
| x7 S-box | 42 | 42 | 88/68 | 88/68 | unchanged |
| S/E formation | 8 | 8 | 4/12 | 4/12 | unchanged |
| element products | 60 | 60 | 240/120 | 240/120 | unchanged W/F work |
| element add/fold ALU | 51 | 51 | 0/102 | 0/102 | unchanged |
| carry materialization | 12 | **0** | 24/0 | **0/0** | removed |
| **total** | **194** | **182** | **358/336** | **334/336** | -12 instr, -24 F |

The directed scheduler optimizer used 331 evaluations from three starts. On
64 deterministic simulator seeds it moves the partial body from raw `379.19`
to `363.72` cycles. After normalizing against D's simulator bias, the modeled
cycle value is `363.745`; this closes 74.3% of the gap between the default E
schedule and the certified `358.4` resource floor.

Full round:

| process | D instr | E2 instr | D F/A | E2 F/A |
|---|---:|---:|---:|---:|
| 12 x7 S-boxes | 492 | 492 | 984/816 | 984/816 |
| M4/setup | 90 | 90 | 42/138 | 42/138 |
| column sums | 12 | 12 | 0/24 | 0/24 |
| output sum/reduction | 84 | **75** | 72/144 | **54/144** |
| constants + control | 20 | 20 | 0/8 | 0/8 |
| **total** | **698** | **689** | **1098/1130** | **1080/1130** |

Best-first acquisition evaluated 90 of the 4096 masks; the top candidates
were rechecked with 64 seeds. Mask `4081` gives raw `1248.212` vs D
`1256.583`, normalized to `1248.629`. This is the best observed feasible
schedule, **not** a proof of the globally best schedule.

Whole-kernel resources remain within 64 registers, 6 predicates, 6
scoreboards and zero spills. E2 removes 663 instructions per nonce in the
weighted model and lowers modeled energy from `509.000` to `503.630 nJ`.
The multiplier is deliberately 15, not 16: the generated L body has 15 common
full slots while its special `lin8` path remains D.

## Forecast, bounds, and remaining gap

| point | full cycles | partial cycles | physics E/D | residual-v2 E/D | LCB95 |
|---|---:|---:|---:|---:|---:|
| E2 best feasible | 1248.629 | 363.745 | 1.03751 | **1.04266** | **1.03085** |
| + partial resource floor | 1248.629 | 358.400 | 1.04259 | 1.04845 | 1.03639 |
| + full resource floor | 1205.333 | 358.400 | 1.05692 | 1.06481 | 1.05191 |
| + all 12 full fusions | 1205.333 | 358.400 | 1.05713 | **1.06505** | **1.05213** |

The last row is a certified resource upper bound over all 24 binary fusion
choices. Each fusion removes one F-pipe instruction and energy while leaving
A work unchanged; monotonic branch-and-bound therefore closes the root box
without enumerating `2^24` assignments. The `120/128` mixing ceiling supplies
the cycle floors. Residual-v2 numbers around that bound are forecasts, not
part of the arithmetic proof.

The remaining corrected gap from E2 to the bound is 2.147%. Most of it is the
full-round schedule gap (~1.64 percentage points of corrected E/D), then the
partial schedule gap (~0.58 pp); the last three full fusions contribute only
~0.024 pp through the energy model. This makes full-round scheduling the next
model-only target.

## Confidence and useful data

Residual v2 is frozen from S15: 18 paired observations in six independent
groups, coefficient `0.1343555`, covariance `0.00289469`, model log-SD
`0.00419068`, candidate log-SD `0.0035`. Leave-one-group-out central forecasts
are `1.04146..1.04342`, with LCB95 `1.02909..1.03181`; every omission retains
the existing GO decision.

Variance shares at E2 are 52.1% historical residual, 36.3% candidate transfer,
and 11.6% coefficient estimation. The realistic model-only reductions are:

| evidence change | projected LCB95 |
|---|---:|
| current | 1.03085 |
| candidate SD 0.0035 -> 0.002 via exact SASS/static evidence | 1.03239 |
| twice as much independent paired history | 1.03290 |
| both | 1.03484 |

More D-only telemetry helps normalize clocks, temperature and throttling, but
cannot identify E/D. The most valuable additional inputs are independent
paired histories carrying the same arithmetic/schedule features, exact target
SASS register/spill/critical-path evidence, and—only after the model decision—a
randomized D/E ABBA with clock, power and temperature telemetry. Repeated runs
on one unchanged host are not independent historical groups.

## Offline verification and reproduction

The committed combined body passes:

- 8 control paths / 14,162 instructions, zero static latency, scoreboard,
  seam or reuse problems;
- 20,000 full-slot states, zero bit differences from the reference and zero
  differences from ptxas on 2,000 sampled comparisons;
- 20,000 states x 22 partial rounds, zero bit differences and zero ptxas-loop
  differences on 2,000 sampled comparisons.

Pure reports and tests:

```sh
python3.11 -m miner.qtc_e_model.fused_reduction
python3.11 -m miner.qtc_e_model.optimizer_v3
python3.11 -m pytest -q miner/qtc_e_model
```

With the ignored S10 pilot present, regenerate and verify the body:

```sh
python3.11 -m miner.qtc_e_model.fused_reduction \
  --pilot-root pilots/miner/qtc_own_miner \
  --write-sass /tmp/frag_partial_E_fused.sass
python3.14 pilots/miner/qtc_own_miner/s10/frag/check.py \
  /tmp/frag_partial_E_fused.sass
python3.14 pilots/miner/qtc_own_miner/s10/frag/verify.py \
  /tmp/frag_partial_E_fused.sass 20000
python3.14 pilots/miner/qtc_own_miner/s10/frag/verify_partial.py \
  /tmp/frag_partial_E_fused.sass 20000
```

Before any GPU use: assemble with S10, confirm the same resource ceilings in
real SASS, and pass the existing 33,554,432-nonce correctness gate. Only then
can a short randomized D/E ABBA be considered.
