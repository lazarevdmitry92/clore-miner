# QTC E/V100: fused linear reduction

This is the model-only successor to ignored pilot `s15`. It neither inspects
QP nor touches a GPU, network, image, miner wiring, or production.

## Objective and decision

Owner: `codex-01a10d3e › qtc-e-carry`, branch `pilots/qtc-e-carry`.

The paired objective is accepted work `E/D` at the V100 power cap. The
existing model gate is central `>=1.015` and one-sided `LCB95>=1.010`; the new
product goal is an independently confirmed `>=1.04`. QP is outside this
optimizer: E must first beat D in a randomized paired test.

The best feasible model point, **E3**, is central `1.04625 E/D` and LCB95
`1.03429`. It passes the existing model gate and clears +4% centrally, but it
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

E3 applies this identity to all 12 partial outputs and full-round outputs
`0,2,3,4,5,6,8,9,11`, then makes one dependency-safe adjacent schedule move
and greedily recolors non-pinned live intervals only when bank conflicts fall.
The mask was selected by best-first acquisition, not a Cartesian grid.

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

| process | D instr | E3 instr | D F/A | E3 F/A |
|---|---:|---:|---:|---:|
| 12 x7 S-boxes | 492 | 492 | 984/816 | 984/816 |
| M4/setup | 90 | 90 | 42/138 | 42/138 |
| column sums | 12 | 12 | 0/24 | 0/24 |
| output sum/reduction | 84 | **75** | 72/144 | **54/144** |
| constants + control | 20 | 20 | 0/8 | 0/8 |
| **total** | **698** | **689** | **1098/1130** | **1080/1130** |

Best-first acquisition evaluated 90 masks and the bank-aware stage rechecked
five leading masks with 64 seeds. Mask `2941`, one legal neighbor and nine
live-interval recolors reduce bank conflicts `14 -> 4`; raw cycles are
`1238.459` vs D `1256.583`, normalized to `1238.876`. This is the best
observed feasible schedule, **not** a proof of the globally best schedule.

Whole-kernel resources remain within 64 registers, 6 predicates, 6
scoreboards and zero spills. E3 removes 663 instructions per nonce in the
weighted model and lowers modeled energy from `509.000` to `503.630 nJ`.
The multiplier is deliberately 15, not 16: the generated L body has 15 common
full slots while its special `lin8` path remains D.

## Forecast, bounds, and remaining gap

| point | full cycles | partial cycles | physics E/D | residual-v2 E/D | LCB95 |
|---|---:|---:|---:|---:|---:|
| E3 best feasible | 1238.876 | 363.745 | 1.04066 | **1.04625** | **1.03429** |
| + partial resource floor | 1238.876 | 358.400 | 1.04576 | 1.05209 | 1.03986 |
| + full resource floor | 1205.333 | 358.400 | 1.05692 | 1.06481 | 1.05191 |
| + all 12 full fusions | 1205.333 | 358.400 | 1.05713 | **1.06505** | **1.05213** |

The last row is a certified resource upper bound over all 24 binary fusion
choices. Each fusion removes one F-pipe instruction and energy while leaving
A work unchanged; monotonic branch-and-bound therefore closes the root box
without enumerating `2^24` assignments. The `120/128` mixing ceiling supplies
the cycle floors. Residual-v2 numbers around that bound are forecasts, not
part of the arithmetic proof.

The remaining corrected gap from E3 to the resource bound is 1.796%. E3 still
does not pass the new `LCB95>=1.04` goal: at the current partial schedule it
requires normalized full cycles `<=1222.785`.

For the fixed E3 instruction/control order, even optimistic execution does not
close that gap: deleting every bank penalty gives `1235.954`; hypothetical
9-warps/SMSP plus zero bank penalties gives `1231.939`; even 10 warps plus
zero penalties gives `1228.741`, LCB95 `1.03788`. This is a conservative
blocker for the current recolor/occupancy family only; it does not bound a new
instruction order or new arithmetic. The next optimizer must jointly choose
the scheduler and bank-aware allocator. A whole-kernel recolor to declared
`<=56` registers is the next occupancy gate; until proven, E2 remains fallback
and E3 is not promoted.

The first joint local frontier is exhausted: all 461 dependency- and
physical-register-safe adjacent neighbors around bank-aware E3 were evaluated,
then the five leaders were confirmed with 64 seeds. The best moved raw cycles
only `1238.459 -> 1238.405` and LCB95 `1.03429 -> 1.03431`; this immaterial
change is not a new incumbent. The next search must use multi-operation
relocate/beam nodes jointly with recoloring, not another single-swap pass.

## Confidence and useful data

Residual v2 is frozen from S15: 18 paired observations in six independent
groups, coefficient `0.1343555`, covariance `0.00289469`, model log-SD
`0.00419068`, candidate log-SD `0.0035`. Leave-one-group-out central forecasts
are `1.04495..1.04708`, with LCB95 `1.03241..1.03518`; every omission retains
the existing GO decision.

Variance shares at E3 are 51.0% historical residual, 35.6% candidate transfer,
and 13.4% coefficient estimation. The realistic model-only reductions are:

| evidence change | projected LCB95 |
|---|---:|
| current | 1.03429 |
| candidate SD 0.0035 -> 0.002 via exact SASS/static evidence | 1.03582 |
| twice as much independent paired history | 1.03639 |
| both | 1.03832 |

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

The pilot adapter is idempotent: it stores the native full-output builder once
and every selective mask wraps that native function. A real S10 regression
patches, resets and generates twice in one interpreter and requires identical
`683`-instruction main and `197`-instruction lin8 IR bodies.

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
