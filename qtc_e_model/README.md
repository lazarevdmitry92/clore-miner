# QTC E/V100: fused carry correction

This is the model-only successor to ignored pilot `s15`. It does not inspect
QP and does not touch a GPU, network, image, miner wiring, or production.

## Owner and objective

Owner: `codex-01a10d3e › qtc-e-carry`, branch `pilots/qtc-e-carry`.

The paired objective is accepted work `E/D` at the V100 power cap. Promotion
requires central `>=1.015` and one-sided model `LCB95>=1.010`. QP is not part
of this optimization: a promoted E must later beat D in ABBA, and only then
enter the separate product crossover.

## Reproduced S15 anchors

The unmodified S15 scripts were rerun before this step:

- D full round: `F/A=1098/1130`, `sim2=1255.93`, 698 instructions;
- current 60-form family: exact B&B over `2^60`, central upper `1.016677`,
  LCB95 upper `1.008953`, so it cannot pass both gates;
- residual model v2: 18 observations in 6 independent groups; group-CV
  RMSE `0.388%`, MAE `0.299%`, worst `0.825%`, coverage `88.9%`.

## New arithmetic

For every partial-round element, D first materializes `c=r3+k` with
`IMAD.X`, then subtracts `c` from the low word. E uses the identity

```text
lo - r3 - k = lo + ~r3 + !k              (mod 2^32)
hi' = hi + k - borrow
```

The first `IADD3.X` consumes the inverted wide-carry predicate and emits the
no-borrow predicate. The second consumes no-borrow and `k`. This removes one
F-pipe instruction from every output without moving it to ALU.

The identity also relies on the reachable bound `r3+k <= 0xffffffff`. The
generator verifier checks edge states from the real partial-round domain; the
pure model rejects the wider, unreachable wrapping case instead of silently
claiming equivalence there.

The candidate fuses all 12 partial outputs and deliberately leaves the full
round as D: fusing all 12 full outputs reduced resources but made the current
heuristic schedule worse (`1255.93 -> 1268.97`), so that piece is not in E.

## Bounds and decision

The real S10 allocator/scheduler, eight `sim2` seeds, gives:

| quantity | D | E partial fusion |
|---|---:|---:|
| partial F/A | 358 / 336 | **334 / 336** |
| partial sim2 | 399.00 | **379.24** |
| instructions | 194 | **182** |
| partial-body registers / predicates / scoreboards / spills | — | **60 / 5 / 1 / 0** |
| kernel energy model | 509.00 nJ | **504.72 nJ** |
| physical E/D before residual calibration | 1 | **1.01990** |
| residual-v2 corrected E/D | 1 | **1.02260** |
| residual-v2 LCB95 | — | **1.01152** |

The feasible model candidate therefore passes the model gate. This is not a
GPU performance claim. The combined body remains at D's whole-kernel ceiling
of 64 registers, 6 predicates, 6 scoreboards and zero spills.

The generated combined body passed the independent S10 checks: eight control
paths / 14,218 instructions with zero latency, scoreboard, seam or reuse
problems; 20,000 states x 22 partial rounds (one quarter edge-only) with zero
bit differences from the reference and zero differences from the ptxas loop
on the 2,000 sampled comparisons.

The central and lower-bound decision uses the actual S15 residual-v2 contract,
not a manual arm-noise approximation: coefficient `0.1343555`, covariance
`0.00289469`, model log-SD `0.00419068`, plus candidate log-SD `0.0035`.

For the 24 binary fusion choices, every choice removes one F-pipe `IMAD.X`
and 8.1 pJ while leaving A work unchanged. Full D is A-bound. Partial reaches
its A floor after 11 choices and the twelfth still lowers energy. Monotonicity
therefore certifies the resource optimum at all 24 choices without enumerating
`2^24` assignments. With the measured `120/128` mixing ceiling its upper
physical upper bound is `1.05834`. Its residual-v2 projection is reported
separately by the tool; it is a calibrated forecast around the resource bound,
not part of the mathematical certificate.

## Reproduction and next gates

Pure model and unit tests:

```sh
python3.11 -m miner.qtc_e_model.fused_reduction
python3.11 -m pytest -q miner/qtc_e_model/test_fused_reduction.py
```

With the ignored S10 pilot present, generate a combined D-full/E-partial SASS
body and run its existing static and bit-exact checks:

```sh
python3.11 -m miner.qtc_e_model.fused_reduction \
  --pilot-root pilots/miner/qtc_own_miner \
  --write-sass /tmp/frag_partial_E_fused.sass
python3.14 pilots/miner/qtc_own_miner/s10/frag/check.py \
  /tmp/frag_partial_E_fused.sass
python3.14 pilots/miner/qtc_own_miner/s10/frag/verify_partial.py \
  /tmp/frag_partial_E_fused.sass 20000
```

Before GPU use: run `check.py` on all loop paths, assemble with S10, confirm
the same resource ceilings in real SASS, and run the existing 33,554,432-nonce
correctness gate. Only then is a short D/E ABBA justified.
