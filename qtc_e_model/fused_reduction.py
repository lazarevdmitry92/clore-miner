#!/usr/bin/env python3
"""Model and generate the V100 QTC E fused-reduction candidate.

The candidate removes the temporary ``c = r3 + k`` instruction from every
partial-round output fold and selected full-round linear outputs.  The
low-word subtraction consumes ``!k`` directly and produces the no-borrow
predicate used by the high word.  This is an arithmetic change plus a
separately optimized schedule, not a permutation of D's arithmetic forms.

The pure functions and report are self-contained.  ``--write-sass`` is an
optional adapter for the ignored S10 pilot: it patches the existing IR builder
in memory, generates the candidate with the real scheduler/allocator and
writes a body that the existing verifier can consume.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
from pathlib import Path
import sys


M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
EPS = M32
MIXING_CEILING = 120.0 / 128.0
Z95 = 1.959963984540054


@dataclasses.dataclass(frozen=True)
class Anchor:
    measured_t: float = 287.5
    energy_nj: float = 509.0
    full_f: float = 1098.0
    full_a: float = 1130.0
    full_cycles: float = 1257.0
    partial_f: float = 358.0
    partial_a: float = 336.0
    partial_cycles: float = 399.0
    # The generated L body has 15 common full slots plus the special lin8
    # path.  E changes only the common body; lin8 remains D.
    full_repeats: int = 15
    partial_repeats: int = 44
    phase_scale: float = 128.0
    max_clock_mhz: float = 1530.0
    static_power_w: float = 83.3
    power_cap_w: float = 289.0
    voltage_gamma: float = 1.93


@dataclasses.dataclass(frozen=True)
class Candidate:
    name: str
    fused_full_outputs: int
    fused_partial_outputs: int
    full_cycles: float
    partial_cycles: float
    full_f: float
    full_a: float
    partial_f: float
    partial_a: float
    instructions_removed_per_nonce: int
    registers: int
    predicates: int
    scoreboards: int
    spills: int
    energy_nj: float
    projected_t: float
    physics_ed: float
    corrected_ed: float
    lcb95_ed: float
    residual_log_sd: float
    passes_gate: bool


@dataclasses.dataclass(frozen=True)
class BoundCertificate:
    variables: int
    assignments: int
    exact_resource_optimum: bool
    fused_full_outputs: int
    fused_partial_outputs: int
    full_cycle_floor: float
    partial_cycle_floor: float
    physics_ed_upper: float
    corrected_ed_upper: float
    corrected_lcb95_upper: float
    candidate_to_corrected_upper_gap: float


@dataclasses.dataclass(frozen=True)
class Report:
    objective: str
    arithmetic_identity: str
    candidate: Candidate
    certificate: BoundCertificate
    decision: str


def fold_words_materialized(low64: int, high32: int, r3: int = 0) -> int:
    """D's fold after explicitly materializing ``c = r3 + k``.

    ``low64 + high32 * EPS`` is the result of IMAD.WIDE.  Its carry ``k``
    and ``r3`` are bounded so their sum fits one word in the real kernels.
    """
    wide = (low64 & M64) + (high32 & M32) * EPS
    w = wide & M64
    k = wide >> 64
    c = (r3 + k) & M32
    lo = w & M32
    hi = w >> 32
    borrow = int(lo < c)
    return (((hi + k - borrow) & M32) << 32) | ((lo - c) & M32)


def fused_correction(w: int, k: int, r3: int = 0) -> int:
    """Two-IADD3 correction used by E, expressed as integer semantics.

    The low instruction evaluates ``w.lo + ~r3 + !k``.  Its carry predicate
    is exactly ``not borrow``.  The high instruction evaluates
    ``w.hi - 1 + no_borrow + k``.  No register is used for ``r3 + k``.
    """
    if k not in (0, 1):
        raise ValueError("wide carry must be a bit")
    if not 0 <= r3 <= M32 or r3 + k > M32:
        raise ValueError("candidate requires the reachable no-wrap r3+k bound")
    lo = w & M32
    hi = (w >> 32) & M32
    low_total = lo + ((~r3) & M32) + (1 - k)
    out_lo = low_total & M32
    no_borrow = (low_total >> 32) & 1
    out_hi = (hi + M32 + no_borrow + k) & M32
    return out_hi << 32 | out_lo


def fold_words_fused(low64: int, high32: int, r3: int = 0) -> int:
    wide = (low64 & M64) + (high32 & M32) * EPS
    return fused_correction(wide & M64, wide >> 64, r3)


def _frequency_mhz(t: float, energy_nj: float, anchor: Anchor) -> float:
    def power(clock: float) -> float:
        active = energy_nj * 1e-9 * 80 * clock * 1e6 / t
        return (anchor.static_power_w + active) * (clock / anchor.max_clock_mhz) ** anchor.voltage_gamma

    if power(anchor.max_clock_mhz) <= anchor.power_cap_w:
        return anchor.max_clock_mhz
    lo, hi = 1.0, anchor.max_clock_mhz
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if power(mid) <= anchor.power_cap_w:
            lo = mid
        else:
            hi = mid
    return lo


def throughput_ratio(t: float, energy_nj: float, anchor: Anchor = Anchor()) -> float:
    d_clock = _frequency_mhz(anchor.measured_t, anchor.energy_nj, anchor)
    e_clock = _frequency_mhz(t, energy_nj, anchor)
    return (e_clock / t) / (d_clock / anchor.measured_t)


@dataclasses.dataclass(frozen=True)
class ResidualV2:
    """Frozen sm70 fit reproduced from S15 ``model_v2.py`` and history.

    The fit learns the residual around the physical log-ratio from 18 paired
    observations in six independent groups.  Candidate uncertainty is added
    exactly as in ``model_v2.project``; the old hand-written D/E noise shortcut
    is deliberately not part of this contract.
    """

    coefficient: float = 0.13435549676440053
    covariance: float = 0.0028946939923161492
    model_log_sd: float = 0.004190681206093888
    candidate_log_sd: float = 0.0035


def residual_v2_projection(physics_ratio: float, fit: ResidualV2 = ResidualV2()) -> tuple[float, float, float]:
    if not math.isfinite(physics_ratio) or physics_ratio <= 0:
        raise ValueError("physics_ratio must be finite and positive")
    physics_log = math.log(physics_ratio)
    corrected_log = physics_log * (1.0 + fit.coefficient)
    log_sd = math.sqrt(
        physics_log * physics_log * fit.covariance
        + fit.model_log_sd * fit.model_log_sd
        + fit.candidate_log_sd * fit.candidate_log_sd
    )
    corrected = math.exp(corrected_log)
    return corrected, math.exp(corrected_log - Z95 * log_sd), log_sd


def project(
    *,
    fused_full_outputs: int,
    fused_partial_outputs: int,
    full_cycles: float,
    partial_cycles: float,
    registers: int,
    predicates: int,
    scoreboards: int,
    spills: int,
    anchor: Anchor = Anchor(),
    name: str = "E-fused-partial",
) -> Candidate:
    if not 0 <= fused_full_outputs <= 12 or not 0 <= fused_partial_outputs <= 12:
        raise ValueError("fusion counts must be in 0..12")
    full_f = anchor.full_f - 2 * fused_full_outputs
    partial_f = anchor.partial_f - 2 * fused_partial_outputs
    removed = anchor.full_repeats * fused_full_outputs + anchor.partial_repeats * fused_partial_outputs
    energy = anchor.energy_nj - removed * 8.1 / 1000.0
    t = anchor.measured_t + (
        anchor.full_repeats * (full_cycles - anchor.full_cycles)
        + anchor.partial_repeats * (partial_cycles - anchor.partial_cycles)
    ) / anchor.phase_scale
    physics_ratio = throughput_ratio(t, energy, anchor)
    corrected, lower, residual_log_sd = residual_v2_projection(physics_ratio)
    valid = registers <= 64 and predicates <= 6 and scoreboards <= 6 and spills == 0
    return Candidate(
        name=name,
        fused_full_outputs=fused_full_outputs,
        fused_partial_outputs=fused_partial_outputs,
        full_cycles=full_cycles,
        partial_cycles=partial_cycles,
        full_f=full_f,
        full_a=anchor.full_a,
        partial_f=partial_f,
        partial_a=anchor.partial_a,
        instructions_removed_per_nonce=removed,
        registers=registers,
        predicates=predicates,
        scoreboards=scoreboards,
        spills=spills,
        energy_nj=energy,
        projected_t=t,
        physics_ed=physics_ratio,
        corrected_ed=corrected,
        lcb95_ed=lower,
        residual_log_sd=residual_log_sd,
        passes_gate=valid and corrected >= 1.015 and lower >= 1.010,
    )


def build_report(anchor: Anchor = Anchor()) -> Report:
    # Best feasible point found by the bounded search in optimizer_v3.  Raw
    # simulator deltas are normalized to the measured D anchors so the model
    # does not confuse simulator bias with an E/D effect.
    candidate = project(
        fused_full_outputs=9,
        fused_partial_outputs=12,
        full_cycles=1248.6289419421269,
        partial_cycles=363.74544839988715,
        # Both generated bodies remain within D's existing whole-kernel
        # 64/6/6 ceiling.
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        anchor=anchor,
        name="E2-fused-linear",
    )

    # Every fusion removes one F-pipe IMAD.X and 8.1 pJ while leaving A work
    # unchanged.  Full D is already A-bound; partial D reaches A=336 after 11
    # fusions and the twelfth keeps the same floor while reducing energy.
    # Thus all 24 binary choices dominate their zero alternative for the
    # resource/energy upper-bound objective: the root box is certified without
    # enumerating 2**24 assignments.
    full_floor = anchor.full_a / MIXING_CEILING
    partial_floor = anchor.partial_a / MIXING_CEILING
    upper = project(
        fused_full_outputs=12,
        fused_partial_outputs=12,
        full_cycles=full_floor,
        partial_cycles=partial_floor,
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        anchor=anchor,
        name="resource-bound",
    )
    certificate = BoundCertificate(
        variables=24,
        assignments=2**24,
        exact_resource_optimum=True,
        fused_full_outputs=12,
        fused_partial_outputs=12,
        full_cycle_floor=full_floor,
        partial_cycle_floor=partial_floor,
        physics_ed_upper=upper.physics_ed,
        corrected_ed_upper=upper.corrected_ed,
        corrected_lcb95_upper=upper.lcb95_ed,
        candidate_to_corrected_upper_gap=upper.corrected_ed / candidate.corrected_ed - 1.0,
    )
    return Report(
        objective="maximize paired accepted E/D under V100 power cap; residual-v2 corrected>=1.015 and LCB95>=1.010",
        arithmetic_identity="lo-r3-k = lo+~r3+!k; hi'=hi+k-borrow",
        candidate=candidate,
        certificate=certificate,
        decision=(
            "MODEL-GO: bit-exact fused-linear candidate passes the model gate; "
            "central exceeds +4%, LCB95 does not; requires SASS and GPU gates"
        ),
    )


def _patch_pilot(pilot_root: Path):
    frag = pilot_root / "s10" / "frag"
    if not (frag / "gen.py").is_file():
        raise FileNotFoundError(f"S10 frag tools not found under {pilot_root}")
    sys.path.insert(0, str(frag))
    gen = importlib.import_module("gen")
    partial = importlib.import_module("partial")

    full_fused = frozenset((0, 4, 5, 6, 7, 8, 9, 10, 11))
    full_index = 0
    original_out_fold = gen.out_fold

    def selective_full_fold(block, y, s, rcp, chain, out0, out1):
        nonlocal full_index
        index = full_index
        full_index += 1
        if index not in full_fused:
            return original_out_fold(block, y, s, rcp, chain, out0, out1)
        w = gen.Pair("w")
        lane = gen.add3_ll(block, y, s, (rcp.lo, rcp.hi, None), chain, dst=w)
        pk, pb = gen.VP("k"), gen.VP("b")
        block.op(
            "IMAD.WIDE.U32",
            [("dp", w), ("pd", pk), ("s", lane[2]), gen.L(gen.EPS), ("sp", w)],
            "W",
            chain,
        )
        block.op(
            "IADD3.X",
            [
                ("d", out0),
                ("pd", pb),
                ("s", w.lo),
                gen.L("-0x1"),
                gen.L("RZ"),
                ("ps", pk, True),
                ("plit", "!PT"),
            ],
            "A",
            chain,
        )
        block.op(
            "IADD3.X",
            [
                ("d", out1),
                gen.L("RZ"),
                gen.L("-0x1"),
                ("s", w.hi),
                ("ps", pb, False),
                ("ps", pk, False),
            ],
            "A",
            chain,
        )

    def fused_fold_out(block, chain, low_pair, r2, r3, out0, out1, _cform, r3zero=False):
        pk, pb = gen.VP("k"), gen.VP("b")
        block.op(
            "IMAD.WIDE.U32",
            [("dp", low_pair), ("pd", pk), ("s", r2), gen.L(partial.EPSP), ("sp", low_pair)],
            "W",
            chain,
        )
        neg_r3 = gen.L("-0x1") if r3zero else ("s", r3, "~")
        block.op(
            "IADD3.X",
            [
                ("d", out0),
                ("pd", pb),
                ("s", low_pair.lo),
                neg_r3,
                gen.L("RZ"),
                ("ps", pk, True),
                ("plit", "!PT"),
            ],
            "A",
            chain,
        )
        block.op(
            "IADD3.X",
            [
                ("d", out1),
                gen.L("RZ"),
                gen.L("-0x1"),
                ("s", low_pair.hi),
                ("ps", pb, False),
                ("ps", pk, False),
            ],
            "A",
            chain,
        )

    gen.out_fold = selective_full_fold
    partial.fold_out = fused_fold_out
    return gen, partial, importlib.import_module("body_io")


def write_partial_sass(pilot_root: Path, output: Path) -> None:
    gen, partial, body_io = _patch_pilot(pilot_root)
    opts = gen.Opts(lam=0.35, w_imb=4.0, w_stall=1.5, window=2, ldc_max=2, rho=1.0, w_h=0.02, w_chain=3.0)
    full, labels, _blocks = gen.generate("L", lambda _i: ("F",) * 6, opts_s=opts, opts_m=opts)
    partial_opts = gen.Opts(
        lam=0.1,
        w_imb=4.0,
        w_stall=0.5,
        w_h=0.05,
        w_chain=1.0,
        w_press=0.0,
        rho=1.0,
        window=7,
        regs=57,
    )
    loop, _loop_labels, _block = partial.generate(cforms=lambda _i: "F", opts=partial_opts)
    labels = dict(labels, **{".L_x_9": len(full)})
    output.parent.mkdir(parents=True, exist_ok=True)
    body_io.write(
        str(output),
        full + loop,
        labels,
        header=(
            "QTC E2 model candidate: fused carry correction in full outputs "
            "0,4,5,6,7,8,9,10,11 and all 12 partial outputs"
        ),
    )


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sass", type=Path)
    parser.add_argument("--pilot-root", type=Path)
    args = parser.parse_args()
    if args.write_sass:
        if args.pilot_root is None:
            parser.error("--write-sass requires --pilot-root")
        write_partial_sass(args.pilot_root, args.write_sass)
    print(json.dumps(_jsonable(build_report()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
