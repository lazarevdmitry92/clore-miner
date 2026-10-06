#!/usr/bin/env python3
"""Model and generate the V100 QTC E fused-reduction candidate.

The candidate removes the temporary ``c = r3 + k`` instruction from every
partial-round output fold.  The low-word subtraction consumes ``!k``
directly and produces the no-borrow predicate used by the high word.  This is
an arithmetic change, not a rescheduling of D.

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
    full_repeats: int = 16
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
    central_ed: float
    lcb95_ed: float
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
    central_ed_upper: float
    lcb95_ed_upper: float
    candidate_to_upper_gap: float


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


def lcb95(ratio: float, d_sd: float = 0.0017, e_sd: float = 0.0035) -> float:
    return math.exp(math.log(ratio) - Z95 * math.hypot(d_sd, e_sd))


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
    ratio = throughput_ratio(t, energy, anchor)
    lower = lcb95(ratio)
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
        central_ed=ratio,
        lcb95_ed=lower,
        passes_gate=valid and ratio >= 1.015 and lower >= 1.010,
    )


def build_report(anchor: Anchor = Anchor()) -> Report:
    # Real S10 scheduler/allocator result, eight deterministic sim2 seeds.
    candidate = project(
        fused_full_outputs=0,
        fused_partial_outputs=12,
        full_cycles=anchor.full_cycles,
        partial_cycles=379.24000724755894,
        # The fused partial body alone is 60/5/1.  The unchanged D full body
        # fixes the whole combined body at its existing 64/6/6 ceiling.
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        anchor=anchor,
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
        central_ed_upper=upper.central_ed,
        lcb95_ed_upper=upper.lcb95_ed,
        candidate_to_upper_gap=upper.central_ed / candidate.central_ed - 1.0,
    )
    return Report(
        objective="maximize paired accepted E/D under V100 power cap; central>=1.015 and LCB95>=1.010",
        arithmetic_identity="lo-r3-k = lo+~r3+!k; hi'=hi+k-borrow",
        candidate=candidate,
        certificate=certificate,
        decision="MODEL-GO: bit-exact partial fusion passes the model gate; requires SASS and GPU gates",
    )


def _patch_pilot(pilot_root: Path):
    frag = pilot_root / "s10" / "frag"
    if not (frag / "gen.py").is_file():
        raise FileNotFoundError(f"S10 frag tools not found under {pilot_root}")
    sys.path.insert(0, str(frag))
    gen = importlib.import_module("gen")
    partial = importlib.import_module("partial")

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

    partial.fold_out = fused_fold_out
    return gen, partial, importlib.import_module("body_io")


def write_partial_sass(pilot_root: Path, output: Path) -> None:
    gen, partial, body_io = _patch_pilot(pilot_root)
    opts = gen.Opts(lam=0.35, w_imb=4.0, w_stall=1.5, window=2, ldc_max=2, rho=1.0, w_h=0.02, w_chain=3.0)
    full, labels, _blocks = gen.generate("L", lambda _i: ("F",) * 6, opts_s=opts, opts_m=opts)
    loop, _loop_labels, _block = partial.generate(cforms=lambda _i: "F")
    labels = dict(labels, **{".L_x_9": len(full)})
    output.parent.mkdir(parents=True, exist_ok=True)
    body_io.write(
        str(output),
        full + loop,
        labels,
        header="QTC E model candidate: D full body + fused carry correction in all 12 partial-round outputs",
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
