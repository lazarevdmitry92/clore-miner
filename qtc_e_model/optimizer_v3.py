#!/usr/bin/env python3
"""Reproducible D -> E process model and bounded logical search report.

The optimizer has two distinct contracts.  ``best_first_masks`` searches the
discrete fusion graph by admissible one-bit acquisitions; ``pattern_search``
does directed coordinate/pattern moves in scheduler space.  Neither is a
Cartesian parameter grid.  The committed result records the expensive S10
simulator observations; the small pure functions make the search logic
independently testable without the ignored pilot or a GPU.
"""

from __future__ import annotations

import dataclasses
import heapq
import json
import math
from collections.abc import Callable, Iterable, Mapping

from .fused_reduction import Anchor, ResidualV2, Z95, build_report, project


FULL_MASK = 0b111111110001  # outputs 0,4,5,6,7,8,9,10,11
FULL_INDICES = (0, 4, 5, 6, 7, 8, 9, 10, 11)


@dataclasses.dataclass(frozen=True)
class ProcessCost:
    process: str
    d_instructions: int
    e_instructions: int
    d_f: int
    e_f: int
    d_a: int
    e_a: int


PARTIAL_PROCESSES = (
    ProcessCost("control/head", 6, 6, 2, 2, 4, 4),
    ProcessCost("row sum", 15, 15, 0, 0, 30, 30),
    ProcessCost("x7 S-box", 42, 42, 88, 88, 68, 68),
    ProcessCost("S/E formation", 8, 8, 4, 4, 12, 12),
    ProcessCost("element products", 60, 60, 240, 240, 120, 120),
    ProcessCost("element add/fold ALU", 51, 51, 0, 0, 102, 102),
    ProcessCost("carry materialization", 12, 0, 24, 0, 0, 0),
)

FULL_PROCESSES_D = (
    ProcessCost("12 x7 S-boxes", 492, 492, 984, 984, 816, 816),
    ProcessCost("M4/setup", 90, 90, 42, 42, 138, 138),
    ProcessCost("column sums", 12, 12, 0, 0, 24, 24),
    ProcessCost("output sum/reduction", 84, 75, 72, 54, 144, 144),
    ProcessCost("round constants", 12, 12, 0, 0, 0, 0),
    ProcessCost("control", 8, 8, 0, 0, 8, 8),
)


@dataclasses.dataclass(frozen=True)
class SearchEvidence:
    method: str
    evaluations: int
    search_space: int
    globally_proven_schedule_optimum: bool
    best_mask: int
    best_indices: tuple[int, ...]
    d_raw_cycles: float
    e_raw_cycles: float
    normalized_cycles: float


@dataclasses.dataclass(frozen=True)
class LadderPoint:
    name: str
    full_cycles: float
    partial_cycles: float
    full_fusions: int
    physics_ed: float
    corrected_ed: float
    lcb95_ed: float


def sum_processes(rows: Iterable[ProcessCost], side: str) -> tuple[int, int, int]:
    if side not in ("d", "e"):
        raise ValueError("side must be d or e")
    return tuple(sum(getattr(row, f"{side}_{field}") for row in rows) for field in ("instructions", "f", "a"))


def best_first_masks(width: int, budget: int, evaluate: Callable[[int], float]) -> list[tuple[int, float]]:
    """Minimize ``evaluate`` by acquiring one bit from promising masks.

    The queue uses the observed objective, not an enumeration order.  A
    separate monotone resource proof supplies the global upper bound: this
    finite-budget acquisition search only establishes feasible points.
    """
    if width < 1 or budget < 1:
        raise ValueError("width and budget must be positive")
    seen = {0}
    score0 = evaluate(0)
    queue = [(score0, 0)]
    observations = [(0, score0)]
    while queue and len(observations) < budget:
        _score, mask = heapq.heappop(queue)
        for bit in range(width):
            child = mask | (1 << bit)
            if child in seen:
                continue
            seen.add(child)
            score = evaluate(child)
            observations.append((child, score))
            heapq.heappush(queue, (score, child))
            if len(observations) == budget:
                break
    return observations


def pattern_search(
    initial: Mapping[str, float],
    steps: Mapping[str, float],
    evaluate: Callable[[Mapping[str, float]], float],
    rounds: int = 8,
) -> tuple[dict[str, float], float, int]:
    """Deterministic coordinate/pattern descent, with shrinking steps."""
    point = dict(initial)
    best = evaluate(point)
    evaluations = 1
    deltas = dict(steps)
    direction = {key: 0 for key in point}
    for _ in range(rounds):
        improved = False
        for key in point:
            for sign in (direction[key], -1, 1):
                if sign == 0:
                    continue
                candidate = dict(point)
                candidate[key] += sign * deltas[key]
                score = evaluate(candidate)
                evaluations += 1
                if score < best:
                    point, best = candidate, score
                    direction[key] = sign
                    improved = True
                    break
        if not improved:
            deltas = {key: value / 2.0 for key, value in deltas.items()}
    return point, best, evaluations


def _point(name: str, full_cycles: float, partial_cycles: float, full_fusions: int) -> LadderPoint:
    candidate = project(
        fused_full_outputs=full_fusions,
        fused_partial_outputs=12,
        full_cycles=full_cycles,
        partial_cycles=partial_cycles,
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        name=name,
    )
    return LadderPoint(
        name,
        full_cycles,
        partial_cycles,
        full_fusions,
        candidate.physics_ed,
        candidate.corrected_ed,
        candidate.lcb95_ed,
    )


def gap_ladder(anchor: Anchor = Anchor()) -> tuple[LadderPoint, ...]:
    return (
        _point("best-feasible", 1248.6289419421269, 363.74544839988715, 9),
        _point("partial-resource-floor", 1248.6289419421269, 358.4, 9),
        _point("full-and-partial-resource-floors", 1130 / (120 / 128), 358.4, 9),
        _point("certified-resource-bound", 1130 / (120 / 128), 358.4, 12),
    )


def residual_variance(physics_ratio: float, fit: ResidualV2 = ResidualV2()) -> dict[str, float]:
    physics_log = math.log(physics_ratio)
    parts = {
        "coefficient": physics_log * physics_log * fit.covariance,
        "model": fit.model_log_sd * fit.model_log_sd,
        "candidate": fit.candidate_log_sd * fit.candidate_log_sd,
    }
    total = sum(parts.values())
    return {**parts, "total": total, "log_sd": math.sqrt(total)}


def lcb_with_sd(corrected_ratio: float, log_sd: float) -> float:
    return math.exp(math.log(corrected_ratio) - Z95 * log_sd)


LOO_SENSITIVITY = {
    "without-s10-a": (1.04145779, 1.02909020),
    "without-s10-abc": (1.04229905, 1.03037014),
    "without-s14": (1.04300199, 1.03123907),
    "without-s6-b": (1.04298926, 1.03104673),
    "without-s6-c1": (1.04342207, 1.03142545),
    "without-s6-n": (1.04287239, 1.03180684),
}


def build_optimizer_report() -> dict[str, object]:
    report = build_report()
    ladder = gap_ladder()
    variance = residual_variance(report.candidate.physics_ed)
    sensitivity = {
        "current": report.candidate.lcb95_ed,
        "candidate_sd_0.002": lcb_with_sd(
            report.candidate.corrected_ed,
            math.sqrt(variance["coefficient"] + variance["model"] + 0.002**2),
        ),
        "double_independent_history": lcb_with_sd(
            report.candidate.corrected_ed,
            math.sqrt((variance["coefficient"] + variance["model"]) / 2 + variance["candidate"]),
        ),
        "both": lcb_with_sd(
            report.candidate.corrected_ed,
            math.sqrt((variance["coefficient"] + variance["model"]) / 2 + 0.002**2),
        ),
    }
    return {
        "objective": report.objective,
        "partial_processes": [dataclasses.asdict(row) for row in PARTIAL_PROCESSES],
        "partial_totals_d": sum_processes(PARTIAL_PROCESSES, "d"),
        "partial_totals_e": sum_processes(PARTIAL_PROCESSES, "e"),
        "full_processes": [dataclasses.asdict(row) for row in FULL_PROCESSES_D],
        "full_totals_d": sum_processes(FULL_PROCESSES_D, "d"),
        "full_totals_e": sum_processes(FULL_PROCESSES_D, "e"),
        "search": {
            "full": dataclasses.asdict(
                SearchEvidence(
                    "best-first one-bit acquisition + 64-seed confirmation",
                    90,
                    2**12,
                    False,
                    FULL_MASK,
                    FULL_INDICES,
                    1256.583165159796,
                    1248.2121071019228,
                    1248.6289419421269,
                )
            ),
            "partial": {
                "method": "directed coordinate/pattern descent, three starts, 64-seed confirmation",
                "evaluations": 331,
                "raw_d_cycles": 398.9756511201952,
                "raw_default_e_cycles": 379.1866989282294,
                "raw_optimized_e_cycles": 363.72109952008236,
                "normalized_cycles": 363.74544839988715,
                "resource_floor": 358.4,
                "fraction_of_default_gap_closed": 0.743154543868186,
                "scheduler": {
                    "lam": 0.1,
                    "w_imb": 4.0,
                    "w_stall": 0.5,
                    "w_h": 0.05,
                    "w_chain": 1.0,
                    "w_press": 0.0,
                    "rho": 1.0,
                    "window": 7,
                    "regs": 57,
                },
            },
        },
        "best_feasible": dataclasses.asdict(report.candidate),
        "gap_ladder": [dataclasses.asdict(point) for point in ladder],
        "best_to_bound_corrected_gap": ladder[-1].corrected_ed / ladder[0].corrected_ed - 1,
        "residual_variance": variance,
        "residual_sensitivity": sensitivity,
        "leave_one_group_out": LOO_SENSITIVITY,
        "decision": (
            "MODEL-GO for the existing gate; central E/D exceeds 1.04, but LCB95 does not. "
            "The resource bound is global; the finite-budget schedule point is only best observed."
        ),
    }


def main() -> None:
    print(json.dumps(build_optimizer_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
