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


FULL_MASK = 2941
FULL_INDICES = (0, 2, 3, 4, 5, 6, 8, 9, 11)


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
        _point("best-feasible-E3", 1238.8756527329024, 363.74544839988715, 9),
        _point("partial-resource-floor", 1238.8756527329024, 358.4, 9),
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
    "without-s10-a": (1.04494933, 1.03240977),
    "without-s10-abc": (1.04586298, 1.03377689),
    "without-s14": (1.04662647, 1.03468811),
    "without-s6-b": (1.04661264, 1.03450400),
    "without-s6-c1": (1.04708275, 1.03489265),
    "without-s6-n": (1.04648570, 1.03517663),
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
                    "best-first mask acquisition + bank-aware recolor + one legal neighbor",
                    95,
                    2**12,
                    False,
                    FULL_MASK,
                    FULL_INDICES,
                    1256.583165159796,
                    1238.4588178926983,
                    1238.8756527329024,
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
        "current_family_blocker": {
            "scope": "E3 fixed instruction/control order with register-bank recolor and occupancy only",
            "required_normalized_full_cycles_for_lcb_1_04": 1222.7854173827698,
            "feasible_normalized_full_cycles": 1238.8756527329024,
            "zero_bank_8_warp_normalized_cycles": 1235.9540978673654,
            "zero_bank_9_warp_normalized_cycles": 1231.939244682243,
            "zero_bank_10_warp_normalized_cycles": 1228.7413546667158,
            "zero_bank_10_warp_lcb95": 1.0378816475650245,
            "reaches_lcb_1_04": False,
            "note": "does not bound a new instruction order or new arithmetic",
        },
        "joint_allocator_scheduler_frontier": {
            "method": "all dependency/physical-safe adjacent neighbors around bank-aware E3",
            "evaluated_neighbors": 461,
            "confirmation_seeds": 64,
            "e3_raw_cycles": 1238.4588178926983,
            "best_neighbor_raw_cycles": 1238.4046041675053,
            "best_neighbor_normalized_cycles": 1238.8214390077096,
            "best_neighbor_lcb95": 1.0343112596298996,
            "accepted_as_new_incumbent": False,
            "reason": "0.054-cycle gain is immaterial and does not change the model decision",
        },
        "joint_relocate_beam": {
            "method": "depth-3 multi-operation relocate beam with per-node bank-aware recolor",
            "generated_edges": 312,
            "unique_states": 298,
            "coarse_seeds": 1,
            "confirmed_states": 6,
            "confirmation_seeds": 64,
            "moves": ((47, 43), (64, 69), (627, 629)),
            "e3_raw_cycles": 1238.4588178926983,
            "best_raw_cycles": 1237.8403765930216,
            "best_normalized_cycles": 1238.2572114332256,
            "best_lcb95": 1.0345106240695512,
            "reaches_lcb_1_04": False,
            "scope": "best observed over the visited directed relocate/recolor graph, not a schedule bound",
        },
        "occupancy_56_gate": {
            "target_declared_registers": 56,
            "maximum_allowed_physical_register": 53,
            "maximum_fixed_physical_register": 56,
            "lower_bound_declared_registers": 59,
            "reachable_without_frame_remap": False,
            "scope": "certified for the unchanged S10 frame/ABI",
        },
        "saturation": {
            "criterion": "stop after 3 confirmed starts when every absolute LCB gain is <0.0005 and one basin",
            "depths": "2..4",
            "starts": 3,
            "unique_states": (202, 201, 200),
            "generated_edges": (208, 208, 208),
            "confirmed_candidates": (4, 4, 4),
            "confirmation_seeds": 64,
            "lcb95_results": (1.0344211047124616, 1.0343980705706526, 1.0343980705706526),
            "maximum_gain_over_e3": 0.00012899797190835116,
            "result_spread": 2.303414180904184e-05,
            "common_bank_conflicts": 4,
            "same_resource_basin": True,
            "stagnated": True,
            "current_family_stopped": True,
            "arithmetic_resource_upper_lcb95": 1.0521332850266203,
            "arithmetic_restructure_required": True,
        },
        "residual_variance": variance,
        "residual_sensitivity": sensitivity,
        "leave_one_group_out": LOO_SENSITIVITY,
        "decision": (
            "E4 relocate/beam is the best observed schedule point but remains model-only: central exceeds "
            "1.04, LCB95 does not. E3 remains the committed fallback artifact; the unchanged-frame "
            "occupancy gate is impossible and new A-pipe arithmetic is required."
        ),
    }


def main() -> None:
    print(json.dumps(build_optimizer_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
