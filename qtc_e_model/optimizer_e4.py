#!/usr/bin/env python3
"""Deterministic joint relocate/beam and bank-recolor search for QTC E.

The pure beam engine is self-contained.  The optional pilot adapter operates
on the ignored S10 IR and simulator: every edge relocates one instruction
across at least two pairwise-commuting instructions, then greedily recolors
live intervals before scoring.  This is a directed graph search, not a
Cartesian parameter sweep and not another adjacent-swap pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import importlib
import json
from pathlib import Path
import sys
from typing import Callable, Generic, Iterable, TypeVar

from . import fused_reduction as fused


State = TypeVar("State")


@dataclasses.dataclass(frozen=True)
class BeamConfig:
    depth: int = 3
    width: int = 6
    max_span: int = 10
    hot_positions: int = 12
    branch_budget: int = 24
    coarse_seeds: int = 1
    confirmation_seeds: int = 64


@dataclasses.dataclass(frozen=True)
class BeamObservation(Generic[State]):
    state: State
    score: float
    depth: int
    parent: State | None
    move: tuple[int, int] | None


@dataclasses.dataclass(frozen=True)
class SearchSummary:
    method: str
    depth: int
    width: int
    max_span: int
    generated_edges: int
    unique_states: int
    coarse_evaluations: int
    confirmed_states: int
    baseline_raw_cycles: float
    best_raw_cycles: float
    best_normalized_cycles: float
    best_lcb95: float
    reaches_lcb_1_04: bool
    best_moves: tuple[tuple[int, int], ...]
    best_bank_conflicts: int


@dataclasses.dataclass(frozen=True)
class OccupancyGate:
    target_declared_registers: int
    body_register_overhead: int
    maximum_allowed_physical_register: int
    maximum_fixed_physical_register: int
    lower_bound_declared_registers: int
    reachable_without_frame_remap: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class SaturationDecision:
    goal_lcb95: float
    minimum_material_gain: float
    starts: int
    maximum_gain_over_baseline: float
    result_spread: float
    same_resource_basin: bool
    stagnated: bool
    current_family_feasible: bool
    current_family_stopped: bool
    arithmetic_resource_upper_lcb95: float
    arithmetic_restructure_required: bool
    reason: str


def beam_search(
    initial: State,
    *,
    neighbors: Callable[[State], Iterable[tuple[State, tuple[int, int]]]],
    evaluate: Callable[[State], float],
    depth: int,
    width: int,
) -> tuple[tuple[BeamObservation[State], ...], int]:
    """Best-first beam over graph states, retaining provenance and deduping.

    The queue ordering is the objective value and a stable serial tie-breaker.
    A state is evaluated at most once across all depths.
    """
    if depth < 1 or width < 1:
        raise ValueError("depth and width must be positive")
    first = BeamObservation(initial, evaluate(initial), 0, None, None)
    seen = {initial}
    observations = [first]
    frontier = [first]
    generated = 0
    serial = 0
    for level in range(1, depth + 1):
        queue: list[tuple[float, int, BeamObservation[State]]] = []
        for parent in frontier:
            for state, move in neighbors(parent.state):
                generated += 1
                if state in seen:
                    continue
                seen.add(state)
                score = evaluate(state)
                observation = BeamObservation(state, score, level, parent.state, move)
                observations.append(observation)
                serial += 1
                heapq.heappush(queue, (score, serial, observation))
        frontier = [heapq.heappop(queue)[2] for _ in range(min(width, len(queue)))]
        if not frontier:
            break
    return tuple(observations), generated


def relocate(state: tuple[int, ...], source: int, target: int) -> tuple[int, ...]:
    """Move one item; ``target`` is its final index in the returned state."""
    if not 0 <= source < len(state) or not 0 <= target < len(state):
        raise IndexError("relocation outside state")
    values = list(state)
    item = values.pop(source)
    values.insert(target, item)
    return tuple(values)


def occupancy_gate(
    fixed_registers: Iterable[int],
    *,
    target_declared: int = 56,
    overhead: int = 3,
) -> OccupancyGate:
    """Certify the unchanged-frame register gate from pinned ABI words."""
    fixed = tuple(fixed_registers)
    highest = max(fixed, default=-1)
    allowed = target_declared - overhead
    lower = highest + overhead
    reachable = highest <= allowed
    return OccupancyGate(
        target_declared_registers=target_declared,
        body_register_overhead=overhead,
        maximum_allowed_physical_register=allowed,
        maximum_fixed_physical_register=highest,
        lower_bound_declared_registers=lower,
        reachable_without_frame_remap=reachable,
        reason=(
            "all pinned ABI words fit the target"
            if reachable
            else f"pinned R{highest} exceeds the R{allowed} body ceiling; scheduling/recoloring cannot move it"
        ),
    )


def assess_saturation(
    lcb95_results: Iterable[float],
    bank_conflicts: Iterable[int],
    occupancy: OccupancyGate,
    *,
    baseline_lcb95: float = 1.0342921067405533,
    goal_lcb95: float = 1.04,
    minimum_material_gain: float = 0.0005,
    arithmetic_resource_upper_lcb95: float = 1.0521332850266203,
) -> SaturationDecision:
    """Apply the frozen stop rule to independent, confirmed search starts."""
    results = tuple(lcb95_results)
    banks = tuple(bank_conflicts)
    if len(results) < 3 or len(results) != len(banks):
        raise ValueError("saturation requires at least three matched confirmed starts")
    maximum_gain = max(results) - baseline_lcb95
    spread = max(results) - min(results)
    same_basin = len(set(banks)) == 1 and spread < minimum_material_gain
    stagnated = maximum_gain < minimum_material_gain and same_basin
    feasible = occupancy.reachable_without_frame_remap
    stopped = not feasible or stagnated
    restructure = arithmetic_resource_upper_lcb95 >= goal_lcb95
    reasons = []
    if not feasible:
        reasons.append("unchanged-frame feasible set is empty at declared<=56")
    if stagnated:
        reasons.append("three confirmed starts improved <0.0005 and converged in one resource basin")
    if restructure:
        reasons.append("arithmetic resource upper LCB remains >=1.04")
    return SaturationDecision(
        goal_lcb95=goal_lcb95,
        minimum_material_gain=minimum_material_gain,
        starts=len(results),
        maximum_gain_over_baseline=maximum_gain,
        result_spread=spread,
        same_resource_basin=same_basin,
        stagnated=stagnated,
        current_family_feasible=feasible,
        current_family_stopped=stopped,
        arithmetic_resource_upper_lcb95=arithmetic_resource_upper_lcb95,
        arithmetic_restructure_required=restructure,
        reason="; ".join(reasons),
    )


def _resources(op):
    return (
        {word.phys for _slot, word in op.uses},
        {word.phys for word in op.defs},
        {predicate.phys for predicate in op.puses},
        {predicate.phys for predicate in op.pdefs},
    )


def _independent(left, right) -> bool:
    lr, lw, lp, lq = _resources(left)
    rr, rw, rp, rq = _resources(right)
    return not (lw & (rr | rw) or rw & (lr | lw) or lq & (rp | rq) or rq & (lp | lq))


def _safe_relocations(
    state: tuple[int, ...],
    operations: tuple[object, ...],
    hot: tuple[int, ...],
    max_span: int,
) -> Iterable[tuple[tuple[int, ...], tuple[int, int]]]:
    """Relocate through >=2 commuting operations near objective hotspots."""
    sources = set(hot)
    for position in hot:
        sources.update(range(max(0, position - max_span), min(len(state), position + max_span + 1)))
    for source in sorted(sources):
        moving = operations[state[source]]
        for direction in (-1, 1):
            for distance in range(1, max_span + 1):
                target = source + direction * distance
                if not 0 <= target < len(state):
                    break
                crossed = operations[state[target]]
                if not _independent(moving, crossed):
                    break
                if distance >= 2:
                    yield relocate(state, source, target), (source, target)


def _words(order) -> tuple[object, ...]:
    return tuple(
        {word.id: word for op in order for word in op.defs + [word for _slot, word in op.uses]}.values()
    )


def _greedy_recolor(gen, order) -> tuple[tuple[str, int, int], ...]:
    """Bank-aware parity recolor for one candidate's changed live intervals."""
    words = _words(order)
    first: dict[object, int] = {}
    last: dict[object, int] = {}
    for index, op in enumerate(order):
        for _slot, word in op.uses:
            last[word] = index
            first.setdefault(word, -1)
        for word in op.defs:
            first.setdefault(word, index)
            last[word] = max(last.get(word, index), index)
    for word in words:
        if word.fixed is not None and first[word] >= 0:
            last[word] = len(order)

    by_register: dict[int, list[object]] = {}
    for word in words:
        by_register.setdefault(word.phys, []).append(word)

    def available(word, register: int) -> bool:
        return register in gen.AVAIL and all(
            other is word
            or last[other] < first[word]
            or last[word] < first[other]
            for other in by_register.get(register, ())
        )

    fused._refresh_reuse(gen, order)
    readers: dict[object, list[object]] = {word: [] for word in words}
    for op in order:
        for _slot, word in op.uses:
            readers[word].append(op)

    def conflict(op) -> bool:
        registers = {word.phys for slot, word in op.uses if slot not in op.reuse}
        return any(sum(register % 2 == parity for register in registers) > 2 for parity in (0, 1))

    moves = []
    while True:
        incumbent = fused._bank_conflicts(order)
        best = None
        for word in words:
            if word.fixed is not None or word.pair is not None:
                continue
            old = word.phys
            affected = readers[word]
            old_local = sum(conflict(op) for op in affected)
            for register in gen.AVAIL:
                if register % 2 == old % 2 or not available(word, register):
                    continue
                word.phys = register
                conflicts = incumbent - old_local + sum(conflict(op) for op in affected)
                key = (conflicts, word.id, register)
                if conflicts < incumbent and (best is None or key < best[0]):
                    best = (key, word, old, register)
                word.phys = old
        if best is None:
            break
        _key, word, old, register = best
        word.phys = register
        by_register[old].remove(word)
        by_register.setdefault(register, []).append(word)
        moves.append((word.name, old, register))
    fused._refresh_reuse(gen, order)
    return tuple(moves)


class PilotSearch:
    """Mutable adapter isolated behind immutable schedule-index states."""

    def __init__(
        self,
        pilot_root: Path,
        config: BeamConfig,
        initial_moves: Iterable[tuple[int, int]] = (),
    ):
        self.config = config
        self.gen, self.partial, self.body_io = fused._patch_pilot(pilot_root)
        self.gen.qtc_e_reset_fusion()
        opts = self.gen.Opts(
            lam=0.35,
            w_imb=4.0,
            w_stall=1.5,
            window=2,
            ldc_max=2,
            rho=1.0,
            w_h=0.02,
            w_chain=3.0,
        )
        _lines, _labels, blocks = self.gen.generate(
            "L", lambda _i: ("F",) * 6, opts_s=opts, opts_m=opts
        )
        self.main, self.lin8 = blocks
        fused.apply_e3_bank_schedule(self.gen, self.main)
        self.operations = tuple(self.main.order)
        self.identity = tuple(range(len(self.operations)))
        self.initial_moves = tuple(initial_moves)
        self.initial = state_after_moves(len(self.operations), self.initial_moves)
        self.words = _words(self.operations)
        self.base_phys = {word: word.phys for word in self.words}
        self.lin8_lines = self.gen.seal(list(self.lin8.order), min_last_stall=1)
        frag = pilot_root / "s10" / "frag"
        if str(frag) not in sys.path:
            sys.path.insert(0, str(frag))
        self.isa = importlib.import_module("isa")
        self.sim2 = importlib.import_module("sim2")
        self.ana = importlib.import_module("ana")
        self.cache: dict[tuple[int, ...], tuple[float, int, tuple[int, ...], tuple[int, ...]]] = {}
        self.metadata: dict[tuple[int, ...], tuple[int, tuple[int, ...], tuple[int, ...]]] = {}

    def _restore(self) -> None:
        for word, register in self.base_phys.items():
            word.phys = register

    def _lines(self, state: tuple[int, ...]):
        self._restore()
        order = [self.operations[index] for index in state]
        _greedy_recolor(self.gen, order)
        main_lines = self.gen.seal(order)
        head = self.gen.HEAD[:2] + ['[B------:R-:W-:-:S05] @P0 BRA `(.L_x_100) ;']
        lines = head + self.lin8_lines + ['[B------:R-:W-:-:S05] BRA `(.L_x_8) ;'] + main_lines + self.gen.TAIL
        labels = {".L_x_10": 0, ".L_x_6": len(head), ".L_x_100": len(head) + len(self.lin8_lines) + 1}
        bank_positions = []
        for index, op in enumerate(order):
            registers = {word.phys for slot, word in op.uses if slot not in op.reuse}
            if any(sum(register % 2 == parity for register in registers) > 2 for parity in (0, 1)):
                bank_positions.append(index)
        return lines, labels, order, main_lines, tuple(bank_positions)

    def evaluate(self, state: tuple[int, ...], *, seeds: int | None = None) -> float:
        if seeds is None and state in self.cache:
            return self.cache[state][0]
        lines, labels, order, main_lines, bank_positions = self._lines(state)
        program = [self.isa.parse(line) for line in lines]
        sequence = self.sim2.trace_of(program, labels)
        use_seeds = seeds if seeds is not None else self.config.coarse_seeds
        cycles = sum(
            self.sim2.simulate(program, sequence, seed=seed, reps=3 if seeds is None else 6)
            for seed in range(use_seeds)
        ) / use_seeds
        if seeds is None:
            parsed_main = [self.isa.parse(line) for line in main_lines]
            stalls = tuple(ins.ctrl["stall"] for ins in parsed_main)
            self.cache[state] = (cycles, fused._bank_conflicts(order), stalls, bank_positions)
            self.metadata[state] = (fused._bank_conflicts(order), stalls, bank_positions)
        return cycles

    def prepare_metadata(self, state: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        if state not in self.metadata:
            _lines, _labels, order, main_lines, bank_positions = self._lines(state)
            parsed_main = [self.isa.parse(line) for line in main_lines]
            stalls = tuple(ins.ctrl["stall"] for ins in parsed_main)
            self.metadata[state] = (fused._bank_conflicts(order), stalls, bank_positions)
        return self.metadata[state]

    def hot(self, state: tuple[int, ...]) -> tuple[int, ...]:
        _banks, stalls, bank_positions = self.prepare_metadata(state)
        ranked = sorted(range(len(stalls)), key=lambda index: (-stalls[index], index))
        return tuple(dict.fromkeys(bank_positions + tuple(ranked[: self.config.hot_positions])))

    def neighbors(self, state: tuple[int, ...]):
        hot = self.hot(state)
        # ``evaluate`` mutates shared IR word colors.  Reconstruct the parent
        # allocation before testing commuting edges, then materialize every
        # edge before child evaluation can recolor the shared objects.
        self._restore()
        parent_order = [self.operations[index] for index in state]
        _greedy_recolor(self.gen, parent_order)
        candidates = list(_safe_relocations(state, self.operations, hot, self.config.max_span))

        def proxy(item):
            candidate, move = item
            order = [self.operations[index] for index in candidate]
            # Prefer a balanced local alternation of the two integer pipes and
            # longer non-adjacent moves; the actual beam ordering still uses
            # sim2 after recoloring.
            same_pipe = sum(left.pipe == right.pipe for left, right in zip(order, order[1:]))
            same_chain = sum(left.chain == right.chain for left, right in zip(order, order[1:]))
            distance = abs(move[1] - move[0])
            return (same_pipe + 0.05 * same_chain, -distance, move)

        candidates.sort(key=proxy)
        yield from candidates[: self.config.branch_budget]

    def occupancy(self) -> OccupancyGate:
        return occupancy_gate(word.fixed for word in self.words if word.fixed is not None)

    def write(self, state: tuple[int, ...], output: Path) -> None:
        full, labels, _order, _main_lines, _bank_positions = self._lines(state)
        partial_opts = self.gen.Opts(
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
        loop, _loop_labels, _block = self.partial.generate(cforms=lambda _i: "F", opts=partial_opts)
        labels = dict(labels, **{".L_x_9": len(full)})
        output.parent.mkdir(parents=True, exist_ok=True)
        self.body_io.write(
            str(output),
            full + loop,
            labels,
            header="QTC E4 model candidate: multi-operation relocate beam with bank-aware recolor",
        )


def _provenance(best, by_state) -> tuple[tuple[int, int], ...]:
    moves = []
    cursor = best
    while cursor.parent is not None:
        moves.append(cursor.move)
        cursor = by_state[cursor.parent]
    return tuple(reversed(moves))


def state_after_moves(size: int, moves: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    state = tuple(range(size))
    for source, target in moves:
        state = relocate(state, source, target)
    return state


def run_pilot_search(
    pilot_root: Path,
    config: BeamConfig = BeamConfig(),
    initial_moves: Iterable[tuple[int, int]] = (),
) -> tuple[SearchSummary, OccupancyGate]:
    adapter = PilotSearch(pilot_root, config, initial_moves)
    observations, generated = beam_search(
        adapter.initial,
        neighbors=adapter.neighbors,
        evaluate=adapter.evaluate,
        depth=config.depth,
        width=config.width,
    )
    coarse_best = sorted(observations, key=lambda item: (item.score, item.depth))[: config.width]
    confirmed = [(adapter.evaluate(item.state, seeds=config.confirmation_seeds), item) for item in coarse_best]
    best_cycles, best = min(confirmed, key=lambda item: item[0])
    baseline_cycles = adapter.evaluate(adapter.identity, seeds=config.confirmation_seeds)
    # Normalize against the already frozen 64-seed E3 value, retaining only
    # the paired simulator delta found by this search.
    normalized = 1238.8756527329024 + best_cycles - baseline_cycles
    candidate = fused.project(
        fused_full_outputs=9,
        fused_partial_outputs=12,
        full_cycles=normalized,
        partial_cycles=363.74544839988715,
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        name="E4-relocate-beam",
    )
    by_state = {item.state: item for item in observations}
    adapter.evaluate(best.state)
    bank_conflicts = adapter.cache[best.state][1]
    return SearchSummary(
        method="objective-directed multi-operation relocate beam + per-node bank-aware live-interval recolor",
        depth=config.depth,
        width=config.width,
        max_span=config.max_span,
        generated_edges=generated,
        unique_states=len(observations),
        coarse_evaluations=len(observations),
        confirmed_states=len(confirmed),
        baseline_raw_cycles=baseline_cycles,
        best_raw_cycles=best_cycles,
        best_normalized_cycles=normalized,
        best_lcb95=candidate.lcb95_ed,
        reaches_lcb_1_04=candidate.lcb95_ed >= 1.04,
        best_moves=adapter.initial_moves + _provenance(best, by_state),
        best_bank_conflicts=bank_conflicts,
    ), adapter.occupancy()


def write_candidate(pilot_root: Path, output: Path, moves: Iterable[tuple[int, int]]) -> None:
    adapter = PilotSearch(pilot_root, BeamConfig())
    adapter.write(state_after_moves(len(adapter.initial), moves), output)


def _parse_move(value: str) -> tuple[int, int]:
    try:
        source, target = value.split(":", 1)
        return int(source), int(target)
    except ValueError as error:
        raise argparse.ArgumentTypeError("move must be SOURCE:TARGET") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=BeamConfig.depth)
    parser.add_argument("--width", type=int, default=BeamConfig.width)
    parser.add_argument("--max-span", type=int, default=BeamConfig.max_span)
    parser.add_argument("--coarse-seeds", type=int, default=BeamConfig.coarse_seeds)
    parser.add_argument("--confirmation-seeds", type=int, default=BeamConfig.confirmation_seeds)
    parser.add_argument("--branch-budget", type=int, default=BeamConfig.branch_budget)
    parser.add_argument("--write-sass", type=Path)
    parser.add_argument("--initial-move", action="append", type=_parse_move, default=[])
    args = parser.parse_args()
    config = BeamConfig(
        depth=args.depth,
        width=args.width,
        max_span=args.max_span,
        coarse_seeds=args.coarse_seeds,
        confirmation_seeds=args.confirmation_seeds,
        branch_budget=args.branch_budget,
    )
    summary, gate = run_pilot_search(args.pilot_root, config, args.initial_move)
    if args.write_sass:
        write_candidate(args.pilot_root, args.write_sass, summary.best_moves)
    print(json.dumps({"search": dataclasses.asdict(summary), "occupancy_gate": dataclasses.asdict(gate)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
