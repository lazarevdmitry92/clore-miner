#!/usr/bin/env python3
"""CPU/GPU batch contract for QTC E frontier scoring.

CPU remains authoritative for dependency legality, branching, recoloring,
control generation and admissible pruning.  Only independent sim2
candidate-by-seed jobs are packed for a CUDA scorer.  The CPU implementation
below is the executable reference for rental parity and break-even profiling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import time
from typing import Iterable

from . import fused_reduction as fused
from .optimizer_e4 import BeamConfig, BeamObservation, PilotSearch, _provenance


MAGIC = b"QTCE4GPU"
VERSION = 1
PIPE_CODE = {"W": 0, "F": 1, "A": 2, "C": 3, "B": 4}
HEADER = struct.Struct("<8s7I")


@dataclasses.dataclass(frozen=True)
class EncodedTrace:
    pipe: bytes
    stall: bytes
    wait: bytes
    write_barrier: bytes
    bank_conflict: bytes

    @property
    def length(self) -> int:
        return len(self.pipe)


@dataclasses.dataclass(frozen=True)
class Batch:
    traces: tuple[EncodedTrace, ...]
    seeds: tuple[int, ...]
    nwarps: int = 8
    reps: int = 6


@dataclasses.dataclass(frozen=True)
class Profile:
    candidates: int
    seeds: int
    frontier_prepare_seconds: float
    cpu_score_seconds: float
    gpu_eligible_share: float
    cpu_candidate_seed_jobs_per_second: float
    modeled_gpu_jobs_per_second_low: float
    modeled_gpu_jobs_per_second_high: float
    modeled_speedup_low: float
    modeled_speedup_high: float
    modeled_break_even_jobs: int
    note: str


def offsets_for_seed(seed: int, length: int, nwarps: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.randrange(length) for _ in range(nwarps))


def score_trace(
    trace: EncodedTrace,
    seed: int,
    *,
    nwarps: int = 8,
    reps: int = 6,
    ldc_lat: int = 26,
    ldc_gap: int = 2,
) -> float:
    """Integer-equivalent reference for ``sim2.simulate`` (RR/random)."""
    length = trace.length
    pc = list(offsets_for_seed(seed, length, nwarps))
    iterations = [0] * nwarps
    next_issue = [0] * nwarps
    barriers = [[0] * 6 for _ in range(nwarps)]
    fma_free = alu_free = mio_free = 0
    cycle = 0
    last = -1
    issued = 0
    target = reps * length
    while min(iterations[warp] * length + pc[warp] for warp in range(nwarps)) < target + length:
        chosen = -1
        for offset in range(1, nwarps + 1):
            warp = (last + offset) % nwarps
            if next_issue[warp] > cycle:
                continue
            index = pc[warp]
            wait = trace.wait[index]
            if any(barriers[warp][barrier] > cycle for barrier in range(6) if (wait >> barrier) & 1):
                continue
            pipe = trace.pipe[index]
            if pipe == 0 and (fma_free > cycle or alu_free > cycle):
                continue
            if pipe == 1 and fma_free > cycle:
                continue
            if pipe == 2 and alu_free > cycle:
                continue
            if pipe == 3 and mio_free > cycle:
                continue
            chosen = warp
            break
        if chosen >= 0:
            warp = chosen
            index = pc[warp]
            pipe = trace.pipe[index]
            extra = trace.bank_conflict[index]
            if pipe == 0:
                fma_free, alu_free = cycle + 4 + extra, cycle + 2 + extra
            elif pipe == 1:
                fma_free = cycle + 2 + extra
            elif pipe == 2:
                alu_free = cycle + 2 + extra
            elif pipe == 3:
                mio_free = cycle + ldc_gap
            write_barrier = trace.write_barrier[index]
            if write_barrier != 7:
                barriers[warp][write_barrier] = cycle + ldc_lat
            next_issue[warp] = cycle + max(trace.stall[index], 1) + extra
            pc[warp] += 1
            issued += 1
            if pc[warp] == length:
                pc[warp] = 0
                iterations[warp] += 1
            last = warp
        cycle += 1
        if cycle > 50_000_000:
            raise RuntimeError("reference scorer made no progress")
    return cycle * length / issued


def score_batch(batch: Batch) -> tuple[float, ...]:
    return tuple(
        score_trace(trace, seed, nwarps=batch.nwarps, reps=batch.reps)
        for trace in batch.traces
        for seed in batch.seeds
    )


def encode_lines(adapter: PilotSearch, lines: list[str], labels: dict[str, int]) -> EncodedTrace:
    program = [adapter.isa.parse(line) for line in lines]
    sequence = adapter.sim2.trace_of(program, labels)
    code, bank_flags = adapter.sim2.prepare(program, sequence)
    return EncodedTrace(
        pipe=bytes(PIPE_CODE[instruction.pipe()] for instruction in code),
        stall=bytes(instruction.ctrl["stall"] for instruction in code),
        wait=bytes(instruction.ctrl["wait"] for instruction in code),
        write_barrier=bytes(instruction.ctrl["wb"] for instruction in code),
        bank_conflict=bytes(bank_flags),
    )


def encode_state(adapter: PilotSearch, state: tuple[int, ...]) -> EncodedTrace:
    lines, labels, _order, _main, _banks = adapter._lines(state)
    return encode_lines(adapter, lines, labels)


def write_batch(path: Path, batch: Batch, expected: Iterable[float] | None = None) -> None:
    if not batch.traces:
        raise ValueError("batch must contain at least one trace")
    length = batch.traces[0].length
    if any(trace.length != length for trace in batch.traces):
        raise ValueError("CUDA contract requires equal trace lengths")
    values = tuple(expected) if expected is not None else score_batch(batch)
    if len(values) != len(batch.traces) * len(batch.seeds):
        raise ValueError("one expected value is required per candidate and seed")
    with path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, VERSION, len(batch.traces), length, len(batch.seeds), batch.nwarps, batch.reps, 5))
        for trace in batch.traces:
            for field in dataclasses.fields(trace):
                output.write(getattr(trace, field.name))
        for seed in batch.seeds:
            for offset in offsets_for_seed(seed, length, batch.nwarps):
                output.write(struct.pack("<H", offset))
        output.write(struct.pack(f"<{len(values)}d", *values))


def write_unscored_batch(path: Path, batch: Batch) -> None:
    jobs = len(batch.traces) * len(batch.seeds)
    write_batch(path, batch, (float("nan"),) * jobs)


def read_scores(path: Path, jobs: int) -> tuple[float, ...]:
    data = path.read_bytes()
    if len(data) != jobs * 8:
        raise ValueError(f"expected {jobs * 8} result bytes, got {len(data)}")
    return struct.unpack(f"<{jobs}d", data)


def score_batch_cuda(
    batch: Batch,
    executable: Path,
    *,
    timeout_seconds: int = 900,
) -> tuple[float, ...]:
    """Run the rental scorer; no CPU scores are computed on this path."""
    with tempfile.TemporaryDirectory(prefix="qtc-e-gpu-") as directory:
        root = Path(directory)
        input_path, output_path = root / "frontier.bin", root / "scores.bin"
        write_unscored_batch(input_path, batch)
        run_with_watchdog(
            (str(executable), str(input_path), str(output_path)),
            timeout_seconds=timeout_seconds,
        )
        return read_scores(output_path, len(batch.traces) * len(batch.seeds))


def run_with_watchdog(
    command: tuple[str, ...],
    *,
    timeout_seconds: int,
    watchdog_seconds: int = 15,
) -> None:
    """Run a rental command with visible 15-second liveness heartbeats."""
    process = subprocess.Popen(command)
    started = time.monotonic()
    while True:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            process.kill()
            process.wait()
            raise TimeoutError(f"rental command exceeded {timeout_seconds}s")
        try:
            code = process.wait(timeout=min(watchdog_seconds, remaining))
        except subprocess.TimeoutExpired:
            print(json.dumps({"event": "watchdog", "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            continue
        if code:
            raise subprocess.CalledProcessError(code, command)
        return


def hybrid_search(
    pilot_root: Path,
    config: BeamConfig,
    *,
    gpu_binary: Path | None,
) -> dict[str, object]:
    """End-to-end batched beam; GPU is optional only for local smoke."""
    adapter = PilotSearch(pilot_root, config)
    seen = {adapter.initial}
    parents: dict[tuple[int, ...], BeamObservation[tuple[int, ...]]] = {}
    edges = 0

    def score_states(states: tuple[tuple[int, ...], ...], seeds: int) -> tuple[float, ...]:
        batch = Batch(tuple(encode_state(adapter, state) for state in states), tuple(range(seeds)))
        values = score_batch_cuda(batch, gpu_binary) if gpu_binary else score_batch(batch)
        return tuple(
            sum(values[index * seeds : (index + 1) * seeds]) / seeds
            for index in range(len(states))
        )

    (initial_score,) = score_states((adapter.initial,), config.coarse_seeds)
    initial = BeamObservation(adapter.initial, initial_score, 0, None, None)
    parents[adapter.initial] = initial
    observations = [initial]
    frontier = [initial]
    for depth in range(1, config.depth + 1):
        pending: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, int]]] = []
        for parent in frontier:
            for state, move in adapter.neighbors(parent.state):
                edges += 1
                if state in seen:
                    continue
                seen.add(state)
                pending.append((state, parent.state, move))
        if not pending:
            break
        states = tuple(item[0] for item in pending)
        scores = score_states(states, config.coarse_seeds)
        level = []
        for (state, parent, move), score in zip(pending, scores):
            observation = BeamObservation(state, score, depth, parent, move)
            parents[state] = observation
            observations.append(observation)
            level.append(observation)
        frontier = sorted(level, key=lambda item: (item.score, item.state))[: config.width]

    leaders = tuple(sorted(observations, key=lambda item: (item.score, item.depth))[: config.width])
    confirmed = score_states(tuple(item.state for item in leaders), config.confirmation_seeds)
    best_index = min(range(len(leaders)), key=lambda index: confirmed[index])
    best, best_raw = leaders[best_index], confirmed[best_index]
    (baseline_raw,) = score_states((adapter.identity,), config.confirmation_seeds)
    normalized = 1238.8756527329024 + best_raw - baseline_raw
    candidate = fused.project(
        fused_full_outputs=9,
        fused_partial_outputs=12,
        full_cycles=normalized,
        partial_cycles=363.74544839988715,
        registers=64,
        predicates=6,
        scoreboards=6,
        spills=0,
        name="E-rental-hybrid",
    )
    return {
        "method": "CPU dependency/branch/recolor/prune + batched candidate-seed scorer",
        "scorer": "cuda" if gpu_binary else "cpu-smoke",
        "depth": config.depth,
        "width": config.width,
        "branch_budget": config.branch_budget,
        "edges": edges,
        "unique_states": len(observations),
        "confirmed_candidates": len(leaders),
        "confirmation_seeds": config.confirmation_seeds,
        "baseline_raw_cycles": baseline_raw,
        "best_raw_cycles": best_raw,
        "best_normalized_cycles": normalized,
        "best_lcb95": candidate.lcb95_ed,
        "reaches_lcb_1_04": candidate.lcb95_ed >= 1.04,
        "best_moves": _provenance(best, parents),
    }


def read_batch(path: Path) -> tuple[Batch, tuple[float, ...]]:
    data = memoryview(path.read_bytes())
    magic, version, count, length, seed_count, nwarps, reps, fields = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or fields != 5:
        raise ValueError("unsupported QTC E GPU batch")
    position = HEADER.size
    traces = []
    for _ in range(count):
        arrays = []
        for _field in range(fields):
            arrays.append(bytes(data[position : position + length]))
            position += length
        traces.append(EncodedTrace(*arrays))
    # Offsets are derived from seeds in the in-memory contract.  The binary
    # stores the exact offsets instead, so seed identifiers are informational.
    position += seed_count * nwarps * 2
    values = struct.unpack_from(f"<{count * seed_count}d", data, position)
    return Batch(tuple(traces), tuple(range(seed_count)), nwarps, reps), values


def profile(pilot_root: Path, candidates: int = 8, seeds: int = 8) -> Profile:
    adapter = PilotSearch(pilot_root, BeamConfig(branch_budget=max(candidates - 1, 1)))
    states = [adapter.identity]
    states.extend(state for state, _move in adapter.neighbors(adapter.identity))
    states = states[:candidates]
    traces = []
    start = time.perf_counter()
    for state in states:
        traces.append(encode_state(adapter, state))
    prepare_seconds = time.perf_counter() - start
    batch = Batch(tuple(traces), tuple(range(seeds)))
    start = time.perf_counter()
    score_batch(batch)
    score_seconds = time.perf_counter() - start
    total = prepare_seconds + score_seconds
    jobs = len(states) * seeds
    cpu_rate = jobs / score_seconds
    # Conservative launch+copy envelope for the scalar CUDA kernel.  The
    # rental parity run replaces this model with measured rates.
    low_rate, high_rate = 5_000.0, 15_000.0
    low_total = prepare_seconds + jobs / low_rate
    high_total = prepare_seconds + jobs / high_rate
    # 20 ms launch/copy overhead divided by the CPU time saved per job.
    saved_per_job = max(1 / cpu_rate - 1 / low_rate, 1e-9)
    break_even = max(1, int(0.020 / saved_per_job + 0.999))
    return Profile(
        candidates=len(states),
        seeds=seeds,
        frontier_prepare_seconds=prepare_seconds,
        cpu_score_seconds=score_seconds,
        gpu_eligible_share=score_seconds / total,
        cpu_candidate_seed_jobs_per_second=cpu_rate,
        modeled_gpu_jobs_per_second_low=low_rate,
        modeled_gpu_jobs_per_second_high=high_rate,
        modeled_speedup_low=total / low_total,
        modeled_speedup_high=total / high_total,
        modeled_break_even_jobs=break_even,
        note="GPU rates are conservative model inputs until the rental parity run measures them",
    )


def export_frontier(pilot_root: Path, output: Path, candidates: int, seeds: int) -> None:
    adapter = PilotSearch(pilot_root, BeamConfig(branch_budget=max(candidates - 1, 1)))
    states = [adapter.identity]
    states.extend(state for state, _move in adapter.neighbors(adapter.identity))
    traces = tuple(encode_state(adapter, state) for state in states[:candidates])
    batch = Batch(traces, tuple(range(seeds)))
    write_batch(output, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "export", "check", "gpu-check", "hybrid-search"))
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--gpu-binary", type=Path)
    parser.add_argument("--unscored", action="store_true")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--branch-budget", type=int, default=48)
    args = parser.parse_args()
    if args.command in {"profile", "export", "hybrid-search"} and args.pilot_root is None:
        parser.error("profile/export/hybrid-search require --pilot-root")
    if args.command in {"export", "check", "gpu-check"} and args.batch is None:
        parser.error("export/check/gpu-check require --batch")
    if args.command in {"gpu-check", "hybrid-search"} and args.gpu_binary is None:
        parser.error("gpu-check/hybrid-search requires --gpu-binary")
    if args.command == "profile":
        print(json.dumps(dataclasses.asdict(profile(args.pilot_root, args.candidates, args.seeds)), indent=2))
    elif args.command == "export":
        if args.unscored:
            adapter = PilotSearch(args.pilot_root, BeamConfig(branch_budget=max(args.candidates - 1, 1)))
            states = [adapter.identity]
            states.extend(state for state, _move in adapter.neighbors(adapter.identity))
            traces = tuple(encode_state(adapter, state) for state in states[: args.candidates])
            write_unscored_batch(args.batch, Batch(traces, tuple(range(args.seeds))))
        else:
            export_frontier(args.pilot_root, args.batch, args.candidates, args.seeds)
    elif args.command == "check":
        batch, expected = read_batch(args.batch)
        actual = score_batch(batch)
        maximum = max(abs(left - right) for left, right in zip(actual, expected))
        print(json.dumps({"jobs": len(actual), "max_abs_difference": maximum, "passes": maximum <= 1e-12}))
        if maximum > 1e-12:
            raise SystemExit(1)
    elif args.command == "gpu-check":
        run_with_watchdog((str(args.gpu_binary), str(args.batch)), timeout_seconds=900)
    else:
        config = BeamConfig(
            depth=args.depth,
            width=args.width,
            branch_budget=args.branch_budget,
            coarse_seeds=args.seeds,
            confirmation_seeds=64,
        )
        print(json.dumps(hybrid_search(args.pilot_root, config, gpu_binary=args.gpu_binary), indent=2))


if __name__ == "__main__":
    main()
