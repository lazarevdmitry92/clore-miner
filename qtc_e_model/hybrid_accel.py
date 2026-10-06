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
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
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
MAX_CANDIDATES = 1_000_000
MAX_TRACE_LENGTH = 65_535
MAX_SEEDS = 4_096
MAX_JOBS = 10_000_000
MAX_BATCH_BYTES = 1 << 30
MAX_REPS = 64
GLOBAL_SEARCH_TIMEOUT_SECONDS = 30 * 60
COMMAND_TIMEOUT_SECONDS = 15 * 60
PARITY_TOLERANCE = 1e-12
PARITY_SCHEMA = "qtc-e-cuda-parity-v2"
MANIFEST_SCHEMA = "qtc-e-rental-run-v1"


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
class VerifiedParity:
    artifact_sha256: str
    binary_sha256: str
    coarse_batch_sha256: str
    confirmation_batch_sha256: str
    model_sha256: str


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authoritative_model_sha256(pilot_root: Path) -> str:
    """Hash every source that defines the authoritative score contract."""
    paths = (
        Path(__file__),
        Path(__file__).with_name("optimizer_e4.py"),
        Path(__file__).with_name("fused_reduction.py"),
        pilot_root / "s10" / "frag" / "sim2.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"authoritative model source is missing: {path}")
        data = path.read_bytes()
        name = path.name.encode("utf-8")
        digest.update(struct.pack("<I", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)
    return digest.hexdigest()


def _checked_layout(count: int, length: int, seed_count: int, nwarps: int) -> tuple[int, int, int, int]:
    if not 1 <= count <= MAX_CANDIDATES:
        raise ValueError(f"candidate count must be in 1..{MAX_CANDIDATES}")
    if not 1 <= length <= MAX_TRACE_LENGTH:
        raise ValueError(f"trace length must be in 1..{MAX_TRACE_LENGTH}")
    if not 1 <= seed_count <= MAX_SEEDS:
        raise ValueError(f"seed count must be in 1..{MAX_SEEDS}")
    if not 1 <= nwarps <= 8:
        raise ValueError("nwarps must be in 1..8")
    jobs = count * seed_count
    if jobs > MAX_JOBS:
        raise ValueError(f"candidate-seed jobs exceed {MAX_JOBS}")
    trace_bytes = count * length * 5
    offset_bytes = seed_count * nwarps * 2
    result_bytes = jobs * 8
    total = HEADER.size + trace_bytes + offset_bytes + result_bytes
    if total > MAX_BATCH_BYTES:
        raise ValueError(f"batch exceeds {MAX_BATCH_BYTES} bytes")
    return jobs, trace_bytes, offset_bytes, total


def _validate_trace(trace: EncodedTrace, expected_length: int | None = None) -> None:
    fields = tuple(getattr(trace, field.name) for field in dataclasses.fields(trace))
    if not fields or not isinstance(fields[0], bytes):
        raise ValueError("trace fields must be bytes")
    length = len(fields[0])
    if expected_length is not None and length != expected_length:
        raise ValueError("CUDA contract requires equal trace lengths")
    if any(not isinstance(field, bytes) or len(field) != length for field in fields):
        raise ValueError("all trace fields must be bytes of the same length")
    if any(value > 4 for value in trace.pipe):
        raise ValueError("pipe codes must be in 0..4")
    if any(value > 15 for value in trace.stall):
        raise ValueError("stall fields must be in 0..15")
    if any(value > 0x3F for value in trace.wait):
        raise ValueError("wait masks must use only barriers 0..5")
    if any(value not in (0, 1, 2, 3, 4, 5, 7) for value in trace.write_barrier):
        raise ValueError("write barriers must be 0..5 or 7")
    if any(value not in (0, 1) for value in trace.bank_conflict):
        raise ValueError("bank-conflict fields must be boolean")


def validate_batch(batch: Batch, expected: Iterable[float] | None = None) -> tuple[float, ...] | None:
    if not batch.traces:
        raise ValueError("batch must contain at least one trace")
    if not 1 <= batch.reps <= MAX_REPS:
        raise ValueError(f"reps must be in 1..{MAX_REPS}")
    if batch.seeds != tuple(range(len(batch.seeds))):
        raise ValueError("binary contract requires contiguous seeds starting at zero")
    length = batch.traces[0].length
    _checked_layout(len(batch.traces), length, len(batch.seeds), batch.nwarps)
    for trace in batch.traces:
        _validate_trace(trace, length)
    if expected is None:
        return None
    values = tuple(expected)
    if len(values) != len(batch.traces) * len(batch.seeds):
        raise ValueError("one expected value is required per candidate and seed")
    if any(math.isinf(value) for value in values):
        raise ValueError("infinite expected scores are forbidden")
    finite = tuple(math.isfinite(value) for value in values)
    if any(finite) and not all(finite):
        raise ValueError("expected scores must be either all finite or all NaN")
    if not any(finite) and any(not math.isnan(value) for value in values):
        raise ValueError("non-finite production sentinels must be NaN")
    return values


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
    validate_batch(batch)
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
    if expected is None:
        values = score_batch(batch)
    else:
        values = tuple(expected)
        validate_batch(batch, values)
    length = batch.traces[0].length
    with path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, VERSION, len(batch.traces), length, len(batch.seeds), batch.nwarps, batch.reps, 5))
        for trace in batch.traces:
            for field in dataclasses.fields(trace):
                output.write(getattr(trace, field.name))
        for seed in batch.seeds:
            for offset in offsets_for_seed(seed, length, batch.nwarps):
                output.write(struct.pack("<H", offset))
        output.write(struct.pack(f"<{len(values)}d", *values))


def _write_unscored_batch(path: Path, batch: Batch) -> None:
    """Internal production transport; only parity-gated hybrid search calls it."""
    jobs = len(batch.traces) * len(batch.seeds)
    write_batch(path, batch, (float("nan"),) * jobs)


def read_scores(path: Path, jobs: int, *, require_finite: bool = True) -> tuple[float, ...]:
    if not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be in 1..{MAX_JOBS}")
    data = path.read_bytes()
    if len(data) != jobs * 8:
        raise ValueError(f"expected {jobs * 8} result bytes, got {len(data)}")
    values = struct.unpack(f"<{jobs}d", data)
    if require_finite and any(not math.isfinite(value) for value in values):
        raise ValueError("CUDA returned a non-finite production score")
    return values


def score_batch_cuda(
    batch: Batch,
    executable: Path,
    *,
    deadline: float,
    archive_dir: Path,
    label: str,
    manifest_entries: list[dict[str, object]],
) -> tuple[float, ...]:
    """Run one archived production batch under the whole-search deadline."""
    validate_batch(batch)
    index = len(manifest_entries)
    input_path = archive_dir / f"{index:03d}-{label}.batch.bin"
    output_path = archive_dir / f"{index:03d}-{label}.scores.bin"
    _write_unscored_batch(input_path, batch)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("whole hybrid search exceeded 1800s")
    run_with_watchdog(
        (str(executable), str(input_path), str(output_path)),
        timeout_seconds=min(COMMAND_TIMEOUT_SECONDS, remaining),
    )
    jobs = len(batch.traces) * len(batch.seeds)
    values = read_scores(output_path, jobs)
    entry = {
        "label": label,
        "batch": input_path.name,
        "batch_sha256": sha256_file(input_path),
        "results": output_path.name,
        "results_sha256": sha256_file(output_path),
        "candidates": len(batch.traces),
        "seeds": len(batch.seeds),
        "nwarps": batch.nwarps,
        "reps": batch.reps,
        "jobs": jobs,
    }
    manifest_entries.append(entry)
    print(json.dumps({"event": "progress", **entry}), flush=True)
    return values


def _terminate_process_group(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    parent_running = process.poll() is None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if parent_running:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    group_deadline = time.monotonic() + 2
    while time.monotonic() < group_deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def run_with_watchdog(
    command: tuple[str, ...],
    *,
    timeout_seconds: int,
    watchdog_seconds: int = 15,
) -> None:
    """Run a rental command with visible 15-second liveness heartbeats."""
    if timeout_seconds <= 0 or watchdog_seconds <= 0:
        raise ValueError("timeouts must be positive")
    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f"rental command exceeded {timeout_seconds}s")
            try:
                code = process.wait(timeout=min(watchdog_seconds, remaining))
            except subprocess.TimeoutExpired:
                print(json.dumps({"event": "watchdog", "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                continue
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
    finally:
        _terminate_process_group(process)


def read_batch(path: Path) -> tuple[Batch, tuple[float, ...]]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated QTC E GPU batch header")
    data = memoryview(raw)
    magic, version, count, length, seed_count, nwarps, reps, fields = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or fields != 5:
        raise ValueError("unsupported QTC E GPU batch")
    if not 1 <= reps <= MAX_REPS:
        raise ValueError(f"reps must be in 1..{MAX_REPS}")
    jobs, _trace_bytes, offset_bytes, total = _checked_layout(count, length, seed_count, nwarps)
    if len(data) != total:
        raise ValueError(f"batch size must be exactly {total} bytes, got {len(data)}")
    position = HEADER.size
    traces = []
    for _ in range(count):
        arrays = []
        for _field in range(fields):
            arrays.append(bytes(data[position : position + length]))
            position += length
        traces.append(EncodedTrace(*arrays))
    stored_offsets = struct.unpack_from(f"<{seed_count * nwarps}H", data, position)
    if any(offset >= length for offset in stored_offsets):
        raise ValueError("stored warp offset is outside the trace")
    derived_offsets = tuple(
        offset
        for seed in range(seed_count)
        for offset in offsets_for_seed(seed, length, nwarps)
    )
    if stored_offsets != derived_offsets:
        raise ValueError("stored warp offsets do not match the authoritative seed contract")
    position += offset_bytes
    values = struct.unpack_from(f"<{jobs}d", data, position)
    batch = Batch(tuple(traces), tuple(range(seed_count)), nwarps, reps)
    validate_batch(batch, values)
    return batch, values


def _write_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def create_parity_artifact(
    artifact_path: Path,
    *,
    executable: Path,
    batch_paths: dict[int, Path],
    pilot_root: Path,
    results: dict[int, tuple[int, float]],
) -> VerifiedParity:
    if set(batch_paths) != {3, 6} or set(results) != {3, 6}:
        raise ValueError("parity suite must contain exactly reps=3 and reps=6")
    runs = []
    for reps in (3, 6):
        jobs, maximum = results[reps]
        batch, expected = read_batch(batch_paths[reps])
        if batch.reps != reps or any(not math.isfinite(value) for value in expected):
            raise ValueError(f"reps={reps} parity batch is not authoritative and fully scored")
        if type(jobs) is not int or jobs != len(expected) or jobs < 1:
            raise ValueError("parity jobs must exactly match their batch")
        if not math.isfinite(maximum) or not 0 <= maximum <= PARITY_TOLERANCE:
            raise ValueError("cannot attest failed or non-finite CUDA parity")
        runs.append({
            "reps": reps,
            "authoritative_batch_sha256": sha256_file(batch_paths[reps]),
            "jobs": jobs,
            "max_abs_difference": maximum,
        })
    payload: dict[str, object] = {
        "schema": PARITY_SCHEMA,
        "created_unix": int(time.time()),
        "gpu_binary_sha256": sha256_file(executable),
        "authoritative_model_sha256": authoritative_model_sha256(pilot_root),
        "tolerance": PARITY_TOLERANCE,
        "runs": runs,
    }
    _write_json(artifact_path, payload)
    return VerifiedParity(
        artifact_sha256=sha256_file(artifact_path),
        binary_sha256=str(payload["gpu_binary_sha256"]),
        coarse_batch_sha256=str(runs[0]["authoritative_batch_sha256"]),
        confirmation_batch_sha256=str(runs[1]["authoritative_batch_sha256"]),
        model_sha256=str(payload["authoritative_model_sha256"]),
    )


def verify_parity_artifact(
    artifact_path: Path,
    *,
    executable: Path,
    batch_paths: dict[int, Path],
    pilot_root: Path,
) -> VerifiedParity:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read parity artifact: {error}") from error
    required = {
        "schema",
        "gpu_binary_sha256",
        "authoritative_model_sha256",
        "tolerance",
        "runs",
    }
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.get("schema") != PARITY_SCHEMA:
        raise ValueError("unsupported or incomplete parity artifact")
    tolerance = payload["tolerance"]
    if type(tolerance) not in (int, float) or tolerance != PARITY_TOLERANCE:
        raise ValueError("parity artifact did not pass the frozen tolerance")
    if set(batch_paths) != {3, 6}:
        raise ValueError("verification requires exactly reps=3 and reps=6 batches")
    actual = {
        "gpu_binary_sha256": sha256_file(executable),
        "authoritative_model_sha256": authoritative_model_sha256(pilot_root),
    }
    for field, digest in actual.items():
        recorded = payload.get(field)
        if not isinstance(recorded, str) or len(recorded) != 64 or recorded != digest:
            raise ValueError(f"parity artifact {field} does not match")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("parity artifact must contain two runs")
    by_reps = {run.get("reps"): run for run in runs if isinstance(run, dict)}
    if set(by_reps) != {3, 6}:
        raise ValueError("parity artifact must cover exactly reps=3 and reps=6")
    batch_hashes = {}
    for reps in (3, 6):
        run = by_reps[reps]
        batch, expected = read_batch(batch_paths[reps])
        if batch.reps != reps or not expected or any(not math.isfinite(value) for value in expected):
            raise ValueError(f"reps={reps} parity batch is not fully scored")
        maximum = run.get("max_abs_difference")
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or not 0 <= maximum <= PARITY_TOLERANCE:
            raise ValueError(f"reps={reps} parity result is invalid")
        if type(run.get("jobs")) is not int or run["jobs"] != len(expected):
            raise ValueError(f"reps={reps} parity job count does not match")
        digest = sha256_file(batch_paths[reps])
        if run.get("authoritative_batch_sha256") != digest:
            raise ValueError(f"reps={reps} parity batch hash does not match")
        batch_hashes[reps] = digest
    return VerifiedParity(
        artifact_sha256=sha256_file(artifact_path),
        binary_sha256=actual["gpu_binary_sha256"],
        coarse_batch_sha256=batch_hashes[3],
        confirmation_batch_sha256=batch_hashes[6],
        model_sha256=actual["authoritative_model_sha256"],
    )


def gpu_check(
    batch_paths: dict[int, Path],
    executable: Path,
    artifact_path: Path,
    pilot_root: Path,
) -> VerifiedParity:
    if set(batch_paths) != {3, 6}:
        raise ValueError("CUDA parity requires exactly reps=3 and reps=6 batches")
    results = {}
    with tempfile.TemporaryDirectory(prefix="qtc-e-parity-") as directory:
        root = Path(directory)
        for reps in (3, 6):
            batch, expected = read_batch(batch_paths[reps])
            if batch.reps != reps or any(not math.isfinite(value) for value in expected):
                raise ValueError(f"CUDA reps={reps} parity requires finite authoritative references")
            output = root / f"scores-{reps}.bin"
            run_with_watchdog(
                (str(executable), str(batch_paths[reps]), str(output)),
                timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            )
            actual = read_scores(output, len(expected))
            maximum = max(abs(left - right) for left, right in zip(actual, expected))
            if not math.isfinite(maximum) or maximum > PARITY_TOLERANCE:
                raise ValueError(f"CUDA reps={reps} parity failed: max_abs_difference={maximum!r}")
            results[reps] = (len(expected), maximum)
    return create_parity_artifact(
        artifact_path,
        executable=executable,
        batch_paths=batch_paths,
        pilot_root=pilot_root,
        results=results,
    )


def _write_run_manifest(
    path: Path,
    *,
    status: str,
    verified: VerifiedParity,
    executable: Path,
    entries: list[dict[str, object]],
    started_unix: int,
    error: str | None = None,
) -> str:
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "started_unix": started_unix,
        "finished_unix": int(time.time()),
        "gpu_binary": str(executable),
        "gpu_binary_sha256": verified.binary_sha256,
        "parity_artifact_sha256": verified.artifact_sha256,
        "authoritative_batches_sha256": {
            "3": verified.coarse_batch_sha256,
            "6": verified.confirmation_batch_sha256,
        },
        "authoritative_model_sha256": verified.model_sha256,
        "batches": entries,
    }
    if error is not None:
        manifest["error"] = error
    _write_json(path, manifest)
    return sha256_file(path)


def hybrid_search(
    pilot_root: Path,
    config: BeamConfig,
    *,
    gpu_binary: Path,
    parity_artifact: Path,
    parity_batches: dict[int, Path],
    archive_dir: Path,
    timeout_seconds: int = GLOBAL_SEARCH_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """End-to-end CUDA beam, fail-closed behind exact parity and archive bindings."""
    if not 1 <= timeout_seconds <= GLOBAL_SEARCH_TIMEOUT_SECONDS:
        raise ValueError("whole-search timeout must be in 1..1800 seconds")
    verified = verify_parity_artifact(
        parity_artifact,
        executable=gpu_binary,
        batch_paths=parity_batches,
        pilot_root=pilot_root,
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, object]] = []
    manifest_path = archive_dir / "manifest.json"
    started_unix = int(time.time())
    deadline = time.monotonic() + timeout_seconds
    try:
        adapter = PilotSearch(pilot_root, config)
        seen = {adapter.initial}
        parents: dict[tuple[int, ...], BeamObservation[tuple[int, ...]]] = {}
        edges = 0

        def score_states(
            states: tuple[tuple[int, ...], ...],
            seeds: int,
            reps: int,
            label: str,
        ) -> tuple[float, ...]:
            if time.monotonic() >= deadline:
                raise TimeoutError("whole hybrid search exceeded 1800s")
            batch = Batch(tuple(encode_state(adapter, state) for state in states), tuple(range(seeds)), reps=reps)
            values = score_batch_cuda(
                batch,
                gpu_binary,
                deadline=deadline,
                archive_dir=archive_dir,
                label=label,
                manifest_entries=entries,
            )
            return tuple(
                sum(values[index * seeds : (index + 1) * seeds]) / seeds
                for index in range(len(states))
            )

        (initial_score,) = score_states((adapter.initial,), config.coarse_seeds, 3, "coarse-initial")
        initial = BeamObservation(adapter.initial, initial_score, 0, None, None)
        parents[adapter.initial] = initial
        observations = [initial]
        frontier = [initial]
        for depth in range(1, config.depth + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("whole hybrid search exceeded 1800s")
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
            scores = score_states(states, config.coarse_seeds, 3, f"coarse-depth-{depth}")
            level = []
            for (state, parent, move), score in zip(pending, scores):
                observation = BeamObservation(state, score, depth, parent, move)
                parents[state] = observation
                observations.append(observation)
                level.append(observation)
            frontier = sorted(level, key=lambda item: (item.score, item.state))[: config.width]

        leaders = tuple(sorted(observations, key=lambda item: (item.score, item.depth))[: config.width])
        confirmed = score_states(tuple(item.state for item in leaders), config.confirmation_seeds, 6, "confirmation")
        best_index = min(range(len(leaders)), key=lambda index: confirmed[index])
        best, best_raw = leaders[best_index], confirmed[best_index]
        (baseline_raw,) = score_states((adapter.identity,), config.confirmation_seeds, 6, "confirmation-baseline")
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
        manifest_sha256 = _write_run_manifest(
            manifest_path,
            status="complete",
            verified=verified,
            executable=gpu_binary,
            entries=entries,
            started_unix=started_unix,
        )
        return {
            "method": "CPU dependency/branch/recolor/prune + parity-bound CUDA candidate-seed scorer",
            "scorer": "cuda",
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
            "archive_manifest": str(manifest_path),
            "archive_manifest_sha256": manifest_sha256,
        }
    except BaseException as error:
        _write_run_manifest(
            manifest_path,
            status="failed",
            verified=verified,
            executable=gpu_binary,
            entries=entries,
            started_unix=started_unix,
            error=f"{type(error).__name__}: {error}",
        )
        raise


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


def authoritative_scores(
    adapter: PilotSearch,
    states: tuple[tuple[int, ...], ...],
    seeds: int,
    *,
    reps: int,
) -> tuple[float, ...]:
    values = []
    for state in states:
        lines, labels, _order, _main, _banks = adapter._lines(state)
        program = [adapter.isa.parse(line) for line in lines]
        sequence = adapter.sim2.trace_of(program, labels)
        values.extend(
            adapter.sim2.simulate(program, sequence, seed=seed, reps=reps)
            for seed in range(seeds)
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("authoritative sim2 returned a non-finite score")
    return tuple(values)


def authoritative_self_check(pilot_root: Path) -> dict[str, object]:
    """Exercise the real S10 model and binary codec for both search modes."""
    adapter = PilotSearch(pilot_root, BeamConfig())
    state = adapter.identity
    trace = encode_state(adapter, state)
    maxima = {}
    with tempfile.TemporaryDirectory(prefix="qtc-e-self-check-") as directory:
        root = Path(directory)
        for reps in (3, 6):
            batch = Batch((trace,), (0, 1), reps=reps)
            expected = authoritative_scores(adapter, (state,), 2, reps=reps)
            actual = score_batch(batch)
            maximum = max(abs(left - right) for left, right in zip(actual, expected))
            if not math.isfinite(maximum) or maximum > PARITY_TOLERANCE:
                raise ValueError(f"authoritative reps={reps} self-check failed: {maximum!r}")
            path = root / f"reps-{reps}.bin"
            write_batch(path, batch, expected)
            restored, restored_expected = read_batch(path)
            if restored.reps != reps or restored_expected != expected:
                raise ValueError(f"authoritative reps={reps} codec round trip failed")
            maxima[str(reps)] = maximum
    return {"self_check": "PASS", "jobs_per_reps": 2, "max_abs_difference": maxima}


def export_frontier(
    pilot_root: Path,
    output: Path,
    candidates: int,
    seeds: int,
    *,
    reps: int = 6,
) -> None:
    adapter = PilotSearch(pilot_root, BeamConfig(branch_budget=max(candidates - 1, 1)))
    states = [adapter.identity]
    states.extend(state for state, _move in adapter.neighbors(adapter.identity))
    chosen = tuple(states[:candidates])
    traces = tuple(encode_state(adapter, state) for state in chosen)
    batch = Batch(traces, tuple(range(seeds)), reps=reps)
    write_batch(output, batch, authoritative_scores(adapter, chosen, seeds, reps=reps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "self-check", "export", "check", "gpu-check", "hybrid-search"))
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--gpu-binary", type=Path)
    parser.add_argument("--parity-artifact", type=Path)
    parser.add_argument("--coarse-parity-batch", type=Path)
    parser.add_argument("--confirmation-parity-batch", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--branch-budget", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=int, default=GLOBAL_SEARCH_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.command in {"profile", "self-check", "export", "gpu-check", "hybrid-search"} and args.pilot_root is None:
        parser.error("profile/self-check/export/gpu-check/hybrid-search require --pilot-root")
    if args.command in {"export", "check"} and args.batch is None:
        parser.error("export/check require --batch")
    if args.command in {"gpu-check", "hybrid-search"} and args.gpu_binary is None:
        parser.error("gpu-check/hybrid-search requires --gpu-binary")
    if args.command == "gpu-check" and any(value is None for value in (
        args.parity_artifact, args.coarse_parity_batch, args.confirmation_parity_batch
    )):
        parser.error("gpu-check requires --parity-artifact and both parity batches")
    if args.command == "hybrid-search" and any(
        value is None for value in (
            args.parity_artifact,
            args.coarse_parity_batch,
            args.confirmation_parity_batch,
            args.archive_dir,
        )
    ):
        parser.error("hybrid-search requires --parity-artifact, both parity batches and --archive-dir")
    if args.command == "profile":
        print(json.dumps(dataclasses.asdict(profile(args.pilot_root, args.candidates, args.seeds)), indent=2))
    elif args.command == "self-check":
        print(json.dumps(authoritative_self_check(args.pilot_root), indent=2))
    elif args.command == "export":
        export_frontier(
            args.pilot_root,
            args.batch,
            args.candidates,
            args.seeds,
            reps=args.reps,
        )
    elif args.command == "check":
        batch, expected = read_batch(args.batch)
        actual = score_batch(batch)
        if any(not math.isfinite(value) for value in expected):
            raise ValueError("check requires a fully scored authoritative batch")
        maximum = max(abs(left - right) for left, right in zip(actual, expected))
        passes = math.isfinite(maximum) and maximum <= PARITY_TOLERANCE
        print(json.dumps({"jobs": len(actual), "max_abs_difference": maximum, "passes": passes}))
        if not passes:
            raise SystemExit(1)
    elif args.command == "gpu-check":
        verified = gpu_check(
            {3: args.coarse_parity_batch, 6: args.confirmation_parity_batch},
            args.gpu_binary,
            args.parity_artifact,
            args.pilot_root,
        )
        print(json.dumps({"parity": "PASS", **dataclasses.asdict(verified)}, indent=2))
    else:
        config = BeamConfig(
            depth=args.depth,
            width=args.width,
            branch_budget=args.branch_budget,
            coarse_seeds=args.seeds,
            confirmation_seeds=64,
        )
        result = hybrid_search(
            args.pilot_root,
            config,
            gpu_binary=args.gpu_binary,
            parity_artifact=args.parity_artifact,
            parity_batches={3: args.coarse_parity_batch, 6: args.confirmation_parity_batch},
            archive_dir=args.archive_dir,
            timeout_seconds=args.timeout_seconds,
        )
        if args.result is not None:
            _write_json(args.result, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
