from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

import pytest

from miner.qtc_e_model import hybrid_accel as hybrid
from miner.qtc_e_model.optimizer_e4 import BeamConfig, PilotSearch


def _trace() -> hybrid.EncodedTrace:
    return hybrid.EncodedTrace(
        pipe=bytes((1, 2, 3, 4)),
        stall=bytes((1, 2, 1, 5)),
        wait=bytes((0, 0, 0, 0)),
        write_barrier=bytes((7, 7, 0, 7)),
        bank_conflict=bytes((0, 1, 0, 0)),
    )


def test_reference_batch_is_deterministic_and_seed_sensitive():
    batch = hybrid.Batch((_trace(),), (0, 1, 2), nwarps=2, reps=2)
    first = hybrid.score_batch(batch)
    assert first == hybrid.score_batch(batch)
    assert len(first) == 3
    assert all(value > 0 for value in first)


def test_binary_round_trip_preserves_cpu_parity(tmp_path: Path):
    batch = hybrid.Batch((_trace(), _trace()), (0, 1), nwarps=2, reps=2)
    path = tmp_path / "frontier.bin"
    hybrid.write_batch(path, batch)
    restored, expected = hybrid.read_batch(path)
    assert hybrid.score_batch(restored) == expected


def test_unscored_batch_contains_no_accidental_cpu_claim(tmp_path: Path):
    batch = hybrid.Batch((_trace(),), (0, 1), nwarps=2, reps=2)
    path = tmp_path / "frontier.bin"
    hybrid._write_unscored_batch(path, batch)
    _restored, expected = hybrid.read_batch(path)
    assert all(value != value for value in expected)  # NaN reference means GPU production mode


def test_offsets_match_python_random_contract():
    assert hybrid.offsets_for_seed(0, 10, 4) == (6, 6, 0, 4)


def test_read_batch_rejects_wrong_size_offsets_domains_and_nonfinite_values(tmp_path: Path):
    batch = hybrid.Batch((_trace(),), (0,), nwarps=2, reps=3)
    good = tmp_path / "good.bin"
    hybrid.write_batch(good, batch)
    payload = good.read_bytes()

    trailing = tmp_path / "trailing.bin"
    trailing.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="exactly"):
        hybrid.read_batch(trailing)

    bad_pipe = tmp_path / "pipe.bin"
    changed = bytearray(payload)
    changed[hybrid.HEADER.size] = 5
    bad_pipe.write_bytes(changed)
    with pytest.raises(ValueError, match="pipe"):
        hybrid.read_batch(bad_pipe)

    bad_offset = tmp_path / "offset.bin"
    changed = bytearray(payload)
    offset_position = hybrid.HEADER.size + 5 * _trace().length
    original_offset = struct.unpack_from("<H", changed, offset_position)[0]
    struct.pack_into("<H", changed, offset_position, (original_offset + 1) % _trace().length)
    bad_offset.write_bytes(changed)
    with pytest.raises(ValueError, match="authoritative seed"):
        hybrid.read_batch(bad_offset)

    infinite = tmp_path / "infinite.bin"
    changed = bytearray(payload)
    struct.pack_into("<d", changed, len(changed) - 8, math.inf)
    infinite.write_bytes(changed)
    with pytest.raises(ValueError, match="infinite"):
        hybrid.read_batch(infinite)


@pytest.mark.parametrize(("header_index", "value", "message"), (
    (2, 0, "candidate"),
    (3, 0, "trace length"),
    (4, 0, "seed count"),
    (5, 0, "nwarps"),
    (5, 9, "nwarps"),
    (6, 0, "reps"),
))
def test_read_batch_rejects_unsafe_dimensions(
    tmp_path: Path,
    header_index: int,
    value: int,
    message: str,
):
    path = tmp_path / f"bad-{header_index}-{value}.bin"
    batch = hybrid.Batch((_trace(),), (0,), nwarps=2, reps=3)
    hybrid.write_batch(path, batch)
    changed = bytearray(path.read_bytes())
    header = list(hybrid.HEADER.unpack_from(changed))
    header[header_index] = value
    hybrid.HEADER.pack_into(changed, 0, *header)
    path.write_bytes(changed)
    with pytest.raises(ValueError, match=message):
        hybrid.read_batch(path)


def test_read_scores_rejects_nonfinite_cuda_output(tmp_path: Path):
    output = tmp_path / "scores.bin"
    output.write_bytes(struct.pack("<d", math.nan))
    with pytest.raises(ValueError, match="non-finite"):
        hybrid.read_scores(output, 1)


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    for relative in hybrid.AUTHORITATIVE_PILOT_SOURCES:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# authoritative test model: {relative}\n", encoding="utf-8")
    return root


def test_parity_artifact_binds_exact_binary_batch_and_model(tmp_path: Path):
    binary = tmp_path / "scorer"
    binary.write_bytes(b"binary-v1")
    batch_paths = {3: tmp_path / "parity-3.bin", 6: tmp_path / "parity-6.bin"}
    for reps, batch_path in batch_paths.items():
        batch = hybrid.Batch((_trace(),), (0,), nwarps=2, reps=reps)
        hybrid.write_batch(batch_path, batch)
    pilot_root = _model_root(tmp_path)
    artifact = tmp_path / "parity.json"
    hybrid.create_parity_artifact(
        artifact,
        executable=binary,
        batch_paths=batch_paths,
        pilot_root=pilot_root,
        results={3: (1, 0.0), 6: (1, 0.0)},
    )
    hybrid.verify_parity_artifact(
        artifact,
        executable=binary,
        batch_paths=batch_paths,
        pilot_root=pilot_root,
    )

    binary.write_bytes(b"binary-v2")
    with pytest.raises(ValueError, match="gpu_binary_sha256"):
        hybrid.verify_parity_artifact(
            artifact,
            executable=binary,
            batch_paths=batch_paths,
            pilot_root=pilot_root,
        )

    binary.write_bytes(b"binary-v1")
    control_model = pilot_root / "s10" / "tools" / "lat_sm70.json"
    original_control_model = control_model.read_text(encoding="utf-8")
    control_model.write_text(original_control_model + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative_model_sha256"):
        hybrid.verify_parity_artifact(
            artifact,
            executable=binary,
            batch_paths=batch_paths,
            pilot_root=pilot_root,
        )
    control_model.write_text(original_control_model, encoding="utf-8")

    batch_paths[3].write_bytes(batch_paths[3].read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="reps=3 parity batch hash"):
        hybrid.verify_parity_artifact(
            artifact,
            executable=binary,
            batch_paths=batch_paths,
            pilot_root=pilot_root,
        )


def test_parity_suite_refuses_a_missing_repetition(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly reps=3 and reps=6"):
        hybrid.create_parity_artifact(
            tmp_path / "parity.json",
            executable=tmp_path / "missing",
            batch_paths={},
            pilot_root=tmp_path,
            results={},
        )


def test_gpu_check_runs_and_attests_both_repetition_modes(tmp_path: Path):
    scorer = tmp_path / "scorer"
    scorer.write_text(
        f"#!{sys.executable}\n"
        "import struct, sys\n"
        "data = open(sys.argv[1], 'rb').read()\n"
        "_m, _v, candidates, _length, seeds, _warps, _reps, _fields = struct.unpack_from('<8s7I', data)\n"
        "open(sys.argv[2], 'wb').write(data[-candidates * seeds * 8:])\n",
        encoding="utf-8",
    )
    scorer.chmod(0o755)
    batch_paths = {3: tmp_path / "parity-3.bin", 6: tmp_path / "parity-6.bin"}
    for reps, path in batch_paths.items():
        hybrid.write_batch(path, hybrid.Batch((_trace(),), (0, 1), nwarps=2, reps=reps))
    artifact = tmp_path / "parity.json"
    verified = hybrid.gpu_check(batch_paths, scorer, artifact, _model_root(tmp_path))
    assert verified.coarse_batch_sha256 == hybrid.sha256_file(batch_paths[3])
    assert verified.confirmation_batch_sha256 == hybrid.sha256_file(batch_paths[6])


def test_watchdog_times_out_and_reaps_process_group():
    with pytest.raises(TimeoutError):
        hybrid.run_with_watchdog(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            timeout_seconds=0.05,
            watchdog_seconds=0.01,
        )


def _authoritative_pilot_root() -> Path | None:
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / "pilots" / "miner" / "qtc_own_miner"]
    marker = repo / ".git"
    if marker.is_file():
        gitdir = Path(marker.read_text(encoding="utf-8").strip().split(":", 1)[1].strip())
        commondir = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
        candidates.append(commondir.parent / "pilots" / "miner" / "qtc_own_miner")
    return next((candidate for candidate in candidates if (candidate / "s10" / "frag" / "sim2.py").is_file()), None)


def test_encoded_real_trace_matches_authoritative_sim2_for_coarse_and_confirmation_reps():
    pilot_root = _authoritative_pilot_root()
    if pilot_root is None:
        pytest.skip("ignored authoritative S10 pilot is not installed")
    adapter = PilotSearch(pilot_root, BeamConfig())
    lines, labels, _order, _main, _banks = adapter._lines(adapter.identity)
    program = [adapter.isa.parse(line) for line in lines]
    sequence = adapter.sim2.trace_of(program, labels)
    trace = hybrid.encode_lines(adapter, lines, labels)
    for reps in (3, 6):
        for seed in (0, 1):
            authoritative = adapter.sim2.simulate(program, sequence, seed=seed, reps=reps)
            encoded = hybrid.score_trace(trace, seed, nwarps=8, reps=reps)
            assert encoded == pytest.approx(authoritative, abs=1e-12)


def test_self_check_runs_from_extracted_minimal_pilot_payload(tmp_path: Path):
    pilot_root = _authoritative_pilot_root()
    if pilot_root is None:
        pytest.skip("ignored authoritative S10 pilot is not installed")
    kernel = pilot_root / "s6" / "s7" / "mining_n1n3s3e.cu"
    if not kernel.is_file():
        pytest.fail(f"authoritative S6/S7 dependency is missing: {kernel}")

    archive = tmp_path / "pilot-model.tar.gz"
    prefix = Path("pilots/miner/qtc_own_miner")
    with tarfile.open(archive, "w:gz") as output:
        output.add(pilot_root / "s10", arcname=prefix / "s10")
        output.add(kernel, arcname=prefix / "s6/s7/mining_n1n3s3e.cu")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(extracted, filter="data")

    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "miner.qtc_e_model.hybrid_accel",
            "self-check",
            "--pilot-root",
            str(extracted / prefix),
        ),
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"self_check": "PASS"' in result.stdout


def test_rental_entrypoint_is_syntax_checked_and_done_is_after_manifest_gate():
    root = Path(__file__).parent
    entrypoint = root / "rental_entrypoint.sh"
    subprocess.run(("bash", "-n", str(entrypoint)), check=True)
    source = entrypoint.read_text(encoding="utf-8")
    assert "platforms.clore" not in source and "platforms.vast" not in source
    assert "apt-get" not in source and "pytest" not in source
    assert 'pilot-model.tar.gz' in source
    assert 's6/s7/mining_n1n3s3e.cu' in source
    assert all(name in source for name in ("tools/guard.py", "tools/sass_ir.py", "tools/lat_sm70.json"))
    assert "! -name run.log ! -name http.log ! -name compute-timeout.log" in source
    assert source.index('manifest.get("status") != "complete"') < source.index('mv "$DONE.tmp" "$DONE"')
    dockerfile = (root / "rental_image.Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM nvidia/cuda:12.6.3-devel-ubuntu24.04\n")
    assert "pilot-model.tar.gz" in dockerfile and "pilot-s10.tar.gz" not in dockerfile


def test_cli_has_no_public_unscored_export_path():
    result = subprocess.run(
        (sys.executable, "-m", "miner.qtc_e_model.hybrid_accel", "--help"),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--unscored" not in result.stdout
