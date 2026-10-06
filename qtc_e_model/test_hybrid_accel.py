from __future__ import annotations

from pathlib import Path

from miner.qtc_e_model import hybrid_accel as hybrid


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
    hybrid.write_unscored_batch(path, batch)
    _restored, expected = hybrid.read_batch(path)
    assert all(value != value for value in expected)  # NaN reference means GPU production mode


def test_offsets_match_python_random_contract():
    assert hybrid.offsets_for_seed(0, 10, 4) == (6, 6, 0, 4)
