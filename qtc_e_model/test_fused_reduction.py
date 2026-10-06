from __future__ import annotations

import random

import pytest

from miner.qtc_e_model import fused_reduction as model


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("r3", [0, 1, model.M32 - 1])
@pytest.mark.parametrize("lo", [0, 1, model.M32 - 1, model.M32])
@pytest.mark.parametrize("hi", [0, 1, model.M32 - 1, model.M32])
def test_fused_correction_matches_materialized_boundary_states(k, r3, lo, hi):
    w = hi << 32 | lo
    c = (r3 + k) & model.M32
    borrow = int(lo < c)
    expected = (((hi + k - borrow) & model.M32) << 32) | ((lo - c) & model.M32)
    assert model.fused_correction(w, k, r3) == expected


def test_fused_correction_rejects_the_unreachable_wrapping_carry_word():
    with pytest.raises(ValueError, match="no-wrap"):
        model.fused_correction(0, 1, model.M32)


def test_fused_fold_matches_materialized_for_random_words():
    rng = random.Random(0xE)
    for _ in range(20_000):
        low = rng.getrandbits(64)
        high = rng.getrandbits(32)
        # The real partial arithmetic keeps r3+k in one word.
        r3 = rng.randrange(model.M32)
        assert model.fold_words_fused(low, high, r3) == model.fold_words_materialized(low, high, r3)


def test_partial_candidate_passes_model_gate_and_constraints():
    candidate = model.build_report().candidate
    assert candidate.instructions_removed_per_nonce == 528
    assert candidate.partial_f == 334
    assert candidate.partial_a == 336
    assert candidate.registers == 64
    assert candidate.predicates == 6
    assert candidate.scoreboards == 6
    assert candidate.spills == 0
    assert candidate.central_ed == pytest.approx(1.0198986887600312)
    assert candidate.lcb95_ed == pytest.approx(1.0121502697963802)
    assert candidate.passes_gate


def test_resource_bound_is_exact_for_the_binary_fusion_family():
    certificate = model.build_report().certificate
    assert certificate.variables == 24
    assert certificate.assignments == 2**24
    assert certificate.exact_resource_optimum
    assert certificate.fused_full_outputs == 12
    assert certificate.fused_partial_outputs == 12
    assert certificate.central_ed_upper == pytest.approx(1.0583434947392545)
    assert certificate.lcb95_ed_upper == pytest.approx(1.0503030012127217)
