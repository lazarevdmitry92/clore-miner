from __future__ import annotations

import pytest

from miner.qtc_e_model import optimizer_v3 as opt


def test_process_decomposition_reconstructs_real_ir_totals():
    assert opt.sum_processes(opt.PARTIAL_PROCESSES, "d") == (194, 358, 336)
    assert opt.sum_processes(opt.PARTIAL_PROCESSES, "e") == (182, 334, 336)
    assert opt.sum_processes(opt.FULL_PROCESSES_D, "d") == (698, 1098, 1130)
    assert opt.sum_processes(opt.FULL_PROCESSES_D, "e") == (689, 1080, 1130)


def test_best_first_is_objective_directed_not_mask_enumeration():
    target = 0b1011
    observations = opt.best_first_masks(4, 12, lambda mask: (mask ^ target).bit_count())
    assert min(observations, key=lambda item: item[1]) == (target, 0)
    assert [mask for mask, _score in observations[:5]] != list(range(5))


def test_pattern_search_descends_without_cartesian_grid():
    point, score, evaluations = opt.pattern_search(
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        lambda p: (p["x"] - 3.0) ** 2 + (p["y"] + 2.0) ** 2,
        rounds=8,
    )
    assert point == {"x": 3.0, "y": -2.0}
    assert score == 0
    assert evaluations <= 40


def test_gap_ladder_is_monotone_and_matches_certificate():
    ladder = opt.gap_ladder()
    assert [point.corrected_ed for point in ladder] == sorted(point.corrected_ed for point in ladder)
    assert ladder[0].corrected_ed == pytest.approx(1.0426574796546892)
    assert ladder[-1].corrected_ed == pytest.approx(1.065047994784599)
    assert ladder[-1].corrected_ed / ladder[0].corrected_ed - 1 == pytest.approx(0.02147446842977141)


def test_residual_uncertainty_is_dominated_by_model_and_candidate_transfer():
    report = opt.build_optimizer_report()
    variance = report["residual_variance"]
    assert variance["model"] / variance["total"] == pytest.approx(0.5205, abs=1e-4)
    assert variance["candidate"] / variance["total"] == pytest.approx(0.3631, abs=1e-4)
    assert report["residual_sensitivity"]["both"] > report["residual_sensitivity"]["current"]
    assert min(lower for _central, lower in report["leave_one_group_out"].values()) > 1.029


def test_report_distinguishes_global_resource_bound_from_schedule_search():
    report = opt.build_optimizer_report()
    assert report["search"]["full"]["best_mask"] == 4081
    assert report["search"]["full"]["globally_proven_schedule_optimum"] is False
    assert report["best_feasible"]["corrected_ed"] > 1.04
    assert report["best_feasible"]["lcb95_ed"] < 1.04
