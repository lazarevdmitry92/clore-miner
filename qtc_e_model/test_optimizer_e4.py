from __future__ import annotations

from miner.qtc_e_model import optimizer_e4 as opt


def test_relocate_is_non_adjacent_and_preserves_every_item():
    state = (0, 1, 2, 3, 4)
    moved = opt.relocate(state, 1, 4)
    assert moved == (0, 2, 3, 4, 1)
    assert sorted(moved) == list(state)


def test_beam_can_find_a_two_move_result_without_cartesian_enumeration():
    initial = (0, 1, 2, 3, 4)

    def neighbors(state):
        yield opt.relocate(state, 0, 2), (0, 2)
        yield opt.relocate(state, 4, 2), (4, 2)

    target = (1, 2, 4, 0, 3)
    observations, generated = opt.beam_search(
        initial,
        neighbors=neighbors,
        evaluate=lambda state: sum(left != right for left, right in zip(state, target)),
        depth=2,
        width=2,
    )
    assert generated == 6
    assert len(observations) == 7 < 120  # graph-directed search, not all 5! permutations
    best = min(observations, key=lambda item: item.score)
    assert best.depth == 2
    assert best.state == target


def test_fixed_frame_certifies_56_register_gate_is_unreachable():
    gate = opt.occupancy_gate((1, 6, 52, 53, 55, 56))
    assert gate.maximum_allowed_physical_register == 53
    assert gate.maximum_fixed_physical_register == 56
    assert gate.lower_bound_declared_registers == 59
    assert not gate.reachable_without_frame_remap


def test_saturation_stops_schedule_family_but_opens_arithmetic_cycle():
    gate = opt.occupancy_gate((1, 6, 56))
    decision = opt.assess_saturation(
        (1.03451, 1.03447, 1.03444),
        (4, 4, 4),
        gate,
    )
    assert decision.same_resource_basin
    assert decision.stagnated
    assert decision.current_family_stopped
    assert decision.arithmetic_restructure_required
    assert decision.maximum_gain_over_baseline < 0.0005


def test_recorded_moves_replay_deterministically():
    moves = ((47, 43), (64, 69), (627, 629))
    state = opt.state_after_moves(683, moves)
    assert len(state) == 683
    assert len(set(state)) == 683
    assert state != tuple(range(683))
