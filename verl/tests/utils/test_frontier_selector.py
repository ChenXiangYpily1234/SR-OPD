# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the cost-aware Frontier trajectory selector (FF-Cost).

The score under test is

    J_j = (d_j + eps) * c_j^0.5
    log S_j = -log(d_j + eps) - 0.5 * log(c_j)

with d_j the raw nearest-positive profile distance and c_j the raw Teacher
cost (prompt + response tokens). argmin J == argmax log S.
"""

import math

import pytest

from verl.utils.frontier_selector import (
    FRONTIER_SELECTOR,
    FrontierTrajectorySelector,
)


def _constant_row(length: int, value: float) -> list[float]:
    return [float(value)] * length


def _run_select(selector, log_prob_rows, correct, valid, costs, prompt_uid="p"):
    masks = [[True] * len(row) for row in log_prob_rows]
    return selector.select(
        response_masks=masks,
        verifier_correct=correct,
        rollout_valid=valid,
        student_sampled_log_probs=log_prob_rows,
        prompt_uid=prompt_uid,
        attempt_id=1,
        candidate_costs=costs,
    )


def test_cost_aware_picks_cheaper_sibling_when_distance_is_close():
    """A is nearest but expensive; B is slightly farther yet much cheaper, so
    the raw FF-Cost objective must switch from A to B."""
    rows = [
        _constant_row(10, 0.0),  # positive
        _constant_row(8, 0.50),  # A: nearest (d=0.50), most expensive (1600)
        _constant_row(12, 0.5075),  # B: d=0.5075, cheap (400)
        _constant_row(6, 1.0),  # C: far (d=1.0), medium cost (200)
    ]
    correct = [1, 0, 0, 0]
    valid = [True] * 4
    costs = [100.0, 1600.0, 400.0, 200.0]

    result = _run_select(FrontierTrajectorySelector(seed=42, cost_aware=True), rows, correct, valid, costs)

    assert result.selection_mode == FRONTIER_SELECTOR + "_cost"
    assert not result.fallback_used
    # Strict nearest-positive distances are the constant gaps.
    assert result.candidate_nearest_positive_distances == pytest.approx({1: 0.50, 2: 0.5075, 3: 1.0}, abs=1e-6)
    expected_objectives = {
        1: 0.50 * math.sqrt(1600.0),
        2: 0.5075 * math.sqrt(400.0),
        3: 1.0 * math.sqrt(200.0),
    }
    assert result.candidate_objectives == pytest.approx(expected_objectives, abs=1e-6)
    expected_log_scores = {
        1: -math.log(0.50 + 1e-8) - 0.5 * math.log(1600.0),
        2: -math.log(0.5075 + 1e-8) - 0.5 * math.log(400.0),
        3: -math.log(1.0 + 1e-8) - 0.5 * math.log(200.0),
    }
    assert result.candidate_log_scores == pytest.approx(expected_log_scores, abs=1e-6)
    # B wins despite A being the nearest positive sibling.
    assert result.selected_negative_idx == 2
    assert result.nearest_only_selected_idx == 1
    assert result.lowest_cost_selected_idx == 3
    assert result.cost_switch
    assert result.selected_objective == pytest.approx(expected_objectives[2], abs=1e-6)
    assert result.selected_log_score == pytest.approx(expected_log_scores[2], abs=1e-6)
    assert result.selected_cost == pytest.approx(400.0)
    assert result.nearest_only_cost == pytest.approx(1600.0)
    assert result.distance_regret == pytest.approx(0.0075, abs=1e-6)
    assert result.relative_cost_reduction == pytest.approx(0.75, abs=1e-6)
    # The matched positive is the only correct sibling.
    assert result.matched_nearest_positive_idx == 0


def test_nearest_only_mode_ignores_costs():
    """The ablation baseline (cost_aware=False) must pick the nearest sibling
    regardless of Teacher cost, and report no switch."""
    rows = [
        _constant_row(10, 0.0),
        _constant_row(8, 0.50),
        _constant_row(12, 0.5075),
        _constant_row(6, 1.0),
    ]
    selector = FrontierTrajectorySelector(seed=42, cost_aware=False)
    result = _run_select(selector, rows, [1, 0, 0, 0], [True] * 4, [100.0, 1600.0, 400.0, 200.0])

    assert result.selection_mode == FRONTIER_SELECTOR
    assert result.selected_negative_idx == 1
    assert result.nearest_only_selected_idx == 1
    assert not result.cost_aware
    assert not result.cost_switch
    assert result.distance_regret == 0.0
    assert result.relative_cost_reduction == 0.0
    assert result.candidate_objectives == pytest.approx({1: 0.50, 2: 0.5075, 3: 1.0}, abs=1e-6)
    assert result.candidate_log_scores == pytest.approx({1: -0.50, 2: -0.5075, 3: -1.0}, abs=1e-6)


def test_selection_distance_is_min_not_mean_over_positives():
    """With several positives, d_{p,j} = min_i D(y^-_j, y^+_i): the negative
    sitting next to one positive (but far from the other) must beat the
    negative that is mediocre against both, and matched_nearest_positive must
    be the argmin row."""
    rows = [
        _constant_row(10, 0.0),  # positive 1
        _constant_row(10, 10.0),  # positive 2
        _constant_row(10, 1.0),  # X: min distance 1.0, mean 5.0
        _constant_row(10, 6.0),  # Y: min distance 4.0, mean 5.0
    ]
    correct = [1, 1, 0, 0]
    valid = [True] * 4
    # Uniform costs make the cost term a constant, so ranking is pure distance.
    costs = [50.0, 50.0, 50.0, 50.0]
    selector = FrontierTrajectorySelector(seed=42, cost_aware=True)
    result = _run_select(selector, rows, correct, valid, costs)

    assert result.candidate_nearest_positive_distances == pytest.approx({2: 1.0, 3: 4.0}, abs=1e-6)
    assert result.candidate_mean_positive_distances == pytest.approx({2: 5.0, 3: 5.0}, abs=1e-6)
    assert result.selected_negative_idx == 2
    assert result.matched_nearest_positive_idx == 0
    assert not result.cost_switch  # identical costs cannot induce a switch


def test_single_negative_direct_short_circuits_scoring():
    selector = FrontierTrajectorySelector(seed=42, cost_aware=True)
    rows = [_constant_row(10, 0.0), _constant_row(8, 1.0)]
    result = _run_select(selector, rows, [1, 0], [True, True], [10.0, 5.0])

    assert result.selection_mode == "single_negative_direct"
    assert result.selected_negative_idx == 1
    assert result.nearest_only_selected_idx == 1
    assert result.lowest_cost_selected_idx == 1
    assert result.selected_cost == pytest.approx(5.0)
    assert result.nearest_only_cost == pytest.approx(5.0)
    assert not result.cost_switch
    assert result.distance_regret == 0.0
    assert result.relative_cost_reduction == 0.0


def test_equal_distance_prefers_lower_cost_in_selector():
    """Equal distances must not fall back to uniform: the cheaper candidate
    wins through the raw objective itself."""
    rows = [
        _constant_row(10, 0.0),
        _constant_row(10, 1.0),  # d=1.0, cost 1000
        _constant_row(10, 1.0),  # d=1.0, cost 500
    ]
    selector = FrontierTrajectorySelector(seed=42, cost_aware=True)
    result = _run_select(selector, rows, [1, 0, 0], [True] * 3, [9.0, 1000.0, 500.0])

    assert not result.fallback_used
    assert result.selection_mode == FRONTIER_SELECTOR + "_cost"
    assert result.selected_negative_idx == 2
    # Nearest-only ties break on the rollout index.
    assert result.nearest_only_selected_idx == 1
    assert result.lowest_cost_selected_idx == 2
    assert result.cost_switch
    assert result.distance_regret == pytest.approx(0.0, abs=1e-8)
    assert result.relative_cost_reduction == pytest.approx(0.5, abs=1e-6)


def test_cost_aware_requires_candidate_costs():
    selector = FrontierTrajectorySelector(seed=42, cost_aware=True)
    rows = [
        _constant_row(10, 0.0),
        _constant_row(8, 0.5),
        _constant_row(12, 0.55),
    ]
    with pytest.raises(ValueError, match="candidate_costs"):
        _run_select(selector, rows, [1, 0, 0], [True] * 3, None)


