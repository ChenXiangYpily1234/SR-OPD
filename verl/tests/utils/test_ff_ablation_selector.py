# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import pytest
import torch

from verl.utils.boundary_opd import is_boundary_selector_mode
from verl.utils.ff_opd import FFOPDConfig, FFOPDQueueManager
from verl.utils.frontier_selector import FF_SELECTOR_MODES, select_ff_rollouts

DIRECT_SELECTOR_MODES = tuple(
    mode
    for mode in FF_SELECTOR_MODES
    if not is_boundary_selector_mode(mode)
    and mode not in {"shortest_wrong", "all_wrong", "frontier_tlr"}
)
PAYLOAD_FREE_MODES = tuple(mode for mode in FF_SELECTOR_MODES if not is_boundary_selector_mode(mode))


@pytest.fixture
def selector_inputs():
    valid = torch.tensor(
        [
            [1, 1, 1, 1],  # frontier
            [1, 1, 1, 0],  # frontier
            [1, 1, 1, 1],  # all correct
            [1, 1, 0, 0],  # all wrong
        ],
        dtype=torch.bool,
    )
    correct = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    distance = torch.tensor(
        [
            [math.nan, 0.1, 0.8, math.nan],
            [0.4, math.nan, 0.2, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
            [math.nan, math.nan, math.nan, math.nan],
        ],
        dtype=torch.float64,
    )
    cost = torch.tensor(
        [
            [100, 900, 100, 100],
            [100, 100, 800, 100],
            [100, 100, 100, 100],
            [100, 100, 100, 100],
        ],
        dtype=torch.float64,
    )
    return valid, correct, distance, cost


def _select(inputs, mode, seed=7, step=3):
    valid, correct, distance, cost = inputs
    return select_ff_rollouts(
        valid_mask=valid,
        correct_mask=correct,
        distance=distance,
        teacher_cost=cost,
        mode=mode,
        base_seed=seed,
        global_step=step,
    )


def test_random_wrong_selects_one_wrong_per_frontier(selector_inputs):
    selected, stats = _select(selector_inputs, "random_wrong")
    assert selected.sum(dim=1).tolist() == [1, 1, 0, 0]
    assert not (selected & selector_inputs[1]).any()
    assert stats["actual_query_count"] == 2


def test_random_correct_selects_one_correct_per_frontier(selector_inputs):
    selected, stats = _select(selector_inputs, "random_correct")
    assert selected.sum(dim=1).tolist() == [1, 1, 0, 0]
    assert not (selected & ~selector_inputs[1]).any()
    assert stats["selected_correct_count"] == 2


def test_all_wrong_selects_every_valid_wrong_rollout_from_frontier_prompts(selector_inputs):
    selected, stats = _select(selector_inputs, "all_wrong")
    assert selected.sum(dim=1).tolist() == [2, 2, 0, 0]
    assert not (selected & selector_inputs[1]).any()
    assert stats["target_query_count"] == 4
    assert stats["actual_query_count"] == 4
    assert stats["selected_wrong_count"] == 4


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("nearest_only", [(0, 1), (1, 2)]),
        ("cost_only", [(0, 2), (1, 0)]),
        ("farthest_only", [(0, 2), (1, 0)]),
    ],
)
def test_deterministic_selector_semantics(selector_inputs, mode, expected):
    selected, _ = _select(selector_inputs, mode)
    assert set(map(tuple, selected.nonzero(as_tuple=False).tolist())) == set(expected)


def test_global_random_count_equals_frontier_and_only_valid(selector_inputs):
    selected, stats = _select(selector_inputs, "global_random")
    assert selected.sum().item() == 2
    assert not (selected & ~selector_inputs[0]).any()
    assert stats["selected_valid_count"] == stats["actual_query_count"]


@pytest.mark.parametrize(
    ("valid", "correct"),
    [
        (torch.ones(4, 4, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool)),
        (torch.ones(4, 4, dtype=torch.bool), torch.zeros(4, 4, dtype=torch.bool)),
        (torch.zeros(4, 4, dtype=torch.bool), torch.zeros(4, 4, dtype=torch.bool)),
    ],
)
@pytest.mark.parametrize("mode", DIRECT_SELECTOR_MODES)
def test_no_frontier_edge_cases(valid, correct, mode):
    distance = torch.full((4, 4), math.nan, dtype=torch.float64)
    cost = torch.ones(4, 4, dtype=torch.float64)
    selected, stats = select_ff_rollouts(
        valid_mask=valid,
        correct_mask=correct,
        distance=distance,
        teacher_cost=cost,
        mode=mode,
        base_seed=1,
        global_step=0,
    )
    assert not selected.any()
    assert stats["zero_frontier_step"] == 1


@pytest.mark.parametrize("mode", ["nearest_only", "cost_only", "farthest_only"])
def test_seeded_tie_break_is_reproducible_and_can_change(mode):
    valid = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    correct = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    distance = torch.tensor([[math.nan, 0.5, 0.5, 0.5]], dtype=torch.float64)
    cost = torch.tensor([[1.0, 100.0, 100.0, 100.0]], dtype=torch.float64)

    picks = []
    for seed in range(8):
        args = dict(
            valid_mask=valid,
            correct_mask=correct,
            distance=distance,
            teacher_cost=cost,
            mode=mode,
            base_seed=seed,
            global_step=4,
        )
        first, stats = select_ff_rollouts(**args)
        second, _ = select_ff_rollouts(**args)
        assert torch.equal(first, second)
        assert stats["tie_count"] == 1
        picks.append(first.nonzero(as_tuple=False)[0, 1].item())
    assert len(set(picks)) > 1


@pytest.mark.parametrize("mode", ["random_wrong", "nearest_only", "cost_only", "farthest_only"])
def test_non_global_never_selects_correct_invalid_or_nonfrontier(selector_inputs, mode):
    selected, _ = _select(selector_inputs, mode)
    valid, correct, _, _ = selector_inputs
    wrong = valid & ~correct
    frontier = (valid & correct).any(dim=1) & wrong.any(dim=1)
    assert not (selected & ~wrong).any()
    assert not (selected & ~frontier[:, None]).any()


def test_selected_mask_filters_teacher_batch_exactly(selector_inputs):
    selected, stats = _select(selector_inputs, "nearest_only")
    teacher_batch = torch.arange(16).reshape(4, 4)[selected]
    assert teacher_batch.numel() == selected.sum().item()
    assert teacher_batch.numel() == stats["actual_query_count"]


@pytest.mark.parametrize("mode", ["nearest_only", "farthest_only"])
def test_missing_profile_distance_uses_seeded_uniform_fallback(mode):
    valid = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    correct = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    distance = torch.full((1, 4), math.nan, dtype=torch.float64)
    cost = torch.tensor([[10, 20, 30, 40]], dtype=torch.float64)
    first, stats = select_ff_rollouts(
        valid_mask=valid,
        correct_mask=correct,
        distance=distance,
        teacher_cost=cost,
        mode=mode,
        base_seed=7,
        global_step=3,
    )
    second, _ = select_ff_rollouts(
        valid_mask=valid,
        correct_mask=correct,
        distance=distance,
        teacher_cost=cost,
        mode=mode,
        base_seed=7,
        global_step=3,
    )
    assert torch.equal(first, second)
    assert first.sum() == 1
    assert stats["tie_count"] == 1
    assert math.isfinite(stats["selected_distance_mean"])


@pytest.mark.parametrize("mode", DIRECT_SELECTOR_MODES)
def test_three_step_smoke_invariants(selector_inputs, mode):
    for step in range(3):
        selected, stats = _select(selector_inputs, mode, seed=42, step=step)
        assert torch.isfinite(torch.tensor(list(stats.values()))).all()
        assert stats["actual_query_count"] == stats["frontier_prompt_count"]
        if mode == "global_random":
            assert stats["selected_valid_count"] == stats["actual_query_count"]
        elif mode == "random_correct":
            assert stats["selected_correct_count"] == stats["actual_query_count"]
            assert stats["selected_nonfrontier_count"] == 0
        else:
            assert stats["selected_correct_count"] == 0
            assert stats["selected_nonfrontier_count"] == 0
        assert selected.sum().item() == stats["actual_query_count"]


def test_invalid_mode_is_rejected(selector_inputs):
    with pytest.raises(ValueError, match="algorithm.ff_selector_mode"):
        _select(selector_inputs, "not_a_mode")


@pytest.mark.parametrize("mode", PAYLOAD_FREE_MODES)
def test_queue_manager_crops_teacher_batch_before_forward(tmp_path, mode):
    manager = FFOPDQueueManager(
        FFOPDConfig(
            k_rollouts=4,
            selector_mode=mode,
            csv_path=str(tmp_path / f"{mode}.jsonl"),
            boundary_opd=None,
        ),
        run_name=mode,
    )
    correct_rows = (
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
    )
    valid_rows = (
        (1, 1, 1, 1),
        (1, 1, 1, 0),
        (1, 1, 1, 1),
        (1, 1, 0, 0),
    )
    profiles = torch.tensor(
        [
            [-0.10, -0.20],
            [-0.12, -0.21],
            [-0.80, -0.90],
            [-0.20, -0.10],
        ]
    )
    prompts = []
    for prompt_index in range(4):
        prompts.append(
            {
                "prompt_uid": f"prompt-{prompt_index}",
                "source_index": str(prompt_index),
                "rollout_indices": list(range(prompt_index * 4, prompt_index * 4 + 4)),
                "verifier_correct": correct_rows[prompt_index],
                "rollout_valid": valid_rows[prompt_index],
                "processing_tokens": [100, 200, 300, 400],
                "response_tokens": [2, 2, 2, 2],
                "response_masks": torch.ones(4, 2, dtype=torch.bool),
                "sampled_token_log_probs": profiles,
                "global_step": 2,
                "queue_source": "fresh",
            }
        )

    result = manager.route_prompt_attempts(prompts)
    expected_count = 4 if mode == "all_wrong" else 2
    assert len(result.selected_indices) == len(result.candidates) == expected_count
    assert result.metrics["ff_ablation/actual_query_count"] == expected_count
    assert result.metrics["opd_query/full_rollout_count"] == 16
    assert all(candidate.rollout_valid for candidate in result.candidates)
    if mode == "random_correct":
        assert all(candidate.verifier_correct for candidate in result.candidates)
    elif mode != "global_random":
        assert all(not candidate.verifier_correct for candidate in result.candidates)
        assert all(candidate.bucket == "frontier" for candidate in result.candidates)
