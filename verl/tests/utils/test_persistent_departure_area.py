# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from verl.utils.boundary_calibration import BoundaryCalibrationAccumulator
from verl.utils.boundary_opd import (
    BoundaryOPDSettings,
    build_centered_hidden_states,
    build_uniform_response_state_indices,
    compute_persistent_departure_area_scores_from_cosine,
    normalized_cosine_similarity,
    persistent_departure_area,
)
from verl.utils.ff_opd import FFOPDConfig


def _score(cosine, lengths=(10,), rollout_ids=(0,), prompt_length=5):
    return compute_persistent_departure_area_scores_from_cosine(
        torch.tensor(cosine, dtype=torch.float32),
        prompt_length,
        torch.tensor(lengths),
        torch.tensor(rollout_ids),
    )


def test_normalized_cosine_range():
    assert torch.equal(normalized_cosine_similarity(torch.tensor([-1.0, 0.0, 1.0])), torch.tensor([0.0, 0.5, 1.0]))


def test_zero_departure_and_running_max_anchor():
    _, _, area = persistent_departure_area(torch.tensor([0.2, 0.4, 0.6, 0.8]))
    assert area.item() == 0.0
    running, departure, _ = persistent_departure_area(torch.tensor([0.4, 0.8, 0.5, 0.5]))
    assert torch.isclose(running[1], torch.tensor(0.8))
    assert torch.allclose(departure[2:], torch.tensor([0.3, 0.3]))


def test_persistent_and_early_departures_have_larger_area():
    _, _, transient = persistent_departure_area(torch.tensor([0.9, 0.5, 0.88, 0.90]))
    _, _, persistent = persistent_departure_area(torch.tensor([0.9, 0.55, 0.50, 0.47]))
    assert persistent > transient
    _, _, early = persistent_departure_area(torch.tensor([0.9, 0.5, 0.5, 0.5, 0.5]))
    _, _, late = persistent_departure_area(torch.tensor([0.9, 0.9, 0.9, 0.5, 0.5]))
    assert early > late


def test_area_cost_and_score_bounds_randomized():
    generator = torch.Generator().manual_seed(42)
    similarity = torch.rand(4096, 32, generator=generator)
    _, _, area = persistent_departure_area(similarity)
    assert bool(((0.0 <= area) & (area <= 1.0)).all())
    cosine = 2.0 * torch.rand(7, 3, 32, generator=generator) - 1.0
    result = compute_persistent_departure_area_scores_from_cosine(
        cosine, 20, torch.tensor([1, 2, 3, 5, 8, 13, 21]), torch.arange(7)
    )
    assert result["cost_ratio"][0].item() == 1.0
    assert bool(((0.0 < result["cost_ratio"]) & (result["cost_ratio"] <= 1.0)).all())
    assert bool(((0.0 <= result["final_score"]) & (result["final_score"] <= 1.0)).all())


def test_cost_tradeoff_max_positive_and_tie_breaks():
    # normalized similarities: candidate 0 has area .4 but costs 4x candidate 1;
    # candidate 1 has area .2 and wins by final score.
    result = _score(
        [
            [[0.8, 0.0, 0.0], [0.2, -0.2, -0.2]],
            [[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        lengths=(35, 5),
        rollout_ids=(4, 3),
        prompt_length=5,
    )
    assert result["pair_area"][0, 0] > result["pair_area"][0, 1]
    assert int(result["selected_local_index"]) == 1

    zero = _score(
        [[[0.0, 0.2, 0.4]], [[-0.2, 0.0, 0.2]], [[0.4, 0.6, 0.8]]],
        lengths=(8, 4, 4),
        rollout_ids=(0, 9, 2),
    )
    assert bool(zero["boundary_degenerate_to_cost_only"])
    assert int(zero["selected_rollout_id"]) == 2


def test_min_positive_aggregation_uses_smallest_pair_area():
    cosine = torch.tensor([[[0.8, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    maximum = compute_persistent_departure_area_scores_from_cosine(
        cosine, 5, torch.tensor([5]), torch.tensor([7])
    )
    minimum = compute_persistent_departure_area_scores_from_cosine(
        cosine,
        5,
        torch.tensor([5]),
        torch.tensor([7]),
        positive_aggregation="min",
    )

    assert int(maximum["best_positive_local_index"]) == 0
    assert int(minimum["best_positive_local_index"]) == 1
    assert torch.equal(minimum["pda_area"], minimum["pair_area"].min(dim=1).values)
    assert minimum["positive_aggregation"] == "min"


def test_pda_min_mode_is_independent_from_original_mode():
    original = BoundaryOPDSettings(
        score_mode="persistent_departure_area",
        similarity_metric="centered_hidden_state_cosine",
        num_boundaries=32,
        calibration_artifact_path="manifest.json",
        calibration_artifact_sha256="0" * 64,
    )
    minimum = BoundaryOPDSettings(
        score_mode="persistent_departure_area_min",
        similarity_metric="centered_hidden_state_cosine",
        num_boundaries=32,
        calibration_artifact_path="manifest.json",
        calibration_artifact_sha256="0" * 64,
    )
    original.validate()
    minimum.validate()
    assert original.score_mode != minimum.score_mode


def test_unique_wrong_is_selected():
    result = _score([[[0.9, 0.1, 0.0, -0.1]]], rollout_ids=(7,))
    assert int(result["selected_rollout_id"]) == 7


def test_uniform_positions_use_direct_response_tokens_and_allow_repeats():
    response_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    prompt_mask = torch.ones(2, 3, dtype=torch.bool)
    indices, valid = build_uniform_response_state_indices(response_mask, prompt_mask, 5)
    assert indices[0].tolist() == [3, 4, 5, 6, 7]
    assert indices[1].tolist() == [3, 3, 3, 3, 3]
    assert valid.all()


def test_direct_hidden_path_never_builds_differences():
    positive = torch.tensor(
        [[0.97145, 0.21144, 0.10761], [-0.40906, 0.03078, -0.91199],
         [0.18878, 0.56558, -0.80279], [0.58831, 0.31422, 0.74509]]
    )
    negatives = torch.tensor(
        [[[0.38307, 0.57877, 0.71992], [-0.05578, -0.57178, 0.81851],
          [-0.04163, -0.96878, -0.24439], [0.98378, -0.16763, -0.06389]],
         [[0.23762, -0.85278, 0.46509], [-0.15282, 0.58941, -0.79325],
          [-0.81352, 0.40917, -0.41323], [-0.26437, 0.67610, 0.68775]]]
    )
    states = torch.cat((negatives, positive.unsqueeze(0)))
    direct = build_centered_hidden_states(states, torch.ones(3, 4, dtype=torch.bool), mean=torch.zeros(3))
    cosine = torch.einsum("nmd,pmd->npm", direct[:2], direct[2:])
    result = compute_persistent_departure_area_scores_from_cosine(
        cosine, 5, torch.tensor([4, 4]), torch.tensor([0, 1])
    )
    direct_winner = int(result["selected_local_index"])
    positive_delta = torch.nn.functional.normalize(positive[1:] - positive[:-1], dim=-1)
    negative_delta = torch.nn.functional.normalize(negatives[:, 1:] - negatives[:, :-1], dim=-1)
    delta_cosine = torch.einsum("nmd,md->nm", negative_delta, positive_delta)
    _, _, delta_area = persistent_departure_area(normalized_cosine_similarity(delta_cosine))
    assert direct_winner == 0
    assert int(delta_area.argmax()) == 1
    assert result["representation_domain"] == "direct_hidden_state"
    assert result["uses_hidden_difference"] is False


def test_pda_settings_are_independent_of_k_rollouts():
    settings = BoundaryOPDSettings(
        num_boundaries=32,
        similarity_metric="centered_hidden_state_cosine",
        calibration_artifact_path="manifest.json",
        calibration_artifact_sha256="0" * 64,
        score_mode="persistent_departure_area",
    )
    settings.validate()
    FFOPDConfig(k_rollouts=4, max_no_success_retries=0, boundary_opd=settings).validate()
    FFOPDConfig(k_rollouts=8, max_no_success_retries=0, boundary_opd=settings).validate()
    FFOPDConfig(k_rollouts=16, max_no_success_retries=0, boundary_opd=settings).validate()
    with pytest.raises(ValueError, match="max_no_success_retries=0"):
        FFOPDConfig(k_rollouts=4, max_no_success_retries=1, boundary_opd=settings).validate()
    FFOPDConfig(k_rollouts=4, fresh_step_limit=2, boundary_opd=settings).validate()
    with pytest.raises(ValueError, match="fresh_step_limit"):
        FFOPDConfig(k_rollouts=4, fresh_step_limit=-1, boundary_opd=settings).validate()


def test_direct_state_calibration_manifest_cannot_be_used_as_transition_statistics(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    settings = BoundaryOPDSettings(
        num_boundaries=32,
        similarity_metric="raw_hidden_cosine",
        calibration_collect=True,
        calibration_artifact_path=str(manifest_path),
        calibration_model_hash="1" * 64,
        calibration_data_hash="2" * 64,
        calibration_num_prompts=1,
        score_mode="persistent_departure_area",
    )
    accumulator = BoundaryCalibrationAccumulator(settings, k_rollouts=4)
    generator = torch.Generator().manual_seed(42)
    accumulator.update(
        torch.randn(4, 32, 3, generator=generator),
        torch.ones(4, 32, dtype=torch.bool),
        ["prompt"] * 4,
        [0, 1, 2, 3],
    )
    artifact = accumulator.finalize()
    assert artifact.manifest["representation_domain"] == "direct_hidden_state"
    assert artifact.manifest["uses_hidden_difference"] is False
