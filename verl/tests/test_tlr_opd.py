import pytest
import torch

from verl.utils.tlr_opd import (
    TLRConfig,
    select_tlr_trajectories,
    validate_method_exclusivity,
)


def _select(lengths=None, entropies=None, eligible=None):
    prompts, k = 3, 4
    if lengths is None:
        lengths = torch.tensor(
            [
                16, 20, 30, 40,
                40, 30, 20, 16,
                16, 20, 30, 40,
            ],
            dtype=torch.float32,
        )
    if entropies is None:
        entropies = torch.tensor(
            [
                0.1, 0.2, 0.3, 0.4,
                0.4, 0.3, 0.2, 0.1,
                0.4, 0.1, 0.2, 0.3,
            ],
            dtype=torch.float32,
        )
    if eligible is None:
        eligible = torch.ones(prompts * k, dtype=torch.bool)
    return select_tlr_trajectories(
        prompt_ids=[f"prompt-{prompt}" for prompt in range(prompts) for _ in range(k)],
        eligible_mask=eligible,
        response_lengths=lengths,
        student_entropy_mean=entropies,
    )


def test_bon4_maximizes_short_low_entropy_score():
    selection = _select()
    assert selection.selected_indices.tolist() == [0, 7, 9]
    for group in range(3):
        group_scores = selection.scores[group * 4 : (group + 1) * 4]
        assert int(torch.argmax(group_scores)) == selection.selected_indices[group] - group * 4


def test_short_rollout_below_floor_is_excluded_before_scoring():
    selection = select_tlr_trajectories(
        prompt_ids=["a"] * 4,
        eligible_mask=torch.ones(4, dtype=torch.bool),
        response_lengths=torch.tensor([1, 20, 30, 40]),
        student_entropy_mean=torch.tensor([0.9, 0.2, 0.3, 0.4]),
    )
    assert selection.selected_indices.tolist() == [1]
    assert not selection.eligible_mask[0]
    assert selection.scores[0].item() == float("-inf")


def test_prompt_is_skipped_when_every_rollout_is_below_floor():
    selection = select_tlr_trajectories(
        prompt_ids=["a"] * 4,
        eligible_mask=torch.ones(4, dtype=torch.bool),
        response_lengths=torch.tensor([1, 4, 8, 15]),
        student_entropy_mean=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    assert selection.selected_indices.numel() == 0
    assert not selection.eligible_mask.any()


def test_exactly_one_of_four_per_prompt_and_quarter_teacher_ratio():
    selection = _select()
    selected_groups = selection.prompt_inverse[selection.selected_indices]
    assert torch.bincount(selected_groups, minlength=3).tolist() == [1, 1, 1]
    assert selection.selected_indices.numel() / selection.eligible_mask.numel() == 0.25


def test_equal_entropy_tie_is_deterministic_first_rollout():
    selection = select_tlr_trajectories(
        prompt_ids=["a"] * 4,
        eligible_mask=torch.ones(4, dtype=torch.bool),
        response_lengths=torch.full((4,), 16.0),
        student_entropy_mean=torch.ones(4),
    )
    assert selection.scores.tolist() == pytest.approx([0.25] * 4)
    assert selection.selected_indices.tolist() == [0]


def test_invalid_rollouts_are_excluded_and_empty_prompt_is_skipped():
    eligible = torch.tensor(
        [
            1, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 0, 0,
        ],
        dtype=torch.bool,
    )
    selection = _select(eligible=eligible)
    assert selection.selected_indices.tolist() == [0, 8]
    assert not (~selection.eligible_mask[selection.selected_indices]).any()
    assert selection.selected_indices.numel() == 2


def test_nonfinite_eligible_entropy_is_rejected():
    entropies = torch.ones(4)
    entropies[2] = float("nan")
    with pytest.raises(ValueError, match="entropies"):
        select_tlr_trajectories(
            prompt_ids=["a"] * 4,
            eligible_mask=torch.ones(4, dtype=torch.bool),
            response_lengths=torch.ones(4),
            student_entropy_mean=entropies,
        )


def test_config_supports_variable_k_and_requires_exact_rollout_n():
    TLRConfig(enabled=True).validate(rollout_n=4)
    TLRConfig(enabled=True, rollouts_per_prompt=8).validate(rollout_n=8)
    with pytest.raises(ValueError, match="rollout.n"):
        TLRConfig(enabled=True, rollouts_per_prompt=8).validate(rollout_n=4)


def test_bon8_selects_exactly_one_short_low_entropy_rollout():
    selection = select_tlr_trajectories(
        prompt_ids=["a"] * 8,
        eligible_mask=torch.ones(8, dtype=torch.bool),
        response_lengths=torch.arange(16, 24),
        student_entropy_mean=torch.tensor([0.8, 0.7, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2]),
        rollouts_per_prompt=8,
    )
    assert selection.selected_indices.tolist() == [2]


def test_selector_rejects_incomplete_groups():
    with pytest.raises(ValueError, match="exactly 4"):
        select_tlr_trajectories(
            prompt_ids=["a"] * 3,
            eligible_mask=torch.ones(3, dtype=torch.bool),
            response_lengths=torch.ones(3),
            student_entropy_mean=torch.ones(3),
        )


@pytest.mark.parametrize("method", ["ff_opd", "gq_opd", "random_query_opd"])
def test_method_mutual_exclusion(method):
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_method_exclusivity(True, **{method: True})
