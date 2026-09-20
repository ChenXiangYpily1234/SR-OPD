import math

import pytest
import torch

from verl.utils.rollout_gradient_diagnostics import (
    compute_rollout_gradient_diagnostics,
    summarize_rollout_gradient_diagnostics,
)


def _rollout_tensors(grad_mass):
    grad_mass = torch.tensor(grad_mass, dtype=torch.float32)
    ones = torch.ones_like(grad_mass)
    return {
        "rollout_grad_mass": grad_mass,
        "rollout_grad_mass_per_token": grad_mass,
        "rollout_response_length": ones,
        "rollout_mean_abs_advantage": grad_mass,
        "rollout_mean_student_prob": torch.zeros_like(grad_mass),
        "rollout_mean_gradient_leverage": ones,
        "rollout_mean_student_logprob": torch.zeros_like(grad_mass),
        "rollout_mean_teacher_entropy": torch.zeros_like(grad_mass),
    }


@pytest.mark.parametrize(
    ("grad_mass", "expected"),
    [
        ([4, 3, 2, 1], (0.4, 0.7, 0.9)),
        ([1, 1, 1, 1], (0.25, 0.5, 0.75)),
        ([10, 0, 0, 0], (1.0, 1.0, 1.0)),
    ],
)
def test_oracle_gmc_hand_calculated_examples(grad_mass, expected):
    metrics, _ = summarize_rollout_gradient_diagnostics(
        ["prompt"] * 4, _rollout_tensors(grad_mass), expected_group_size=4
    )

    assert metrics["grad_concentration/oracle_gmc_25"] == pytest.approx(expected[0])
    assert metrics["grad_concentration/oracle_gmc_50"] == pytest.approx(expected[1])
    assert metrics["grad_concentration/oracle_gmc_75"] == pytest.approx(expected[2])


def test_grouping_uses_uid_not_row_order_and_reports_bad_group_sizes():
    metrics, records = summarize_rollout_gradient_diagnostics(
        ["a", "b", "a", "short", "b", "a", "long", "b", "a", "b", "long", "long", "long", "long"],
        _rollout_tensors([4, 1, 3, 8, 4, 2, 1, 3, 1, 2, 1, 1, 1, 1]),
        expected_group_size=4,
    )

    assert metrics["grad_concentration/valid_group_count"] == 2
    assert metrics["grad_concentration/incomplete_group_count"] == 1
    assert metrics["grad_concentration/overfull_group_count"] == 1
    assert metrics["grad_concentration/oracle_gmc_50"] == pytest.approx((0.7 + 0.7) / 2)
    assert len(records) == 14


def test_zero_mass_group_is_excluded_without_nan():
    metrics, _ = summarize_rollout_gradient_diagnostics(
        ["zero"] * 4, _rollout_tensors([0, 0, 0, 0]), expected_group_size=4
    )

    assert metrics["grad_concentration/zero_grad_group_fraction"] == 1.0
    assert metrics["grad_concentration/oracle_gmc_50"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_token_gradient_mass_uses_mask_and_fp32_detached_values():
    log_p = torch.log(torch.tensor([[0.5, 0.25, 0.5]], dtype=torch.float16, requires_grad=True))
    log_q = torch.log(torch.tensor([[0.25, 0.5, 0.1]], dtype=torch.float16))
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    entropy = torch.tensor([[0.7, 0.2, 99.0]], dtype=torch.float32)

    result = compute_rollout_gradient_diagnostics(log_p, log_q, mask, teacher_entropy=entropy)
    expected = abs(math.log(0.25) - math.log(0.5)) * 0.5
    expected += abs(math.log(0.5) - math.log(0.25)) * 0.75

    assert result["rollout_grad_mass"].dtype == torch.float32
    assert not result["rollout_grad_mass"].requires_grad
    assert result["rollout_grad_mass"].item() == pytest.approx(expected, rel=2e-3)
    assert result["rollout_response_length"].item() == 2
    # masked mean: the padded 99.0 entropy position must be excluded
    assert result["rollout_mean_teacher_entropy"].item() == pytest.approx((0.7 + 0.2) / 2)
    assert result["rollout_mean_student_logprob"].item() == pytest.approx(
        (math.log(0.5) + math.log(0.25)) / 2, rel=2e-3
    )


def test_missing_teacher_entropy_fails_closed():
    with pytest.raises(ValueError, match="teacher_entropy is required"):
        compute_rollout_gradient_diagnostics(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2))


def test_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="identical shapes"):
        compute_rollout_gradient_diagnostics(
            torch.zeros(2, 3), torch.zeros(2, 2), torch.ones(2, 3), teacher_entropy=torch.ones(2, 3)
        )


def test_correctness_metrics_and_records():
    # group A: the only correct rollout carries the largest mass;
    # group B: all rollouts wrong, so it is excluded from delta-G.
    uids = ["a", "a", "a", "a", "b", "b", "b", "b"]
    masses = [4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    metrics, records = summarize_rollout_gradient_diagnostics(
        uids, _rollout_tensors(masses), expected_group_size=4, true_rewards=rewards
    )

    assert metrics["grad_concentration/correct_rollout_fraction"] == pytest.approx(1 / 8)
    assert metrics["grad_concentration/mean_grad_mass_correct"] == pytest.approx(4.0)
    assert metrics["grad_concentration/mean_grad_mass_wrong"] == pytest.approx(10 / 7)
    # argmax of group A is correct (mass 4), argmax of group B is wrong -> 1/2
    assert metrics["grad_concentration/argmax_g_correct_rate"] == pytest.approx(0.5)
    # within prompt A: mean wrong mass (3+2+1)/3 = 2 < correct mass 4 -> delta negative
    assert metrics["grad_concentration/within_prompt_mixed_group_count"] == 1.0
    assert metrics["grad_concentration/within_prompt_delta_g_mean"] == pytest.approx(-2.0)
    assert metrics["grad_concentration/within_prompt_delta_g_positive_rate"] == pytest.approx(0.0)

    correct_records = [record for record in records if record["correct"]]
    assert len(correct_records) == 1
    assert correct_records[0]["true_reward"] == 1.0
    assert correct_records[0]["grad_mass"] == pytest.approx(4.0)
    assert all("true_reward" in record and "mean_teacher_entropy" in record for record in records)


def test_correctness_count_mismatch_fails_closed():
    with pytest.raises(ValueError, match="true_rewards and rollout diagnostic counts must match"):
        summarize_rollout_gradient_diagnostics(
            ["a"] * 4, _rollout_tensors([1, 1, 1, 1]), expected_group_size=4, true_rewards=[1.0, 0.0, 0.0]
        )
