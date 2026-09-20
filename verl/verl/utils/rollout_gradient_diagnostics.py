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

from collections import defaultdict

import numpy as np
import torch

ROLLOUT_DIAGNOSTIC_KEYS = (
    "rollout_grad_mass",
    "rollout_grad_mass_per_token",
    "rollout_response_length",
    "rollout_mean_abs_advantage",
    "rollout_mean_student_prob",
    "rollout_mean_gradient_leverage",
    "rollout_mean_student_logprob",
    "rollout_mean_teacher_entropy",
)


def compute_rollout_gradient_diagnostics(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    eps: float = 1e-8,
    teacher_entropy: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute detached sampled-token logit-gradient mass for each rollout."""
    if teacher_entropy is None:
        raise ValueError(
            "teacher_entropy is required for rollout gradient diagnostics; "
            "set reward_model.compute_teacher_entropy=True"
        )
    with torch.no_grad():
        tensors = []
        for tensor in (student_log_probs, teacher_log_probs, loss_mask, teacher_entropy):
            if tensor.is_nested:
                tensor = tensor.to_padded_tensor(0.0)
            tensors.append(tensor.detach().float())
        log_p, log_q, mask, entropy = tensors

        if log_p.shape != log_q.shape or log_p.shape != mask.shape or log_p.shape != entropy.shape:
            raise ValueError(
                "student_log_probs, teacher_log_probs, loss_mask, and teacher_entropy must have identical shapes; "
                f"got {tuple(log_p.shape)}, {tuple(log_q.shape)}, {tuple(mask.shape)}, and {tuple(entropy.shape)}"
            )
        if log_p.ndim != 2:
            raise ValueError(f"rollout gradient diagnostics require [batch, response] tensors, got {log_p.ndim}D")

        if not torch.isfinite(entropy).all():
            raise ValueError("teacher entropy must be finite")
        entropy = entropy.clamp_min(0.0)

        raw_student_prob = log_p.exp()
        invalid_probability = (
            not torch.isfinite(raw_student_prob).all()
            or (raw_student_prob < 0).any()
            or (raw_student_prob > 1.0001).any()
        )
        if invalid_probability:
            raise ValueError("student sampled-token probabilities must be finite and in [0, 1]")
        student_prob = raw_student_prob.clamp(0.0, 1.0)
        abs_advantage = (log_q - log_p).abs()
        gradient_leverage = (1.0 - student_prob).clamp(0.0, 1.0)
        token_grad_mass = abs_advantage * gradient_leverage * mask
        rollout_length = mask.sum(dim=-1)
        safe_length = rollout_length.clamp_min(eps)
        rollout_grad_mass = token_grad_mass.sum(dim=-1)

        if not torch.isfinite(rollout_grad_mass).all() or (rollout_grad_mass < 0).any():
            raise ValueError("rollout gradient mass must be finite and non-negative")

        return {
            "rollout_grad_mass": rollout_grad_mass,
            "rollout_grad_mass_per_token": rollout_grad_mass / safe_length,
            "rollout_response_length": rollout_length,
            "rollout_mean_abs_advantage": (abs_advantage * mask).sum(dim=-1) / safe_length,
            "rollout_mean_student_prob": (student_prob * mask).sum(dim=-1) / safe_length,
            "rollout_mean_gradient_leverage": (gradient_leverage * mask).sum(dim=-1) / safe_length,
            "rollout_mean_student_logprob": (log_p * mask).sum(dim=-1) / safe_length,
            "rollout_mean_teacher_entropy": (entropy * mask).sum(dim=-1) / safe_length,
        }


def summarize_rollout_gradient_diagnostics(
    uids,
    rollout_tensors: dict[str, torch.Tensor],
    expected_group_size: int,
    eps: float = 1e-8,
    true_rewards=None,
) -> tuple[dict[str, float], list[dict]]:
    """Group per-rollout diagnostics by UID and compute OracleGMC metrics."""
    if expected_group_size < 2:
        raise ValueError("expected_group_size must be at least 2")

    values = {key: value.detach().float().cpu().numpy() for key, value in rollout_tensors.items()}
    sample_count = len(uids)
    if any(len(value) != sample_count for value in values.values()):
        raise ValueError("UID and rollout diagnostic counts must match")

    correct_flags = None
    if true_rewards is not None:
        if torch.is_tensor(true_rewards):
            true_rewards = true_rewards.detach().float().cpu().numpy()
        true_rewards = np.asarray(true_rewards, dtype=np.float64).reshape(-1)
        if len(true_rewards) != sample_count:
            raise ValueError("true_rewards and rollout diagnostic counts must match")
        # The rule-based math verifier emits binary scores (1.0 = correct).
        correct_flags = true_rewards > 0.5

    grouped = defaultdict(list)
    for sample_index, uid in enumerate(uids):
        grouped[str(uid)].append(sample_index)

    complete_groups = []
    incomplete_count = 0
    overfull_count = 0
    for indices in grouped.values():
        if len(indices) == expected_group_size:
            complete_groups.append(indices)
        elif len(indices) < expected_group_size:
            incomplete_count += 1
        else:
            overfull_count += 1

    grad_mass = values["rollout_grad_mass"]
    nonzero_group_shares = []
    cv_values = []
    max_to_mean_values = []
    zero_group_count = 0
    for indices in complete_groups:
        group_mass = np.asarray(grad_mass[indices], dtype=np.float64)
        total = float(group_mass.sum())
        if total <= eps:
            zero_group_count += 1
            continue
        shares = np.cumsum(np.sort(group_mass)[::-1]) / total
        nonzero_group_shares.append(shares)
        mean = float(group_mass.mean())
        max_to_mean_values.append(float(group_mass.max() / (mean + eps)))
        cv_values.append(float(group_mass.std(ddof=0) / (mean + eps)))

    metrics = {
        "grad_concentration/valid_group_count": float(len(complete_groups)),
        "grad_concentration/incomplete_group_count": float(incomplete_count),
        "grad_concentration/overfull_group_count": float(overfull_count),
        "grad_concentration/zero_grad_group_fraction": float(zero_group_count / max(len(complete_groups), 1)),
        "grad_concentration/grad_mass_mean": float(grad_mass.mean()) if sample_count else 0.0,
        "grad_concentration/grad_mass_max": float(grad_mass.max()) if sample_count else 0.0,
        "grad_concentration/grad_mass_min": float(grad_mass.min()) if sample_count else 0.0,
        "grad_concentration/grad_mass_per_token_mean": (
            float(values["rollout_grad_mass_per_token"].mean()) if sample_count else 0.0
        ),
    }

    if nonzero_group_shares:
        shares = np.stack(nonzero_group_shares)
        tolerance = 1e-6
        random_baseline = np.arange(1, expected_group_size + 1, dtype=np.float64) / expected_group_size
        if (
            np.any(shares < -tolerance)
            or np.any(shares > 1.0 + tolerance)
            or np.any(np.diff(shares, axis=1) < -tolerance)
            or np.any(shares < random_baseline[None, :] - tolerance)
        ):
            raise ValueError("OracleGMC shares violate concentration invariants")
        for top_b in range(1, expected_group_size + 1):
            metrics[f"grad_concentration/oracle_gmc_top{top_b}"] = float(shares[:, top_b - 1].mean())
        metrics["grad_concentration/max_share"] = metrics["grad_concentration/oracle_gmc_top1"]
        metrics["grad_concentration/max_to_mean"] = float(np.mean(max_to_mean_values))
        metrics["grad_concentration/cv"] = float(np.mean(cv_values))
        if expected_group_size == 4:
            for percentile, top_b in ((25, 1), (50, 2), (75, 3)):
                metrics[f"grad_concentration/oracle_gmc_{percentile}"] = metrics[
                    f"grad_concentration/oracle_gmc_top{top_b}"
                ]
            top2 = shares[:, 1]
            metrics["grad_concentration/oracle_gmc_50_mean"] = float(top2.mean())
            for percentile in (25, 50, 75, 90):
                metrics[f"grad_concentration/oracle_gmc_50_p{percentile}"] = float(
                    np.percentile(top2, percentile)
                )
    else:
        for top_b in range(1, expected_group_size + 1):
            metrics[f"grad_concentration/oracle_gmc_top{top_b}"] = 0.0
        metrics.update(
            {
                "grad_concentration/max_share": 0.0,
                "grad_concentration/max_to_mean": 0.0,
                "grad_concentration/cv": 0.0,
            }
        )
        if expected_group_size == 4:
            for key in ("25", "50", "75", "50_mean", "50_p25", "50_p50", "50_p75", "50_p90"):
                metrics[f"grad_concentration/oracle_gmc_{key}"] = 0.0

    if correct_flags is not None:
        correct_mass = grad_mass[correct_flags]
        wrong_mass = grad_mass[~correct_flags]
        metrics["grad_concentration/correct_rollout_fraction"] = (
            float(correct_flags.mean()) if sample_count else 0.0
        )
        metrics["grad_concentration/mean_grad_mass_correct"] = (
            float(correct_mass.mean()) if correct_mass.size else 0.0
        )
        metrics["grad_concentration/mean_grad_mass_wrong"] = float(wrong_mass.mean()) if wrong_mass.size else 0.0

        # Whether the largest-G rollout inside each sibling group is correct, and
        # whether wrong rollouts carry more gradient mass than correct ones
        # within the same prompt (delta G controls for prompt difficulty).
        argmax_correct = []
        delta_values = []
        for indices in complete_groups:
            group_mass = np.asarray(grad_mass[indices], dtype=np.float64)
            group_correct = correct_flags[indices]
            if float(group_mass.sum()) <= eps:
                continue
            argmax_correct.append(bool(group_correct[int(np.argmax(group_mass))]))
            if group_correct.any() and not group_correct.all():
                delta_values.append(float(group_mass[~group_correct].mean() - group_mass[group_correct].mean()))
        metrics["grad_concentration/argmax_g_group_count"] = float(len(argmax_correct))
        metrics["grad_concentration/argmax_g_correct_rate"] = float(np.mean(argmax_correct)) if argmax_correct else 0.0
        metrics["grad_concentration/within_prompt_mixed_group_count"] = float(len(delta_values))
        metrics["grad_concentration/within_prompt_delta_g_mean"] = (
            float(np.mean(delta_values)) if delta_values else 0.0
        )
        metrics["grad_concentration/within_prompt_delta_g_positive_rate"] = (
            float(np.mean([delta > 0 for delta in delta_values])) if delta_values else 0.0
        )

    records = []
    for uid, indices in grouped.items():
        for rollout_index, sample_index in enumerate(indices):
            record = {
                "uid": uid,
                "rollout_index_within_uid": rollout_index,
                "grad_mass": float(values["rollout_grad_mass"][sample_index]),
                "response_length": int(values["rollout_response_length"][sample_index]),
                "grad_mass_per_token": float(values["rollout_grad_mass_per_token"][sample_index]),
                "mean_abs_advantage": float(values["rollout_mean_abs_advantage"][sample_index]),
                "mean_student_prob": float(values["rollout_mean_student_prob"][sample_index]),
                "mean_gradient_leverage": float(values["rollout_mean_gradient_leverage"][sample_index]),
                "mean_student_logprob": float(values["rollout_mean_student_logprob"][sample_index]),
                "mean_teacher_entropy": float(values["rollout_mean_teacher_entropy"][sample_index]),
            }
            if correct_flags is not None:
                record["true_reward"] = float(true_rewards[sample_index])
                record["correct"] = bool(correct_flags[sample_index])
            records.append(record)
    return metrics, records
