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

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

TA_OPD_PATH = Path(__file__).parents[1] / "verl" / "utils" / "ta_opd.py"
TA_OPD_SPEC = importlib.util.spec_from_file_location("ta_opd_under_test", TA_OPD_PATH)
assert TA_OPD_SPEC is not None and TA_OPD_SPEC.loader is not None
TA_OPD_MODULE = importlib.util.module_from_spec(TA_OPD_SPEC)
TA_OPD_SPEC.loader.exec_module(TA_OPD_MODULE)

aggregate_ta_opd_loss = TA_OPD_MODULE.aggregate_ta_opd_loss
compute_ta_opd_mask = TA_OPD_MODULE.compute_ta_opd_mask
robust_normalize = TA_OPD_MODULE.robust_normalize
select_opd_training_reward = TA_OPD_MODULE.select_opd_training_reward


def _topk_inputs(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int) -> dict:
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_topk_logp, student_topk_ids = torch.topk(student_logp, k=k, dim=-1)
    teacher_topk_logp, teacher_topk_ids = torch.topk(teacher_logp, k=k, dim=-1)
    return {
        "student_topk_log_probs": student_topk_logp,
        "student_topk_indices": student_topk_ids,
        "teacher_topk_log_probs": teacher_topk_logp,
        "teacher_topk_indices": teacher_topk_ids,
        "teacher_on_student_log_probs": teacher_logp.gather(-1, student_topk_ids),
        "student_on_teacher_log_probs": student_logp.gather(-1, teacher_topk_ids),
    }


def _manual_local_teacher_student_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int) -> float:
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_ids = student_logp.topk(k, dim=-1).indices[0, 0].tolist()
    teacher_ids = teacher_logp.topk(k, dim=-1).indices[0, 0].tolist()
    union_ids = torch.tensor(list(dict.fromkeys(student_ids + teacher_ids))).view(1, 1, -1)
    student_union = student_logp.gather(-1, union_ids)
    teacher_union = teacher_logp.gather(-1, union_ids)
    student_local = student_union - torch.logsumexp(student_union, dim=-1, keepdim=True)
    teacher_local = teacher_union - torch.logsumexp(teacher_union, dim=-1, keepdim=True)
    return (teacher_local.exp() * (teacher_local - student_local)).sum().item()


def test_ta_training_reward_uses_only_sampled_rollout_tokens():
    sampled = torch.randn(2, 4)
    topk = torch.randn(2, 4, 32)

    selected = select_opd_training_reward(sampled, topk, ta_opd_enable=True)

    assert selected is sampled
    assert selected.shape == (2, 4)


def test_ta_training_reward_rejects_missing_or_topk_shaped_sampled_reward():
    topk = torch.randn(2, 4, 32)

    with pytest.raises(RuntimeError, match="requires sampled_opd_rm_scores"):
        select_opd_training_reward(None, topk, ta_opd_enable=True)
    with pytest.raises(RuntimeError, match=r"shape \[B, T\]"):
        select_opd_training_reward(topk, topk, ta_opd_enable=True)


def test_non_ta_training_reward_uses_same_sampled_token_target():
    sampled = torch.randn(2, 4)
    topk = torch.randn(2, 4, 16)

    selected = select_opd_training_reward(sampled, topk, ta_opd_enable=False)

    assert selected is sampled


def test_robust_normalize_uses_only_valid_tokens():
    values = torch.tensor([[0.0, 1.0, 2.0, 1_000_000.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    normalized = robust_normalize(values, mask)

    expected_low = torch.quantile(torch.tensor([0.0, 1.0, 2.0]), 0.05)
    expected_high = torch.quantile(torch.tensor([0.0, 1.0, 2.0]), 0.95)
    expected = ((values - expected_low) / (expected_high - expected_low + 1e-8)).clamp(0.0, 1.0)
    assert torch.allclose(normalized, expected)


def test_topk_path_matches_deduplicated_union_and_exact_compatibility():
    student_logits = torch.tensor([[[3.0, 2.0, 0.0, -1.0]]])
    teacher_logits = torch.tensor([[[0.0, 3.0, 2.0, -1.0]]])
    inputs = _topk_inputs(student_logits, teacher_logits, k=2)
    response_mask = torch.ones(1, 1)

    _, _, stats = compute_ta_opd_mask(**inputs, response_mask=response_mask, topk=2, retain_ratio=1.0)

    expected_kl = _manual_local_teacher_student_kl(student_logits, teacher_logits, k=2)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    expected_c = teacher_logp.gather(-1, inputs["student_topk_indices"]).exp().sum().item()
    intersection_c = inputs["teacher_topk_log_probs"][..., :1].exp().sum().item()
    assert stats["ta_opd/D_mean"] == pytest.approx(expected_kl, rel=1e-6, abs=1e-7)
    assert stats["ta_opd/C_mean"] == pytest.approx(expected_c, rel=1e-6, abs=1e-7)
    assert expected_c > intersection_c


def test_full_logits_path_uses_local_forward_kl():
    student_logits = torch.tensor([[[3.0, 2.0, 0.0, -1.0]]])
    teacher_logits = torch.tensor([[[0.0, 3.0, 2.0, -1.0]]])

    _, _, stats = compute_ta_opd_mask(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=torch.ones(1, 1),
        topk=2,
        retain_ratio=1.0,
    )

    expected_forward_kl = _manual_local_teacher_student_kl(student_logits, teacher_logits, k=2)
    reverse_kl = _manual_local_teacher_student_kl(teacher_logits, student_logits, k=2)
    assert stats["ta_opd/D_mean"] == pytest.approx(expected_forward_kl, rel=1e-6, abs=1e-7)
    assert not math.isclose(expected_forward_kl, reverse_kl, rel_tol=1e-3, abs_tol=1e-3)


def test_configured_topk_truncates_larger_input_tensors():
    student_logits = torch.tensor([[[4.0, 3.0, 2.0, 0.0, -1.0]]])
    teacher_logits = torch.tensor([[[0.0, 4.0, 3.0, 2.0, -1.0]]])
    inputs = _topk_inputs(student_logits, teacher_logits, k=3)

    _, _, stats = compute_ta_opd_mask(
        **inputs,
        response_mask=torch.ones(1, 1),
        topk=2,
        retain_ratio=1.0,
    )

    expected_kl = _manual_local_teacher_student_kl(student_logits, teacher_logits, k=2)
    assert stats["ta_opd/D_mean"] == pytest.approx(expected_kl, rel=1e-6, abs=1e-7)
    assert stats["ta_opd/topk"] == 2


def test_topk_path_rejects_missing_cross_support_probabilities():
    logits = torch.tensor([[[2.0, 1.0, 0.0]]])
    inputs = _topk_inputs(logits, logits, k=2)
    inputs.pop("student_on_teacher_log_probs")

    with pytest.raises(ValueError, match="cross-support"):
        compute_ta_opd_mask(**inputs, response_mask=torch.ones(1, 1), topk=2)


def test_global_top_ratio_uses_all_valid_batch_tokens():
    student_logits = torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(1))
    teacher_logits = torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(2))
    response_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])

    selected_mask, _, _ = compute_ta_opd_mask(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=response_mask,
        topk=3,
        retain_ratio=0.30,
        normalize_scope="batch",
    )

    assert selected_mask.sum().item() == math.ceil(response_mask.sum().item() * 0.30)
    assert not selected_mask[~response_mask.bool()].any()


def test_paper_retention_uses_batch_ceil():
    student_logits = torch.randn(2, 3, 7, generator=torch.Generator().manual_seed(11))
    teacher_logits = torch.randn(2, 3, 7, generator=torch.Generator().manual_seed(12))
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    selected_mask, _, stats = compute_ta_opd_mask(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=response_mask,
        topk=3,
        retain_ratio=0.30,
        normalize_scope="batch",
    )

    assert selected_mask.sum().item() == 2
    assert stats["ta_opd/coverage_is_exact"] == 1
    for name in ("D", "C", "score"):
        assert f"ta_opd/{name}_q05" in stats
        assert f"ta_opd/{name}_q50" in stats
        assert f"ta_opd/{name}_q95" in stats


def test_positive_ratio_keeps_one_token_in_a_one_token_batch():
    logits = torch.randn(1, 1, 7)
    selected_mask, _, _ = compute_ta_opd_mask(
        student_logits=logits,
        teacher_logits=logits,
        response_mask=torch.ones(1, 1),
        topk=3,
        retain_ratio=0.05,
        normalize_scope="batch",
    )

    assert selected_mask.sum().item() == 1


def test_paper_retention_selects_exactly_ten_of_one_hundred_tokens():
    generator = torch.Generator().manual_seed(17)
    selected_mask, _, _ = compute_ta_opd_mask(
        student_logits=torch.randn(4, 25, 7, generator=generator),
        teacher_logits=torch.randn(4, 25, 7, generator=generator),
        response_mask=torch.ones(4, 25),
        topk=3,
        retain_ratio=0.10,
        normalize_scope="batch",
    )

    assert selected_mask.sum().item() == 10


def test_paper_retention_rejects_per_sample_selection():
    logits = torch.randn(2, 3, 7)
    with pytest.raises(ValueError, match="normalize_scope='batch'"):
        compute_ta_opd_mask(
            student_logits=logits,
            teacher_logits=logits,
            response_mask=torch.ones(2, 3),
            topk=3,
            retain_ratio=0.10,
            normalize_scope="sample",
        )


def test_selected_token_scale_recovers_global_mean_across_microbatches_and_ranks():
    # Rank 0 has two selected tokens split across micro-batches; rank 1 has one.
    # DDP/FSDP averages rank gradients, so each local numerator is multiplied by
    # world_size / global_selected_tokens.
    world_size = 2
    global_selected_tokens = 3
    scale = world_size / global_selected_tokens

    rank0_micro_losses = [torch.tensor([[1.0]]), torch.tensor([[3.0]])]
    rank1_micro_losses = [torch.tensor([[10.0]])]

    def rank_contribution(micro_losses):
        return sum(
            aggregate_ta_opd_loss(
                per_token_loss=loss,
                response_mask=torch.ones_like(loss),
                selected_mask=torch.ones_like(loss),
                normalize_by="selected_tokens",
                selected_token_scale=scale,
            )
            for loss in micro_losses
        )

    distributed_average = (rank_contribution(rank0_micro_losses) + rank_contribution(rank1_micro_losses)) / world_size
    assert distributed_average.item() == pytest.approx((1.0 + 3.0 + 10.0) / 3.0)
