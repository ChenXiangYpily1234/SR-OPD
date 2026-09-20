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
"""
TA-OPD: Teachability-Aware On-Policy Distillation

This module implements the TA-OPD token selection algorithm.
TA-OPD selects which tokens to supervise in OPD loss based on
the teachability score: D_norm * C_norm, where:
  - D_t: local KL disagreement between teacher and student on union support
  - C_t: compatibility mass (teacher probability on student top-K)
  - Both are robustly normalized per batch.

The algorithm operates AFTER teacher scoring (full teacher query),
and only reduces the number of supervised tokens, NOT the number
of teacher calls or teacher-scored tokens.
"""

from typing import Any, Dict, Optional, Tuple

import math

import torch
import torch.nn.functional as F


def select_opd_training_reward(
    sampled_opd_rm_scores: Optional[torch.Tensor],
    topk_rm_scores: Optional[torch.Tensor],
    ta_opd_enable: bool,
) -> torch.Tensor:
    """Return the shared sampled-token Actor target for every OPD method.

    Top-k rewards are auxiliary signals for selectors and diagnostics; they
    must not silently replace the Full/TA/PG training objective.
    """
    del topk_rm_scores, ta_opd_enable
    if sampled_opd_rm_scores is None:
        raise RuntimeError(
            "OPD requires sampled_opd_rm_scores from the Teacher worker; "
            "top-k rewards are auxiliary-only."
        )
    if sampled_opd_rm_scores.dim() != 2:
        raise RuntimeError(
            "OPD sampled reward must have shape [B, T], got "
            f"{tuple(sampled_opd_rm_scores.shape)}"
        )
    return sampled_opd_rm_scores


def robust_normalize(
    values: torch.Tensor,
    mask: torch.Tensor,
    q_low: float = 0.05,
    q_high: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Robust batch-wise normalization using quantile clipping.

    Args:
        values: [B, T] tensor of raw scores
        mask: [B, T] boolean or float mask (1 = valid, 0 = ignore)
        q_low: lower quantile (default 0.05)
        q_high: upper quantile (default 0.95)
        eps: small value for numerical stability

    Returns:
        normalized: [B, T] tensor, clipped to [0, 1]
    """
    if mask.sum() == 0:
        return torch.zeros_like(values)

    # Quantiles are fitted only on valid response tokens. Compute in fp32
    # because torch.quantile does not support bf16 on all backends.
    valid_values = values[mask.bool()].float()
    q_low_val = torch.quantile(valid_values, q_low)
    q_high_val = torch.quantile(valid_values, q_high)

    denom = q_high_val - q_low_val + eps
    normalized = (values.float() - q_low_val) / denom
    normalized = torch.clamp(normalized, 0.0, 1.0)

    return normalized.to(values.dtype)


def compute_ta_opd_mask(
    student_logits: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    student_topk_indices: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    teacher_on_student_log_probs: Optional[torch.Tensor] = None,
    student_on_teacher_log_probs: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    topk: int = 16,
    retain_ratio: float = 0.10,
    retain_token_count: Optional[int] = None,
    normalize_scope: str = "batch",
    eps: float = 1e-8,
    mode: str = "teachability",
    detach_score: bool = True,
    return_details: bool = False,
    score_normalization_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]] | Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute TA-OPD token selection mask.

    Args:
        student_logits: [B, T, V] full student logits (optional, for full logits mode)
        teacher_logits: [B, T, V] full teacher logits (optional, for full logits mode)
        student_topk_log_probs: [B, T, K_s] student top-k log probs
        student_topk_indices: [B, T, K_s] student top-k token indices
        teacher_topk_log_probs: [B, T, K_t] teacher top-k log probs
        teacher_topk_indices: [B, T, K_t] teacher top-k token indices
        teacher_on_student_log_probs: [B, T, K_s] teacher log probs
            evaluated at student top-k token ids
        student_on_teacher_log_probs: [B, T, K_t] student log probs
            evaluated at teacher top-k token ids
        response_mask: [B, T] mask for valid response tokens
        score_normalization_mask: optional [B, T] subset used only to fit
            robust score-normalization quantiles; scores are still produced
            for every valid response token
        topk: K for top-k computation (default 16)
        retain_ratio: fraction of valid tokens to retain (default 0.10)
        retain_token_count: exact number of tokens to retain (overrides retain_ratio)
        normalize_scope: must be "batch" for the paper TA-OPD estimator
        eps: small value for numerical stability
        mode: selector mode - "teachability", "compatibility", "disagreement",
              "entropy", "tip", "random"
        detach_score: whether to detach all scores from compute graph
        return_details: when True, return ``(selected_mask, details)`` for
            PRIOR-OPD; the default keeps the historical three-value API.

    Returns:
        With ``return_details=False``: ``(selected_mask, score, stats)``.
        With ``return_details=True``: ``(selected_mask, details)``.
    """
    if normalize_scope != "batch":
        raise ValueError(
            "Paper TA-OPD requires normalize_scope='batch'; per-sample selection changes the estimator"
        )

    use_full_logits = (student_logits is not None) and (teacher_logits is not None)
    use_topk_approx = not use_full_logits

    B, T = response_mask.shape
    device = response_mask.device
    dtype = response_mask.dtype

    # --- Initialize fallback flag ---
    fallback_used = 0
    missing_teacher_prob_count = 0

    # --- Compute D_t (local KL disagreement) and C_t (compatibility mass) ---
    if use_full_logits:
        # Full logits mode: exact computation on the deduplicated top-k union.
        student_logits_f = student_logits.float()
        teacher_logits_f = teacher_logits.float()

        if topk <= 0 or topk > student_logits_f.shape[-1]:
            raise ValueError(f"topk must be in [1, vocab_size], got {topk}")

        student_logp = F.log_softmax(student_logits_f, dim=-1)
        teacher_logp = F.log_softmax(teacher_logits_f, dim=-1)
        student_topk_logp, S_topk_idx = torch.topk(student_logp, k=topk, dim=-1)
        teacher_topk_logp, T_topk_idx = torch.topk(teacher_logp, k=topk, dim=-1)
        teacher_on_student = torch.gather(teacher_logp, dim=-1, index=S_topk_idx)
        student_on_teacher = torch.gather(student_logp, dim=-1, index=T_topk_idx)

        matches = S_topk_idx.unsqueeze(-1) == T_topk_idx.unsqueeze(-2)
        teacher_only = ~matches.any(dim=-2)
        union_valid = torch.cat([torch.ones_like(S_topk_idx, dtype=torch.bool), teacher_only], dim=-1)
        student_union_logp = torch.cat([student_topk_logp, student_on_teacher], dim=-1)
        teacher_union_logp = torch.cat([teacher_on_student, teacher_topk_logp], dim=-1)

        student_union_logp = student_union_logp.masked_fill(~union_valid, float("-inf"))
        teacher_union_logp = teacher_union_logp.masked_fill(~union_valid, float("-inf"))
        student_local_logp = student_union_logp - torch.logsumexp(student_union_logp, dim=-1, keepdim=True)
        teacher_local_logp = teacher_union_logp - torch.logsumexp(teacher_union_logp, dim=-1, keepdim=True)
        teacher_local_prob = teacher_local_logp.exp()
        local_log_ratio = torch.where(
            union_valid,
            teacher_local_logp - student_local_logp,
            torch.zeros_like(teacher_local_logp),
        )
        D_t = (teacher_local_prob * local_log_ratio).sum(dim=-1)
        C_t = teacher_on_student.exp().sum(dim=-1)
        entropy_flat = -(student_topk_logp.exp() * student_topk_logp).sum(dim=-1)
        topk_overlap_flat = matches.any(dim=-1).float().sum(dim=-1)
    else:
        # Memory-efficient exact computation on the deduplicated top-k union.

        if any(
            value is None
            for value in (
                student_topk_log_probs,
                teacher_topk_log_probs,
                student_topk_indices,
                teacher_topk_indices,
                teacher_on_student_log_probs,
                student_on_teacher_log_probs,
            )
        ):
            raise ValueError(
                "top-k TA-OPD requires student/teacher top-k ids and log-probs plus "
                "both cross-support log-prob tensors"
            )
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if student_topk_log_probs.shape[-1] < topk or teacher_topk_log_probs.shape[-1] < topk:
            raise ValueError(
                f"top-k tensors must contain at least topk={topk} entries; got "
                f"student={student_topk_log_probs.shape[-1]}, teacher={teacher_topk_log_probs.shape[-1]}"
            )

        s_logp = student_topk_log_probs[..., :topk].float()
        t_logp = teacher_topk_log_probs[..., :topk].float()
        s_idx = student_topk_indices[..., :topk]
        t_idx = teacher_topk_indices[..., :topk]
        t_on_s = teacher_on_student_log_probs[..., :topk].float()
        s_on_t = student_on_teacher_log_probs[..., :topk].float()

        for name, value, expected in (
            ("student_topk_indices", s_idx, s_logp),
            ("teacher_topk_indices", t_idx, t_logp),
            ("teacher_on_student_log_probs", t_on_s, s_logp),
            ("student_on_teacher_log_probs", s_on_t, t_logp),
        ):
            if value.shape != expected.shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected.shape}")

        matches = s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)
        teacher_only = ~matches.any(dim=-2)
        union_valid = torch.cat([torch.ones_like(s_idx, dtype=torch.bool), teacher_only], dim=-1)
        student_union_logp = torch.cat([s_logp, s_on_t], dim=-1).masked_fill(
            ~union_valid, float("-inf")
        )
        teacher_union_logp = torch.cat([t_on_s, t_logp], dim=-1).masked_fill(
            ~union_valid, float("-inf")
        )

        student_local_logp = student_union_logp - torch.logsumexp(student_union_logp, dim=-1, keepdim=True)
        teacher_local_logp = teacher_union_logp - torch.logsumexp(teacher_union_logp, dim=-1, keepdim=True)
        teacher_local_prob = teacher_local_logp.exp()
        local_log_ratio = torch.where(
            union_valid,
            teacher_local_logp - student_local_logp,
            torch.zeros_like(teacher_local_logp),
        )
        D_t = (teacher_local_prob * local_log_ratio).sum(dim=-1)
        C_t = t_on_s.exp().sum(dim=-1)
        entropy_flat = -(s_logp.exp() * s_logp).sum(dim=-1)
        topk_overlap_flat = matches.any(dim=-1).float().sum(dim=-1)

    # --- Robust normalization ---
    normalization_mask = response_mask
    normalization_fallback = 0
    if score_normalization_mask is not None:
        if score_normalization_mask.shape != response_mask.shape:
            raise ValueError("score_normalization_mask and response_mask must have identical shapes")
        requested_mask = score_normalization_mask.bool() & response_mask.bool()
        if requested_mask.any():
            normalization_mask = requested_mask
        else:
            normalization_fallback = 1
    D_norm = robust_normalize(D_t, normalization_mask, q_low=0.05, q_high=0.95, eps=eps)
    C_norm = robust_normalize(C_t, normalization_mask, q_low=0.05, q_high=0.95, eps=eps)
    H_norm = robust_normalize(entropy_flat, normalization_mask, q_low=0.05, q_high=0.95, eps=eps)

    # --- Compute score based on mode ---
    if mode == "teachability":
        raw_score = D_t * C_t
        score = D_norm * C_norm
    elif mode == "compatibility":
        raw_score = C_t
        score = C_norm
    elif mode == "disagreement":
        raw_score = D_t
        score = D_norm
    elif mode == "entropy":
        raw_score = entropy_flat
        score = H_norm
    elif mode == "tip":
        raw_score = entropy_flat + D_t - entropy_flat * D_t
        score = H_norm + D_norm - H_norm * D_norm
    elif mode == "random":
        score = torch.rand_like(D_norm)
        raw_score = score
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Detach if requested ---
    if detach_score:
        score = score.detach()
        D_t = D_t.detach()
        C_t = C_t.detach()
        D_norm = D_norm.detach()
        C_norm = C_norm.detach()
        raw_score = raw_score.detach()

    # --- Token selection ---
    # Masked score: set score to -inf for invalid tokens
    mask_bool = response_mask.bool()
    masked_score = torch.where(mask_bool, score, torch.full_like(score, float('-inf')))

    # Flatten the entire batch and select exactly ceil(rho * N_valid), as in
    # the TA-OPD paper.
    flat_score = masked_score.reshape(-1)
    flat_mask = mask_bool.reshape(-1)

    n_valid = flat_mask.sum().item()
    if n_valid == 0:
        selected_mask = torch.zeros(B, T, dtype=dtype, device=device)
        n_selected = 0
    else:
        if retain_token_count is not None:
            n_selected = min(retain_token_count, n_valid)
        else:
            n_selected = math.ceil(retain_ratio * n_valid)

        _, top_indices = torch.topk(flat_score, k=n_selected)
        selected_flat = torch.zeros(flat_score.shape[0], dtype=dtype, device=device)
        selected_flat[top_indices] = 1.0
        selected_mask = selected_flat.reshape(B, T)

    if detach_score:
        selected_mask = selected_mask.detach()

    # --- Compute selection threshold ---
    if mask_bool.sum() > 0:
        selected_scores = masked_score[selected_mask.bool()]
        if selected_scores.numel() > 0:
            selection_threshold_tensor = selected_scores.min().detach()
            selection_threshold = selection_threshold_tensor.item()
        else:
            selection_threshold_tensor = torch.zeros((), device=device, dtype=score.dtype)
            selection_threshold = 0.0
    else:
        selection_threshold_tensor = torch.zeros((), device=device, dtype=score.dtype)
        selection_threshold = 0.0

    # --- Compute stats ---
    valid_count = mask_bool.sum().item()
    selected_count = selected_mask.sum().item()

    stats = {}
    # Token selection stats
    stats["ta_opd/selected_token_count"] = selected_count
    stats["ta_opd/valid_response_token_count"] = valid_count
    stats["ta_opd/selected_token_ratio"] = selected_count / max(valid_count, 1)
    stats["ta_opd/unselected_token_count"] = valid_count - selected_count
    stats["ta_opd/selection_threshold"] = float(selection_threshold)

    # Score stats
    valid_score = score[mask_bool]
    if valid_score.numel() > 0:
        stats["ta_opd/score_mean"] = valid_score.mean().item()
        stats["ta_opd/score_max"] = valid_score.max().item()
        stats["ta_opd/score_min"] = valid_score.min().item()
        stats["ta_opd/score_std"] = valid_score.std(unbiased=False).item()
    else:
        stats["ta_opd/score_mean"] = 0.0
        stats["ta_opd/score_max"] = 0.0
        stats["ta_opd/score_min"] = 0.0
        stats["ta_opd/score_std"] = 0.0

    # D_t stats
    valid_D = D_t[mask_bool]
    if valid_D.numel() > 0:
        stats["ta_opd/D_mean"] = valid_D.mean().item()
        stats["ta_opd/D_max"] = valid_D.max().item()
        stats["ta_opd/D_min"] = valid_D.min().item()
        stats["ta_opd/D_std"] = valid_D.std(unbiased=False).item()
    else:
        stats["ta_opd/D_mean"] = 0.0
        stats["ta_opd/D_max"] = 0.0
        stats["ta_opd/D_min"] = 0.0
        stats["ta_opd/D_std"] = 0.0

    # C_t stats
    valid_C = C_t[mask_bool]
    if valid_C.numel() > 0:
        stats["ta_opd/C_mean"] = valid_C.mean().item()
        stats["ta_opd/C_max"] = valid_C.max().item()
        stats["ta_opd/C_min"] = valid_C.min().item()
        stats["ta_opd/C_std"] = valid_C.std(unbiased=False).item()
    else:
        stats["ta_opd/C_mean"] = 0.0
        stats["ta_opd/C_max"] = 0.0
        stats["ta_opd/C_min"] = 0.0
        stats["ta_opd/C_std"] = 0.0

    # D_norm / C_norm stats
    valid_Dn = D_norm[mask_bool]
    valid_Cn = C_norm[mask_bool]
    stats["ta_opd/D_norm_mean"] = valid_Dn.mean().item() if valid_Dn.numel() > 0 else 0.0
    stats["ta_opd/C_norm_mean"] = valid_Cn.mean().item() if valid_Cn.numel() > 0 else 0.0

    # Entropy stats
    valid_H = entropy_flat[mask_bool]
    stats["ta_opd/entropy_mean"] = valid_H.mean().item() if valid_H.numel() > 0 else 0.0

    # Selected/unselected score stats
    sel_mask_bool = selected_mask.bool()
    selected_scores_arr = score[sel_mask_bool]
    unselected_scores_arr = score[mask_bool & ~sel_mask_bool]
    stats["ta_opd/selected_score_mean"] = selected_scores_arr.mean().item() if selected_scores_arr.numel() > 0 else 0.0
    stats["ta_opd/unselected_score_mean"] = unselected_scores_arr.mean().item() if unselected_scores_arr.numel() > 0 else 0.0

    # Selected D/C stats
    selected_D = D_t[sel_mask_bool]
    unselected_D = D_t[mask_bool & ~sel_mask_bool]
    selected_C = C_t[sel_mask_bool]
    unselected_C = C_t[mask_bool & ~sel_mask_bool]
    stats["ta_opd/selected_D_mean"] = selected_D.mean().item() if selected_D.numel() > 0 else 0.0
    stats["ta_opd/unselected_D_mean"] = unselected_D.mean().item() if unselected_D.numel() > 0 else 0.0
    stats["ta_opd/selected_C_mean"] = selected_C.mean().item() if selected_C.numel() > 0 else 0.0
    stats["ta_opd/unselected_C_mean"] = unselected_C.mean().item() if unselected_C.numel() > 0 else 0.0

    # Selected D_norm / C_norm
    selected_Dn = D_norm[sel_mask_bool]
    selected_Cn = C_norm[sel_mask_bool]
    stats["ta_opd/selected_D_norm_mean"] = selected_Dn.mean().item() if selected_Dn.numel() > 0 else 0.0
    stats["ta_opd/selected_C_norm_mean"] = selected_Cn.mean().item() if selected_Cn.numel() > 0 else 0.0

    # Configuration stats
    stats["ta_opd/topk"] = topk
    stats["ta_opd/retain_ratio"] = retain_ratio
    # The selector receives exact Teacher log-probabilities evaluated on the
    # Student top-k support, so C_t is exact coverage rather than an
    # intersection-only lower bound. Keep this numeric for scalar loggers.
    stats["ta_opd/coverage_is_exact"] = 1

    def add_quantiles(prefix: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            stats[f"ta_opd/{prefix}_q05"] = 0.0
            stats[f"ta_opd/{prefix}_q50"] = 0.0
            stats[f"ta_opd/{prefix}_q95"] = 0.0
            return
        tensor = tensor.float()
        stats[f"ta_opd/{prefix}_q05"] = torch.quantile(tensor, 0.05).item()
        stats[f"ta_opd/{prefix}_q50"] = torch.quantile(tensor, 0.50).item()
        stats[f"ta_opd/{prefix}_q95"] = torch.quantile(tensor, 0.95).item()

    add_quantiles("D", valid_D)
    add_quantiles("C", valid_C)
    add_quantiles("score", valid_score)
    # Note: mode is a string, not logged as scalar metric to avoid TensorBoard errors

    # Fallback stats
    stats["ta_opd/fallback_used"] = fallback_used
    if use_topk_approx:
        valid_overlap = topk_overlap_flat[mask_bool]
        stats["ta_opd/topk_overlap_mean"] = valid_overlap.mean().item() if valid_overlap.numel() > 0 else 0.0
    stats["ta_opd/missing_teacher_prob_count"] = missing_teacher_prob_count
    stats["ta_opd/normalization_token_count"] = int(normalization_mask.sum().item())
    stats["ta_opd/normalization_fallback"] = normalization_fallback

    if return_details:
        details = {
            "raw_score": raw_score.detach() if detach_score else raw_score,
            "normalized_score": score.detach() if detach_score else score,
            "selection_threshold": selection_threshold_tensor,
            "valid_response_mask": response_mask.detach(),
            "selected_mask": selected_mask,
            "stats": stats,
        }
        return selected_mask, details
    return selected_mask, score, stats


def aggregate_ta_opd_loss(
    per_token_loss: torch.Tensor,
    selected_mask: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_by: str = "selected_tokens",
    selected_token_scale: torch.Tensor | float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Aggregate per-token loss using TA-OPD selected mask.

    Args:
        per_token_loss: [B, T] or [B, T, K] per-token loss
        selected_mask: [B, T] mask of selected tokens
        response_mask: [B, T] original response mask
        normalize_by: normalization mode
            - "selected_tokens": divide by sum(selected_mask)
            - "all_tokens": divide by sum(response_mask)
            - "original": uses original masked_mean (response_mask)
        selected_token_scale: optional distributed multiplier for the local
            selected-token numerator
        eps: small value for numerical stability

    Returns:
        aggregated loss (scalar)
    """
    if per_token_loss.dim() == 3:
        # 3D case: sum over K dimension first
        per_token_loss = per_token_loss.sum(dim=-1)

    if normalize_by == "selected_tokens":
        final_mask = response_mask * selected_mask
        numerator = (per_token_loss * final_mask).sum()
        if selected_token_scale is not None:
            return numerator * selected_token_scale
        total = final_mask.sum()
        if total == 0:
            # Preserve a zero-valued autograd path when an explicit count is 0.
            return numerator
        loss = numerator / total
    elif normalize_by == "all_tokens":
        total = response_mask.sum()
        if total == 0:
            return torch.tensor(0.0, device=per_token_loss.device, dtype=per_token_loss.dtype)
        loss = (per_token_loss * selected_mask * response_mask).sum() / total
    elif normalize_by == "original":
        # Use original response_mask only
        total = response_mask.sum()
        if total == 0:
            return torch.tensor(0.0, device=per_token_loss.device, dtype=per_token_loss.dtype)
        loss = (per_token_loss * selected_mask * response_mask).sum() / total
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")

    return loss


def compute_opd_metrics(
    response_mask: torch.Tensor,
    selected_mask: Optional[torch.Tensor] = None,
    prompt_token_count: Optional[int] = None,
    teacher_scored_response_token_count: Optional[int] = None,
    teacher_processed_input_token_count: Optional[int] = None,
    per_token_loss: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    ta_opd_enabled: bool = False,
    full_query_teacher_scored: Optional[int] = None,
    full_query_teacher_processed: Optional[int] = None,
    full_valid_response_token_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute unified OPD metrics (shared across Pure OPD, TA-OPD, and future selective OPD).

    Args:
        response_mask: [B, T] response mask
        selected_mask: [B, T] TA-OPD selected mask (None for Pure OPD)
        prompt_token_count: total prompt tokens in batch
        teacher_scored_response_token_count: teacher-scored response tokens
        teacher_processed_input_token_count: teacher-processed input tokens
        per_token_loss: [B, T] per-token loss
        student_topk_log_probs: [B, T, K] student top-k log probs
        teacher_topk_log_probs: [B, T, K] teacher top-k log probs
        ta_opd_enabled: whether TA-OPD is active
        full_query_teacher_scored: full-query teacher scored tokens (baseline)
        full_query_teacher_processed: full-query teacher processed tokens (baseline)

    Returns:
        dict of metrics
    """
    metrics = {}

    mask_bool = response_mask.bool()
    valid_count = mask_bool.sum().item()

    # Token counts
    metrics["opd/response_token_count"] = response_mask.shape[0] * response_mask.shape[1]
    metrics["opd/valid_response_token_count"] = valid_count
    metrics["opd/padding_token_count"] = (
        response_mask.shape[0] * response_mask.shape[1] - valid_count
    )

    if prompt_token_count is not None:
        metrics["opd/prompt_token_count"] = prompt_token_count
        metrics["opd/total_token_count"] = prompt_token_count + metrics["opd/response_token_count"]

    # Supervised token counts
    if ta_opd_enabled and selected_mask is not None:
        supervised_count = selected_mask.sum().item()
    else:
        supervised_count = valid_count

    metrics["opd/supervised_token_count"] = supervised_count
    post_query_supervised_ratio = supervised_count / max(valid_count, 1)
    full_valid_count = (
        valid_count if full_valid_response_token_count is None else full_valid_response_token_count
    )
    metrics["opd/post_query_supervised_token_ratio"] = post_query_supervised_ratio
    metrics["opd/global_supervised_token_ratio"] = supervised_count / max(full_valid_count, 1)
    metrics["opd/teacher_query_token_ratio"] = valid_count / max(full_valid_count, 1)
    # Deprecated compatibility alias. This retains its historical post-query
    # denominator; new dashboards must use one of the explicit metrics above.
    metrics["opd/supervised_token_ratio"] = post_query_supervised_ratio
    metrics["opd/unsupervised_token_count"] = valid_count - supervised_count
    metrics["opd/unsupervised_token_ratio"] = (valid_count - supervised_count) / max(valid_count, 1)

    # Teacher scoring token counts
    if teacher_scored_response_token_count is not None:
        metrics["opd/teacher_scored_response_token_count"] = teacher_scored_response_token_count
    if teacher_processed_input_token_count is not None:
        metrics["opd/teacher_processed_input_token_count"] = teacher_processed_input_token_count
    if teacher_scored_response_token_count is not None and valid_count > 0:
        metrics["opd/teacher_scored_token_ratio"] = (
            teacher_scored_response_token_count / max(valid_count, 1)
        )

    # Loss metrics
    if per_token_loss is not None:
        if per_token_loss.dim() == 3:
            pt_loss = per_token_loss.sum(dim=-1)
        else:
            pt_loss = per_token_loss

        valid_loss = pt_loss[mask_bool]
        if valid_loss.numel() > 0:
            metrics["opd/distillation_advantage_mean"] = valid_loss.mean().item()
            metrics["opd/distillation_advantage_max"] = valid_loss.max().item()
            metrics["opd/distillation_advantage_min"] = valid_loss.min().item()

        if ta_opd_enabled and selected_mask is not None:
            sel_mask_bool = selected_mask.bool()
            selected_loss = pt_loss[sel_mask_bool]
            unselected_loss = pt_loss[mask_bool & ~sel_mask_bool]
            if selected_loss.numel() > 0:
                metrics["opd/selected_distillation_advantage_mean"] = selected_loss.mean().item()
            else:
                metrics["opd/selected_distillation_advantage_mean"] = 0.0
            if unselected_loss.numel() > 0:
                metrics["opd/unselected_distillation_advantage_mean"] = unselected_loss.mean().item()
            else:
                metrics["opd/unselected_distillation_advantage_mean"] = 0.0
        else:
            metrics["opd/selected_distillation_advantage_mean"] = metrics.get(
                "opd/distillation_advantage_mean", 0.0
            )
            metrics["opd/unselected_distillation_advantage_mean"] = 0.0

    # Distribution metrics from top-k
    if student_topk_log_probs is not None:
        s_probs = torch.exp(student_topk_log_probs)  # [B, T, K]
        s_entropy = -(s_probs * torch.log(s_probs + 1e-8)).sum(dim=-1)  # [B, T]
        valid_s_entropy = s_entropy[mask_bool]
        if valid_s_entropy.numel() > 0:
            metrics["opd/topk_student_entropy_mean"] = valid_s_entropy.mean().item()

    if teacher_topk_log_probs is not None:
        t_probs = torch.exp(teacher_topk_log_probs)
        t_entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1)
        valid_t_entropy = t_entropy[mask_bool]
        if valid_t_entropy.numel() > 0:
            metrics["opd/topk_teacher_entropy_mean"] = valid_t_entropy.mean().item()

    if student_topk_log_probs is not None and teacher_topk_log_probs is not None:
        s_probs = torch.exp(student_topk_log_probs)
        t_probs = torch.exp(teacher_topk_log_probs)
        # Approximate KL on top-K
        kl_approx = (t_probs * (torch.log(t_probs + 1e-8) - torch.log(s_probs + 1e-8))).sum(dim=-1)
        valid_kl = kl_approx[mask_bool]
        if valid_kl.numel() > 0:
            metrics["opd/topk_student_teacher_kl_mean"] = valid_kl.mean().item()

    # Method ID
    if ta_opd_enabled:
        metrics["opd/method_id"] = 1  # ta_opd
    else:
        metrics["opd/method_id"] = 0  # pure_opd

    # Saving ratios
    if full_query_teacher_scored is not None and teacher_scored_response_token_count is not None:
        metrics["cost/teacher_scored_response_token_saving_ratio"] = max(
            0.0,
            1.0 - teacher_scored_response_token_count / max(full_query_teacher_scored, 1),
        )
    if full_query_teacher_processed is not None and teacher_processed_input_token_count is not None:
        metrics["cost/teacher_processed_input_token_saving_ratio"] = max(
            0.0,
            1.0 - teacher_processed_input_token_count / max(full_query_teacher_processed, 1),
        )

    # Supervised reduction ratio
    metrics["cost/supervised_token_reduction_ratio"] = max(
        0.0, 1.0 - supervised_count / max(valid_count, 1)
    )

    if teacher_scored_response_token_count is not None:
        metrics["cost/supervised_over_teacher_scored_ratio"] = (
            supervised_count / max(teacher_scored_response_token_count, 1)
        )
    if teacher_processed_input_token_count is not None:
        metrics["cost/supervised_over_teacher_processed_ratio"] = (
            supervised_count / max(teacher_processed_input_token_count, 1)
        )

    return metrics
