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

"""Token-level confidence-trajectory selection for Frontier FF-OPD."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from verl.utils.boundary_opd import BOUNDARY_SELECTOR_MODES

FRONTIER_SELECTOR = "token_confidence_trajectory_nearest_positive"
SELECTOR_FORMULA_VERSION = "ff_cost_raw_v1"
FF_SELECTOR_MODES = (
    "global_random",
    "random_wrong",
    "all_wrong",
    "random_correct",
    "nearest_only",
    "cost_only",
    "farthest_only",
    "shortest_wrong",
    # Frontier-TLR: matched-support baseline for Frontier-PDA.
    # Same mixed-only gating and wrong-only candidate set as Frontier-PDA /
    # Frontier-Shortest; only the selector is replaced with the TLR score
    # Score_LH = (1 - L̂) * (1 - Ĥ).
    "frontier_tlr",
    *BOUNDARY_SELECTOR_MODES,
)


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean().item()) if finite.numel() else 0.0


def select_ff_rollouts(
    *,
    valid_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    distance: torch.Tensor,
    teacher_cost: torch.Tensor,
    mode: str,
    base_seed: int,
    global_step: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select the FF-OPD Teacher queries for one global rollout batch.

    Randomness is generated on CPU so a selection made by the global-batch
    owner is independent of its accelerator type. Callers must broadcast the
    returned mask if the global Teacher batch is replicated across ranks.
    """
    if mode not in FF_SELECTOR_MODES:
        raise ValueError(f"algorithm.ff_selector_mode must be one of {FF_SELECTOR_MODES}, got {mode!r}")
    if valid_mask.ndim != 2:
        raise ValueError("FF selector inputs must have shape [B, K]")
    if any(tensor.shape != valid_mask.shape for tensor in (correct_mask, distance, teacher_cost)):
        raise ValueError("valid_mask, correct_mask, distance, and teacher_cost must have identical shapes")
    if int(base_seed) < 0 or int(global_step) < 0:
        raise ValueError("base_seed and global_step must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    device = valid_mask.device
    valid = valid_mask.detach().bool().cpu()
    correct = correct_mask.detach().bool().cpu()
    distances = distance.detach().to(dtype=torch.float64, device="cpu")
    costs = teacher_cost.detach().to(dtype=torch.float64, device="cpu")
    if torch.any(valid & (~torch.isfinite(costs) | (costs <= 0))):
        raise ValueError("teacher_cost must be finite and positive for valid rollouts")

    wrong = valid & ~correct
    positive = valid & correct
    frontier = positive.any(dim=1) & wrong.any(dim=1)
    target_query_count = int(
        ((wrong & frontier[:, None]).sum() if mode == "all_wrong" else frontier.sum()).item()
    )
    selected = torch.zeros_like(valid)
    tie_count = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(base_seed) * 1_000_003 + int(global_step))

    if mode == "all_wrong":
        selected = wrong & frontier[:, None]
    elif mode == "global_random":
        valid_flat = valid.flatten().nonzero(as_tuple=False).flatten()
        if target_query_count > int(valid_flat.numel()):
            raise AssertionError("frontier prompt count cannot exceed the number of valid rollouts")
        if target_query_count:
            chosen = valid_flat[torch.randperm(valid_flat.numel(), generator=generator)[:target_query_count]]
            selected.flatten()[chosen] = True
    else:
        for prompt_index in frontier.nonzero(as_tuple=False).flatten().tolist():
            candidate_mask = positive if mode == "random_correct" else wrong
            candidates = candidate_mask[prompt_index].nonzero(as_tuple=False).flatten()
            if mode in {"random_wrong", "random_correct"}:
                chosen = candidates[torch.randint(candidates.numel(), (1,), generator=generator).item()]
            else:
                candidate_distances = distances[prompt_index, candidates]
                candidate_costs = costs[prompt_index, candidates]
                finite_distances = torch.isfinite(candidate_distances)
                if mode != "cost_only" and not finite_distances.any():
                    # Preserve the legacy uniform fallback when token-profile
                    # alignment cannot produce a distance for this prompt.
                    tie_count += int(candidates.numel() > 1)
                    chosen = candidates[torch.randint(candidates.numel(), (1,), generator=generator).item()]
                    selected[prompt_index, chosen] = True
                    continue
                if mode != "cost_only" and (not finite_distances.all() or (candidate_distances < 0).any()):
                    raise ValueError("distance must be finite and non-negative for frontier wrong candidates")
                if mode == "nearest_only":
                    scores = candidate_distances
                    optimum = scores.min()
                elif mode == "cost_only":
                    scores = candidate_costs
                    optimum = scores.min()
                elif mode == "farthest_only":
                    scores = candidate_distances
                    optimum = scores.max()
                elif mode in BOUNDARY_SELECTOR_MODES or mode in {"shortest_wrong", "frontier_tlr"}:
                    raise ValueError(f"{mode} is selected by its dedicated rollout selector")
                else:
                    raise AssertionError(f"unhandled FF selector mode: {mode}")
                tied = candidates[scores == optimum]
                tie_count += int(tied.numel() > 1)
                chosen = tied[torch.randint(tied.numel(), (1,), generator=generator).item()]
            selected[prompt_index, chosen] = True

    selected_count = int(selected.sum().item())
    selected_frontier = selected & frontier[:, None]
    objective = (distances + eps) * torch.sqrt(costs)
    finite_wrong_distance = distances[wrong & torch.isfinite(distances)]
    stats = {
        "frontier_prompt_count": float(target_query_count),
        "target_query_count": float(target_query_count),
        "actual_query_count": float(selected_count),
        "selected_valid_count": float((selected & valid).sum().item()),
        "selected_correct_count": float((selected & valid & correct).sum().item()),
        "selected_wrong_count": float((selected & wrong).sum().item()),
        "selected_frontier_count": float(selected_frontier.sum().item()),
        "selected_nonfrontier_count": float((selected & ~frontier[:, None]).sum().item()),
        "selected_distance_mean": _finite_mean(distances[selected]),
        "selected_cost_mean": _finite_mean(costs[selected]),
        "selected_objective_mean": _finite_mean(objective[selected]),
        "all_wrong_distance_mean": _finite_mean(finite_wrong_distance),
        "all_wrong_cost_mean": _finite_mean(costs[wrong]),
        "tie_count": float(tie_count),
        "zero_frontier_step": float(target_query_count == 0),
        "boundary_fallback_prompt_count": 0.0,
    }

    if selected_count != target_query_count:
        raise AssertionError("FF selector query count differs from the frontier prompt count")
    if (selected & ~valid).any():
        raise AssertionError("FF selector chose an invalid rollout")
    if mode != "global_random":
        expected_candidates = positive if mode == "random_correct" else wrong
        if (selected & (~expected_candidates | ~frontier[:, None])).any():
            candidate_type = "correct" if mode == "random_correct" else "wrong"
            raise AssertionError(f"{mode} must select only {candidate_type} rollouts from frontier prompts")
        if mode == "all_wrong" and not torch.equal(selected, wrong & frontier[:, None]):
            raise AssertionError("all_wrong must select every valid wrong rollout from frontier prompts")
        if mode != "all_wrong" and target_query_count and not torch.equal(selected.sum(dim=1).bool(), frontier):
            raise AssertionError(f"{mode} must select exactly one rollout per frontier prompt")
    return selected.to(device=device), stats


@dataclass
class FrontierSelectionResult:
    selected_negative_idx: Optional[int] = None
    matched_nearest_positive_idx: Optional[int] = None
    selector_type: str = FRONTIER_SELECTOR
    positive_length: Optional[int] = None
    negative_length: Optional[int] = None
    common_profile_length: Optional[int] = None
    nearest_positive_profile_distance: Optional[float] = None
    mean_positive_profile_distance: Optional[float] = None
    distance_min: Optional[float] = None
    distance_second_min: Optional[float] = None
    distance_gap: Optional[float] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    selection_mode: str = "none"
    all_pair_count: int = 0
    # Sibling bucketing of the attempt, kept for per-attempt CSV auditing.
    positive_indices: tuple[int, ...] = ()
    negative_indices: tuple[int, ...] = ()
    # rollout index -> mean absolute profile distance against every positive.
    # Audit only; the selection distance is candidate_nearest_positive_distances.
    candidate_mean_positive_distances: dict[int, float] = field(default_factory=dict)
    # rollout index -> strict nearest-positive distance d_j = min_i D(y^-_j, y^+_i).
    candidate_nearest_positive_distances: dict[int, float] = field(default_factory=dict)
    # rollout index -> Teacher processing cost c_j = prompt + response tokens.
    candidate_costs: dict[int, float] = field(default_factory=dict)
    # rollout index -> J_j = (d_j + eps) * sqrt(c_j) under FF-Cost, or the raw
    # distance under the nearest-only ablation.
    candidate_objectives: dict[int, float] = field(default_factory=dict)
    # rollout index -> log S_j = -log(d_j + eps) - 0.5 * log(c_j), or -d_j
    # under the nearest-only ablation.
    candidate_log_scores: dict[int, float] = field(default_factory=dict)
    selected_objective: Optional[float] = None
    selected_log_score: Optional[float] = None
    # The nearest-only (argmin d) pick, always computed as the ablation control.
    nearest_only_selected_idx: Optional[int] = None
    # The pure lowest-cost pick, always computed for audit only.
    lowest_cost_selected_idx: Optional[int] = None
    # Teacher processing cost of the selected negative.
    selected_cost: Optional[float] = None
    # Teacher processing cost of the nearest-only control pick.
    nearest_only_cost: Optional[float] = None
    # True when the cost-aware pick differs from the nearest-only pick.
    cost_aware: bool = False
    cost_switch: bool = False
    # selected distance minus the nearest-only distance; must be >= 0.
    distance_regret: float = 0.0
    # 1 - selected_cost / nearest_only_cost; > 0 only when a cost switch pays off.
    relative_cost_reduction: float = 0.0
    # Only populated for sampled audit rows; never written to ff.csv.
    positive_profile: Optional[list[float]] = None
    negative_profile: Optional[list[float]] = None
    positive_profile_resampled: Optional[list[float]] = None
    negative_profile_resampled: Optional[list[float]] = None


@dataclass(frozen=True)
class _Trajectory:
    student_log_probs: list[float]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _stable_int_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


class FrontierTrajectorySelector:
    """Select the negative whose aligned confidence curve is nearest to all positives."""

    def __init__(
        self,
        *,
        seed: int = 42,
        cost_aware: bool = False,
        cost_alpha: float = 0.5,
        score_eps: float = 1e-8,
    ):
        self.seed = seed
        self.cost_aware = bool(cost_aware)
        self.cost_alpha = float(cost_alpha)
        self.score_eps = float(score_eps)

    def select(
        self,
        *,
        response_masks: Any,
        verifier_correct: Sequence[int],
        rollout_valid: Sequence[bool],
        student_sampled_log_probs: Any,
        prompt_uid: str,
        attempt_id: int = 1,
        global_step: int = 0,
        collect_profiles: bool = False,
        candidate_costs: Optional[Sequence[float]] = None,
    ) -> FrontierSelectionResult:
        del global_step
        correct = [bool(value) for value in verifier_correct]
        valid = [bool(value) for value in rollout_valid]
        if len(valid) != len(correct):
            return self._uniform_fallback([], prompt_uid, attempt_id, "token_alignment_failed:rollout_metadata_length")

        positives = [index for index in range(len(correct)) if valid[index] and correct[index]]
        negatives = [index for index in range(len(correct)) if valid[index] and not correct[index]]
        buckets = {"positive_indices": tuple(positives), "negative_indices": tuple(negatives)}
        if not positives:
            return FrontierSelectionResult(fallback_reason="no_valid_correct", **buckets)
        if not negatives:
            return FrontierSelectionResult(fallback_reason="no_valid_incorrect", **buckets)

        costs: dict[int, float] = {}
        if candidate_costs is not None:
            if len(candidate_costs) != len(correct):
                raise ValueError("candidate_costs must align with verifier_correct")
            costs = {int(index): float(candidate_costs[index]) for index in negatives}

        trajectories, alignment_error = self._build_trajectories(
            response_masks,
            student_sampled_log_probs,
            len(correct),
        )
        if alignment_error:
            return self._uniform_fallback(negatives, prompt_uid, attempt_id, alignment_error, positives)

        all_profiles = [trajectories[index].student_log_probs for index in positives + negatives]
        if any(not profile for profile in all_profiles):
            return self._uniform_fallback(negatives, prompt_uid, attempt_id, "empty_response", positives)
        common_length = max(len(profile) for profile in all_profiles)
        try:
            positive_profiles = torch.stack(
                [
                    self._resample_to_common_length(trajectories[index].student_log_probs, common_length)
                    for index in positives
                ]
            )
            negative_profiles = torch.stack(
                [
                    self._resample_to_common_length(trajectories[index].student_log_probs, common_length)
                    for index in negatives
                ]
            )
        except (RuntimeError, TypeError, ValueError):
            return self._uniform_fallback(
                negatives, prompt_uid, attempt_id, "token_alignment_failed:resample", positives
            )

        pair_distances = self._compute_profile_l1_distance(positive_profiles, negative_profiles)
        if not torch.isfinite(pair_distances).all():
            return self._uniform_fallback(
                negatives, prompt_uid, attempt_id, "token_alignment_failed:non_finite", positives
            )
        # Keep the positive-set mean distance for audit, but the selection
        # distance is the strict nearest-positive one:
        #   d_{p,j} = min_{i in P_p} D(y^-_{p,j}, y^+_{p,i}).
        mean_distance_tensor = pair_distances.mean(dim=0)
        min_distance_tensor, nearest_positive_rows = pair_distances.min(dim=0)
        mean_positive_distances = {
            int(negatives[column]): float(mean_distance_tensor[column].item()) for column in range(len(negatives))
        }
        nearest_positive_distances = {
            int(negatives[column]): float(min_distance_tensor[column].item()) for column in range(len(negatives))
        }
        candidate_distances = [nearest_positive_distances[int(index)] for index in negatives]
        if not all(math.isfinite(distance) and distance >= 0 for distance in candidate_distances):
            return self._uniform_fallback(negatives, prompt_uid, attempt_id, "invalid_profile_distance", positives)

        sorted_distances, _ = torch.sort(min_distance_tensor)
        distance_min = float(sorted_distances[0].item())
        distance_second_min = float(sorted_distances[1].item()) if len(negatives) > 1 else None
        distance_gap = None if distance_second_min is None else distance_second_min - distance_min

        # Nearest-only control pick: argmin_j d_{p,j} with deterministic
        # tie-break on the rollout index. Always computed so the cost-aware
        # mode can report the cost-induced switch against it.
        nearest_only_column = min(
            range(len(negatives)),
            key=lambda column: (candidate_distances[column], negatives[column]),
        )

        costs_active = bool(self.cost_aware)
        if costs_active and candidate_costs is None:
            raise ValueError("cost-aware frontier selection requires per-rollout candidate_costs")
        if costs_active and not costs:
            raise ValueError("cost-aware frontier selection requires per-rollout candidate_costs")

        # Pure lowest-cost pick, audit only.
        if costs:
            lowest_cost_column = min(
                range(len(negatives)),
                key=lambda column: (costs[int(negatives[column])], negatives[column]),
            )
        else:
            lowest_cost_column = nearest_only_column

        if costs_active:
            objectives = []
            log_scores = []
            for distance, cost in zip(candidate_distances, [costs[int(index)] for index in negatives], strict=True):
                objectives.append((distance + self.score_eps) * (cost**self.cost_alpha))
                log_scores.append(-math.log(distance + self.score_eps) - self.cost_alpha * math.log(cost))
            candidate_objectives = {int(negatives[column]): objectives[column] for column in range(len(negatives))}
            candidate_log_scores = {int(negatives[column]): log_scores[column] for column in range(len(negatives))}
            selected_column = min(
                range(len(negatives)),
                key=lambda column: (objectives[column], negatives[column]),
            )
            selection_mode = FRONTIER_SELECTOR + "_cost"
        else:
            candidate_objectives = {
                int(negatives[column]): candidate_distances[column] for column in range(len(negatives))
            }
            candidate_log_scores = {
                int(negatives[column]): -candidate_distances[column] for column in range(len(negatives))
            }
            selected_column = nearest_only_column
            selection_mode = FRONTIER_SELECTOR
        if len(negatives) == 1:
            selection_mode = "single_negative_direct"

        selected_negative = negatives[selected_column]
        nearest_negative = negatives[nearest_only_column]
        selected_distance = float(candidate_distances[selected_column])
        nearest_distance = float(candidate_distances[nearest_only_column])
        selected_cost = costs.get(int(selected_negative))
        nearest_cost = costs.get(int(nearest_negative))
        cost_switch = bool(selected_column != nearest_only_column)
        distance_regret = selected_distance - nearest_distance
        if distance_regret < -1e-8:
            raise AssertionError("FF-Cost distance regret must be non-negative")
        if nearest_cost is not None and selected_cost is not None:
            relative_cost_reduction = 1.0 - selected_cost / max(nearest_cost, 1.0)
        else:
            relative_cost_reduction = 0.0

        # The matched positive is exactly the argmin row defining d_{p,j}.
        nearest_positive_row = int(nearest_positive_rows[selected_column].item())
        matched_nearest_positive = positives[nearest_positive_row]
        result = FrontierSelectionResult(
            selected_negative_idx=selected_negative,
            matched_nearest_positive_idx=matched_nearest_positive,
            selector_type=selection_mode,
            positive_length=len(trajectories[matched_nearest_positive].student_log_probs),
            negative_length=len(trajectories[selected_negative].student_log_probs),
            common_profile_length=common_length,
            nearest_positive_profile_distance=float(pair_distances[nearest_positive_row, selected_column].item()),
            mean_positive_profile_distance=float(mean_distance_tensor[selected_column].item()),
            distance_min=distance_min,
            distance_second_min=distance_second_min,
            distance_gap=distance_gap,
            selection_mode=selection_mode,
            all_pair_count=len(positives) * len(negatives),
            candidate_mean_positive_distances=mean_positive_distances,
            candidate_nearest_positive_distances=nearest_positive_distances,
            candidate_costs=costs,
            candidate_objectives=candidate_objectives,
            candidate_log_scores=candidate_log_scores,
            selected_objective=candidate_objectives.get(int(selected_negative)),
            selected_log_score=candidate_log_scores.get(int(selected_negative)),
            nearest_only_selected_idx=int(nearest_negative),
            lowest_cost_selected_idx=int(negatives[lowest_cost_column]),
            selected_cost=selected_cost,
            nearest_only_cost=nearest_cost,
            cost_aware=costs_active,
            cost_switch=cost_switch,
            distance_regret=float(distance_regret),
            relative_cost_reduction=float(relative_cost_reduction),
            **buckets,
        )
        if collect_profiles:
            # Audit-only payload: raw curves plus the interpolated curves the
            # distance was actually computed on.
            result.positive_profile = list(trajectories[matched_nearest_positive].student_log_probs)
            result.negative_profile = list(trajectories[selected_negative].student_log_probs)
            result.positive_profile_resampled = [float(value) for value in positive_profiles[nearest_positive_row]]
            result.negative_profile_resampled = [float(value) for value in negative_profiles[selected_column]]
        return result

    @staticmethod
    def _extract_token_logprob_sequence(mask_row: Any, logprob_row: Any) -> list[float]:
        masks = [bool(value) for value in _as_list(mask_row)]
        log_probs = [float(value) for value in _as_list(logprob_row)]
        if len(masks) != len(log_probs):
            raise ValueError("response mask and sampled log-prob lengths differ")
        return [log_probs[position] for position, is_valid in enumerate(masks) if is_valid]

    @staticmethod
    def _resample_to_common_length(profile: Sequence[float], common_length: int) -> torch.Tensor:
        values = torch.as_tensor(list(profile), dtype=torch.float64)
        if values.numel() == 0 or common_length < 1:
            raise ValueError("cannot resample an empty confidence profile")
        if values.numel() == common_length:
            return values
        if values.numel() == 1:
            return values.expand(common_length).clone()
        return F.interpolate(values.view(1, 1, -1), size=common_length, mode="linear", align_corners=True).view(-1)

    @staticmethod
    def _compute_profile_l1_distance(positive_profiles: torch.Tensor, negative_profiles: torch.Tensor) -> torch.Tensor:
        return torch.abs(positive_profiles[:, None, :] - negative_profiles[None, :, :]).mean(dim=-1)

    def _build_trajectories(
        self,
        response_masks: Any,
        student_log_probs: Any,
        size: int,
    ) -> tuple[dict[int, _Trajectory], Optional[str]]:
        try:
            mask_rows = _as_list(response_masks)
            student_rows = _as_list(student_log_probs)
        except (TypeError, ValueError):
            return {}, "token_alignment_failed:unreadable_tensor"
        if not (len(mask_rows) == len(student_rows) == size):
            return {}, "token_alignment_failed:batch_length"

        trajectories: dict[int, _Trajectory] = {}
        for index in range(size):
            try:
                sequence = self._extract_token_logprob_sequence(mask_rows[index], student_rows[index])
            except (TypeError, ValueError):
                continue
            trajectories[index] = _Trajectory(student_log_probs=sequence)
        if len(trajectories) != size:
            return trajectories, "token_alignment_failed:sequence_length"
        return trajectories, None

    def _uniform_fallback(
        self,
        negatives: Sequence[int],
        prompt_uid: str,
        attempt_id: int,
        reason: str,
        positives: Sequence[int] = (),
    ) -> FrontierSelectionResult:
        selected = None
        if negatives:
            seed = _stable_int_hash(f"{self.seed}:{prompt_uid}:{attempt_id}:uniform_incorrect")
            selected = random.Random(seed).choice(list(negatives))
        return FrontierSelectionResult(
            selected_negative_idx=selected,
            selector_type="uniform_incorrect",
            selection_mode="uniform_fallback",
            fallback_used=True,
            fallback_reason=reason,
            positive_indices=tuple(positives),
            negative_indices=tuple(negatives),
        )
