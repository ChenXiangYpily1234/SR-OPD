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

"""Boundary-state utilities for Frontier-First OPD rollout selection."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

BOUNDARY_SELECTOR_MODES = (
    "boundary_opd",
    "bc_0p4n_random",
)

_BOUNDARY_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_BOUNDARY_CAPTURE_METHODS = {
    "auto",
    "final_norm_hook",
    "lm_head_pre_hook",
}
BOUNDARY_SCORE_MODES = {
    "legacy",
    "safr_n1",
    "persistent_departure_area",
    "persistent_departure_area_min",
    "dtw_persistent_departure_area",
    "dtw_persistent_departure_area_min",
    "gap_regularized_dtw_persistent_departure_area",
    "success_manifold_pda",
}
BOUNDARY_SIMILARITY_METRICS = {
    "raw_hidden_cosine",
    "centered_hidden_cosine",
    "centered_hidden_state_cosine",
    "whitened_hidden_cosine",
    "lm_head_cosine",
    "centered_lm_head_cosine",
    "whitened_lm_head_cosine",
    "hidden_span_wasserstein",
    "centered_lm_head_span_wasserstein",
}
BOUNDARY_CALIBRATED_HIDDEN_METRICS = {
    "centered_hidden_cosine",
    "whitened_hidden_cosine",
}
BOUNDARY_CALIBRATED_DIRECT_STATE_METRICS = {"centered_hidden_state_cosine"}
# Metrics that require a frozen calibration artifact.  This is a superset of
# BOUNDARY_CALIBRATED_HIDDEN_METRICS: ``whitened_lm_head_cosine`` consumes the
# artifact on the worker (before the LM-head projection) while its pairwise
# cosine still travels through the worker-pairwise selector path, so it must
# not join the hidden-metric set that drives selector-side transition builds.
BOUNDARY_CALIBRATED_METRICS = BOUNDARY_CALIBRATED_HIDDEN_METRICS | {
    "whitened_lm_head_cosine",
}
BOUNDARY_CALIBRATION_ARTIFACT_METRICS = BOUNDARY_CALIBRATED_METRICS | BOUNDARY_CALIBRATED_DIRECT_STATE_METRICS
BOUNDARY_SPAN_METRICS = {
    "hidden_span_wasserstein",
    "centered_lm_head_span_wasserstein",
}

L8_WINDOW_RATIO = 1.0 / 16.0
L8_LENGTH_NORMALIZATION_EPSILON = 1.0e-6
L8_L13_0P4N_HT_SCORE_MODES = {"l8_adaptive_early_fork_ht_l13_0p4n"}
L8_HT_SCORE_MODES = {"l8_adaptive_early_fork_ht"} | L8_L13_0P4N_HT_SCORE_MODES
L8_SCORE_MODES = {
    "l8_adaptive_early_fork",
    *L8_HT_SCORE_MODES,
    "l8_adaptive_early_fork_no_alignment",
    "l8_adaptive_early_fork_no_drop",
    "l8_adaptive_early_fork_no_divergence",
    "l8_adaptive_early_fork_no_length_efficiency",
    "l8_adaptive_peak_fork_length",
    "l8_adaptive_peak_fork_only",
    # L23 keeps the exact L8 rollout selection (scoring, tie-breaks, routing)
    # and only changes the downstream OPD loss: the already-located fork
    # position is exported so the Actor supervises the post-fork span only.
    "l8_adaptive_early_fork_post_fork_span",
}
L9_SCORE_MODES = {"l9_bounded_cost_adaptive_early_fork"}
L10_SCORE_MODES = {"l10_cost_ratio_adaptive_early_fork"}
L11_SCORE_MODES = {"persistent_drop"}
L12_SCORE_MODES = {"interior_nearest_failure"}
L13_SCORE_MODES = {"most_divergent_failure"}
L14_SCORE_MODES = {"l14_peer_calibrated_early_fork", "l14_peer_calibrated_early_fork_h1"}
N1_SCORE_MODES = {"safr_n1"}
# L15 keeps the exact L8 per-prompt utility/cost scoring (it lives in
# ADAPTIVE_EARLY_FORK_SCORE_MODES so compute_boundary_contrast_scores_from_cosine
# takes the L8 branch) and only changes the batch-global Teacher-query
# allocation: acquisition score S_L15 = G_x * U_BC * C_i with the frontier
# weight G_x = 4 p_x (1 - p_x), then a global Top-B pick under a fixed
# Teacher-query budget B = floor(r * N_prompt * K) defined against Full OPD
# (one query per rollout), with at most two queries per prompt.
L15_SCORE_MODES = {"l15_frontier_budgeted_early_fork"}
L15_MAX_QUERIES_PER_PROMPT = 2
# Fixed Teacher-query ratio against the Full-OPD query count N_prompt * K.
# With the training configuration N=64, K=4 the budget is floor(0.1*256)=25
# queries per step (<= 9.77% of Full OPD); floor keeps the realized ratio
# <= r for every batch shape.  This is a fixed design constant, not a knob.
L15_TEACHER_QUERY_RATIO = 0.10
L16_SCORE_MODES = {"l16_frontier_expected_consensus"}
L16_TEACHER_QUERY_RATIO = 0.10
L17_SCORE_MODES = {"l17_oci_opd"}
L17_TEACHER_QUERY_RATIO = 0.10
L17_LENGTH_EPSILON = 1.0e-6
L18_SCORE_MODES = {"l18_success_conditioned_early_change"}
L18_TEACHER_QUERY_RATIO = 0.10
L18_LENGTH_EPSILON = 1.0e-6
# L19 = Centered-LM-head Latent Path Efficiency x Normalized Length Cost.
# It is a Student-only, trajectory-intrinsic selector: it never compares a
# wrong rollout against its positive siblings and never builds fork/drop
# features.  Over the first 50% of the response it captures M=16 latent states,
# maps each to the centered LM-head space z = C W_U h, and scores every wrong
# rollout by its Latent Path Efficiency E = ||z_M - z_1|| / (sum ||z_{m+1}-z_m||)
# times the normalized Teacher-cost efficiency (1 - L_resp / L_max).  Positive
# siblings are used only to decide whether a prompt sits on the Student
# competence frontier (mixed 1P3N/2P2N/3P1N group).
L19_SCORE_MODES = {"l19_cost_aware_latent_path_efficiency"}
L19_TEACHER_QUERY_RATIO = 0.10
L19_LENGTH_EPSILON = 1.0e-6
# Epsilon guarding the Latent Path Efficiency ratio (spec section 11).
L19_PATH_EPSILON = 1.0e-8
# Fixed number of centered LM-head states over the first 50% of the response.
L19_NUM_STATES = 16
L20_SCORE_MODES = {"l20_trajectory_departure"}
L21_SCORE_MODES = {"l21_entropy_weighted_trajectory_departure"}
L26_SCORE_MODES = {"l26_uncertainty_weighted_trajectory_departure"}
L27_SCORE_MODES = {"l27_cost_aware_uncertainty_weighted_trajectory_departure"}
L28_SCORE_MODES = {"l28_entropy_weighted_mid_drop_trajectory_departure"}
L29_SCORE_MODES = {"l29_entropy_weighted_potential_departure"}
L30_SCORE_MODES = {"l30_entropy_weighted_ratio_cost_trajectory_departure"}
L31_SCORE_MODES = {"l31_student_only_positive_sibling_routing"}
PDA_SCORE_MODES = {
    "persistent_departure_area",
    "persistent_departure_area_min",
    "dtw_persistent_departure_area",
    "dtw_persistent_departure_area_min",
    "gap_regularized_dtw_persistent_departure_area",
}
GAP_REGULARIZED_DTW_PDA_SCORE_MODES = {"gap_regularized_dtw_persistent_departure_area"}
DTW_PDA_SCORE_MODES = {
    "dtw_persistent_departure_area",
    "dtw_persistent_departure_area_min",
    *GAP_REGULARIZED_DTW_PDA_SCORE_MODES,
}
MIN_POSITIVE_PDA_SCORE_MODES = {"persistent_departure_area_min", "dtw_persistent_departure_area_min"}
SM_PDA_SCORE_MODES = {"success_manifold_pda"}
SM_PDA_NUM_SPANS = 16
SM_PDA_MAX_POINTS_PER_SPAN = 32
UWTD_SCORE_MODES = L26_SCORE_MODES | L27_SCORE_MODES
TRAJECTORY_DEPARTURE_SCORE_MODES = (
    L20_SCORE_MODES
    | L21_SCORE_MODES
    | UWTD_SCORE_MODES
    | L28_SCORE_MODES
    | L29_SCORE_MODES
    | L30_SCORE_MODES
)
UWTD_EPSILON = 1.0e-6
# L23 = Post-Fork Span Supervised OPD.  Rollout selection is byte-identical
# to L8 (the score mode lives in L8_SCORE_MODES so every L8 branch applies);
# the only change is loss-side: the L8 fork location is converted to a token
# position and the Actor's sampled-token reverse-KL is restricted to the
# post-fork span through the existing TA-OPD selected-mask channel.
L23_SCORE_MODES = {"l8_adaptive_early_fork_post_fork_span"}
# Buffer (in response tokens) subtracted from the fork position when opening
# the supervised span, absorbing the state-quantization error of the 16-state
# early grid (each state spans early_end/15 tokens of the first L/8 window).
L23_PRE_FORK_BUFFER_TOKENS = 32
# Responses whose post-fork span would be shorter than this fall back to
# full-sequence supervision (the fork carries no usable span information).
L23_MIN_SPAN_TOKENS = 64
SASB_SCORE_MODES = {
    "sasb_shortest_only",
    "sasb_near_no_cost",
    "sasb_band_no_cost",
    "sasb_band_shortest_cost",
    "sasb_shortest_anchored_success_boundary",
}
SASB_MAIN_SCORE_MODE = "sasb_shortest_anchored_success_boundary"
SASB_TOP_K = 32
SASB_TAU_S = 0.05
SASB_EMA_BETA = 0.1
SASB_ETA = 0.25
SASB_MEDOID_WEIGHT = 0.15
SASB_COST_NORMALIZER = 8192.0
SASB_EPSILON = 1.0e-8
WHOLE_RESPONSE_STATE_SCORE_MODES = L11_SCORE_MODES | L12_SCORE_MODES | L13_SCORE_MODES | L31_SCORE_MODES
# L22 = Token-Budgeted Knapsack Allocation.  Per-candidate scoring is frozen
# to L15 (S_L22 = G_x * U_BC * C_i, identical fields); only the allocator
# changes: the budget unit becomes Teacher input tokens (rho of the Full-OPD
# token total of the step) and the greedy order becomes unit-token utility
# S_c / teacher_cost_c, i.e. a cost-aware knapsack instead of a query-count
# Top-B.
L22_SCORE_MODES = {"l22_token_budgeted_knapsack"}
L22_MAX_QUERIES_PER_PROMPT = 2
L22_TEACHER_TOKEN_RATIO = 0.10
ADAPTIVE_EARLY_FORK_SCORE_MODES = (
    L8_SCORE_MODES
    | L9_SCORE_MODES
    | L10_SCORE_MODES
    | L14_SCORE_MODES
    | L15_SCORE_MODES
    | L16_SCORE_MODES
    | L22_SCORE_MODES
    | TRAJECTORY_DEPARTURE_SCORE_MODES
)


@dataclass(frozen=True)
class BoundaryOPDSettings:
    """Validated runtime settings for the Boundary-OPD selector."""

    num_boundaries: int = 16
    hidden_layer: int = -1
    capture_location: str = "pre_lm_head"
    capture_method: str = "auto"
    hidden_storage_dtype: str = "float16"
    distance_compute_dtype: str = "float32"
    hidden_norm_epsilon: float = 1.0e-6
    similarity_metric: str = "raw_hidden_cosine"
    calibration_artifact_path: str = ""
    calibration_artifact_sha256: str = ""
    calibration_model_hash: str = ""
    calibration_data_hash: str = ""
    calibration_collect: bool = False
    calibration_num_prompts: int = 64
    whitening_regularization: float = 1.0e-4
    lm_head_vocab_chunk_size: int = 2048
    sinkhorn_epsilon: float = 0.05
    sinkhorn_max_iterations: int = 2000
    sinkhorn_tolerance: float = 1.0e-5
    dtw_gap_penalty: float = 0.0
    score_mode: str = "legacy"
    window_ratio: float = 0.125
    detach_hidden: bool = True
    fallback_to_ff_cost: bool = False
    debug_store_pairwise: bool = False
    representation_dynamics_m64: bool = False
    sm_pda_compute_original_pda_diagnostics: bool = True

    @classmethod
    def from_mapping(cls, values: Any) -> BoundaryOPDSettings:
        if values is None:
            settings = cls()
        else:
            resolved = {
                name: values.get(name, config_field.default) for name, config_field in cls.__dataclass_fields__.items()
            }
            settings = cls(**resolved)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.num_boundaries not in {2, 4, 8, 16, 32, 64}:
            raise ValueError("algorithm.boundary_opd.num_boundaries must be one of {2, 4, 8, 16, 32, 64}")
        if self.hidden_layer != -1:
            raise ValueError("Boundary-OPD currently captures only hidden_layer=-1")
        if self.capture_location != "pre_lm_head":
            raise ValueError("Boundary-OPD requires capture_location=pre_lm_head")
        if self.capture_method not in _BOUNDARY_CAPTURE_METHODS:
            raise ValueError(f"algorithm.boundary_opd.capture_method must be one of {_BOUNDARY_CAPTURE_METHODS}")
        if self.hidden_storage_dtype not in _BOUNDARY_STORAGE_DTYPES:
            raise ValueError("algorithm.boundary_opd.hidden_storage_dtype must be float16 or bfloat16")
        if self.distance_compute_dtype != "float32":
            raise ValueError("Boundary-OPD requires distance_compute_dtype=float32")
        if self.hidden_norm_epsilon <= 0:
            raise ValueError("Boundary-OPD hidden_norm_epsilon must be positive")
        if self.similarity_metric not in BOUNDARY_SIMILARITY_METRICS:
            raise ValueError(
                f"algorithm.boundary_opd.similarity_metric must be one of {sorted(BOUNDARY_SIMILARITY_METRICS)}"
            )
        if self.calibration_collect:
            if self.similarity_metric != "raw_hidden_cosine":
                raise ValueError("Boundary calibration collection requires similarity_metric=raw_hidden_cosine")
            if not self.calibration_artifact_path:
                raise ValueError("Boundary calibration collection requires calibration_artifact_path")
            if self.calibration_num_prompts <= 0:
                raise ValueError("Boundary calibration_num_prompts must be positive")
            for name, value in (
                ("calibration_model_hash", self.calibration_model_hash),
                ("calibration_data_hash", self.calibration_data_hash),
            ):
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                    raise ValueError(f"Boundary {name} must be a SHA-256 digest during calibration")
        elif self.similarity_metric in BOUNDARY_CALIBRATION_ARTIFACT_METRICS:
            if not self.calibration_artifact_path:
                raise ValueError(f"{self.similarity_metric} requires calibration_artifact_path")
            if len(self.calibration_artifact_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.calibration_artifact_sha256.lower()
            ):
                raise ValueError(f"{self.similarity_metric} requires calibration_artifact_sha256")
        if self.whitening_regularization <= 0:
            raise ValueError("Boundary whitening_regularization must be positive")
        if self.lm_head_vocab_chunk_size <= 0:
            raise ValueError("Boundary lm_head_vocab_chunk_size must be positive")
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("Boundary sinkhorn_epsilon must be positive")
        if self.sinkhorn_max_iterations <= 0:
            raise ValueError("Boundary sinkhorn_max_iterations must be positive")
        if self.sinkhorn_tolerance <= 0:
            raise ValueError("Boundary sinkhorn_tolerance must be positive")
        if not math.isfinite(self.dtw_gap_penalty) or self.dtw_gap_penalty < 0:
            raise ValueError("Boundary dtw_gap_penalty must be finite and non-negative")
        if self.score_mode not in BOUNDARY_SCORE_MODES:
            raise ValueError(f"algorithm.boundary_opd.score_mode must be one of {sorted(BOUNDARY_SCORE_MODES)}")
        if self.score_mode in GAP_REGULARIZED_DTW_PDA_SCORE_MODES:
            if self.dtw_gap_penalty <= 0:
                raise ValueError(f"{self.score_mode} requires dtw_gap_penalty > 0")
        elif self.dtw_gap_penalty != 0:
            raise ValueError("dtw_gap_penalty must be 0 outside gap-regularized DTW-PDA")
        if not math.isfinite(self.window_ratio) or not 0.0 < self.window_ratio < 0.25:
            raise ValueError("algorithm.boundary_opd.window_ratio must satisfy 0 < window_ratio < 0.25")
        if self.score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
            if self.similarity_metric != "centered_lm_head_cosine":
                raise ValueError(f"{self.score_mode} requires centered_lm_head_cosine")
            if not math.isclose(self.window_ratio, L8_WINDOW_RATIO, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"{self.score_mode} requires window_ratio=0.0625")
        if self.score_mode in L8_L13_0P4N_HT_SCORE_MODES:
            if self.num_boundaries != 16:
                raise ValueError(f"{self.score_mode} requires num_boundaries=16")
            if self.representation_dynamics_m64:
                raise ValueError(f"{self.score_mode} does not support representation_dynamics_m64")
        if self.score_mode in TRAJECTORY_DEPARTURE_SCORE_MODES:
            if self.num_boundaries != 16:
                raise ValueError(f"{self.score_mode} requires num_boundaries=16")
            if self.representation_dynamics_m64:
                raise ValueError(f"{self.score_mode} does not support representation_dynamics_m64")
        if self.representation_dynamics_m64:
            if self.score_mode not in L8_SCORE_MODES:
                raise ValueError("representation_dynamics_m64 requires an L8 score mode")
            if self.num_boundaries != 64:
                raise ValueError("representation_dynamics_m64 requires num_boundaries=64")
            if self.similarity_metric != "centered_lm_head_cosine":
                raise ValueError("representation_dynamics_m64 requires centered_lm_head_cosine")
        if self.score_mode in WHOLE_RESPONSE_STATE_SCORE_MODES - L31_SCORE_MODES:
            if self.num_boundaries != 16:
                raise ValueError(f"{self.score_mode} requires num_boundaries=16")
            if self.similarity_metric != "centered_lm_head_cosine":
                raise ValueError(f"{self.score_mode} requires centered_lm_head_cosine")
        if self.score_mode in N1_SCORE_MODES:
            if self.num_boundaries != 16:
                raise ValueError("safr_n1 requires num_boundaries=16")
            if self.similarity_metric != "raw_hidden_cosine":
                raise ValueError("safr_n1 requires raw_hidden_cosine (prompt-wise centering is applied by the router)")
        if self.score_mode in SASB_SCORE_MODES:
            if self.num_boundaries != 16:
                raise ValueError("SASB requires num_boundaries=16")
            if not math.isclose(self.window_ratio, L8_WINDOW_RATIO, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("SASB requires window_ratio=0.0625")
        if self.score_mode in PDA_SCORE_MODES | SM_PDA_SCORE_MODES:
            if self.num_boundaries not in {8, 16, 32, 64}:
                raise ValueError("PDA score modes require num_boundaries in {8, 16, 32, 64}")
            if not self.calibration_collect and self.similarity_metric != "centered_hidden_state_cosine":
                raise ValueError("PDA score modes require centered_hidden_state_cosine")
        if self.score_mode in SM_PDA_SCORE_MODES:
            if self.num_boundaries != SM_PDA_NUM_SPANS:
                raise ValueError("success_manifold_pda requires num_boundaries=16")
            if not math.isclose(self.sinkhorn_epsilon, 4.5, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("success_manifold_pda requires sinkhorn_epsilon=4.5")
            if self.sinkhorn_max_iterations != 50:
                raise ValueError("success_manifold_pda requires sinkhorn_max_iterations=50")
            if not math.isclose(self.sinkhorn_tolerance, 1.0e-3, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("success_manifold_pda requires sinkhorn_tolerance=1.0e-3")
        if not self.detach_hidden:
            raise ValueError("Boundary-OPD requires detach_hidden=true")
        if self.fallback_to_ff_cost:
            raise ValueError("Boundary-Contrast OPD requires fallback_to_ff_cost=false")

    @property
    def storage_dtype(self) -> torch.dtype:
        return _BOUNDARY_STORAGE_DTYPES[self.hidden_storage_dtype]


def is_boundary_selector_mode(mode: str) -> bool:
    return mode in BOUNDARY_SELECTOR_MODES


def boundary_score_mode_requires_hidden_capture(score_mode: str) -> bool:
    """Whether a Boundary score mode requires selector-only Actor states."""

    return score_mode not in L17_SCORE_MODES | SASB_SCORE_MODES


def boundary_score_mode_requires_span_capture(score_mode: str) -> bool:
    """Whether the selector needs the detached full response hidden tensor."""

    return score_mode in SM_PDA_SCORE_MODES


def boundary_score_mode_requires_policy_capture(score_mode: str) -> bool:
    return score_mode in SASB_SCORE_MODES


def boundary_transition_count(settings: BoundaryOPDSettings) -> int:
    if settings.score_mode in PDA_SCORE_MODES | SM_PDA_SCORE_MODES:
        return settings.num_boundaries
    if settings.representation_dynamics_m64:
        return 2 * settings.num_boundaries
    if settings.score_mode in L8_L13_0P4N_HT_SCORE_MODES:
        # 15 L8 early-state deltas followed by 16 L13 whole-response states.
        return 31
    if settings.score_mode in WHOLE_RESPONSE_STATE_SCORE_MODES:
        return 16
    if settings.score_mode in N1_SCORE_MODES:
        return settings.num_boundaries
    if settings.score_mode in L18_SCORE_MODES:
        return 16
    if settings.score_mode in L19_SCORE_MODES:
        # Per-state validity for the M=16 captured latent states (L18-style;
        # every state carries one validity flag rather than a movement flag).
        return L19_NUM_STATES
    if settings.score_mode in SASB_SCORE_MODES:
        return 16
    if settings.score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        return 15
    return settings.num_boundaries


def boundary_state_count(settings: BoundaryOPDSettings) -> int:
    if settings.score_mode in PDA_SCORE_MODES | SM_PDA_SCORE_MODES:
        return settings.num_boundaries
    if settings.representation_dynamics_m64:
        return 2 * (settings.num_boundaries + 1)
    if settings.score_mode in L8_L13_0P4N_HT_SCORE_MODES:
        return 32
    if settings.score_mode in WHOLE_RESPONSE_STATE_SCORE_MODES:
        return 16
    if settings.score_mode in N1_SCORE_MODES:
        return settings.num_boundaries
    if settings.score_mode in L18_SCORE_MODES:
        return 16
    if settings.score_mode in L19_SCORE_MODES:
        return L19_NUM_STATES
    if settings.score_mode in SASB_SCORE_MODES:
        return 16
    if settings.score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        return 16
    return settings.num_boundaries + 1


def build_uniform_response_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    num_states: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample direct response-token states at round(j * (L-1) / (M-1))."""

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [B, response_length]")
    if num_states < 2:
        raise ValueError("num_states must be at least two")
    response_mask = response_mask.bool()
    batch_size, response_width = response_mask.shape
    if prompt_mask is None:
        prompt_width = 0
        prompt_valid = torch.ones(batch_size, dtype=torch.bool, device=response_mask.device)
    else:
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
            raise ValueError("prompt_mask must align with response_mask")
        prompt_width = prompt_mask.shape[1]
        prompt_valid = prompt_mask.bool().any(dim=-1)
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    response_valid = lengths > 0
    if response_width == 0:
        return (
            torch.zeros(batch_size, num_states, dtype=torch.long, device=response_mask.device),
            torch.zeros(batch_size, num_states, dtype=torch.bool, device=response_mask.device),
        )
    positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
    ordered = torch.where(response_mask, positions, torch.full_like(positions, response_width)).sort(dim=-1).values
    normalized_positions = torch.linspace(0.0, 1.0, num_states, device=response_mask.device)
    ordinals = torch.round(normalized_positions.unsqueeze(0) * (lengths - 1).clamp_min(0).unsqueeze(1)).long()
    ordinals = torch.minimum(ordinals, (lengths - 1).clamp_min(0).unsqueeze(1))
    local = ordered.gather(1, ordinals).clamp(min=0, max=response_width - 1)
    indices = local + prompt_width
    valid = (prompt_valid & response_valid).unsqueeze(1).expand(-1, num_states)
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    if valid.any() and (indices[valid] < prompt_width).any():
        raise AssertionError("PDA state selection included a prompt or padding token")
    return indices.detach(), valid.detach()


def build_unique_uniform_response_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    num_states: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``min(M, T)`` unique, uniformly spaced response-token states.

    Valid states occupy the leading slots. Remaining fixed-width payload slots
    repeat the final valid index but are masked out; DTW therefore sees only
    the unique trajectory observations and may align sequences of unequal
    lengths without changing the Actor-worker payload shape.
    """

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [B, response_length]")
    if num_states < 2:
        raise ValueError("num_states must be at least two")
    response_mask = response_mask.bool()
    batch_size, response_width = response_mask.shape
    if prompt_mask is None:
        prompt_width = 0
        prompt_valid = torch.ones(batch_size, dtype=torch.bool, device=response_mask.device)
    else:
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
            raise ValueError("prompt_mask must align with response_mask")
        prompt_width = prompt_mask.shape[1]
        prompt_valid = prompt_mask.bool().any(dim=-1)
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    indices = torch.zeros(batch_size, num_states, dtype=torch.long, device=response_mask.device)
    valid = torch.zeros(batch_size, num_states, dtype=torch.bool, device=response_mask.device)
    if response_width == 0:
        return indices.detach(), valid.detach()
    positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
    ordered = torch.where(response_mask, positions, torch.full_like(positions, response_width)).sort(dim=-1).values
    for row in range(batch_size):
        length = int(lengths[row].item())
        if not bool(prompt_valid[row]) or length == 0:
            continue
        count = min(num_states, length)
        if count == 1:
            ordinals = torch.zeros(1, dtype=torch.long, device=response_mask.device)
        else:
            ordinals = torch.round(
                torch.linspace(0.0, float(length - 1), count, device=response_mask.device)
            ).long()
        if int(torch.unique_consecutive(ordinals).numel()) != count:
            raise AssertionError("DTW-PDA unique state sampling produced a duplicate ordinal")
        local = ordered[row].index_select(0, ordinals)
        indices[row, :count] = local + prompt_width
        indices[row, count:] = local[-1] + prompt_width
        valid[row, :count] = True
    if valid.any() and (indices[valid] < prompt_width).any():
        raise AssertionError("DTW-PDA state selection included a prompt or padding token")
    return indices.detach(), valid.detach()


def build_boundary_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    num_boundaries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build absolute sequence positions for the prompt end and response boundaries.

    Returns:
        state_indices: ``[B, M + 1]``. Column zero is the last valid prompt
            token and the remaining columns are response boundary tokens.
        transition_valid_mask: ``[B, M]``. Repeated response boundaries and
            rows with an empty prompt or response are invalid transitions.
    """

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [B, response_length]")
    if num_boundaries < 1:
        raise ValueError("num_boundaries must be positive")

    response_mask = response_mask.bool()
    batch_size, response_width = response_mask.shape
    if prompt_mask is None:
        prompt_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=response_mask.device,
        )
    if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
        raise ValueError("prompt_mask must have shape [B, prompt_width]")
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    prompt_width = prompt_mask.shape[1]
    sequence_width = prompt_width + response_width
    if sequence_width < 1:
        raise ValueError("prompt and response widths cannot both be zero")

    prompt_positions = torch.arange(
        prompt_width,
        device=response_mask.device,
        dtype=torch.long,
    ).expand(batch_size, -1)
    prompt_valid = prompt_mask.any(dim=-1)
    if prompt_width:
        prompt_last = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values
    else:
        prompt_last = torch.full(
            (batch_size,),
            -1,
            device=response_mask.device,
            dtype=torch.long,
        )
    prompt_last = prompt_last.clamp_min(0)

    response_lengths = response_mask.sum(dim=-1, dtype=torch.long)
    response_valid = response_lengths > 0
    if response_width:
        response_positions = torch.arange(
            response_width,
            device=response_mask.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        # Sorting moves the sentinel after every real response position while
        # retaining the original response-token order.
        ordered_response_positions = (
            torch.where(
                response_mask,
                response_positions,
                torch.full_like(response_positions, response_width),
            )
            .sort(dim=-1)
            .values
        )
    else:
        ordered_response_positions = torch.zeros(
            batch_size,
            1,
            device=response_mask.device,
            dtype=torch.long,
        )

    boundary_number = torch.arange(
        1,
        num_boundaries + 1,
        device=response_mask.device,
        dtype=torch.long,
    )
    end_pos = (boundary_number.unsqueeze(0) * response_lengths.unsqueeze(1) + num_boundaries - 1) // num_boundaries
    local_ordinals = (end_pos - 1).clamp_min(0)
    if response_width:
        local_ordinals = torch.minimum(
            local_ordinals,
            (response_lengths - 1).clamp_min(0).unsqueeze(1),
        )
        boundary_local = ordered_response_positions.gather(dim=1, index=local_ordinals)
        boundary_local = boundary_local.clamp(min=0, max=response_width - 1)
    else:
        boundary_local = torch.zeros_like(local_ordinals)

    boundary_absolute = boundary_local + prompt_width
    safe_prompt_last = prompt_last.clamp(max=sequence_width - 1)
    boundary_absolute = torch.where(
        (prompt_valid & response_valid).unsqueeze(1),
        boundary_absolute,
        safe_prompt_last.unsqueeze(1),
    ).clamp(min=0, max=sequence_width - 1)
    state_indices = torch.cat([safe_prompt_last.unsqueeze(1), boundary_absolute], dim=1)

    transition_valid_mask = (state_indices[:, 1:] != state_indices[:, :-1]) & (prompt_valid & response_valid).unsqueeze(
        1
    )
    if state_indices.min().item() < 0 or state_indices.max().item() >= sequence_width:
        raise AssertionError("Boundary-OPD produced an out-of-range state index")
    if response_valid.any():
        last_response_positions = ordered_response_positions[
            response_valid,
            response_lengths[response_valid] - 1,
        ]
        expected_last = prompt_width + last_response_positions
        if not torch.equal(state_indices[response_valid, -1], expected_last):
            raise AssertionError("the final boundary is not the final valid response token")
    if (state_indices[:, 2:] < state_indices[:, 1:-1]).any():
        raise AssertionError("response boundary indices must be monotonic")
    return state_indices, transition_valid_mask


def build_reasoning_span_end_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    *,
    min_span_tokens: int = 4,
    max_span_tokens: int = 128,
    max_spans: int = 16,
) -> torch.Tensor:
    """Deterministically segment response tokens into lightweight reasoning spans.

    Natural boundaries are newlines, sentence-final punctuation, and common
    displayed/inline math closers. Very short spans are merged forward; very
    long spans are cut at ``max_span_tokens``. If more than ``max_spans`` are
    produced, the shortest adjacent pair is repeatedly merged (earliest tie).
    The returned boolean mask marks inclusive response-token endpoints.
    """
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must have identical [B, L] shapes")
    if min_span_tokens < 1 or max_span_tokens < min_span_tokens or max_spans < 1:
        raise ValueError("invalid reasoning-span limits")

    endpoints = torch.zeros_like(response_mask, dtype=torch.bool)
    masks = response_mask.detach().bool().cpu()
    token_rows = responses.detach().cpu()
    for row in range(responses.shape[0]):
        positions = masks[row].nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            continue
        spans: list[tuple[int, int]] = []
        start_offset = 0
        for offset, position in enumerate(positions):
            piece = tokenizer.decode([int(token_rows[row, position])], skip_special_tokens=True)
            stripped = piece.rstrip()
            natural = (
                "\n" in piece
                or stripped.endswith((".", "?", "!", ";", "。", "？", "！", "；"))
                or stripped.endswith(("\\]", "\\)", "$$"))
            )
            span_length = offset - start_offset + 1
            is_last = offset == len(positions) - 1
            if is_last or span_length >= max_span_tokens or (natural and span_length >= min_span_tokens):
                spans.append((start_offset, offset))
                start_offset = offset + 1

        if len(spans) > 1 and spans[-1][1] - spans[-1][0] + 1 < min_span_tokens:
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()
        while len(spans) > max_spans:
            merge_at = min(
                range(len(spans) - 1),
                key=lambda index: (spans[index + 1][1] - spans[index][0] + 1, index),
            )
            spans[merge_at] = (spans[merge_at][0], spans[merge_at + 1][1])
            spans.pop(merge_at + 1)
        for _, end_offset in spans:
            endpoints[row, positions[end_offset]] = True
    return endpoints.to(device=response_mask.device)


def build_reasoning_span_state_indices(
    span_end_mask: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    *,
    max_spans: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert response-relative span endpoints to padded absolute indices."""
    if span_end_mask.ndim != 2:
        raise ValueError("span_end_mask must have shape [B, response_length]")
    batch_size, response_width = span_end_mask.shape
    if prompt_mask is None:
        prompt_width = 0
    else:
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
            raise ValueError("prompt_mask must align with span_end_mask")
        prompt_width = prompt_mask.shape[1]
    indices = torch.zeros(batch_size, max_spans, dtype=torch.long, device=span_end_mask.device)
    valid = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=span_end_mask.device)
    for row in range(batch_size):
        relative = span_end_mask[row].nonzero(as_tuple=False).flatten()
        if relative.numel() > max_spans:
            raise ValueError(f"reasoning span count exceeds max_spans={max_spans}")
        count = int(relative.numel())
        if count:
            absolute = relative + prompt_width
            indices[row, :count] = absolute
            indices[row, count:] = absolute[-1]
            valid[row, :count] = True
    if response_width == 0:
        return indices, valid
    return indices, valid


def build_normalized_hidden_transitions(
    boundary_states: torch.Tensor,
    transition_valid_mask: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Build detached FP32 unit transition directions from boundary states."""

    if boundary_states.ndim != 3:
        raise ValueError("boundary_states must have shape [B, M + 1, D]")
    expected = (boundary_states.shape[0], boundary_states.shape[1] - 1)
    if transition_valid_mask.shape != expected:
        raise ValueError("transition_valid_mask must have shape [B, M] aligned with boundary_states")
    if eps <= 0:
        raise ValueError("eps must be positive")

    transitions = boundary_states[:, 1:, :] - boundary_states[:, :-1, :]
    transitions = F.normalize(
        transitions.float(),
        p=2,
        dim=-1,
        eps=eps,
    )
    transitions = transitions.masked_fill(
        ~transition_valid_mask.bool().unsqueeze(-1),
        0.0,
    )
    transitions = transitions.detach()
    if transitions.requires_grad:
        raise AssertionError("Boundary-OPD transitions must be detached")
    if not torch.isfinite(transitions).all():
        raise FloatingPointError("Boundary-OPD transitions contain NaN or Inf")
    return transitions


def build_calibrated_hidden_transitions(
    boundary_states: torch.Tensor,
    transition_valid_mask: torch.Tensor,
    *,
    mean: torch.Tensor,
    cholesky: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Center, optionally whiten, and L2-normalize hidden transitions.

    ``mean`` and ``cholesky`` are frozen calibration statistics. Whitening is
    implemented as the triangular solve ``L^-1 (delta - mean)``; no inverse is
    formed or retained.
    """

    if boundary_states.ndim != 3:
        raise ValueError("boundary_states must have shape [B, M + 1, D]")
    expected = (boundary_states.shape[0], boundary_states.shape[1] - 1)
    if transition_valid_mask.shape != expected:
        raise ValueError("transition_valid_mask must have shape [B, M] aligned with boundary_states")
    hidden_dim = boundary_states.shape[-1]
    if mean.shape != (hidden_dim,):
        raise ValueError(f"calibration mean must have shape {(hidden_dim,)}")
    centered = boundary_states[:, 1:, :].detach().float() - boundary_states[:, :-1, :].detach().float()
    centered = centered - mean.detach().to(device=centered.device, dtype=torch.float32)
    if cholesky is not None:
        if cholesky.shape != (hidden_dim, hidden_dim):
            raise ValueError(f"calibration Cholesky factor must have shape {(hidden_dim, hidden_dim)}")
        factor = cholesky.detach().to(device=centered.device, dtype=torch.float32)
        flat = centered.reshape(-1, hidden_dim)
        centered = torch.linalg.solve_triangular(factor, flat.T, upper=False).T.reshape_as(centered)
    transformed = F.normalize(centered, p=2, dim=-1, eps=eps)
    transformed = transformed.masked_fill(~transition_valid_mask.bool().unsqueeze(-1), 0.0).detach()
    if transformed.requires_grad:
        raise AssertionError("calibrated Boundary transitions must be detached")
    if not torch.isfinite(transformed).all():
        raise FloatingPointError("calibrated Boundary transitions contain NaN or Inf")
    return transformed


def build_centered_hidden_states(
    boundary_states: torch.Tensor,
    state_valid_mask: torch.Tensor,
    *,
    mean: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Center and normalize direct hidden states without forming differences."""

    if boundary_states.ndim != 3:
        raise ValueError("boundary_states must have shape [B, M, D]")
    if state_valid_mask.shape != boundary_states.shape[:2]:
        raise ValueError("state_valid_mask must have shape [B, M]")
    if mean.ndim != 1 or mean.shape[0] != boundary_states.shape[-1]:
        raise ValueError("mean must align with the hidden dimension")
    centered = boundary_states.detach().float() - mean.to(boundary_states.device, torch.float32)
    centered = F.normalize(centered, p=2, dim=-1, eps=eps)
    centered = centered.masked_fill(~state_valid_mask.bool().unsqueeze(-1), 0.0).detach()
    if not torch.isfinite(centered).all():
        raise FloatingPointError("centered direct hidden states contain NaN or Inf")
    return centered


def normalized_cosine_similarity(cosine: torch.Tensor) -> torch.Tensor:
    """Map finite cosine values from [-1, 1] to [0, 1]."""

    cosine = cosine.detach().float()
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("PDA cosine contains NaN or Inf")
    if ((cosine < -1.000001) | (cosine > 1.000001)).any():
        raise ValueError("PDA cosine lies outside [-1, 1]")
    return (1.0 + cosine.clamp(-1.0, 1.0)) * 0.5


def persistent_departure_area(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return running maximum, departure, and trapezoidal area on [0, 1]."""

    similarity = similarity.detach().float()
    if similarity.ndim < 1 or similarity.shape[-1] < 2:
        raise ValueError("PDA similarity requires at least two states")
    if not torch.isfinite(similarity).all():
        raise FloatingPointError("PDA similarity contains NaN or Inf")
    if ((similarity < -1.0e-6) | (similarity > 1.0 + 1.0e-6)).any():
        raise ValueError("PDA similarity lies outside [0, 1]")
    running_max = torch.cummax(similarity, dim=-1).values
    departure = running_max - similarity
    if (departure < -1.0e-6).any():
        raise FloatingPointError("PDA departure became negative")
    area = 0.5 * (departure[..., :-1] + departure[..., 1:]).mean(dim=-1)
    if ((area < -1.0e-6) | (area > 1.0 + 1.0e-6)).any():
        raise FloatingPointError("PDA area lies outside [0, 1]")
    return running_max.detach(), departure.detach(), area.detach()


def compute_persistent_departure_area_scores_from_cosine(
    cosine: torch.Tensor,
    prompt_length: int | float | torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    *,
    positive_aggregation: str = "max",
) -> dict[str, torch.Tensor | str | bool]:
    """Score wrong rollouts by aggregated-positive PDA times C_min/C_n."""

    if cosine.ndim != 3 or cosine.shape[-1] < 2:
        raise ValueError("cosine must have shape [N, P, M] with M >= 2")
    num_negative, num_positive, _ = cosine.shape
    if num_negative < 1 or num_positive < 1:
        raise ValueError("PDA requires at least one negative and one positive sibling")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("PDA candidate metadata must align with negative rollouts")
    if positive_aggregation not in {"max", "min"}:
        raise ValueError("positive_aggregation must be 'max' or 'min'")
    similarity = normalized_cosine_similarity(cosine)
    running_max, departure, pair_area = persistent_departure_area(similarity)
    if positive_aggregation == "min":
        area, best_positive_local_index = pair_area.min(dim=1)
    else:
        area, best_positive_local_index = pair_area.max(dim=1)
    candidate_index = torch.arange(num_negative, device=cosine.device)
    best_similarity = similarity[candidate_index, best_positive_local_index]
    best_raw_cosine = cosine.detach().float()[candidate_index, best_positive_local_index]
    best_running_max = running_max[candidate_index, best_positive_local_index]
    best_departure = departure[candidate_index, best_positive_local_index]
    teacher_cost = negative_response_lengths.detach().to(cosine.device, torch.float32) + torch.as_tensor(
        prompt_length, device=cosine.device, dtype=torch.float32
    )
    if teacher_cost.ndim != 1 or (teacher_cost <= 0).any() or not torch.isfinite(teacher_cost).all():
        raise ValueError("PDA Teacher costs must be finite and positive")
    cost_ratio = teacher_cost.min() / teacher_cost
    final_score = area * cost_ratio
    zero_area = bool(torch.isclose(area, torch.zeros_like(area), rtol=1.0e-6, atol=1.0e-8).all())
    rollout_ids = negative_rollout_ids.to(cosine.device)
    if zero_area:
        order = sorted(range(num_negative), key=lambda i: (float(teacher_cost[i]), int(rollout_ids[i])))
    else:
        order = sorted(
            range(num_negative),
            key=lambda i: (-float(final_score[i]), -float(area[i]), float(teacher_cost[i]), int(rollout_ids[i])),
        )
    selected = order[0]
    peak_value, peak_index = best_departure.max(dim=-1)
    zeros = torch.zeros_like(area)
    minus_ones = torch.full((num_negative,), -1, dtype=torch.long, device=cosine.device)
    return {
        "boundary_utility": area,
        "pda_area": area,
        "pair_area": pair_area,
        "cost_ratio": cost_ratio,
        "cost_efficiency": cost_ratio,
        "teacher_cost": teacher_cost,
        "teacher_input_len": teacher_cost,
        "final_score": final_score,
        "best_positive_local_index": best_positive_local_index,
        "best_stage_similarity": best_similarity,
        "best_raw_cosine": best_raw_cosine,
        "best_running_max": best_running_max,
        "best_departure": best_departure,
        "max_departure": peak_value,
        "peak_departure_index": peak_index,
        "mean_departure": best_departure.mean(dim=-1),
        "selected_local_index": torch.tensor(selected, device=cosine.device),
        "selected_rollout_id": rollout_ids[selected],
        "boundary_degenerate_to_cost_only": torch.tensor(zero_area, device=cosine.device),
        "cost_only_local_index": torch.argmin(teacher_cost),
        "valid_candidate_mask": torch.ones(num_negative, dtype=torch.bool, device=cosine.device),
        "best_stage_valid_mask": torch.ones_like(best_similarity, dtype=torch.bool),
        "best_split_index": minus_ones,
        "best_pre_similarity": best_similarity[:, 0],
        "best_post_similarity": best_similarity[:, -1],
        "best_fork_drop": peak_value,
        "best_post_divergence": 1.0 - best_similarity[:, -1],
        "early_cosine": best_similarity[:, 0],
        "middle1_cosine": best_similarity[:, -1],
        "middle2_cosine": zeros,
        "aggregated_middle_cosine": best_similarity.mean(dim=-1),
        "early_alignment": zeros,
        "plateau_drop": peak_value,
        "plateau_divergence": 1.0 - best_similarity[:, -1],
        "pair_utility": area,
        "first_span_cosine": best_similarity[:, 0],
        "second_span_cosine": best_similarity[:, -1],
        "group_min_response_len": negative_response_lengths.min(),
        "group_min_teacher_input_len": teacher_cost.min(),
        "normalized_response_length": zeros,
        "length_efficiency": cost_ratio,
        "rollout_count": torch.tensor(num_negative + num_positive, device=cosine.device),
        "valid_split_count": torch.full((num_negative,), cosine.shape[-1] - 1, device=cosine.device),
        "boundary_search_enabled": False,
        "uses_hidden_difference": False,
        "representation_domain": "direct_hidden_state",
        "positive_aggregation": positive_aggregation,
    }


def dtw_align_symmetric(
    cosine_matrix: torch.Tensor,
    *,
    gap_penalty: float = 0.0,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Return deterministic symmetric-weight DTW alignment and normalized cost.

    Diagonal moves consume two observations and therefore carry ``2D``;
    vertical and horizontal moves carry ``D + gap_penalty``, where
    ``D = 1 - cosine``. A positive penalty makes repeated-state warping earn
    enough cosine improvement to pay for every non-diagonal move.
    Exact predecessor ties prefer diagonal, then vertical, then horizontal.
    """

    cosine_matrix = cosine_matrix.detach().float()
    if cosine_matrix.ndim != 2 or min(cosine_matrix.shape) < 1:
        raise ValueError("DTW cosine_matrix must have non-empty shape [M_n, M_p]")
    if not torch.isfinite(cosine_matrix).all():
        raise FloatingPointError("DTW cosine matrix contains NaN or Inf")
    if ((cosine_matrix < -1.000001) | (cosine_matrix > 1.000001)).any():
        raise ValueError("DTW cosine matrix lies outside [-1, 1]")
    if not math.isfinite(gap_penalty) or gap_penalty < 0:
        raise ValueError("DTW gap_penalty must be finite and non-negative")
    distance = (1.0 - cosine_matrix.clamp(-1.0, 1.0)).cpu()
    rows, columns = distance.shape
    accumulated = torch.full((rows, columns), float("inf"), dtype=torch.float64)
    predecessor = torch.full((rows, columns), -1, dtype=torch.int8)
    accumulated[0, 0] = 2.0 * float(distance[0, 0])
    for row in range(1, rows):
        accumulated[row, 0] = accumulated[row - 1, 0] + float(distance[row, 0]) + gap_penalty
        predecessor[row, 0] = 1  # vertical
    for column in range(1, columns):
        accumulated[0, column] = accumulated[0, column - 1] + float(distance[0, column]) + gap_penalty
        predecessor[0, column] = 2  # horizontal
    for row in range(1, rows):
        for column in range(1, columns):
            local = float(distance[row, column])
            choices = (
                (float(accumulated[row - 1, column - 1]) + 2.0 * local, 0),
                (float(accumulated[row - 1, column]) + local + gap_penalty, 1),
                (float(accumulated[row, column - 1]) + local + gap_penalty, 2),
            )
            value, move = min(choices, key=lambda item: (item[0], item[1]))
            accumulated[row, column] = value
            predecessor[row, column] = move
    path: list[tuple[int, int]] = []
    row, column = rows - 1, columns - 1
    while True:
        path.append((row, column))
        if row == 0 and column == 0:
            break
        move = int(predecessor[row, column])
        if move == 0:
            row, column = row - 1, column - 1
        elif move == 1:
            row -= 1
        elif move == 2:
            column -= 1
        else:
            raise RuntimeError("DTW predecessor chain is incomplete")
    path.reverse()
    normalized_cost = accumulated[-1, -1].to(cosine_matrix.device, torch.float32) / float(rows + columns)
    return path, normalized_cost.detach()


def build_dtw_aligned_similarity_curve(
    cosine_matrix: torch.Tensor,
    path: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Mean all DTW-matched positive similarities for each wrong state."""

    if cosine_matrix.ndim != 2 or min(cosine_matrix.shape) < 1:
        raise ValueError("DTW cosine_matrix must have non-empty shape [M_n, M_p]")
    matches: list[list[int]] = [[] for _ in range(cosine_matrix.shape[0])]
    for wrong_index, positive_index in path:
        if not (0 <= wrong_index < cosine_matrix.shape[0] and 0 <= positive_index < cosine_matrix.shape[1]):
            raise ValueError("DTW path contains an out-of-range coordinate")
        matches[wrong_index].append(positive_index)
    if any(not columns for columns in matches):
        raise ValueError("DTW path does not cover every wrong state")
    return torch.stack(
        [cosine_matrix[row, columns].mean() for row, columns in enumerate(matches)]
    ).detach()


def _dtw_path_statistics(
    path: Sequence[tuple[int, int]],
    negative_count: int,
    positive_count: int,
) -> dict[str, float]:
    def progress(index: int, count: int) -> float:
        return 0.0 if count == 1 else index / float(count - 1)

    warp = [abs(progress(row, negative_count) - progress(column, positive_count)) for row, column in path]
    moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    denominator = max(len(moves), 1)
    gap_step_count = sum(move in {(1, 0), (0, 1)} for move in moves)
    return {
        "mean_warp": sum(warp) / len(warp),
        "max_warp": max(warp),
        "diagonal_step_ratio": sum(move == (1, 1) for move in moves) / denominator,
        "vertical_step_ratio": sum(move == (1, 0) for move in moves) / denominator,
        "horizontal_step_ratio": sum(move == (0, 1) for move in moves) / denominator,
        "gap_step_count": float(gap_step_count),
    }


def compute_dtw_persistent_departure_area_scores(
    negative_states: torch.Tensor,
    positive_states: torch.Tensor,
    negative_state_valid: torch.Tensor,
    positive_state_valid: torch.Tensor,
    prompt_length: int | float | torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    *,
    collect_pair_matrices: bool = False,
    positive_aggregation: str = "max",
    gap_penalty: float = 0.0,
) -> dict[str, Any]:
    """Replace PDA's relative-position correspondence with symmetric DTW.

    Representation, persistent-area aggregation, cost ratio, selection order,
    and all downstream Teacher/OPD behavior remain identical to PDA.
    """

    if negative_states.ndim != 3 or positive_states.ndim != 3:
        raise ValueError("DTW-PDA states must have shape [N/P, M, D]")
    if negative_states.shape[1:] != positive_states.shape[1:]:
        raise ValueError("DTW-PDA positive and negative payload shapes must match")
    if negative_state_valid.shape != negative_states.shape[:2]:
        raise ValueError("negative_state_valid must align with negative_states")
    if positive_state_valid.shape != positive_states.shape[:2]:
        raise ValueError("positive_state_valid must align with positive_states")
    num_negative, max_states, _ = negative_states.shape
    num_positive = positive_states.shape[0]
    if num_negative < 1 or num_positive < 1:
        raise ValueError("DTW-PDA requires at least one negative and one positive sibling")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("DTW-PDA candidate metadata must align with negative rollouts")
    if positive_aggregation not in {"max", "min"}:
        raise ValueError("positive_aggregation must be 'max' or 'min'")
    if not math.isfinite(gap_penalty) or gap_penalty < 0:
        raise ValueError("DTW-PDA gap_penalty must be finite and non-negative")

    pair_area = torch.empty(num_negative, num_positive, device=negative_states.device, dtype=torch.float32)
    pair_diagnostics: list[list[dict[str, Any]]] = []
    pair_curves: list[list[torch.Tensor]] = []
    pair_running: list[list[torch.Tensor]] = []
    pair_departure: list[list[torch.Tensor]] = []
    for negative_index in range(num_negative):
        negative_slots = negative_state_valid[negative_index].bool().nonzero(as_tuple=False).flatten()
        if negative_slots.numel() < 2:
            raise ValueError("DTW-PDA requires at least two unique states per negative rollout")
        negative_sequence = negative_states[negative_index].index_select(0, negative_slots)
        diagnostic_row: list[dict[str, Any]] = []
        curve_row: list[torch.Tensor] = []
        running_row: list[torch.Tensor] = []
        departure_row: list[torch.Tensor] = []
        for positive_index in range(num_positive):
            positive_slots = positive_state_valid[positive_index].bool().nonzero(as_tuple=False).flatten()
            if positive_slots.numel() < 2:
                raise ValueError("DTW-PDA requires at least two unique states per positive rollout")
            positive_sequence = positive_states[positive_index].index_select(0, positive_slots)
            cosine_matrix = negative_sequence @ positive_sequence.transpose(0, 1)
            path, dtw_cost = dtw_align_symmetric(cosine_matrix, gap_penalty=gap_penalty)
            raw_curve = build_dtw_aligned_similarity_curve(cosine_matrix, path)
            similarity = normalized_cosine_similarity(raw_curve)
            running, departure, area = persistent_departure_area(similarity)
            pair_area[negative_index, positive_index] = area
            relative_columns = torch.round(
                torch.linspace(
                    0.0,
                    float(positive_sequence.shape[0] - 1),
                    negative_sequence.shape[0],
                    device=cosine_matrix.device,
                )
            ).long()
            relative_curve = cosine_matrix[
                torch.arange(negative_sequence.shape[0], device=cosine_matrix.device), relative_columns
            ]
            statistics = _dtw_path_statistics(path, negative_sequence.shape[0], positive_sequence.shape[0])
            pair_diagnostic: dict[str, Any] = {
                "negative_state_count": int(negative_sequence.shape[0]),
                "positive_state_count": int(positive_sequence.shape[0]),
                "dtw_cost": float(dtw_cost),
                "diag_mean_similarity": float(relative_curve.mean()),
                "dtw_mean_similarity": float(raw_curve.mean()),
                "similarity_gain": float(raw_curve.mean() - relative_curve.mean()),
                "pda_diag": float(persistent_departure_area(normalized_cosine_similarity(relative_curve))[2]),
                "pda_dtw": float(area),
                "gap_penalty": gap_penalty,
                "gap_penalty_cost": gap_penalty * statistics["gap_step_count"],
                **statistics,
            }
            if collect_pair_matrices:
                pair_diagnostic.update(
                    {
                        "path": [[int(row), int(column)] for row, column in path],
                        "cosine_matrix": cosine_matrix.detach().cpu().tolist(),
                        "diag_curve": relative_curve.detach().cpu().tolist(),
                        "dtw_curve": raw_curve.detach().cpu().tolist(),
                    }
                )
            diagnostic_row.append(pair_diagnostic)
            curve_row.append(similarity)
            running_row.append(running)
            departure_row.append(departure)
        pair_diagnostics.append(diagnostic_row)
        pair_curves.append(curve_row)
        pair_running.append(running_row)
        pair_departure.append(departure_row)

    if positive_aggregation == "min":
        area, best_positive_local_index = pair_area.min(dim=1)
    else:
        area, best_positive_local_index = pair_area.max(dim=1)
    teacher_cost = negative_response_lengths.detach().to(negative_states.device, torch.float32) + torch.as_tensor(
        prompt_length, device=negative_states.device, dtype=torch.float32
    )
    if teacher_cost.ndim != 1 or (teacher_cost <= 0).any() or not torch.isfinite(teacher_cost).all():
        raise ValueError("DTW-PDA Teacher costs must be finite and positive")
    cost_ratio = teacher_cost.min() / teacher_cost
    final_score = area * cost_ratio
    zero_area = bool(torch.isclose(area, torch.zeros_like(area), rtol=1.0e-6, atol=1.0e-8).all())
    rollout_ids = negative_rollout_ids.to(negative_states.device)
    if zero_area:
        order = sorted(range(num_negative), key=lambda i: (float(teacher_cost[i]), int(rollout_ids[i])))
    else:
        order = sorted(
            range(num_negative),
            key=lambda i: (-float(final_score[i]), -float(area[i]), float(teacher_cost[i]), int(rollout_ids[i])),
        )
    selected = order[0]
    padded_similarity = torch.zeros(num_negative, max_states, device=negative_states.device)
    padded_raw = torch.zeros_like(padded_similarity)
    padded_running = torch.zeros_like(padded_similarity)
    padded_departure = torch.zeros_like(padded_similarity)
    best_valid = torch.zeros(num_negative, max_states, dtype=torch.bool, device=negative_states.device)
    best_pair_diagnostics: list[dict[str, Any]] = []
    peak_value = torch.zeros(num_negative, device=negative_states.device)
    peak_index = torch.zeros(num_negative, dtype=torch.long, device=negative_states.device)
    for negative_index in range(num_negative):
        positive_index = int(best_positive_local_index[negative_index])
        similarity = pair_curves[negative_index][positive_index]
        running = pair_running[negative_index][positive_index]
        departure = pair_departure[negative_index][positive_index]
        count = similarity.numel()
        padded_similarity[negative_index, :count] = similarity
        padded_raw[negative_index, :count] = 2.0 * similarity - 1.0
        padded_running[negative_index, :count] = running
        padded_departure[negative_index, :count] = departure
        best_valid[negative_index, :count] = True
        peak_value[negative_index], peak_index[negative_index] = departure.max(dim=0)
        best_pair_diagnostics.append(pair_diagnostics[negative_index][positive_index])
    zeros = torch.zeros_like(area)
    minus_ones = torch.full((num_negative,), -1, dtype=torch.long, device=negative_states.device)
    return {
        "boundary_utility": area,
        "pda_area": area,
        "pair_area": pair_area,
        "cost_ratio": cost_ratio,
        "cost_efficiency": cost_ratio,
        "teacher_cost": teacher_cost,
        "teacher_input_len": teacher_cost,
        "final_score": final_score,
        "best_positive_local_index": best_positive_local_index,
        "best_stage_similarity": padded_similarity,
        "best_stage_valid_mask": best_valid,
        "best_raw_cosine": padded_raw,
        "best_running_max": padded_running,
        "best_departure": padded_departure,
        "max_departure": peak_value,
        "peak_departure_index": peak_index,
        "mean_departure": padded_departure.sum(dim=-1) / best_valid.sum(dim=-1),
        "selected_local_index": torch.tensor(selected, device=negative_states.device),
        "selected_rollout_id": rollout_ids[selected],
        "boundary_degenerate_to_cost_only": torch.tensor(zero_area, device=negative_states.device),
        "cost_only_local_index": torch.argmin(teacher_cost),
        "valid_candidate_mask": torch.ones(num_negative, dtype=torch.bool, device=negative_states.device),
        "best_split_index": minus_ones,
        "best_pre_similarity": padded_similarity[:, 0],
        "best_post_similarity": torch.stack(
            [padded_similarity[index, best_valid[index].sum() - 1] for index in range(num_negative)]
        ),
        "best_fork_drop": peak_value,
        "best_post_divergence": 1.0 - torch.stack(
            [padded_similarity[index, best_valid[index].sum() - 1] for index in range(num_negative)]
        ),
        "early_cosine": padded_similarity[:, 0],
        "middle1_cosine": torch.stack(
            [padded_similarity[index, best_valid[index].sum() - 1] for index in range(num_negative)]
        ),
        "middle2_cosine": zeros,
        "aggregated_middle_cosine": padded_similarity.sum(dim=-1) / best_valid.sum(dim=-1),
        "early_alignment": zeros,
        "plateau_drop": peak_value,
        "plateau_divergence": 1.0 - torch.stack(
            [padded_similarity[index, best_valid[index].sum() - 1] for index in range(num_negative)]
        ),
        "pair_utility": area,
        "first_span_cosine": padded_similarity[:, 0],
        "second_span_cosine": torch.stack(
            [padded_similarity[index, best_valid[index].sum() - 1] for index in range(num_negative)]
        ),
        "group_min_response_len": negative_response_lengths.min(),
        "group_min_teacher_input_len": teacher_cost.min(),
        "normalized_response_length": zeros,
        "length_efficiency": cost_ratio,
        "rollout_count": torch.tensor(num_negative + num_positive, device=negative_states.device),
        "valid_split_count": best_valid.sum(dim=-1) - 1,
        "boundary_search_enabled": False,
        "uses_hidden_difference": False,
        "representation_domain": "direct_hidden_state",
        "positive_aggregation": positive_aggregation,
        "pair_diagnostics": pair_diagnostics,
        "best_pair_diagnostics": best_pair_diagnostics,
    }


def _lm_head_cosine_products(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Accumulate exact uncentered LM-head dot products by vocabulary chunk."""

    if query_transitions.ndim != 3:
        raise ValueError("query_transitions must have shape [B, M, D]")
    if candidate_transitions.ndim != 4:
        raise ValueError("candidate_transitions must have shape [B, K, M, D]")
    if candidate_transitions.shape[0] != query_transitions.shape[0]:
        raise ValueError("query and candidate batch dimensions must match")
    if candidate_transitions.shape[2:] != query_transitions.shape[1:]:
        raise ValueError("query and candidate transition shapes must align")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != query_transitions.shape[-1]:
        raise ValueError("lm_head_weight must have shape [V, D] aligned with transitions")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    query = query_transitions.detach().float()
    candidates = candidate_transitions.detach().float()
    device = query.device
    cross_sum = torch.zeros(query.shape[0], candidates.shape[1], query.shape[1], dtype=torch.float32, device=device)
    query_square_sum = torch.zeros(query.shape[:2], dtype=torch.float32, device=device)
    candidate_square_sum = torch.zeros(candidates.shape[:3], dtype=torch.float32, device=device)
    for start in range(0, int(lm_head_weight.shape[0]), vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(device=device, dtype=torch.float32)
        query_logits = torch.einsum("bmd,vd->bmv", query, weight_chunk)
        candidate_logits = torch.einsum("bkmd,vd->bkmv", candidates, weight_chunk)
        cross_sum.add_(torch.einsum("bmv,bkmv->bkm", query_logits, candidate_logits))
        query_square_sum.add_(query_logits.square().sum(dim=-1))
        candidate_square_sum.add_(candidate_logits.square().sum(dim=-1))
        del weight_chunk, query_logits, candidate_logits
    return cross_sum, query_square_sum, candidate_square_sum


def _finalize_lm_head_cosine(statistics: tuple[torch.Tensor, ...], *, eps: float) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be positive")
    cross_sum, query_square_sum, candidate_square_sum = statistics
    denominator = query_square_sum.clamp_min(0.0).sqrt()[:, None, :] * candidate_square_sum.clamp_min(0.0).sqrt()
    cosine = torch.where(
        denominator > eps,
        cross_sum / denominator.clamp_min(eps),
        torch.zeros_like(cross_sum),
    ).clamp(-1.0, 1.0)
    cosine = cosine.detach()
    if cosine.dtype != torch.float32 or cosine.requires_grad:
        raise AssertionError("LM-head cosine must be detached FP32")
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("LM-head cosine contains NaN or Inf")
    return cosine


def compute_lm_head_cosine(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    vocab_chunk_size: int = 2048,
    vocab_process_group=None,
) -> torch.Tensor:
    """Compute exact ``cos(Wu, Wv)`` from the current Actor head."""

    statistics = _lm_head_cosine_products(
        query_transitions,
        candidate_transitions,
        lm_head_weight,
        vocab_chunk_size=vocab_chunk_size,
    )
    if vocab_process_group is not None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("vocab_process_group requires initialized torch.distributed")
        for tensor in statistics:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    return _finalize_lm_head_cosine(statistics, eps=eps)


def _centered_lm_head_cosine_sums(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Accumulate full-vocabulary sums without retaining projected logits."""

    if query_transitions.ndim != 3:
        raise ValueError("query_transitions must have shape [B, M, D]")
    if candidate_transitions.ndim != 4:
        raise ValueError("candidate_transitions must have shape [B, K, M, D]")
    if candidate_transitions.shape[0] != query_transitions.shape[0]:
        raise ValueError("query and candidate batch dimensions must match")
    if candidate_transitions.shape[2:] != query_transitions.shape[1:]:
        raise ValueError("query and candidate transition shapes must align")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != query_transitions.shape[-1]:
        raise ValueError("lm_head_weight must have shape [V, D] aligned with transitions")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    query = query_transitions.detach().float()
    candidates = candidate_transitions.detach().float()
    device = query.device
    batch_size, num_stages = query.shape[:2]
    num_candidates = candidates.shape[1]
    query_sum = torch.zeros(batch_size, num_stages, dtype=torch.float32, device=device)
    candidate_sum = torch.zeros(batch_size, num_candidates, num_stages, dtype=torch.float32, device=device)

    local_vocab_size = int(lm_head_weight.shape[0])
    for start in range(0, local_vocab_size, vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(device=device, dtype=torch.float32)
        query_logits = torch.einsum("bmd,vd->bmv", query, weight_chunk)
        candidate_logits = torch.einsum("bkmd,vd->bkmv", candidates, weight_chunk)
        query_sum.add_(query_logits.sum(dim=-1))
        candidate_sum.add_(candidate_logits.sum(dim=-1))
        del weight_chunk, query_logits, candidate_logits

    vocab_count = torch.tensor(float(local_vocab_size), dtype=torch.float32, device=device)
    return query_sum, candidate_sum, vocab_count


def _centered_lm_head_cosine_products(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    query_mean: torch.Tensor,
    candidate_mean: torch.Tensor,
    *,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Project again, explicitly center each chunk, and accumulate products."""

    query = query_transitions.detach().float()
    candidates = candidate_transitions.detach().float()
    device = query.device
    cross_sum = torch.zeros_like(candidate_mean, dtype=torch.float32, device=device)
    query_square_sum = torch.zeros_like(query_mean, dtype=torch.float32, device=device)
    candidate_square_sum = torch.zeros_like(candidate_mean, dtype=torch.float32, device=device)
    local_vocab_size = int(lm_head_weight.shape[0])
    for start in range(0, local_vocab_size, vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(device=device, dtype=torch.float32)
        query_centered = torch.einsum("bmd,vd->bmv", query, weight_chunk).sub(query_mean.unsqueeze(-1))
        candidate_centered = torch.einsum("bkmd,vd->bkmv", candidates, weight_chunk).sub(candidate_mean.unsqueeze(-1))
        cross_sum.add_(torch.einsum("bmv,bkmv->bkm", query_centered, candidate_centered))
        query_square_sum.add_(query_centered.square().sum(dim=-1))
        candidate_square_sum.add_(candidate_centered.square().sum(dim=-1))
        del weight_chunk, query_centered, candidate_centered
    return cross_sum, query_square_sum, candidate_square_sum


def _finalize_centered_lm_head_cosine(
    statistics: tuple[torch.Tensor, ...],
    *,
    eps: float,
) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be positive")
    (
        centered_cross,
        query_square_sum,
        candidate_square_sum,
    ) = statistics
    query_norm = query_square_sum.clamp_min(0.0).sqrt()[:, None, :]
    candidate_norm = candidate_square_sum.clamp_min(0.0).sqrt()
    denominator = query_norm * candidate_norm
    cosine = torch.where(
        denominator > eps,
        centered_cross / denominator.clamp_min(eps),
        torch.zeros_like(centered_cross),
    ).clamp(-1.0, 1.0)
    cosine = cosine.detach()
    if cosine.dtype != torch.float32:
        raise AssertionError("centered LM-head cosine must be FP32")
    if cosine.requires_grad:
        raise AssertionError("centered LM-head cosine must be detached")
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("centered LM-head cosine contains NaN or Inf")
    return cosine


def compute_centered_lm_head_cosine(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    vocab_chunk_size: int = 2048,
    vocab_process_group=None,
) -> torch.Tensor:
    """Compute exact ``cos(center(Wu), center(Wv))`` without retaining logits.

    ``lm_head_weight`` may be either the full Actor LM head or the local
    vocabulary shard. When ``vocab_process_group`` is provided, all additive
    statistics (including the global vocabulary count) are summed before the
    centered cosine is finalized, so every tensor-parallel rank receives the
    same full-vocabulary result.
    """

    sum_statistics = _centered_lm_head_cosine_sums(
        query_transitions,
        candidate_transitions,
        lm_head_weight,
        vocab_chunk_size=vocab_chunk_size,
    )
    if vocab_process_group is not None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("vocab_process_group requires initialized torch.distributed")
        for tensor in sum_statistics:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    query_sum, candidate_sum, vocab_count = sum_statistics
    if not torch.isfinite(vocab_count) or vocab_count <= 0:
        raise ValueError("global LM-head vocabulary size must be positive")
    product_statistics = _centered_lm_head_cosine_products(
        query_transitions,
        candidate_transitions,
        lm_head_weight,
        query_sum / vocab_count,
        candidate_sum / vocab_count,
        vocab_chunk_size=vocab_chunk_size,
    )
    if vocab_process_group is not None:
        for tensor in product_statistics:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    return _finalize_centered_lm_head_cosine(product_statistics, eps=eps)


def compute_whitened_lm_head_cosine(
    query_transitions: torch.Tensor,
    candidate_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    mean: torch.Tensor,
    cholesky: torch.Tensor | None = None,
    eps: float = 1.0e-6,
    vocab_chunk_size: int = 2048,
) -> torch.Tensor:
    """Apply calibration (center, optionally whiten, L2-norm) to hidden transitions,
    then project through the LM head and compute uncentered cosine.

    ``mean`` and ``cholesky`` (optional) are frozen calibration statistics.
    Whitening is the triangular solve ``L^{-1} (delta - mean)``, matching
    ``build_calibrated_hidden_transitions``.  After calibration the normalized
    hidden transitions are projected via the vocabulary weight and their
    uncentered cosine is computed, which is equivalent to
    ``cos(W * calibrate(u), W * calibrate(v))``.
    """
    query = query_transitions.detach().float()
    candidates = candidate_transitions.detach().float()
    device = query.device
    hidden_dim = query.shape[-1]

    mean_t = mean.detach().to(device=device, dtype=torch.float32)
    if mean_t.shape != (hidden_dim,):
        raise ValueError(f"calibration mean must have shape {(hidden_dim,)}")

    # Center
    query = query - mean_t
    candidates = candidates - mean_t

    # Whiten (optional)
    if cholesky is not None:
        factor = cholesky.detach().to(device=device, dtype=torch.float32)
        if factor.shape != (hidden_dim, hidden_dim):
            raise ValueError(f"calibration Cholesky must have shape {(hidden_dim, hidden_dim)}")
        flat_q = query.reshape(-1, hidden_dim)
        flat_c = candidates.reshape(-1, hidden_dim)
        query = torch.linalg.solve_triangular(factor, flat_q.T, upper=False).T.reshape_as(query)
        candidates = torch.linalg.solve_triangular(factor, flat_c.T, upper=False).T.reshape_as(candidates)

    # L2-normalize
    query = F.normalize(query, p=2, dim=-1, eps=eps)
    candidates = F.normalize(candidates, p=2, dim=-1, eps=eps)

    # Uncentered LM head cosine
    statistics = _lm_head_cosine_products(query, candidates, lm_head_weight, vocab_chunk_size=vocab_chunk_size)
    return _finalize_lm_head_cosine(statistics, eps=eps)


def build_boundary_span_ranges(response_mask: torch.Tensor, num_boundaries: int) -> tuple[torch.Tensor, ...]:
    """Return response-local ``[start, end)`` ranges for the M boundary spans."""

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [B, response_length]")
    if num_boundaries <= 0:
        raise ValueError("num_boundaries must be positive")
    lengths = response_mask.bool().sum(dim=-1, dtype=torch.long)
    boundary = torch.arange(
        0,
        num_boundaries + 1,
        device=response_mask.device,
        dtype=torch.long,
    )
    endpoints = (boundary.unsqueeze(0) * lengths.unsqueeze(1) + num_boundaries - 1) // num_boundaries
    endpoints[:, 0] = 0
    starts = endpoints[:, :-1]
    ends = endpoints[:, 1:]
    valid = ends > starts
    return starts, ends, valid


def build_l8_adaptive_early_indices(response_length: int) -> list[int]:
    """Return stable-unique L8 state counts in the first two M16 spans.

    Counts follow the existing Boundary-OPD convention: zero denotes the last
    prompt state and ``response_length`` denotes the final response state.
    """
    if response_length <= 0:
        return []
    window_tokens = max(1, response_length // 16)
    early_end = min(response_length, 2 * window_tokens)
    indices = [round(j * early_end / 15) for j in range(16)]
    clamped = [min(max(index, 0), early_end) for index in indices]
    return list(dict.fromkeys(clamped))


def build_uwtd_early_indices(response_length: int) -> list[int]:
    """Return stable-unique UWTD state counts on the exact first-L/8 grid."""

    if response_length <= 0:
        return []
    early_end = max(1, response_length // 8)
    indices = [round(stage * early_end / 15) for stage in range(16)]
    return list(dict.fromkeys(min(max(index, 0), early_end) for index in indices))


@dataclass(frozen=True)
class SASBSparsePolicySummary:
    probabilities: torch.Tensor
    support_ids: torch.Tensor
    support_mask: torch.Tensor


@dataclass(frozen=True)
class SASBPromptScoreResult:
    selected_index: int | None
    anchor_index: int | None
    success_distances: torch.Tensor
    z_scores: torch.Tensor
    boundary_q: torch.Tensor
    medoid_r: torch.Tensor
    utilities: torch.Tensor
    cost_penalties: torch.Tensor
    gains: torch.Tensor
    geometry_valid: torch.Tensor
    medoid_enabled: bool = False
    fallback_shortest: bool = False


def build_group_topk_tail_policy_summary(
    logits: torch.Tensor,
    *,
    top_k: int = SASB_TOP_K,
    eps: float = SASB_EPSILON,
) -> SASBSparsePolicySummary:
    """Compress ``[B,K,M,V]`` logits on each sibling-common Top-K union."""

    if logits.ndim != 4:
        raise ValueError("SASB logits must have shape [B, K, M, V]")
    if top_k <= 0 or top_k > logits.shape[-1]:
        raise ValueError("SASB top_k must lie within the vocabulary")
    if eps <= 0:
        raise ValueError("SASB eps must be positive")
    values = logits.detach().float()
    if not torch.isfinite(values).all():
        raise FloatingPointError("SASB logits contain NaN or Inf")
    batch_size, siblings, states, _ = values.shape
    max_support = siblings * top_k
    probabilities = torch.zeros(
        (batch_size, siblings, states, max_support + 1),
        dtype=torch.float32,
        device=values.device,
    )
    support_ids = torch.zeros(
        (batch_size, states, max_support),
        dtype=torch.long,
        device=values.device,
    )
    support_mask = torch.zeros(
        (batch_size, states, max_support),
        dtype=torch.bool,
        device=values.device,
    )
    log_normalizers = torch.logsumexp(values, dim=-1)
    top_ids = torch.topk(values, k=top_k, dim=-1).indices
    for batch_index in range(batch_size):
        for state_index in range(states):
            union = torch.unique(top_ids[batch_index, :, state_index].reshape(-1), sorted=True)
            width = int(union.numel())
            support_ids[batch_index, state_index, :width] = union
            support_mask[batch_index, state_index, :width] = True
            union_logits = values[batch_index, :, state_index].gather(
                1, union.unsqueeze(0).expand(siblings, -1)
            )
            union_probability = torch.exp(
                union_logits - log_normalizers[batch_index, :, state_index].unsqueeze(-1)
            )
            probabilities[batch_index, :, state_index, :width] = union_probability
            probabilities[batch_index, :, state_index, max_support] = (
                1.0 - union_probability.sum(dim=-1)
            ).clamp_min(0.0)
    probabilities.clamp_(min=0.0)
    totals = probabilities.sum(dim=-1, keepdim=True)
    if bool((totals <= eps).any()):
        raise FloatingPointError("SASB compressed distribution has zero mass")
    probabilities.div_(totals)
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise FloatingPointError("SASB compressed distribution is not normalized")
    return SASBSparsePolicySummary(probabilities, support_ids, support_mask)


def compute_jsd_from_sparse_policy(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    eps: float = SASB_EPSILON,
) -> torch.Tensor:
    """Return stable Jensen-Shannon divergence along the final dimension."""

    if p.shape != q.shape:
        raise ValueError("SASB JSD distributions must have identical shapes")
    if eps <= 0:
        raise ValueError("SASB JSD eps must be positive")
    p32 = p.detach().float().clamp_min(0.0)
    q32 = q.detach().float().clamp_min(0.0)
    p32 = p32 / p32.sum(dim=-1, keepdim=True).clamp_min(eps)
    q32 = q32 / q32.sum(dim=-1, keepdim=True).clamp_min(eps)
    mixture = 0.5 * (p32 + q32)
    kl_p = (p32 * (torch.log(p32 + eps) - torch.log(mixture + eps))).sum(dim=-1)
    kl_q = (q32 * (torch.log(q32 + eps) - torch.log(mixture + eps))).sum(dim=-1)
    return (0.5 * (kl_p + kl_q)).clamp(0.0, math.log(2.0))


def compute_pairwise_sibling_jsd(
    probabilities: torch.Tensor,
    state_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric sibling JSD matrix over jointly valid unique states."""

    if probabilities.ndim != 3:
        raise ValueError("SASB probabilities must have shape [K, M, S]")
    if state_valid.shape != probabilities.shape[:2]:
        raise ValueError("SASB state_valid must have shape [K, M]")
    siblings = probabilities.shape[0]
    distances = torch.full(
        (siblings, siblings),
        float("nan"),
        dtype=torch.float32,
        device=probabilities.device,
    )
    pair_valid = torch.zeros((siblings, siblings), dtype=torch.bool, device=probabilities.device)
    for left in range(siblings):
        distances[left, left] = 0.0
        pair_valid[left, left] = bool(state_valid[left].any())
        for right in range(left + 1, siblings):
            valid = state_valid[left].bool() & state_valid[right].bool()
            if not bool(valid.any()):
                continue
            value = compute_jsd_from_sparse_policy(
                probabilities[left, valid], probabilities[right, valid]
            ).mean()
            distances[left, right] = distances[right, left] = value
            pair_valid[left, right] = pair_valid[right, left] = True
    return distances, pair_valid


def compute_success_soft_distance(distances: torch.Tensor, *, tau: float = SASB_TAU_S) -> torch.Tensor:
    """Stable soft minimum used for SASB success proximity."""

    if tau <= 0:
        raise ValueError("SASB success temperature must be positive")
    values = distances.detach().float().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("SASB success distances must be non-empty and finite")
    return -tau * (torch.logsumexp(-values / tau, dim=0) - math.log(values.numel()))


def compute_failure_medoid(
    distances: torch.Tensor,
    pair_valid: torch.Tensor,
    wrong_indices: Sequence[int],
    *,
    sigma: float | None,
) -> tuple[torch.Tensor, bool]:
    """Return the 1P3N medoid factor, disabling it unless all pairs exist."""

    indices = [int(index) for index in wrong_indices]
    neutral = torch.ones(len(indices), dtype=torch.float32, device=distances.device)
    if len(indices) != 3 or sigma is None or sigma <= 0:
        return neutral, False
    pairs = [(0, 1), (0, 2), (1, 2)]
    if not all(bool(pair_valid[indices[left], indices[right]]) for left, right in pairs):
        return neutral, False
    result = torch.empty_like(neutral)
    for local_index, sibling in enumerate(indices):
        peers = [peer for peer in indices if peer != sibling]
        result[local_index] = 0.5 * torch.exp(-distances[sibling, peers] / sigma).sum()
    return result, True


def compute_sasb_prompt_scores(
    *,
    distances: torch.Tensor,
    pair_valid: torch.Tensor,
    correct_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    rollout_ids: torch.Tensor,
    tau_comp: float | None,
    sigma_3: float | None,
    score_mode: str = SASB_MAIN_SCORE_MODE,
) -> SASBPromptScoreResult:
    """Score one K=4 prompt with the shortest-anchored SASB rule."""

    if score_mode not in SASB_SCORE_MODES:
        raise ValueError(f"invalid SASB score mode: {score_mode}")
    shape = correct_mask.shape
    if correct_mask.ndim != 1 or any(
        tensor.shape != shape for tensor in (valid_mask, response_lengths, rollout_ids)
    ):
        raise ValueError("SASB prompt vectors must have identical rank-1 shapes")
    if distances.shape != pair_valid.shape or distances.shape != (shape[0], shape[0]):
        raise ValueError("SASB pairwise matrices must have shape [K, K]")
    device = distances.device
    finite_default = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    negative_default = torch.full(shape, float("-inf"), dtype=torch.float32, device=device)
    zeros = torch.zeros(shape, dtype=torch.float32, device=device)
    correct = correct_mask.bool() & valid_mask.bool()
    wrong = ~correct_mask.bool() & valid_mask.bool()
    positive_indices = correct.nonzero(as_tuple=False).flatten().tolist()
    wrong_indices = wrong.nonzero(as_tuple=False).flatten().tolist()
    if not positive_indices or not wrong_indices:
        return SASBPromptScoreResult(
            None, None, finite_default, finite_default.clone(), zeros, torch.ones_like(zeros),
            zeros, zeros, negative_default, torch.zeros_like(valid_mask, dtype=torch.bool),
        )
    anchor = min(
        wrong_indices,
        key=lambda index: (int(response_lengths[index]), int(rollout_ids[index]), index),
    )
    if score_mode == "sasb_shortest_only":
        gains = negative_default.clone()
        gains[anchor] = 0.0
        geometry_valid = torch.zeros_like(valid_mask, dtype=torch.bool)
        geometry_valid[anchor] = True
        return SASBPromptScoreResult(
            anchor, anchor, finite_default, finite_default.clone(), zeros, torch.ones_like(zeros),
            zeros, zeros, gains, geometry_valid,
        )

    success = finite_default.clone()
    geometry_valid = torch.zeros_like(valid_mask, dtype=torch.bool)
    for sibling in wrong_indices:
        usable = [positive for positive in positive_indices if bool(pair_valid[sibling, positive])]
        if not usable:
            continue
        success[sibling] = compute_success_soft_distance(distances[sibling, usable])
        geometry_valid[sibling] = True
    if not bool(geometry_valid[anchor]) or tau_comp is None or tau_comp <= 0:
        gains = negative_default.clone()
        gains[anchor] = 0.0
        return SASBPromptScoreResult(
            anchor, anchor, success, finite_default.clone(), zeros, torch.ones_like(zeros),
            zeros, zeros, gains, geometry_valid, fallback_shortest=True,
        )

    z_scores = finite_default.clone()
    z_scores[geometry_valid] = success[geometry_valid] / max(float(tau_comp), SASB_EPSILON)
    boundary_q = zeros.clone()
    if score_mode == "sasb_near_no_cost":
        boundary_q[geometry_valid] = torch.exp(-z_scores[geometry_valid])
    else:
        boundary_q[geometry_valid] = z_scores[geometry_valid] * torch.exp(-z_scores[geometry_valid])
    medoid_r = torch.ones_like(zeros)
    medoid_enabled = False
    if score_mode == SASB_MAIN_SCORE_MODE and len(wrong_indices) == 3:
        local_medoid, medoid_enabled = compute_failure_medoid(
            distances, pair_valid, wrong_indices, sigma=sigma_3
        )
        medoid_r[wrong_indices] = local_medoid
    utilities = boundary_q.clone()
    if medoid_enabled:
        utilities[wrong_indices] *= (
            1.0 - SASB_MEDOID_WEIGHT + SASB_MEDOID_WEIGHT * medoid_r[wrong_indices]
        )
    cost_penalties = zeros.clone()
    gains = negative_default.clone()
    for sibling in wrong_indices:
        if not bool(geometry_valid[sibling]):
            continue
        if score_mode in {"sasb_band_shortest_cost", SASB_MAIN_SCORE_MODE}:
            cost_penalties[sibling] = SASB_ETA * (
                float(response_lengths[sibling] - response_lengths[anchor]) / SASB_COST_NORMALIZER
            )
        gains[sibling] = utilities[sibling] - utilities[anchor] - cost_penalties[sibling]
    gains[anchor] = 0.0
    selected = max(
        wrong_indices,
        key=lambda index: (
            float(gains[index]),
            float(utilities[index]),
            float(boundary_q[index]),
            float(medoid_r[index]) if medoid_enabled else 0.0,
            -int(response_lengths[index]),
            -int(rollout_ids[index]),
            -index,
        ),
    )
    return SASBPromptScoreResult(
        selected, anchor, success, z_scores, boundary_q, medoid_r, utilities,
        cost_penalties, gains, geometry_valid, medoid_enabled=medoid_enabled,
    )


def build_l8_adaptive_early_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture 16 padded states for L8's adaptive first-``L/8`` search.

    Integer rounding can produce fewer than 16 unique states for short
    responses. Unique states are packed left, the last state pads the fixed
    capture shape, and padded transitions are marked invalid.
    """
    if response_mask.ndim != 2 or prompt_mask.ndim != 2:
        raise ValueError("response_mask and prompt_mask must be rank-2")
    if response_mask.shape[0] != prompt_mask.shape[0]:
        raise ValueError("response_mask and prompt_mask batch sizes must match")

    response_mask = response_mask.bool()
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    batch_size, response_width = response_mask.shape
    prompt_width = prompt_mask.shape[1]
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    endpoint_counts = torch.zeros((batch_size, 16), dtype=torch.long, device=response_mask.device)
    transition_valid = torch.zeros((batch_size, 15), dtype=torch.bool, device=response_mask.device)
    for row, response_length in enumerate(lengths.tolist()):
        unique_indices = build_l8_adaptive_early_indices(response_length)
        if not unique_indices:
            continue
        unique_count = len(unique_indices)
        endpoint_counts[row, :unique_count] = torch.tensor(unique_indices, device=response_mask.device)
        endpoint_counts[row, unique_count:] = unique_indices[-1]
        transition_valid[row, : unique_count - 1] = True

    prompt_positions = torch.arange(prompt_width, device=response_mask.device).expand(batch_size, -1)
    prompt_last = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values.clamp_min(0)
    response_positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
    ordered = torch.where(response_mask, response_positions, response_width).sort(dim=-1).values
    ordinals = (endpoint_counts - 1).clamp_min(0)
    if response_width:
        response_absolute = ordered.gather(1, ordinals).clamp(max=response_width - 1) + prompt_width
    else:
        response_absolute = prompt_last.unsqueeze(1).expand_as(endpoint_counts)
    state_indices = torch.where(endpoint_counts > 0, response_absolute, prompt_last.unsqueeze(1))
    transition_valid &= prompt_mask.any(dim=-1).unsqueeze(1)
    return state_indices.detach(), transition_valid.detach()


def build_uwtd_early_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture M=16 stable-unique states over exactly ``floor(L/8)`` tokens."""

    if response_mask.ndim != 2 or prompt_mask.ndim != 2:
        raise ValueError("response_mask and prompt_mask must be rank-2")
    if response_mask.shape[0] != prompt_mask.shape[0]:
        raise ValueError("response_mask and prompt_mask batch sizes must match")

    response_mask = response_mask.bool()
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    batch_size, response_width = response_mask.shape
    prompt_width = prompt_mask.shape[1]
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    endpoint_counts = torch.zeros((batch_size, 16), dtype=torch.long, device=response_mask.device)
    transition_valid = torch.zeros((batch_size, 15), dtype=torch.bool, device=response_mask.device)
    for row, response_length in enumerate(lengths.tolist()):
        unique_indices = build_uwtd_early_indices(response_length)
        if not unique_indices:
            continue
        unique_count = len(unique_indices)
        endpoint_counts[row, :unique_count] = torch.tensor(unique_indices, device=response_mask.device)
        endpoint_counts[row, unique_count:] = unique_indices[-1]
        transition_valid[row, : unique_count - 1] = True

    prompt_positions = torch.arange(prompt_width, device=response_mask.device).expand(batch_size, -1)
    prompt_last = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values.clamp_min(0)
    response_positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
    ordered = torch.where(response_mask, response_positions, response_width).sort(dim=-1).values
    ordinals = (endpoint_counts - 1).clamp_min(0)
    if response_width:
        response_absolute = ordered.gather(1, ordinals).clamp(max=response_width - 1) + prompt_width
    else:
        response_absolute = prompt_last.unsqueeze(1).expand_as(endpoint_counts)
    state_indices = torch.where(endpoint_counts > 0, response_absolute, prompt_last.unsqueeze(1))
    transition_valid &= prompt_mask.any(dim=-1).unsqueeze(1)
    return state_indices.detach(), transition_valid.detach()


def build_sasb_l8_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse L8's grid while invalidating every duplicate state slot."""

    state_indices, _ = build_l8_adaptive_early_state_indices(response_mask, prompt_mask)
    row_valid = response_mask.bool().any(dim=-1) & prompt_mask.bool().any(dim=-1)
    state_valid = torch.zeros_like(state_indices, dtype=torch.bool)
    state_valid[:, 0] = row_valid
    state_valid[:, 1:] = row_valid.unsqueeze(1) & (state_indices[:, 1:] != state_indices[:, :-1])
    return state_indices.detach(), state_valid.detach()


def build_sasb_prompt_atomic_order(
    row_workloads: torch.Tensor,
    *,
    dp_size: int,
    k_rollouts: int = 4,
) -> torch.Tensor:
    """Balance whole sibling groups across DP ranks without splitting prompts."""

    if row_workloads.ndim != 1:
        raise ValueError("SASB row workloads must be rank-1")
    if dp_size <= 0 or k_rollouts <= 0:
        raise ValueError("SASB dp_size and k_rollouts must be positive")
    if row_workloads.numel() % k_rollouts:
        raise ValueError("SASB rows must contain complete sibling groups")
    group_count = row_workloads.numel() // k_rollouts
    if group_count < dp_size or group_count % dp_size:
        raise ValueError("SASB prompt count must be divisible by actor DP size")
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

    group_workloads = row_workloads.detach().cpu().reshape(group_count, k_rollouts).sum(dim=1)
    partitions = get_seqlen_balanced_partitions(
        group_workloads.tolist(),
        k_partitions=dp_size,
        equal_size=True,
    )
    order: list[int] = []
    for partition in partitions:
        for group_index in partition:
            start = int(group_index) * k_rollouts
            order.extend(range(start, start + k_rollouts))
    return torch.tensor(order, dtype=torch.long, device=row_workloads.device)


def build_l21_normalized_state_entropies(
    token_entropies: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align Student next-token entropy with L8's 16 early latent states.

    ``token_entropies[:, q]`` is the entropy at the state after ``q`` valid
    response tokens, predicting response token ``q + 1``.  The terminal state
    after the final response token has no next-response-token distribution and
    is therefore marked invalid instead of being imputed.  L8's valid adjacent
    transition pairs never require that terminal entropy.
    """

    if token_entropies.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("L21 token_entropies and response_mask must be rank-2")
    if token_entropies.shape != response_mask.shape:
        raise ValueError("L21 token_entropies and response_mask must have the same shape")
    if vocab_size <= 1:
        raise ValueError("L21 requires lm_head vocab_size > 1")

    entropy = token_entropies.detach().float()
    mask = response_mask.detach().bool().to(device=entropy.device)
    if not torch.isfinite(entropy[mask]).all():
        raise FloatingPointError("L21 valid Student token entropy contains NaN or Inf")

    normalizer = math.log(float(vocab_size))
    normalized = entropy / normalizer
    tolerance = 1.0e-5
    if bool(((normalized[mask] < -tolerance) | (normalized[mask] > 1.0 + tolerance)).any()):
        raise ValueError("L21 normalized Student entropy must lie in [0, 1]")
    normalized = normalized.clamp(0.0, 1.0)

    batch_size, response_width = entropy.shape
    state_entropy = torch.zeros((batch_size, 16), dtype=torch.float32, device=entropy.device)
    state_entropy_valid = torch.zeros((batch_size, 16), dtype=torch.bool, device=entropy.device)
    response_positions = torch.arange(response_width, device=entropy.device).expand(batch_size, -1)
    ordered_positions = torch.where(mask, response_positions, response_width).sort(dim=-1).values

    for row, response_length in enumerate(mask.sum(dim=-1, dtype=torch.long).tolist()):
        state_counts = build_l8_adaptive_early_indices(response_length)
        if not state_counts:
            continue
        padded_counts = state_counts + [state_counts[-1]] * (16 - len(state_counts))
        for slot, state_count in enumerate(padded_counts):
            if state_count >= response_length:
                continue
            token_position = int(ordered_positions[row, state_count].item())
            state_entropy[row, slot] = normalized[row, token_position]
            state_entropy_valid[row, slot] = True
    return state_entropy.detach(), state_entropy_valid.detach()


def build_uwtd_entropy_statistics(
    token_entropies: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align UWTD state entropy and compute full-response entropy background."""

    if token_entropies.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("UWTD token_entropies and response_mask must be rank-2")
    if token_entropies.shape != response_mask.shape:
        raise ValueError("UWTD token_entropies and response_mask must have the same shape")
    if vocab_size <= 1:
        raise ValueError("UWTD requires lm_head vocab_size > 1")

    entropy = token_entropies.detach().float()
    mask = response_mask.detach().bool().to(device=entropy.device)
    if not torch.isfinite(entropy[mask]).all():
        raise FloatingPointError("UWTD valid Student token entropy contains NaN or Inf")
    normalized = entropy / math.log(float(vocab_size))
    tolerance = 1.0e-5
    if bool(((normalized[mask] < -tolerance) | (normalized[mask] > 1.0 + tolerance)).any()):
        raise ValueError("UWTD normalized Student entropy must lie in [0, 1]")
    normalized = normalized.clamp(0.0, 1.0)

    counts = mask.sum(dim=-1)
    global_mean = (normalized * mask.float()).sum(dim=-1) / counts.clamp_min(1)
    if bool((counts == 0).any()):
        raise ValueError("UWTD requires at least one valid response token per rollout")

    batch_size, response_width = entropy.shape
    state_entropy = torch.zeros((batch_size, 16), dtype=torch.float32, device=entropy.device)
    state_valid = torch.zeros((batch_size, 16), dtype=torch.bool, device=entropy.device)
    positions = torch.arange(response_width, device=entropy.device).expand(batch_size, -1)
    ordered_positions = torch.where(mask, positions, response_width).sort(dim=-1).values
    for row, response_length in enumerate(counts.tolist()):
        state_counts = build_uwtd_early_indices(response_length)
        padded_counts = state_counts + [state_counts[-1]] * (16 - len(state_counts))
        for slot, state_count in enumerate(padded_counts):
            if slot >= len(state_counts):
                continue
            # The terminal response state has no next-response-token entropy.
            if state_count >= response_length:
                continue
            token_position = int(ordered_positions[row, state_count].item())
            state_entropy[row, slot] = normalized[row, token_position]
            state_valid[row, slot] = True
    if not torch.isfinite(global_mean).all():
        raise FloatingPointError("UWTD global Student entropy contains NaN or Inf")
    return state_entropy.detach(), state_valid.detach(), global_mean.detach()


def build_l8_l13_0p4n_ht_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the distinct state tracks required by the L8/L13 HT hybrid.

    The first 16 states and first 15 validity slots are L8's adaptive early
    track. The final 16 states and final 16 validity slots are L13's normalized
    whole-response track. Keeping both tracks prevents the 0P4N route from
    computing L13 geometry on L8's early-only states.
    """

    early_states, early_valid = build_l8_adaptive_early_state_indices(response_mask, prompt_mask)
    whole_states, whole_valid, _, _ = build_l11_whole_response_state_indices(response_mask, prompt_mask)
    return (
        torch.cat((early_states, whole_states), dim=1).detach(),
        torch.cat((early_valid, whole_valid), dim=1).detach(),
    )


def build_l18_early_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture L18's 16 normalized early states, including repeated short-rollout positions."""

    state_indices, _ = build_l8_adaptive_early_state_indices(response_mask, prompt_mask)
    state_valid = (
        response_mask.bool().any(dim=-1) & prompt_mask.bool().any(dim=-1)
    ).unsqueeze(1).expand(-1, 16)
    return state_indices, state_valid.detach()


def build_l19_early_indices(response_length: int) -> list[int]:
    """Return L19's M=16 normalized-progress token counts over the first 50%.

    The normalized progress grid is ``u_m = (m - 1) / (2 (M - 1))`` for
    ``m = 1..M`` so that ``u_1 = 0`` (the prompt-end Student state) and
    ``u_M = 0.5`` (half of the response).  Counts follow the Boundary-OPD
    convention where zero denotes the last prompt token and a positive count
    denotes the corresponding response token.  Repeated counts for very short
    responses are preserved on purpose: a duplicated latent state contributes a
    zero-length movement to the path and never breaks the ``D_net <= D_path``
    inequality.
    """
    if response_length <= 0:
        return []
    denominator = 2 * (L19_NUM_STATES - 1)
    counts = [round(step * response_length / denominator) for step in range(L19_NUM_STATES)]
    return [min(max(count, 0), response_length) for count in counts]


def build_l19_early_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture L19's M=16 centered LM-head states over the first 50% of the response.

    Column zero is the prompt-end Student state (``u_1 = 0``); the remaining
    columns are the response tokens nearest ``round(u_m * L)``.  Every state of
    a rollout with a non-empty prompt and response is marked valid because the
    Latent Path Efficiency ratio consumes the full M=16 trajectory.
    """
    if response_mask.ndim != 2 or prompt_mask.ndim != 2:
        raise ValueError("response_mask and prompt_mask must be rank-2")
    if response_mask.shape[0] != prompt_mask.shape[0]:
        raise ValueError("response_mask and prompt_mask batch sizes must match")

    response_mask = response_mask.bool()
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    batch_size, response_width = response_mask.shape
    prompt_width = prompt_mask.shape[1]
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    endpoint_counts = torch.zeros((batch_size, L19_NUM_STATES), dtype=torch.long, device=response_mask.device)
    row_valid = torch.zeros((batch_size,), dtype=torch.bool, device=response_mask.device)
    for row, response_length in enumerate(lengths.tolist()):
        counts = build_l19_early_indices(response_length)
        if not counts:
            continue
        endpoint_counts[row] = torch.tensor(counts, device=response_mask.device)
        row_valid[row] = True

    prompt_positions = torch.arange(prompt_width, device=response_mask.device).expand(batch_size, -1)
    prompt_last = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values.clamp_min(0)
    response_positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
    ordered = torch.where(response_mask, response_positions, response_width).sort(dim=-1).values
    ordinals = (endpoint_counts - 1).clamp_min(0)
    if response_width:
        response_absolute = ordered.gather(1, ordinals).clamp(max=response_width - 1) + prompt_width
    else:
        response_absolute = prompt_last.unsqueeze(1).expand_as(endpoint_counts)
    state_indices = torch.where(endpoint_counts > 0, response_absolute, prompt_last.unsqueeze(1))
    row_valid &= prompt_mask.any(dim=-1)
    state_valid = row_valid.unsqueeze(1).expand(-1, L19_NUM_STATES)
    return state_indices.detach(), state_valid.detach()


def compute_centered_lm_head_path_norms(
    delta_transitions: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_chunk_size: int = 2048,
    vocab_process_group=None,
) -> torch.Tensor:
    """Return the Euclidean norm ``||C W delta||_2`` of each centered LM-head movement.

    ``delta_transitions`` has shape ``[B, S, D]`` and holds hidden-state
    differences (state-to-state movements plus the net start-to-end movement).
    Because centering is linear, ``center(W delta) = C W delta`` and the norm is
    obtained from the identity ``||C y||^2 = sum_v y_v^2 - (sum_v y_v)^2 / V``
    with ``y = W delta``, so the full ``[B, S, V]`` logits are never retained.
    The RMS ``1/sqrt(V)`` factor from the spec cancels in the Latent Path
    Efficiency ratio and is intentionally omitted.
    """
    if delta_transitions.ndim != 3:
        raise ValueError("delta_transitions must have shape [B, S, D]")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != delta_transitions.shape[-1]:
        raise ValueError("lm_head_weight must have shape [V, D] aligned with transitions")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    deltas = delta_transitions.detach().float()
    device = deltas.device
    batch_size, num_transitions = deltas.shape[:2]
    square_sum = torch.zeros(batch_size, num_transitions, dtype=torch.float32, device=device)
    linear_sum = torch.zeros(batch_size, num_transitions, dtype=torch.float32, device=device)
    local_vocab_size = int(lm_head_weight.shape[0])
    for start in range(0, local_vocab_size, vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(
            device=device, dtype=torch.float32
        )
        chunk_logits = torch.einsum("bsd,vd->bsv", deltas, weight_chunk)
        square_sum.add_(chunk_logits.square().sum(dim=-1))
        linear_sum.add_(chunk_logits.sum(dim=-1))
        del weight_chunk, chunk_logits
    vocab_count = torch.tensor(float(local_vocab_size), dtype=torch.float32, device=device)
    if vocab_process_group is not None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("vocab_process_group requires initialized torch.distributed")
        for tensor in (square_sum, linear_sum, vocab_count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    if not torch.isfinite(vocab_count) or vocab_count <= 0:
        raise ValueError("global LM-head vocabulary size must be positive")
    centered_square = (square_sum - linear_sum.square() / vocab_count).clamp_min(0.0)
    norms = centered_square.sqrt().detach()
    if norms.dtype != torch.float32 or norms.requires_grad:
        raise AssertionError("centered LM-head path norms must be detached FP32")
    if not torch.isfinite(norms).all():
        raise FloatingPointError("centered LM-head path norms contain NaN or Inf")
    return norms


def build_l8_representation_dynamics_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_boundaries: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture independent early-L8 and full-response tracks for dynamics analysis.

    The returned state layout is ``[early_0..early_M, full_0..full_M]`` and
    the valid-mask layout is ``[early transitions, full transitions]``.  The
    two tracks must therefore be differenced separately by the consumer.
    """
    if num_boundaries != 64:
        raise ValueError("representation dynamics is fixed to M=64")
    if response_mask.ndim != 2 or prompt_mask.ndim != 2:
        raise ValueError("response_mask and prompt_mask must be rank-2")
    response_mask = response_mask.bool()
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    lengths = response_mask.sum(dim=-1, dtype=torch.long)

    def track(mask: torch.Tensor, effective_lengths: torch.Tensor):
        # Reuse the standard sampler by replacing each row's response mask with
        # its desired prefix. This preserves padding/left-prompt conventions.
        prefix_mask = mask & (mask.long().cumsum(dim=-1) <= effective_lengths.unsqueeze(1))
        return build_boundary_indices(prefix_mask, prompt_mask, num_boundaries)

    early_lengths = torch.where(lengths > 0, ((lengths + 7) // 8).clamp_min(1), lengths)
    early_states, early_valid = track(response_mask, early_lengths)
    full_states, full_valid = build_boundary_indices(response_mask, prompt_mask, num_boundaries)
    states = torch.cat((early_states, full_states), dim=1)
    valid = torch.cat((early_valid, full_valid), dim=1)
    return states.detach(), valid.detach()


def build_l11_whole_response_state_indices(
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    num_states: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture fixed relative state slots across the complete valid response.

    Short responses deliberately retain repeated token indices: the slots are
    relative trajectory positions, not distinct transitions.  Only valid
    response tokens are sampled, so EOS (removed by the caller) and padding
    can never be selected.

    Returns ``(absolute_indices, slot_valid, unique_counts,
    duplicate_fractions)``.  ``slot_valid`` is true for every slot of a
    non-empty response, including repeated slots.
    """
    if response_mask.ndim != 2 or prompt_mask.ndim != 2:
        raise ValueError("response_mask and prompt_mask must be rank-2")
    if response_mask.shape[0] != prompt_mask.shape[0]:
        raise ValueError("response_mask and prompt_mask batch sizes must match")
    if num_states < 2:
        raise ValueError("num_states must be at least two")

    response_mask = response_mask.bool()
    prompt_mask = prompt_mask.to(device=response_mask.device, dtype=torch.bool)
    batch_size, response_width = response_mask.shape
    prompt_width = prompt_mask.shape[1]
    lengths = response_mask.sum(dim=-1, dtype=torch.long)
    slot_valid = (lengths > 0).unsqueeze(1).expand(-1, num_states).clone()
    relative_indices = torch.zeros((batch_size, num_states), dtype=torch.long, device=response_mask.device)
    fractions = torch.arange(num_states, device=response_mask.device, dtype=torch.float64) / (num_states - 1)
    for row, response_length in enumerate(lengths.tolist()):
        if response_length:
            relative_indices[row] = torch.round(fractions * (response_length - 1)).to(torch.long)

    if response_width:
        response_positions = torch.arange(response_width, device=response_mask.device).expand(batch_size, -1)
        ordered = torch.where(response_mask, response_positions, response_width).sort(dim=-1).values
        absolute = ordered.gather(1, relative_indices.clamp(max=max(response_width - 1, 0))) + prompt_width
    else:
        absolute = torch.zeros_like(relative_indices)

    prompt_positions = torch.arange(prompt_width, device=response_mask.device).expand(batch_size, -1)
    prompt_last = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values.clamp_min(0)
    absolute = torch.where(slot_valid, absolute, prompt_last.unsqueeze(1))
    sequence_width = prompt_width + response_width
    if sequence_width < 1 or absolute.min().item() < 0 or absolute.max().item() >= sequence_width:
        raise AssertionError("Persistent Drop produced an out-of-range state index")

    unique_counts = torch.tensor(
        [len(torch.unique(row)) if length else 0 for row, length in zip(relative_indices, lengths.tolist(), strict=True)],
        dtype=torch.long,
        device=response_mask.device,
    )
    duplicate_fractions = torch.where(
        lengths > 0,
        1.0 - unique_counts.float() / float(num_states),
        torch.zeros(batch_size, dtype=torch.float32, device=response_mask.device),
    )
    return absolute.detach(), slot_valid.detach(), unique_counts.detach(), duplicate_fractions.detach()


def log_domain_sinkhorn_uniform_cost(
    ground_cost: torch.Tensor,
    *,
    epsilon: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, float, int, bool]:
    """Entropic OT cost for uniform empirical measures.

    Returns ``(transport_cost, marginal_residual, iterations, converged)``. The
    best iterate within the budget is always returned and ``converged`` reports
    whether the final marginal residual met ``tolerance``. Callers decide
    whether a non-converged iterate is acceptable, because a small marginal
    violation perturbs the transport cost only mildly.

    The number of iterations needed grows with the ground-cost spread measured
    in units of ``epsilon``: peaked cost matrices (many near-duplicate token
    directions plus a few far ones) routinely need several hundred iterations at
    ``epsilon=0.05``, while generic ones converge within a few dozen. Because
    the loop exits as soon as ``tolerance`` is met, a generous ``max_iterations``
    only costs time on the rare hard instances.

    Each iteration ends with the column scaling, so the column marginals match
    ``b`` up to float32 rounding by construction and the reported residual only
    has to measure the row-marginal violation of the current plan. The row
    log-sum-exp that certifies it is exactly the one the next row update needs,
    so the convergence check costs nothing.
    """

    if ground_cost.ndim != 2 or min(ground_cost.shape) <= 0:
        raise ValueError("ground_cost must be a non-empty matrix")
    if epsilon <= 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Sinkhorn epsilon, max_iterations, and tolerance must be positive")
    cost = ground_cost.detach().float()
    if not torch.isfinite(cost).all() or cost.min() < 0:
        raise ValueError("ground_cost must be finite and non-negative")
    rows, columns = cost.shape
    log_a = torch.full((rows,), -math.log(rows), dtype=torch.float32, device=cost.device)
    log_b = torch.full((columns,), -math.log(columns), dtype=torch.float32, device=cost.device)
    log_kernel = -cost / float(epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    log_row = torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
    residual = math.inf
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        log_u = log_a - log_row
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        log_row = torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        row_mass = torch.exp(log_u + log_row)
        residual = float((row_mass - log_a.exp()).abs().max())
        iterations = iteration
        if residual <= tolerance:
            break
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    transport_cost = (plan * cost).sum().detach()
    if not torch.isfinite(transport_cost):
        raise FloatingPointError("Sinkhorn transport cost is NaN or Inf")
    return transport_cost, residual, iterations, bool(residual <= tolerance)


def build_sm_pda_span_ranges(
    response_mask: torch.Tensor,
    num_spans: int = SM_PDA_NUM_SPANS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic non-empty response-token spans using floor(j * L / M).

    Returned indices address the ordered valid response tokens, not padded tensor
    positions. Empty floor intervals remain invalid; tokens are never duplicated.
    """

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [B, response_length]")
    if num_spans <= 0:
        raise ValueError("num_spans must be positive")
    lengths = response_mask.detach().bool().sum(dim=-1, dtype=torch.long)
    grid = torch.arange(num_spans + 1, device=response_mask.device, dtype=torch.long)
    edges = grid.unsqueeze(0) * lengths.unsqueeze(1) // num_spans
    starts = edges[:, :-1]
    ends = edges[:, 1:]
    return starts.detach(), ends.detach(), (ends > starts).detach()


def _sm_pda_cap_indices(start: int, end: int, max_points: int, device: torch.device) -> torch.Tensor:
    count = end - start
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count <= max_points:
        return torch.arange(start, end, dtype=torch.long, device=device)
    return torch.linspace(start, end - 1, max_points, device=device).round().long()


def compute_success_manifold_span_distances(
    response_hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    num_spans: int = SM_PDA_NUM_SPANS,
    max_points_per_span: int = SM_PDA_MAX_POINTS_PER_SPAN,
    norm_epsilon: float = 1.0e-6,
    sinkhorn_epsilon: float = 4.5,
    sinkhorn_max_iterations: int = 50,
    sinkhorn_tolerance: float = 1.0e-3,
    compute_self_distance_diagnostic: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
    """Return all-sibling/all-span entropic-OT discrepancies for one prompt group.

    The output has shape ``[K, K, M, M]``. Invalid span pairs are NaN. Hidden
    vectors stay on their raw scale for the FP32 ground cost and log-domain
    Sinkhorn solve. The resulting transport cost is divided by the prompt-group
    mean token L2 norm. This preserves the configured Sinkhorn epsilon's raw
    hidden-space scale; normalizing the points before an entropic solve would
    change the relative regularization strength.
    """

    if response_hidden.ndim != 3:
        raise ValueError("response_hidden must have shape [K, response_length, D]")
    if response_mask.shape != response_hidden.shape[:2]:
        raise ValueError("response_mask must align with response_hidden")
    if max_points_per_span <= 0:
        raise ValueError("max_points_per_span must be positive")
    hidden = response_hidden.detach()
    mask = response_mask.detach().bool()
    valid_hidden = hidden[mask].float()
    if valid_hidden.numel() == 0:
        raise ValueError("success_manifold_pda requires at least one valid response token")
    if not torch.isfinite(valid_hidden).all():
        raise FloatingPointError("success_manifold_pda hidden contains NaN or Inf")
    prompt_mean_norm = valid_hidden.norm(dim=-1).mean()
    if not torch.isfinite(prompt_mean_norm) or prompt_mean_norm < 0:
        raise FloatingPointError("success_manifold_pda mean hidden norm is invalid")

    span_build_start = time.perf_counter()
    starts, ends, valid = build_sm_pda_span_ranges(mask, num_spans)
    before = (ends - starts).clamp_min(0)
    after = before.clamp_max(max_points_per_span)
    spans: list[list[torch.Tensor | None]] = []
    for rollout in range(hidden.shape[0]):
        ordered = hidden[rollout, mask[rollout]].detach().float()
        rollout_spans: list[torch.Tensor | None] = []
        for span in range(num_spans):
            if not bool(valid[rollout, span]):
                rollout_spans.append(None)
                continue
            indices = _sm_pda_cap_indices(
                int(starts[rollout, span]),
                int(ends[rollout, span]),
                max_points_per_span,
                hidden.device,
            )
            rollout_spans.append(ordered.index_select(0, indices))
        spans.append(rollout_spans)
    span_build_time_ms = (time.perf_counter() - span_build_start) * 1000.0

    distances = torch.full(
        (hidden.shape[0], hidden.shape[0], num_spans, num_spans),
        float("nan"),
        dtype=torch.float32,
        device=hidden.device,
    )
    iterations: list[int] = []
    nonconverged = 0
    failed = 0
    pairwise_cost_time_ms = 0.0
    sinkhorn_time_ms = 0.0
    distance_scale = prompt_mean_norm + norm_epsilon
    for left in range(hidden.shape[0]):
        for right in range(left + 1, hidden.shape[0]):
            for left_span in range(num_spans):
                x = spans[left][left_span]
                if x is None:
                    continue
                for right_span in range(num_spans):
                    y = spans[right][right_span]
                    if y is None:
                        continue
                    cost_start = time.perf_counter()
                    cost = torch.cdist(x, y, p=2).float()
                    pairwise_cost_time_ms += (time.perf_counter() - cost_start) * 1000.0
                    try:
                        sinkhorn_start = time.perf_counter()
                        value, _, count, converged = log_domain_sinkhorn_uniform_cost(
                            cost,
                            epsilon=sinkhorn_epsilon,
                            max_iterations=sinkhorn_max_iterations,
                            tolerance=sinkhorn_tolerance,
                        )
                        sinkhorn_time_ms += (time.perf_counter() - sinkhorn_start) * 1000.0
                    except (FloatingPointError, RuntimeError, ValueError):
                        failed += 1
                        continue
                    normalized_value = value / distance_scale
                    distances[left, right, left_span, right_span] = normalized_value
                    distances[right, left, right_span, left_span] = normalized_value
                    iterations.append(count)
                    nonconverged += int(not converged)

    self_distances: list[float] = []
    if compute_self_distance_diagnostic:
        for rollout_spans in spans:
            valid_indices = [index for index, span in enumerate(rollout_spans) if span is not None]
            if not valid_indices:
                continue
            x = rollout_spans[valid_indices[len(valid_indices) // 2]]
            assert x is not None
            cost_start = time.perf_counter()
            cost = torch.cdist(x, x, p=2).float()
            pairwise_cost_time_ms += (time.perf_counter() - cost_start) * 1000.0
            try:
                sinkhorn_start = time.perf_counter()
                value, _, _, _ = log_domain_sinkhorn_uniform_cost(
                    cost,
                    epsilon=sinkhorn_epsilon,
                    max_iterations=sinkhorn_max_iterations,
                    tolerance=sinkhorn_tolerance,
                )
                sinkhorn_time_ms += (time.perf_counter() - sinkhorn_start) * 1000.0
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            self_distances.append(float(value / distance_scale))
    return distances.detach(), {
        "span_starts": starts,
        "span_ends": ends,
        "span_valid_mask": valid,
        "num_points_before_cap": before,
        "num_points_after_cap": after,
        "prompt_mean_hidden_norm": float(prompt_mean_norm),
        "self_distance_mean": (
            sum(self_distances) / len(self_distances) if self_distances else float("nan")
        ),
        "sinkhorn_calls": len(iterations),
        "sinkhorn_iterations_sum": sum(iterations),
        "sinkhorn_iterations_max": max(iterations, default=0),
        "sinkhorn_nonconverged_count": nonconverged,
        "sinkhorn_failed_count": failed,
        "pairwise_cost_time_ms": pairwise_cost_time_ms,
        "sinkhorn_time_ms": sinkhorn_time_ms,
        "span_build_time_ms": span_build_time_ms,
    }


def compute_success_manifold_pda_scores(
    span_distances: torch.Tensor,
    negative_span_valid: torch.Tensor,
    positive_span_valid: torch.Tensor,
    prompt_length: int | float,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
) -> dict[str, torch.Tensor | bool]:
    """Score negative candidates by nearest-success distance departure only."""

    if span_distances.ndim != 4:
        raise ValueError("span_distances must have shape [N, P, M_negative, M_positive]")
    n_count, p_count, num_spans, positive_spans = span_distances.shape
    if min(n_count, p_count) <= 0 or positive_spans != num_spans:
        raise ValueError("success_manifold_pda requires non-empty N/P square span grids")
    if negative_span_valid.shape != (n_count, num_spans):
        raise ValueError("negative_span_valid does not align with distances")
    if positive_span_valid.shape != (p_count, num_spans):
        raise ValueError("positive_span_valid does not align with distances")

    valid_pairs = (
        negative_span_valid[:, None, :, None].bool()
        & positive_span_valid[None, :, None, :].bool()
        & torch.isfinite(span_distances)
    )
    flat = (
        span_distances.detach()
        .float()
        .masked_fill(~valid_pairs, float("inf"))
        .permute(0, 2, 1, 3)
        .flatten(2, 3)
    )
    best_distance, flat_reference = flat.min(dim=-1)
    has_reference = torch.isfinite(best_distance)
    nearest_positive_local = torch.div(flat_reference, num_spans, rounding_mode="floor")
    nearest_positive_span = flat_reference.remainder(num_spans)
    nearest_positive_local = nearest_positive_local.masked_fill(~has_reference, -1)
    nearest_positive_span = nearest_positive_span.masked_fill(~has_reference, -1)

    area = torch.full((n_count,), float("nan"), device=span_distances.device)
    distance_mean = torch.full_like(area, float("nan"))
    distance_max = torch.full_like(area, float("nan"))
    distance_final = torch.full_like(area, float("nan"))
    distance_min = torch.full_like(area, float("nan"))
    max_departure = torch.full_like(area, float("nan"))
    departure_start_index = torch.full((n_count,), -1, dtype=torch.long, device=span_distances.device)
    running_min = torch.full_like(best_distance, float("nan"))
    departure = torch.full_like(best_distance, float("nan"))
    usable_candidate = torch.zeros(n_count, dtype=torch.bool, device=span_distances.device)
    for candidate in range(n_count):
        valid_indices = has_reference[candidate].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            continue
        curve = best_distance[candidate, valid_indices]
        curve_running_min = torch.cummin(curve, dim=0).values
        curve_departure = curve - curve_running_min
        usable_candidate[candidate] = True
        area[candidate] = curve_departure[1:].mean()
        distance_mean[candidate] = curve.mean()
        distance_max[candidate] = curve.max()
        distance_final[candidate] = curve[-1]
        distance_min[candidate] = curve.min()
        max_departure[candidate] = curve_departure.max()
        threshold = 0.5 * max_departure[candidate]
        departure_start_index[candidate] = int((curve_departure >= threshold).nonzero()[0])
        running_min[candidate, valid_indices] = curve_running_min
        departure[candidate, valid_indices] = curve_departure

    teacher_cost = negative_response_lengths.detach().to(span_distances.device, torch.float32) + float(prompt_length)
    cost_ratio = teacher_cost.min() / teacher_cost
    score = area * cost_ratio
    rollout_ids = negative_rollout_ids.to(span_distances.device)
    candidates = usable_candidate.nonzero(as_tuple=False).flatten().tolist()
    selected = (
        min(candidates, key=lambda i: (-float(score[i]), -float(area[i]), float(teacher_cost[i]), int(rollout_ids[i])))
        if candidates
        else -1
    )
    return {
        "distance_curve": best_distance,
        "distance_valid_mask": has_reference,
        "nearest_positive_local": nearest_positive_local,
        "nearest_positive_span": nearest_positive_span,
        "running_min_curve": running_min,
        "departure_curve": departure,
        "persistent_area": area,
        "distance_mean": distance_mean,
        "distance_max": distance_max,
        "distance_final": distance_final,
        "distance_min": distance_min,
        "max_departure": max_departure,
        "departure_start_index": departure_start_index,
        "teacher_cost": teacher_cost,
        "cost_ratio": cost_ratio,
        "score": score,
        "usable_candidate": usable_candidate,
        "selected_local_index": torch.tensor(selected, device=span_distances.device),
        "fallback_required": selected < 0,
    }


def compute_hidden_span_wasserstein_pairwise(
    span_hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    num_boundaries: int,
    eps: float = 1.0e-6,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_max_iterations: int = 2000,
    sinkhorn_tolerance: float = 1.0e-5,
    sinkhorn_fallback_relative_tolerance: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute exact token-distribution stage similarities for one K-sibling prompt.

    Sinkhorn occasionally exhausts ``sinkhorn_max_iterations`` with a marginal
    residual just above ``sinkhorn_tolerance`` on slow-mixing cost matrices.
    Such iterates are accepted (and counted in ``sinkhorn_unconverged``) when
    the marginal violation stays within
    ``sinkhorn_fallback_relative_tolerance`` of the uniform marginal mass,
    since the induced transport-cost error is negligible next to the [0, 1]
    similarity clamping. Severe violations still raise, preserving the strict
    guard against genuine numerical failure.

    Spans whose token directions cluster tightly with a few far outliers mix
    slowly and need several hundred iterations at ``sinkhorn_epsilon=0.05``, so
    ``sinkhorn_max_iterations`` defaults to a budget that covers them; the loop
    still stops at ``sinkhorn_tolerance``, leaving easy spans as cheap as before.
    """

    if span_hidden.ndim != 3:
        raise ValueError("span_hidden must have shape [K, response_length, D]")
    if response_mask.shape != span_hidden.shape[:2]:
        raise ValueError("response_mask must align with span_hidden")
    if sinkhorn_fallback_relative_tolerance < 0:
        raise ValueError("sinkhorn_fallback_relative_tolerance must be non-negative")
    starts, ends, valid = build_boundary_span_ranges(response_mask, num_boundaries)
    siblings = span_hidden.shape[0]
    similarity = torch.zeros(
        siblings,
        siblings,
        num_boundaries,
        dtype=torch.float32,
        device=span_hidden.device,
    )
    max_residual = 0.0
    max_iterations = 0
    unconverged_solves = 0

    def solve_sinkhorn(cost: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        nonlocal unconverged_solves
        transport_cost, residual, iterations, converged = log_domain_sinkhorn_uniform_cost(
            cost,
            epsilon=sinkhorn_epsilon,
            max_iterations=sinkhorn_max_iterations,
            tolerance=sinkhorn_tolerance,
        )
        if converged:
            return transport_cost, residual, iterations
        fallback_tolerance = sinkhorn_fallback_relative_tolerance * min(
            1.0 / cost.shape[0],
            1.0 / cost.shape[1],
        )
        if residual > fallback_tolerance:
            raise RuntimeError(
                "log-domain Sinkhorn failed to converge: "
                f"shape={tuple(cost.shape)} epsilon={sinkhorn_epsilon} "
                f"max_iterations={sinkhorn_max_iterations} residual={residual:.8g} "
                f"tolerance={sinkhorn_tolerance} fallback_tolerance={fallback_tolerance:.8g}; "
                "raise algorithm.boundary_opd.sinkhorn_max_iterations "
                "(env BOUNDARY_OPD_SINKHORN_MAX_ITERATIONS) or "
                "algorithm.boundary_opd.sinkhorn_epsilon to restore convergence"
            )
        unconverged_solves += 1
        return transport_cost, residual, iterations

    for stage in range(num_boundaries):
        tokens: list[torch.Tensor | None] = []
        self_ot: list[tuple[torch.Tensor, float, int] | None] = []
        for sibling in range(siblings):
            if not bool(valid[sibling, stage]):
                tokens.append(None)
                self_ot.append(None)
                continue
            start = int(starts[sibling, stage])
            end = int(ends[sibling, stage])
            token_directions = span_hidden[sibling, start:end].detach().float()
            token_directions = token_directions[response_mask[sibling, start:end].bool()]
            if token_directions.numel() == 0:
                raise RuntimeError(
                    f"valid Boundary span unexpectedly contains no tokens: sibling={sibling} stage={stage}"
                )
            token_directions = F.normalize(token_directions, p=2, dim=-1, eps=eps)
            token_square = token_directions.square().sum(dim=-1)
            cost = (
                token_square.unsqueeze(1) + token_square.unsqueeze(0) - 2.0 * (token_directions @ token_directions.T)
            ).clamp(0.0, 4.0)
            tokens.append(token_directions)
            self_ot.append(solve_sinkhorn(cost))
            similarity[sibling, sibling, stage] = 1.0
        for left in range(siblings):
            if tokens[left] is None:
                continue
            for right in range(left + 1, siblings):
                if tokens[right] is None:
                    continue
                left_square = tokens[left].square().sum(dim=-1)
                right_square = tokens[right].square().sum(dim=-1)
                cross_cost = (
                    left_square.unsqueeze(1) + right_square.unsqueeze(0) - 2.0 * (tokens[left] @ tokens[right].T)
                ).clamp(0.0, 4.0)
                cross_ot, cross_residual, cross_iterations = solve_sinkhorn(cross_cost)
                assert self_ot[left] is not None and self_ot[right] is not None
                left_ot, left_residual, left_iterations = self_ot[left]
                right_ot, right_residual, right_iterations = self_ot[right]
                divergence = (cross_ot - 0.5 * left_ot - 0.5 * right_ot).clamp_min(0.0)
                value = (1.0 - divergence / 4.0).clamp(0.0, 1.0)
                similarity[left, right, stage] = value
                similarity[right, left, stage] = value
                max_residual = max(max_residual, cross_residual, left_residual, right_residual)
                max_iterations = max(max_iterations, cross_iterations, left_iterations, right_iterations)
    if unconverged_solves:
        logger.warning(
            "hidden-span Sinkhorn accepted %d non-converged iterates within the fallback "
            "marginal tolerance (max residual %.6g, target tolerance %.6g); "
            "transport costs are mildly approximate",
            unconverged_solves,
            max_residual,
            sinkhorn_tolerance,
        )
    return similarity.detach(), {
        "sinkhorn_max_residual": max_residual,
        "sinkhorn_max_iterations": max_iterations,
        "sinkhorn_unconverged": unconverged_solves,
    }


def compute_centered_lm_head_ground_cost(
    query_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    vocab_chunk_size: int = 2048,
    vocab_process_group=None,
) -> torch.Tensor:
    """Exact squared direction cost between centered LM-head token projections."""

    if query_tokens.ndim != 2 or candidate_tokens.ndim != 2:
        raise ValueError("query_tokens and candidate_tokens must have shapes [N, D] and [M, D]")
    if query_tokens.shape[-1] != candidate_tokens.shape[-1]:
        raise ValueError("query and candidate hidden dimensions must match")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[-1] != query_tokens.shape[-1]:
        raise ValueError("lm_head_weight must have shape [V, D]")
    query = query_tokens.detach().float()
    candidate = candidate_tokens.detach().float()
    query_sum = torch.zeros(query.shape[0], dtype=torch.float32, device=query.device)
    candidate_sum = torch.zeros(candidate.shape[0], dtype=torch.float32, device=query.device)
    vocab_count = torch.tensor(float(lm_head_weight.shape[0]), dtype=torch.float32, device=query.device)
    for start in range(0, int(lm_head_weight.shape[0]), vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(query.device, torch.float32)
        query_sum.add_((query @ weight_chunk.T).sum(dim=-1))
        candidate_sum.add_((candidate @ weight_chunk.T).sum(dim=-1))
    sums = (query_sum, candidate_sum, vocab_count)
    if vocab_process_group is not None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("vocab_process_group requires initialized torch.distributed")
        for tensor in sums:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    query_mean = query_sum / vocab_count
    candidate_mean = candidate_sum / vocab_count
    cross = torch.zeros(query.shape[0], candidate.shape[0], dtype=torch.float32, device=query.device)
    query_square = torch.zeros(query.shape[0], dtype=torch.float32, device=query.device)
    candidate_square = torch.zeros(candidate.shape[0], dtype=torch.float32, device=query.device)
    for start in range(0, int(lm_head_weight.shape[0]), vocab_chunk_size):
        weight_chunk = lm_head_weight[start : start + vocab_chunk_size].detach().to(query.device, torch.float32)
        query_centered = query @ weight_chunk.T - query_mean.unsqueeze(1)
        candidate_centered = candidate @ weight_chunk.T - candidate_mean.unsqueeze(1)
        cross.add_(query_centered @ candidate_centered.T)
        query_square.add_(query_centered.square().sum(dim=-1))
        candidate_square.add_(candidate_centered.square().sum(dim=-1))
    products = (cross, query_square, candidate_square)
    if vocab_process_group is not None:
        for tensor in products:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=vocab_process_group)
    query_norm = query_square.clamp_min(0.0).sqrt()
    candidate_norm = candidate_square.clamp_min(0.0).sqrt()
    denominator = query_norm.unsqueeze(1) * candidate_norm.unsqueeze(0)
    normalized_cross = torch.where(
        denominator > eps,
        cross / denominator.clamp_min(eps),
        torch.zeros_like(cross),
    )
    query_unit_square = (query_norm > eps).to(torch.float32)
    candidate_unit_square = (candidate_norm > eps).to(torch.float32)
    cost = (
        (query_unit_square.unsqueeze(1) + candidate_unit_square.unsqueeze(0) - 2.0 * normalized_cross.clamp(-1.0, 1.0))
        .clamp(0.0, 4.0)
        .detach()
    )
    if not torch.isfinite(cost).all():
        raise FloatingPointError("centered LM-head ground cost contains NaN or Inf")
    return cost


def compute_boundary_contrast_scores(
    negative_transitions: torch.Tensor,
    positive_transitions: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    prompt_length: int | torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    score_mode: str = "legacy",
    window_ratio: float = 0.125,
    negative_state_entropies: torch.Tensor | None = None,
    negative_state_entropy_valid_mask: torch.Tensor | None = None,
    negative_global_entropy_mean: torch.Tensor | None = None,
    l27_lambda_cost: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute Boundary-Contrast utilities and deterministically select one negative sibling."""

    if negative_transitions.ndim != 3 or positive_transitions.ndim != 3:
        raise ValueError("Boundary transitions must have shape [N, M, D]")
    if negative_transitions.shape[1:] != positive_transitions.shape[1:]:
        raise ValueError("positive and negative transitions must share [M, D]")
    negative = negative_transitions.detach().float()
    positive = positive_transitions.detach().float()
    stage_cosine = torch.einsum("nmd,pmd->npm", negative, positive).clamp(-1.0, 1.0)
    return compute_boundary_contrast_scores_from_cosine(
        stage_cosine,
        negative_valid_mask,
        positive_valid_mask,
        prompt_length,
        negative_response_lengths,
        negative_rollout_ids,
        score_mode=score_mode,
        window_ratio=window_ratio,
        negative_state_entropies=negative_state_entropies,
        negative_state_entropy_valid_mask=negative_state_entropy_valid_mask,
        negative_global_entropy_mean=negative_global_entropy_mean,
        l27_lambda_cost=l27_lambda_cost,
    )


def compute_l11_persistent_drop_scores_from_cosine(
    stage_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    *,
    equivalence_tolerance: float = 2.0e-6,
) -> dict[str, torch.Tensor]:
    """Rank wrong siblings by anchored, persistent trajectory separation.

    The nearest positive is fixed once from slot 1.  Slots 2--15 always use
    that same reference; slot 16 is retained for diagnostics but excluded to
    avoid the known terminal rebound.  The direct score is mathematically
    equivalent to the weighted adjacent-drop form
    ``sum((14..1) * (c[k] - c[k+1])) / 14``.
    """
    if stage_cosine.ndim != 3 or stage_cosine.shape[-1] != 16:
        raise ValueError("persistent_drop stage_cosine must have shape [N_neg, N_pos, 16]")
    num_negative, num_positive, _ = stage_cosine.shape
    if num_negative < 1 or num_positive < 1:
        raise ValueError("persistent_drop requires at least one negative and one positive")
    if negative_valid_mask.shape != (num_negative, 16):
        raise ValueError("negative_valid_mask must have shape [N_neg, 16]")
    if positive_valid_mask.shape != (num_positive, 16):
        raise ValueError("positive_valid_mask must have shape [N_pos, 16]")
    if negative_rollout_ids.shape != (num_negative,):
        raise ValueError("negative_rollout_ids must have shape [N_neg]")

    cosine = stage_cosine.detach().float().clamp(-1.0, 1.0)
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("persistent_drop cosine contains NaN or Inf")
    negative_valid = negative_valid_mask.detach().bool()
    positive_valid = positive_valid_mask.detach().bool()
    pair_valid = negative_valid[:, None, :] & positive_valid[None, :, :]
    required_valid = pair_valid[..., :15].all(dim=-1)

    anchor_candidates = cosine[..., 0].masked_fill(~required_valid, -float("inf"))
    best_positive_local_index = anchor_candidates.argmax(dim=1)
    candidate_index = torch.arange(num_negative, device=cosine.device)
    fixed_reference_similarity = cosine[candidate_index, best_positive_local_index, :]
    valid_candidate_mask = required_valid[candidate_index, best_positive_local_index]
    anchor_similarity = fixed_reference_similarity[:, 0]
    body_similarity = fixed_reference_similarity[:, 1:15].mean(dim=-1)
    direct_score = anchor_similarity - body_similarity

    deltas = fixed_reference_similarity[:, :14] - fixed_reference_similarity[:, 1:15]
    weights = torch.arange(14, 0, -1, dtype=torch.float32, device=cosine.device) / 14.0
    delta_score = (deltas * weights).sum(dim=-1)
    max_abs_error = (direct_score - delta_score).abs().max()
    if float(max_abs_error) > equivalence_tolerance:
        raise AssertionError(
            f"persistent_drop direct/delta score mismatch: {float(max_abs_error):.3e}"
        )

    score = torch.where(valid_candidate_mask, direct_score, torch.full_like(direct_score, -float("inf")))
    if not valid_candidate_mask.any():
        raise ValueError("persistent_drop has no candidate with 15 valid scoring slots")
    optimum = score.max()
    tied = (score == optimum).nonzero(as_tuple=False).flatten()
    tied_rollout_ids = negative_rollout_ids.detach().to(device=cosine.device).index_select(0, tied)
    selected_local_index = tied[tied_rollout_ids.argmin()]
    sorted_scores = score[valid_candidate_mask].sort(descending=True).values
    margin = sorted_scores[0] - sorted_scores[1] if sorted_scores.numel() > 1 else torch.zeros_like(sorted_scores[0])
    normalized_margin = margin / (sorted_scores[0].abs() + 1.0e-8)
    return {
        "score": direct_score.detach(),
        "direct_score": direct_score.detach(),
        "delta_score": delta_score.detach(),
        "max_abs_equivalence_error": max_abs_error.detach(),
        "anchor_similarity": anchor_similarity.detach(),
        "body_similarity": body_similarity.detach(),
        "persistent_drop": direct_score.detach(),
        "best_positive_local_index": best_positive_local_index.detach(),
        "fixed_reference_similarity": fixed_reference_similarity.detach(),
        "valid_candidate_mask": valid_candidate_mask.detach(),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": negative_rollout_ids.detach().to(device=cosine.device)[selected_local_index],
        "score_margin": margin.detach(),
        "score_margin_normalized": normalized_margin.detach(),
    }


def _safr_dtw_aligned_similarity(negative: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """Return the negative-indexed similarity trajectory for one PN pair."""
    cosine = (negative @ positive.T).clamp(-1.0, 1.0)
    distance = 1.0 - cosine
    rows, columns = distance.shape
    dp = torch.empty_like(distance)
    parent = torch.full((rows, columns), -1, dtype=torch.int8, device=distance.device)
    for j in range(rows):
        for k in range(columns):
            if j == 0 and k == 0:
                dp[j, k] = distance[j, k]
                continue
            choices: list[tuple[torch.Tensor, int]] = []
            # Deterministic tie order: diagonal, vertical, horizontal.
            if j > 0 and k > 0:
                choices.append((dp[j - 1, k - 1], 0))
            if j > 0:
                choices.append((dp[j - 1, k], 1))
            if k > 0:
                choices.append((dp[j, k - 1], 2))
            best_value, best_parent = min(choices, key=lambda item: (float(item[0]), item[1]))
            dp[j, k] = distance[j, k] + best_value
            parent[j, k] = best_parent

    aligned: list[list[int]] = [[] for _ in range(rows)]
    j, k = rows - 1, columns - 1
    while True:
        aligned[j].append(k)
        if j == 0 and k == 0:
            break
        move = int(parent[j, k])
        if move == 0:
            j, k = j - 1, k - 1
        elif move == 1:
            j -= 1
        elif move == 2:
            k -= 1
        else:  # pragma: no cover - guarded by the DP construction
            raise RuntimeError("invalid SAFR DTW backpointer")

    values = []
    for row, positive_indices in enumerate(aligned):
        values.append(cosine[row, positive_indices].mean())
    return torch.stack(values)


def compute_safr_n1_scores(
    states: torch.Tensor,
    state_valid_mask: torch.Tensor,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    rollout_ids: Sequence[int],
    *,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute the complete SAFR-OPD n1 sibling score for one mixed prompt."""
    if states.ndim != 3 or state_valid_mask.shape != states.shape[:2]:
        raise ValueError("SAFR states must be [K, M, D] with an aligned valid mask")
    if not positive_indices or not negative_indices:
        raise ValueError("SAFR requires a mixed prompt")
    if len(rollout_ids) != states.shape[0]:
        raise ValueError("SAFR rollout_ids must align with states")

    hidden = states.detach().float()
    valid = state_valid_mask.detach().bool()
    if not torch.isfinite(hidden[valid]).all():
        raise FloatingPointError("SAFR hidden states contain NaN or Inf")
    rollout_means = []
    for rollout_index in range(states.shape[0]):
        count = int(valid[rollout_index].sum())
        if count < 1:
            raise ValueError("each SAFR rollout must contain at least one reasoning span")
        rollout_means.append(hidden[rollout_index, :count].mean(dim=0))
    center = torch.stack(rollout_means).mean(dim=0)
    centered = F.normalize(hidden - center, dim=-1, eps=eps)

    negative_scores: list[torch.Tensor] = []
    best_positive_locals: list[int] = []
    best_forks: list[int] = []
    best_fork_scores: list[torch.Tensor] = []
    best_pre_similarities: list[torch.Tensor] = []
    best_post_similarities: list[torch.Tensor] = []
    alignment_qualities: list[torch.Tensor] = []
    aligned_profiles: list[torch.Tensor] = []
    valid_candidates: list[bool] = []

    for negative_index in negative_indices:
        negative_count = int(valid[negative_index].sum())
        negative_states = centered[negative_index, :negative_count]
        pair_profiles: list[torch.Tensor] = []
        pair_qualities: list[torch.Tensor] = []
        for positive_index in positive_indices:
            positive_count = int(valid[positive_index].sum())
            profile = _safr_dtw_aligned_similarity(
                negative_states,
                centered[positive_index, :positive_count],
            )
            pair_profiles.append(profile)
            pair_qualities.append(profile.mean())
        best_positive_local = max(
            range(len(positive_indices)),
            key=lambda local: (float(pair_qualities[local]), -int(rollout_ids[positive_indices[local]])),
        )
        profile = pair_profiles[best_positive_local]
        quality = pair_qualities[best_positive_local]

        if negative_count < 3:
            fork_index = -1
            fork_score = torch.zeros((), device=states.device)
            best_pre = torch.zeros((), device=states.device)
            best_post = torch.zeros((), device=states.device)
            score = torch.zeros((), device=states.device)
            candidate_valid = False
        else:
            window = min(2, negative_count // 2)
            fork_values: list[torch.Tensor] = []
            fork_indices = list(range(window - 1, negative_count - window))
            for fork in fork_indices:
                pre = profile[fork - window + 1 : fork + 1].mean()
                post = profile[fork + 1 : fork + window + 1].mean()
                fork_values.append(torch.relu(pre) * torch.relu(pre - post))
            best_fork_local = max(
                range(len(fork_values)),
                key=lambda local: (float(fork_values[local]), -fork_indices[local]),
            )
            fork_index = fork_indices[best_fork_local]
            fork_score = fork_values[best_fork_local]
            best_pre = profile[fork_index - window + 1 : fork_index + 1].mean()
            best_post = profile[fork_index + 1 : fork_index + window + 1].mean()

            score = fork_score
            candidate_valid = True

        negative_scores.append(score)
        best_positive_locals.append(best_positive_local)
        best_forks.append(fork_index)
        best_fork_scores.append(fork_score)
        best_pre_similarities.append(best_pre)
        best_post_similarities.append(best_post)
        alignment_qualities.append(quality)
        padded_profile = torch.zeros(states.shape[1], dtype=torch.float32, device=states.device)
        padded_profile[:negative_count] = profile
        aligned_profiles.append(padded_profile)
        valid_candidates.append(candidate_valid)

    scores = torch.stack(negative_scores)
    selected_local = max(
        range(len(negative_indices)),
        key=lambda local: (float(scores[local]), -int(rollout_ids[negative_indices[local]])),
    )
    return {
        "prompt_center": center.detach(),
        "score": scores.detach(),
        "selected_local_index": torch.tensor(selected_local, device=states.device),
        "selected_rollout_id": torch.tensor(rollout_ids[negative_indices[selected_local]], device=states.device),
        "best_positive_local_index": torch.tensor(best_positive_locals, device=states.device),
        "best_fork_index": torch.tensor(best_forks, device=states.device),
        "fork_score": torch.stack(best_fork_scores).detach(),
        "pre_similarity": torch.stack(best_pre_similarities).detach(),
        "post_similarity": torch.stack(best_post_similarities).detach(),
        "alignment_quality": torch.stack(alignment_qualities).detach(),
        "aligned_similarity": torch.stack(aligned_profiles).detach(),
        "valid_candidate_mask": torch.tensor(valid_candidates, dtype=torch.bool, device=states.device),
    }


def compute_l12_interior_nearest_failure_scores_from_cosine(
    negative_positive_cosine: torch.Tensor,
    negative_negative_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rank wrong siblings with L12's common interior trajectory geometry.

    Slots 2--15 (zero-based slice ``1:15``) are the only geometric signal.
    With positive siblings, ``Q`` is the distance to the nearest positive.  If
    no positive exists, ``Q`` is the mean distance to the other wrong siblings
    (the failure-medoid objective).  Both routes minimize
    ``J = Q / R_len`` and break numerical ties by rollout id, where ``R_len``
    is L8's prompt-local response-length efficiency.
    """
    if negative_positive_cosine.ndim != 3 or negative_positive_cosine.shape[-1] != 16:
        raise ValueError("L12 negative_positive_cosine must have shape [N_neg, N_pos, 16]")
    if negative_negative_cosine.ndim != 3 or negative_negative_cosine.shape[-1] != 16:
        raise ValueError("L12 negative_negative_cosine must have shape [N_neg, N_neg, 16]")
    num_negative, num_positive, _ = negative_positive_cosine.shape
    if num_negative < 1 or negative_negative_cosine.shape[:2] != (num_negative, num_negative):
        raise ValueError("L12 requires a non-empty square wrong/wrong cosine matrix")
    if negative_valid_mask.shape != (num_negative, 16):
        raise ValueError("L12 negative_valid_mask must have shape [N_neg, 16]")
    if positive_valid_mask.shape != (num_positive, 16):
        raise ValueError("L12 positive_valid_mask must have shape [N_pos, 16]")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("L12 negative_response_lengths and negative_rollout_ids must have shape [N_neg]")

    neg_pos = negative_positive_cosine.detach().float().clamp(-1.0, 1.0)
    neg_neg = negative_negative_cosine.detach().float().clamp(-1.0, 1.0)
    response_lengths = negative_response_lengths.detach().to(device=neg_pos.device, dtype=torch.float32)
    if not torch.isfinite(neg_pos).all() or not torch.isfinite(neg_neg).all():
        raise FloatingPointError("L12 cosine contains NaN or Inf")
    if not torch.isfinite(response_lengths).all() or bool((response_lengths <= 0).any()):
        raise ValueError("L12 response lengths must be finite and positive")

    neg_interior_valid = negative_valid_mask.detach().bool()[:, 1:15].all(dim=-1)
    pos_interior_valid = positive_valid_mask.detach().bool()[:, 1:15].all(dim=-1)
    neg_pos_distance = (1.0 - neg_pos[:, :, 1:15].mean(dim=-1)).clamp(0.0, 2.0)
    neg_neg_distance = (1.0 - neg_neg[:, :, 1:15].mean(dim=-1)).clamp(0.0, 2.0)

    if num_positive:
        pair_valid = neg_interior_valid[:, None] & pos_interior_valid[None, :]
        masked = neg_pos_distance.masked_fill(~pair_valid, float("inf"))
        reference_local_index = masked.argmin(dim=1)
        row = torch.arange(num_negative, device=masked.device)
        geometry = masked[row, reference_local_index]
        reference_similarity = neg_pos[row, reference_local_index]
        valid_candidate_mask = torch.isfinite(geometry)
        routing_code = torch.ones((), dtype=torch.long, device=masked.device)
    else:
        if num_negative < 2:
            raise ValueError("L12 failure medoid requires at least two wrong siblings")
        pair_valid = neg_interior_valid[:, None] & neg_interior_valid[None, :]
        diagonal = torch.eye(num_negative, dtype=torch.bool, device=neg_neg.device)
        off_diagonal_valid = pair_valid & ~diagonal
        valid_candidate_mask = off_diagonal_valid.sum(dim=1) == num_negative - 1
        geometry = neg_neg_distance.masked_fill(~off_diagonal_valid, 0.0).sum(dim=1) / (num_negative - 1)
        nearest_other = neg_neg_distance.masked_fill(~off_diagonal_valid, float("inf"))
        reference_local_index = nearest_other.argmin(dim=1)
        row = torch.arange(num_negative, device=nearest_other.device)
        reference_similarity = neg_neg[row, reference_local_index]
        geometry = geometry.masked_fill(~valid_candidate_mask, float("inf"))
        routing_code = torch.zeros((), dtype=torch.long, device=neg_neg.device)

    if not valid_candidate_mask.any():
        raise ValueError("L12 has no candidate with 14 valid interior slots")
    length_range = response_lengths.max() - response_lengths.min()
    normalized_response_length = (response_lengths - response_lengths.min()) / (
        length_range + L8_LENGTH_NORMALIZATION_EPSILON
    )
    length_efficiency = 1.0 - normalized_response_length
    objective = geometry / length_efficiency.clamp_min(L8_LENGTH_NORMALIZATION_EPSILON)
    optimum = objective.min()
    # Cosine reductions can differ by a few ulps for mathematically identical
    # candidates.  Treat those as ties before applying the stable rollout-id
    # rule so device/reduction details cannot change the selected sibling.
    tied = torch.isclose(objective, optimum, rtol=1.0e-6, atol=1.0e-7).nonzero(as_tuple=False).flatten()
    tied_efficiency = length_efficiency.index_select(0, tied)
    tied = tied[torch.isclose(tied_efficiency, tied_efficiency.max(), rtol=1.0e-6, atol=1.0e-7)]
    tied_ids = negative_rollout_ids.detach().to(device=objective.device).index_select(0, tied)
    selected_local_index = tied[tied_ids.argmin()]
    sorted_objectives = objective[valid_candidate_mask].sort().values
    margin = (
        sorted_objectives[1] - sorted_objectives[0]
        if sorted_objectives.numel() > 1
        else torch.zeros_like(sorted_objectives[0])
    )
    return {
        "geometry": geometry.detach(),
        "objective": objective.detach(),
        "cost_factor": length_efficiency.detach(),
        "normalized_response_length": normalized_response_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "reference_local_index": reference_local_index.detach(),
        "reference_similarity": reference_similarity.detach(),
        "valid_candidate_mask": valid_candidate_mask.detach(),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": negative_rollout_ids.detach().to(objective.device)[selected_local_index],
        "objective_margin": margin.detach(),
        "uses_positive_reference": routing_code.detach(),
    }


def compute_l13_most_divergent_failure_scores_from_cosine(
    negative_positive_cosine: torch.Tensor,
    negative_negative_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rank wrong siblings using L13 Most-Divergent Failure Selection.

    The distance is ``1 - mean(cosine[1:15])``. For Mixed prompts each wrong
    rollout is scored by its maximum distance from a correct rollout; wrong/
    wrong distances do not participate. For 0P4N, the score is its mean
    distance from the other wrong rollouts. Both routes maximize geometry
    multiplied by L8's prompt-local response-length efficiency.
    """
    if negative_positive_cosine.ndim != 3 or negative_positive_cosine.shape[-1] != 16:
        raise ValueError("L13 negative_positive_cosine must have shape [N_neg, N_pos, 16]")
    if negative_negative_cosine.ndim != 3 or negative_negative_cosine.shape[-1] != 16:
        raise ValueError("L13 negative_negative_cosine must have shape [N_neg, N_neg, 16]")
    num_negative, num_positive, _ = negative_positive_cosine.shape
    if num_negative < 1 or negative_negative_cosine.shape[:2] != (num_negative, num_negative):
        raise ValueError("L13 requires a non-empty square wrong/wrong cosine matrix")
    if negative_valid_mask.shape != (num_negative, 16):
        raise ValueError("L13 negative_valid_mask must have shape [N_neg, 16]")
    if positive_valid_mask.shape != (num_positive, 16):
        raise ValueError("L13 positive_valid_mask must have shape [N_pos, 16]")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("L13 negative_response_lengths and negative_rollout_ids must have shape [N_neg]")

    neg_pos = negative_positive_cosine.detach().float().clamp(-1.0, 1.0)
    neg_neg = negative_negative_cosine.detach().float().clamp(-1.0, 1.0)
    response_lengths = negative_response_lengths.detach().to(device=neg_pos.device, dtype=torch.float32)
    if not torch.isfinite(neg_pos).all() or not torch.isfinite(neg_neg).all():
        raise FloatingPointError("L13 cosine contains NaN or Inf")
    if not torch.isfinite(response_lengths).all() or bool((response_lengths <= 0).any()):
        raise ValueError("L13 response lengths must be finite and positive")

    neg_valid = negative_valid_mask.detach().bool()[:, 1:15].all(dim=-1)
    pos_valid = positive_valid_mask.detach().bool()[:, 1:15].all(dim=-1)
    neg_pos_distance = (1.0 - neg_pos[:, :, 1:15].mean(dim=-1)).clamp(0.0, 2.0)
    neg_neg_distance = (1.0 - neg_neg[:, :, 1:15].mean(dim=-1)).clamp(0.0, 2.0)

    if num_positive:
        pair_valid = neg_valid[:, None] & pos_valid[None, :]
        masked = neg_pos_distance.masked_fill(~pair_valid, -float("inf"))
        reference_local_index = masked.argmax(dim=1)
        row = torch.arange(num_negative, device=masked.device)
        geometry = masked[row, reference_local_index]
        reference_similarity = neg_pos[row, reference_local_index]
        valid_candidate_mask = torch.isfinite(geometry)
        routing_code = torch.ones((), dtype=torch.long, device=masked.device)
    else:
        if num_negative < 2:
            raise ValueError("L13 failure outlier requires at least two wrong siblings")
        pair_valid = neg_valid[:, None] & neg_valid[None, :]
        diagonal = torch.eye(num_negative, dtype=torch.bool, device=neg_neg.device)
        off_diagonal_valid = pair_valid & ~diagonal
        valid_candidate_mask = off_diagonal_valid.sum(dim=1) == num_negative - 1
        geometry = neg_neg_distance.masked_fill(~off_diagonal_valid, 0.0).sum(dim=1) / (num_negative - 1)
        farthest_other = neg_neg_distance.masked_fill(~off_diagonal_valid, -float("inf"))
        reference_local_index = farthest_other.argmax(dim=1)
        row = torch.arange(num_negative, device=farthest_other.device)
        reference_similarity = neg_neg[row, reference_local_index]
        geometry = geometry.masked_fill(~valid_candidate_mask, -float("inf"))
        routing_code = torch.zeros((), dtype=torch.long, device=neg_neg.device)

    if not valid_candidate_mask.any():
        raise ValueError("L13 has no candidate with 14 valid interior slots")
    length_range = response_lengths.max() - response_lengths.min()
    normalized_response_length = (response_lengths - response_lengths.min()) / (
        length_range + L8_LENGTH_NORMALIZATION_EPSILON
    )
    length_efficiency = 1.0 - normalized_response_length
    objective = (geometry * length_efficiency).masked_fill(~valid_candidate_mask, -float("inf"))
    optimum = objective.max()
    tied = torch.isclose(objective, optimum, rtol=1.0e-6, atol=1.0e-7).nonzero(as_tuple=False).flatten()
    tied_efficiency = length_efficiency.index_select(0, tied)
    tied = tied[torch.isclose(tied_efficiency, tied_efficiency.max(), rtol=1.0e-6, atol=1.0e-7)]
    tied_ids = negative_rollout_ids.detach().to(device=objective.device).index_select(0, tied)
    selected_local_index = tied[tied_ids.argmin()]
    sorted_scores = objective[valid_candidate_mask].sort(descending=True).values
    margin = sorted_scores[0] - sorted_scores[1] if sorted_scores.numel() > 1 else torch.zeros_like(sorted_scores[0])
    return {
        "geometry": geometry.detach(),
        "objective": objective.detach(),
        "cost_factor": length_efficiency.detach(),
        "normalized_response_length": normalized_response_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "reference_local_index": reference_local_index.detach(),
        "reference_similarity": reference_similarity.detach(),
        "valid_candidate_mask": valid_candidate_mask.detach(),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": negative_rollout_ids.detach().to(geometry.device)[selected_local_index],
        "objective_margin": margin.detach(),
        "uses_positive_reference": routing_code.detach(),
    }


def compute_l14_peer_calibrated_scores_from_cosine(
    negative_positive_cosine: torch.Tensor,
    negative_negative_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    prompt_length: int | torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    *,
    persistence_horizon: int = 2,
) -> dict[str, torch.Tensor]:
    """Rank wrong siblings by a peer-calibrated persistent early fork.

    At every stage, ``a_plus`` is the closest correct sibling and ``a_minus``
    is the closest *other* wrong sibling.  Their half difference is the
    correctness-relative margin.  H=1 is the L14-R calibration-only ablation;
    H=2 is the full persistent L14 score.
    """
    if negative_positive_cosine.ndim != 3:
        raise ValueError("L14 negative_positive_cosine must have shape [N_neg, N_pos, M]")
    if negative_negative_cosine.ndim != 3:
        raise ValueError("L14 negative_negative_cosine must have shape [N_neg, N_neg, M]")
    num_negative, num_positive, num_stages = negative_positive_cosine.shape
    if num_negative < 2 or num_positive < 1:
        raise ValueError("L14 scoring requires at least two wrong and one correct sibling")
    if negative_negative_cosine.shape != (num_negative, num_negative, num_stages):
        raise ValueError("L14 wrong/wrong cosine must be square and stage-aligned")
    if negative_valid_mask.shape != (num_negative, num_stages):
        raise ValueError("L14 negative_valid_mask must have shape [N_neg, M]")
    if positive_valid_mask.shape != (num_positive, num_stages):
        raise ValueError("L14 positive_valid_mask must have shape [N_pos, M]")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("L14 negative lengths and rollout ids must have shape [N_neg]")
    if persistence_horizon not in {1, 2} or num_stages <= persistence_horizon:
        raise ValueError("L14 persistence_horizon must be 1 or 2 and leave at least one fork location")

    neg_pos = negative_positive_cosine.detach().float().clamp(-1.0, 1.0)
    neg_neg = negative_negative_cosine.detach().float().clamp(-1.0, 1.0)
    if not torch.isfinite(neg_pos).all() or not torch.isfinite(neg_neg).all():
        raise FloatingPointError("L14 cosine contains NaN or Inf")
    neg_valid = negative_valid_mask.detach().bool()
    pos_valid = positive_valid_mask.detach().bool()

    positive_pair_valid = neg_valid[:, None, :] & pos_valid[None, :, :]
    positive_masked = neg_pos.masked_fill(~positive_pair_valid, -float("inf"))
    a_plus, nearest_positive_index = positive_masked.max(dim=1)
    a_plus_valid = torch.isfinite(a_plus)

    negative_pair_valid = neg_valid[:, None, :] & neg_valid[None, :, :]
    diagonal = torch.eye(num_negative, dtype=torch.bool, device=neg_neg.device).unsqueeze(-1)
    negative_pair_valid &= ~diagonal
    negative_masked = neg_neg.masked_fill(~negative_pair_valid, -float("inf"))
    a_minus, nearest_negative_index = negative_masked.max(dim=1)
    a_minus_valid = torch.isfinite(a_minus)
    stage_valid = a_plus_valid & a_minus_valid
    relative_margin = 0.5 * (a_plus - a_minus)
    relative_margin = torch.where(stage_valid, relative_margin, torch.zeros_like(relative_margin))

    locations = num_stages - persistence_horizon
    pre_margin = relative_margin[:, :locations]
    post_stack = torch.stack(
        [relative_margin[:, offset : offset + locations] for offset in range(1, persistence_horizon + 1)],
        dim=-1,
    )
    post_margin = post_stack.mean(dim=-1)
    fork_valid = stage_valid[:, :locations].clone()
    for offset in range(1, persistence_horizon + 1):
        fork_valid &= stage_valid[:, offset : offset + locations]
    alignment = 1.0 + pre_margin
    fork_drop = torch.relu(pre_margin - post_margin)
    post_divergence = 1.0 - post_margin
    fork_utility = alignment * fork_drop * post_divergence
    masked_utility = fork_utility.masked_fill(~fork_valid, -float("inf"))
    boundary_utility, best_location = masked_utility.max(dim=1)
    valid_candidate_mask = torch.isfinite(boundary_utility)
    boundary_utility = torch.where(valid_candidate_mask, boundary_utility, torch.zeros_like(boundary_utility))
    row = torch.arange(num_negative, device=neg_pos.device)
    safe_location = best_location.clamp(0, locations - 1)
    best_pre_margin = pre_margin[row, safe_location]
    best_post_margin = post_margin[row, safe_location]
    best_fork_drop = fork_drop[row, safe_location]
    best_post_divergence = post_divergence[row, safe_location]
    best_positive_local_index = nearest_positive_index[row, safe_location]
    best_peer_negative_local_index = nearest_negative_index[row, safe_location]

    prompt_tokens = torch.as_tensor(prompt_length, dtype=torch.float32, device=neg_pos.device)
    if prompt_tokens.numel() != 1:
        raise ValueError("L14 prompt_length must be scalar")
    response_lengths = negative_response_lengths.detach().to(device=neg_pos.device, dtype=torch.float32)
    teacher_cost = response_lengths + prompt_tokens
    if (~torch.isfinite(teacher_cost)).any() or (teacher_cost <= 0).any():
        raise ValueError("L14 Teacher costs must be finite and positive")
    normalized_length = (response_lengths - response_lengths.min()) / (
        response_lengths.max() - response_lengths.min() + L8_LENGTH_NORMALIZATION_EPSILON
    )
    rollout_count = num_negative + num_positive
    minimum_efficiency = 1.0 / rollout_count
    length_efficiency = minimum_efficiency + (1.0 - minimum_efficiency) * (1.0 - normalized_length)
    final_score = boundary_utility * length_efficiency

    valid_scores = final_score[valid_candidate_mask]
    degenerate = not bool(valid_candidate_mask.any()) or bool((boundary_utility <= 1.0e-12).all())
    remaining = torch.ones(num_negative, dtype=torch.bool, device=neg_pos.device)
    if degenerate:
        remaining &= torch.isclose(teacher_cost, teacher_cost.min(), rtol=1.0e-6, atol=1.0e-8)
    else:
        for values, maximize in (
            (final_score, True),
            (boundary_utility, True),
            (teacher_cost, False),
            (best_post_margin, False),
            (negative_rollout_ids.detach().to(neg_pos.device), False),
        ):
            optimum = values[remaining].max() if maximize else values[remaining].min()
            remaining &= torch.isclose(values, optimum, rtol=1.0e-6, atol=1.0e-8)
            if int(remaining.sum()) == 1:
                break
    selected_local_index = remaining.nonzero(as_tuple=False).flatten()[0]
    sorted_scores = valid_scores.sort(descending=True).values
    score_margin = (
        sorted_scores[0] - sorted_scores[1] if sorted_scores.numel() > 1 else torch.zeros((), device=neg_pos.device)
    )

    zeros = torch.zeros_like(boundary_utility)
    for tensor in (best_pre_margin, best_post_margin, best_fork_drop, best_post_divergence):
        tensor.masked_fill_(~valid_candidate_mask, 0.0)
    best_positive_local_index = torch.where(
        valid_candidate_mask, best_positive_local_index, torch.full_like(best_positive_local_index, -1)
    )
    best_peer_negative_local_index = torch.where(
        valid_candidate_mask, best_peer_negative_local_index, torch.full_like(best_peer_negative_local_index, -1)
    )
    return {
        "boundary_utility": boundary_utility.detach(),
        "final_score": final_score.detach(),
        "teacher_cost": teacher_cost.detach(),
        "teacher_input_len": teacher_cost.detach(),
        "group_min_response_len": response_lengths.min().detach(),
        "group_min_teacher_input_len": teacher_cost.min().detach(),
        "cost_ratio": zeros,
        "cost_efficiency": length_efficiency.detach(),
        "normalized_response_length": normalized_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "rollout_count": torch.tensor(rollout_count, device=neg_pos.device),
        "best_positive_local_index": best_positive_local_index.detach(),
        "best_peer_negative_local_index": best_peer_negative_local_index.detach(),
        "best_split_index": torch.where(
            valid_candidate_mask, safe_location + 1, torch.full_like(safe_location, -1)
        ).detach(),
        "best_pre_similarity": best_pre_margin.detach(),
        "best_post_similarity": best_post_margin.detach(),
        "best_fork_drop": best_fork_drop.detach(),
        "best_post_divergence": best_post_divergence.detach(),
        "early_cosine": best_pre_margin.detach(),
        "middle1_cosine": best_post_margin.detach(),
        "middle2_cosine": zeros,
        "aggregated_middle_cosine": best_post_margin.detach(),
        "early_alignment": (1.0 + best_pre_margin).detach(),
        "plateau_drop": best_fork_drop.detach(),
        "plateau_divergence": best_post_divergence.detach(),
        "first_span_cosine": best_pre_margin.detach(),
        "second_span_cosine": best_post_margin.detach(),
        "pair_utility": boundary_utility.detach(),
        "best_stage_similarity": relative_margin.detach(),
        "best_stage_valid_mask": stage_valid.detach(),
        "valid_split_count": fork_valid.sum(dim=1).detach(),
        "valid_candidate_mask": valid_candidate_mask.detach(),
        "boundary_degenerate_to_cost_only": torch.tensor(degenerate, device=neg_pos.device),
        "cost_only_local_index": torch.argmin(teacher_cost).detach(),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": negative_rollout_ids.detach().to(neg_pos.device)[selected_local_index],
        "score_mode": (
            "l14_peer_calibrated_early_fork_h1"
            if persistence_horizon == 1
            else "l14_peer_calibrated_early_fork"
        ),
        "boundary_search_enabled": True,
        "a_plus": torch.where(stage_valid, a_plus, torch.zeros_like(a_plus)).detach(),
        "a_minus": torch.where(stage_valid, a_minus, torch.zeros_like(a_minus)).detach(),
        "relative_margin": relative_margin.detach(),
        "post_margin": best_post_margin.detach(),
        "persistence_horizon": torch.tensor(persistence_horizon, device=neg_pos.device),
        "score_margin": score_margin.detach(),
    }


def compute_boundary_contrast_scores_from_cosine(
    stage_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    prompt_length: int | torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
    score_mode: str = "legacy",
    window_ratio: float = 0.125,
    negative_state_entropies: torch.Tensor | None = None,
    negative_state_entropy_valid_mask: torch.Tensor | None = None,
    negative_global_entropy_mean: torch.Tensor | None = None,
    l27_lambda_cost: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Apply legacy scoring or cost-aware Early-Middle Transition Contrast."""

    if score_mode not in BOUNDARY_SCORE_MODES:
        raise ValueError(f"invalid Boundary-Contrast score mode: {score_mode!r}")
    if score_mode in L14_SCORE_MODES:
        raise ValueError("L14 requires wrong/wrong similarities; use compute_l14_peer_calibrated_scores_from_cosine")
    if stage_cosine.ndim != 3:
        raise ValueError("stage_cosine must have shape [N_neg, N_pos, M]")
    num_negative, num_positive, num_stages = stage_cosine.shape
    if num_negative < 1 or num_positive < 1 or num_stages < 2:
        raise ValueError("Boundary-Contrast requires negatives, positives, and at least two stages")
    if negative_valid_mask.shape != (num_negative, num_stages):
        raise ValueError("negative_valid_mask must align with stage_cosine")
    if positive_valid_mask.shape != (num_positive, num_stages):
        raise ValueError("positive_valid_mask must align with stage_cosine")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("negative lengths and rollout ids must have shape [N_neg]")

    cosine = stage_cosine.detach().float()
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("Boundary-Contrast stage cosine contains NaN or Inf")
    cosine = cosine.clamp(-1.0, 1.0)
    negative_valid = negative_valid_mask.detach().bool()
    positive_valid = positive_valid_mask.detach().bool()
    stage_similarity = (1.0 + cosine) / 2.0
    pair_valid = negative_valid[:, None, :] & positive_valid[None, :, :]
    l21_state_entropy = torch.zeros(
        (num_negative, num_stages + 1), dtype=cosine.dtype, device=cosine.device
    )
    l21_state_entropy_valid = torch.zeros(
        (num_negative, num_stages + 1), dtype=torch.bool, device=cosine.device
    )
    entropy_weighted_modes = (
        L21_SCORE_MODES
        | UWTD_SCORE_MODES
        | L28_SCORE_MODES
        | L29_SCORE_MODES
        | L30_SCORE_MODES
    )
    if score_mode in entropy_weighted_modes:
        if negative_state_entropies is None or negative_state_entropy_valid_mask is None:
            raise ValueError(f"{score_mode} requires aligned negative Student state entropies and validity")
        if negative_state_entropies.shape != (num_negative, num_stages + 1):
            raise ValueError("negative_state_entropies must have shape [N_neg, M+1]")
        if negative_state_entropy_valid_mask.shape != (num_negative, num_stages + 1):
            raise ValueError("entropy validity must have shape [N_neg, M+1]")
        l21_state_entropy = negative_state_entropies.detach().to(device=cosine.device, dtype=cosine.dtype)
        l21_state_entropy_valid = negative_state_entropy_valid_mask.detach().to(
            device=cosine.device, dtype=torch.bool
        )
        if not torch.isfinite(l21_state_entropy[l21_state_entropy_valid]).all():
            raise FloatingPointError("aligned Student state entropy contains NaN or Inf")
        if bool(
            (
                (l21_state_entropy[l21_state_entropy_valid] < 0.0)
                | (l21_state_entropy[l21_state_entropy_valid] > 1.0)
            ).any()
        ):
            raise ValueError("aligned Student state entropy must be normalized to [0, 1]")
    uwtd_global_entropy = torch.zeros(num_negative, dtype=cosine.dtype, device=cosine.device)
    if score_mode in UWTD_SCORE_MODES:
        if negative_global_entropy_mean is None or negative_global_entropy_mean.shape != (num_negative,):
            raise ValueError("UWTD requires negative_global_entropy_mean with shape [N_neg]")
        uwtd_global_entropy = negative_global_entropy_mean.detach().to(
            device=cosine.device, dtype=cosine.dtype
        )
        if not torch.isfinite(uwtd_global_entropy).all() or bool(
            ((uwtd_global_entropy < 0.0) | (uwtd_global_entropy > 1.0)).any()
        ):
            raise ValueError("UWTD global entropy mean must be finite and normalized to [0, 1]")
        if score_mode in L27_SCORE_MODES and (not math.isfinite(l27_lambda_cost) or l27_lambda_cost <= 0):
            raise ValueError("L27 requires l27_lambda_cost > 0")
    if score_mode == "legacy":
        similarity_sum = (stage_similarity * pair_valid.float()).cumsum(dim=-1)
        valid_count = pair_valid.long().cumsum(dim=-1)
        total_sum = similarity_sum[..., -1:]
        total_count = valid_count[..., -1:]
        pre_sum = similarity_sum[..., :-1]
        pre_count = valid_count[..., :-1]
        post_sum = total_sum - pre_sum
        post_count = total_count - pre_count
        valid_split = (pre_count > 0) & (post_count > 0)
        pre_similarity = pre_sum / pre_count.clamp_min(1)
        post_similarity = post_sum / post_count.clamp_min(1)
        fork_drop = (pre_similarity - post_similarity).clamp_min(0.0)
        post_divergence = 1.0 - post_similarity
        split_utility = (pre_similarity * fork_drop * post_divergence).masked_fill(~valid_split, -1.0)
        flat_utility = split_utility.flatten(start_dim=1)
        boundary_utility, best_flat = flat_utility.max(dim=1)
        splits_per_positive = num_stages - 1
        best_positive_local_index = torch.div(best_flat, splits_per_positive, rounding_mode="floor")
        best_split_zero_based = best_flat.remainder(splits_per_positive)
        best_split_index = best_split_zero_based + 1
    elif score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        if num_stages < 2:
            raise ValueError(f"{score_mode} requires at least two transition stages")
        if not math.isclose(window_ratio, L8_WINDOW_RATIO, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{score_mode} requires window_ratio=0.0625")

        # Scan all adjacent pairs of native centered-LM-head cosine values.
        # Candidate j compares c_j with c_{j+1}; no averaging or position prior
        # is introduced.
        c_before = cosine[..., :-1].clamp(-1.0, 1.0)
        c_after = cosine[..., 1:].clamp(-1.0, 1.0)
        valid_split = pair_valid[..., :-1] & pair_valid[..., 1:]
        adaptive_drop_all = torch.relu(c_before - c_after)
        adaptive_divergence_all = 1.0 - c_after
        drop_factor = adaptive_drop_all
        divergence_factor = adaptive_divergence_all
        if score_mode in L9_SCORE_MODES:
            # L9 removes L8's early-alignment factor. Its utility contains
            # only the abrupt fork drop and post-fork divergence.
            alignment_factor = torch.ones_like(c_before)
        else:
            alignment_factor = 1.0 + c_before
        if score_mode in {
            "l8_adaptive_early_fork_no_alignment",
            "l8_adaptive_peak_fork_length",
            "l8_adaptive_peak_fork_only",
        }:
            alignment_factor = torch.ones_like(c_before)
        elif score_mode == "l8_adaptive_early_fork_no_drop":
            drop_factor = torch.ones_like(drop_factor)
        elif score_mode == "l8_adaptive_early_fork_no_divergence":
            divergence_factor = torch.ones_like(divergence_factor)
        if score_mode in {"l8_adaptive_peak_fork_length", "l8_adaptive_peak_fork_only"}:
            divergence_factor = torch.ones_like(divergence_factor)
        # L26/L27 contain only the outward-departure factor. L21 retains the
        # original L8 three-factor geometry. L28 keeps that geometry but swaps
        # the endpoints of the alignment/divergence factors exactly as specified.
        if score_mode in UWTD_SCORE_MODES:
            unweighted_pair_utility_all = adaptive_drop_all
        elif score_mode in L28_SCORE_MODES:
            unweighted_pair_utility_all = (
                (1.0 + c_after) * adaptive_drop_all * (1.0 - c_before)
            )
        elif score_mode in L29_SCORE_MODES:
            potential_before = c_before - c_before.pow(3) / 3.0
            potential_after = c_after - c_after.pow(3) / 3.0
            unweighted_pair_utility_all = torch.relu(potential_before - potential_after)
        else:
            unweighted_pair_utility_all = alignment_factor * drop_factor * divergence_factor
        if score_mode in entropy_weighted_modes:
            # c_j is the similarity of latent transition j.  A pair
            # (c_j, c_{j+1}) is weighted by the mean normalized Student entropy
            # at the two transition-start states j and j+1.
            interval_entropy = 0.5 * (
                l21_state_entropy[:, :-2] + l21_state_entropy[:, 1:-1]
            )
            interval_entropy_valid = (
                l21_state_entropy_valid[:, :-2] & l21_state_entropy_valid[:, 1:-1]
            )
            valid_split = valid_split & interval_entropy_valid[:, None, :]
            adaptive_pair_utility_all = unweighted_pair_utility_all * interval_entropy[:, None, :]
        else:
            interval_entropy = torch.ones(
                (num_negative, num_stages - 1), dtype=cosine.dtype, device=cosine.device
            )
            adaptive_pair_utility_all = unweighted_pair_utility_all
        split_utility = adaptive_pair_utility_all.masked_fill(~valid_split, -float("inf"))
        splits_per_positive = num_stages - 1
        strongest_pair_utility, positive_best_split_zero_based = split_utility.max(dim=-1)
        positive_pair_valid = torch.isfinite(strongest_pair_utility) & (strongest_pair_utility >= 0.0)
        if score_mode in TRAJECTORY_DEPARTURE_SCORE_MODES:
            # Trajectory-departure modes aggregate every valid early interval
            # instead of retaining only the single strongest fork. Entropy-
            # weighted variants apply normalized Student entropy above.
            positive_pair_utility = torch.where(
                valid_split,
                adaptive_pair_utility_all,
                torch.zeros_like(adaptive_pair_utility_all),
            ).sum(dim=-1)
            raw_positive_pair_utility = positive_pair_utility.masked_fill(
                ~positive_pair_valid,
                -float("inf"),
            )
        else:
            raw_positive_pair_utility = strongest_pair_utility
            positive_pair_utility = torch.where(
                positive_pair_valid,
                raw_positive_pair_utility,
                torch.zeros_like(raw_positive_pair_utility),
            )
        positive_max_utility, best_positive_local_index = raw_positive_pair_utility.max(dim=1)
        best_split_zero_based = positive_best_split_zero_based.gather(
            1, best_positive_local_index.unsqueeze(1)
        ).squeeze(1)
        if score_mode in L16_SCORE_MODES:
            boundary_utility = positive_pair_utility.mean(dim=1)
            boundary_utility = boundary_utility.masked_fill(~positive_pair_valid.any(dim=1), -float("inf"))
        else:
            boundary_utility = positive_max_utility
        best_split_index = best_split_zero_based + 1
        pre_similarity = c_before
        post_similarity = c_after
        fork_drop = adaptive_drop_all
        post_divergence = adaptive_divergence_all
    else:  # pragma: no cover - score_mode validation makes this unreachable
        raise AssertionError(f"unreachable Boundary-Contrast score mode: {score_mode!r}")
    valid_candidate_mask = boundary_utility >= 0.0
    boundary_utility = torch.where(valid_candidate_mask, boundary_utility, torch.zeros_like(boundary_utility))
    candidate_index = torch.arange(num_negative, device=cosine.device)
    best_pre_similarity = pre_similarity[candidate_index, best_positive_local_index, best_split_zero_based]
    best_post_similarity = post_similarity[candidate_index, best_positive_local_index, best_split_zero_based]
    best_fork_drop = fork_drop[candidate_index, best_positive_local_index, best_split_zero_based]
    best_post_divergence = post_divergence[candidate_index, best_positive_local_index, best_split_zero_based]
    if score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        best_first_span_cosine = c_before[candidate_index, best_positive_local_index, best_split_zero_based]
        best_second_span_cosine = c_after[candidate_index, best_positive_local_index, best_split_zero_based]
        best_early_cosine = best_first_span_cosine
        best_middle1_cosine = best_second_span_cosine
        best_middle2_cosine = torch.zeros_like(boundary_utility)
        best_aggregated_middle_cosine = torch.zeros_like(boundary_utility)
        best_early_alignment = 1.0 + best_first_span_cosine
        best_plateau_drop = adaptive_drop_all[candidate_index, best_positive_local_index, best_split_zero_based]
        best_plateau_divergence = adaptive_divergence_all[
            candidate_index, best_positive_local_index, best_split_zero_based
        ]
        best_pair_utility = adaptive_pair_utility_all[candidate_index, best_positive_local_index, best_split_zero_based]
    else:
        best_early_cosine = torch.zeros_like(boundary_utility)
        best_middle1_cosine = torch.zeros_like(boundary_utility)
        best_middle2_cosine = torch.zeros_like(boundary_utility)
        best_aggregated_middle_cosine = torch.zeros_like(boundary_utility)
        best_early_alignment = torch.zeros_like(boundary_utility)
        best_plateau_drop = torch.zeros_like(boundary_utility)
        best_plateau_divergence = torch.zeros_like(boundary_utility)
        best_pair_utility = torch.zeros_like(boundary_utility)
        best_first_span_cosine = torch.zeros_like(boundary_utility)
        best_second_span_cosine = torch.zeros_like(boundary_utility)
    # Full per-stage profile against the best positive, exported for the
    # Boundary-Contrast per-candidate CSV.
    best_stage_similarity = stage_similarity[candidate_index, best_positive_local_index, :]
    best_stage_valid = pair_valid[candidate_index, best_positive_local_index, :]
    zeros = torch.zeros_like(boundary_utility)
    best_pre_similarity = torch.where(valid_candidate_mask, best_pre_similarity, zeros)
    best_post_similarity = torch.where(valid_candidate_mask, best_post_similarity, zeros)
    best_fork_drop = torch.where(valid_candidate_mask, best_fork_drop, zeros)
    best_post_divergence = torch.where(valid_candidate_mask, best_post_divergence, zeros)
    best_early_cosine = torch.where(valid_candidate_mask, best_early_cosine, zeros)
    best_middle1_cosine = torch.where(valid_candidate_mask, best_middle1_cosine, zeros)
    best_middle2_cosine = torch.where(valid_candidate_mask, best_middle2_cosine, zeros)
    best_aggregated_middle_cosine = torch.where(valid_candidate_mask, best_aggregated_middle_cosine, zeros)
    best_early_alignment = torch.where(valid_candidate_mask, best_early_alignment, zeros)
    best_plateau_drop = torch.where(valid_candidate_mask, best_plateau_drop, zeros)
    best_plateau_divergence = torch.where(valid_candidate_mask, best_plateau_divergence, zeros)
    best_pair_utility = torch.where(valid_candidate_mask, best_pair_utility, zeros)
    best_first_span_cosine = torch.where(valid_candidate_mask, best_first_span_cosine, zeros)
    best_second_span_cosine = torch.where(valid_candidate_mask, best_second_span_cosine, zeros)
    if score_mode in entropy_weighted_modes:
        chosen_interval_entropy = interval_entropy[candidate_index, best_split_zero_based]
        chosen_valid_split = valid_split[candidate_index, best_positive_local_index]
        trajectory_entropy_mean = (
            (interval_entropy * chosen_valid_split.float()).sum(dim=-1)
            / chosen_valid_split.sum(dim=-1).clamp_min(1)
        )
        unweighted_trajectory_utility = (
            unweighted_pair_utility_all[candidate_index, best_positive_local_index]
            * chosen_valid_split.float()
        ).sum(dim=-1)
        chosen_interval_entropy = torch.where(valid_candidate_mask, chosen_interval_entropy, zeros)
        trajectory_entropy_mean = torch.where(valid_candidate_mask, trajectory_entropy_mean, zeros)
        unweighted_trajectory_utility = torch.where(
            valid_candidate_mask, unweighted_trajectory_utility, zeros
        )
        valid_departure_interval_count = chosen_valid_split.sum(dim=-1)
    else:
        chosen_interval_entropy = zeros
        trajectory_entropy_mean = zeros
        unweighted_trajectory_utility = zeros
        valid_departure_interval_count = torch.zeros_like(boundary_utility, dtype=torch.long)
    best_stage_similarity = torch.where(
        valid_candidate_mask.unsqueeze(-1),
        best_stage_similarity,
        torch.zeros_like(best_stage_similarity),
    )
    best_stage_valid = best_stage_valid & valid_candidate_mask.unsqueeze(-1)
    best_positive_local_index = torch.where(
        valid_candidate_mask, best_positive_local_index, torch.full_like(best_positive_local_index, -1)
    )
    best_split_index = torch.where(valid_candidate_mask, best_split_index, torch.full_like(best_split_index, -1))

    prompt_tokens = torch.as_tensor(prompt_length, dtype=torch.float32, device=cosine.device)
    if prompt_tokens.numel() != 1:
        raise ValueError("prompt_length must be scalar for one Frontier prompt")
    teacher_cost = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32) + prompt_tokens
    if (~torch.isfinite(teacher_cost)).any() or (teacher_cost <= 0).any():
        raise ValueError("Boundary-OPD Teacher costs must be finite and positive")
    minimum_response_length = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32).min()
    minimum_cost = teacher_cost.min()
    normalized_response_length = torch.zeros_like(teacher_cost)
    cost_ratio = torch.zeros_like(teacher_cost)
    if score_mode in UWTD_SCORE_MODES:
        response_lengths_float = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32)
        normalized_response_length = (response_lengths_float - response_lengths_float.min()) / (
            response_lengths_float.max() - response_lengths_float.min() + UWTD_EPSILON
        )
        cost_efficiency = torch.ones_like(teacher_cost)
    elif score_mode in {"l8_adaptive_early_fork_no_length_efficiency", "l8_adaptive_peak_fork_only"}:
        cost_efficiency = torch.ones_like(teacher_cost)
    elif score_mode in L10_SCORE_MODES:
        # L10 changes only L8's cost term. Use effective (un-padded) prompt
        # and response token counts to measure utility per Teacher input cost.
        min_teacher_input_len = minimum_cost
        teacher_input_len = teacher_cost
        cost_ratio = min_teacher_input_len / teacher_input_len
        cost_efficiency = cost_ratio
    elif score_mode in L30_SCORE_MODES:
        response_lengths_float = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32)
        length_range = response_lengths_float.max() - response_lengths_float.min()
        normalized_response_length = (response_lengths_float - response_lengths_float.min()) / (
            length_range + L8_LENGTH_NORMALIZATION_EPSILON
        )
        cost_efficiency = minimum_response_length / (
            response_lengths_float + L8_LENGTH_NORMALIZATION_EPSILON
        )
        cost_ratio = cost_efficiency
    elif score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        response_lengths_float = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32)
        length_range = response_lengths_float.max() - response_lengths_float.min()
        normalized_response_length = (response_lengths_float - response_lengths_float.min()) / (
            length_range + L8_LENGTH_NORMALIZATION_EPSILON
        )
        if score_mode in L9_SCORE_MODES:
            rollout_count = num_negative + num_positive
            minimum_efficiency = 1.0 / rollout_count
            cost_efficiency = minimum_efficiency + (1.0 - minimum_efficiency) * (
                1.0 - normalized_response_length
            )
        else:
            cost_efficiency = 1.0 - normalized_response_length
    else:
        cost_efficiency = minimum_cost / teacher_cost
    if score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        positive_pair_scores = positive_pair_utility * cost_efficiency.unsqueeze(1)
        positive_score_mean = positive_pair_scores.mean(dim=1)
        positive_score_max = positive_pair_scores.max(dim=1).values
        positive_score_std = positive_pair_scores.std(dim=1, unbiased=False)
    else:
        positive_pair_scores = torch.zeros(
            num_negative, num_positive, dtype=boundary_utility.dtype, device=boundary_utility.device
        )
        positive_score_mean = torch.zeros_like(boundary_utility)
        positive_score_max = torch.zeros_like(boundary_utility)
        positive_score_std = torch.zeros_like(boundary_utility)
    normalized_teaching_value = torch.zeros_like(boundary_utility)
    normalized_value_within_prompt = torch.zeros_like(boundary_utility)
    if score_mode in UWTD_SCORE_MODES:
        normalized_teaching_value = boundary_utility / (uwtd_global_entropy + UWTD_EPSILON)
        if score_mode in L26_SCORE_MODES:
            final_score = normalized_teaching_value
        else:
            value_min = normalized_teaching_value.min()
            normalized_value_within_prompt = (normalized_teaching_value - value_min) / (
                normalized_teaching_value.max() - value_min + UWTD_EPSILON
            )
            final_score = normalized_value_within_prompt - l27_lambda_cost * normalized_response_length
    else:
        final_score = (
            positive_score_mean if score_mode in L16_SCORE_MODES else boundary_utility * cost_efficiency
        )
    if not all(torch.isfinite(value).all() for value in (boundary_utility, cost_efficiency, final_score)):
        raise FloatingPointError("Boundary-Contrast score contains NaN or Inf")
    if score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES:
        zero_utility = boundary_utility <= 1.0e-12
    else:
        zero_utility = torch.isclose(
            boundary_utility,
            torch.zeros_like(boundary_utility),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    degenerate = bool((~valid_candidate_mask).all() or zero_utility.all())

    remaining = torch.ones(num_negative, dtype=torch.bool, device=cosine.device)
    if score_mode in L16_SCORE_MODES:
        remaining &= final_score == final_score.max()
        rollout_ids_on_device = negative_rollout_ids.detach().to(device=cosine.device)
        minimum_rollout_id = rollout_ids_on_device[remaining].min()
        remaining &= rollout_ids_on_device == minimum_rollout_id
    elif score_mode in UWTD_SCORE_MODES:
        remaining &= torch.isclose(final_score, final_score.max(), rtol=1.0e-6, atol=1.0e-8)
        rollout_ids_on_device = negative_rollout_ids.detach().to(device=cosine.device)
        remaining &= rollout_ids_on_device == rollout_ids_on_device[remaining].min()
    elif degenerate:
        best_cost = teacher_cost.min()
        remaining &= torch.isclose(teacher_cost, best_cost, rtol=1.0e-6, atol=1.0e-8)
    else:
        for values, maximize in (
            (final_score, True),
            (boundary_utility, True),
            (teacher_cost, False),
            (best_post_similarity, False),
            (negative_rollout_ids.detach().to(device=cosine.device), False),
        ):
            eligible = values[remaining]
            optimum = eligible.max() if maximize else eligible.min()
            remaining &= torch.isclose(values, optimum, rtol=1.0e-6, atol=1.0e-8)
            if int(remaining.sum()) == 1:
                break
    selected_local_index = remaining.nonzero(as_tuple=False).flatten()[0]
    valid_split_count = valid_split.sum(dim=(1, 2))
    return {
        "boundary_utility": boundary_utility.detach(),
        "teacher_cost": teacher_cost.detach(),
        "teacher_input_len": teacher_cost.detach(),
        "group_min_response_len": minimum_response_length.detach(),
        "group_min_teacher_input_len": minimum_cost.detach(),
        "cost_ratio": cost_ratio.detach(),
        "cost_efficiency": cost_efficiency.detach(),
        "final_score": final_score.detach(),
        "positive_pair_scores": positive_pair_scores.detach(),
        "positive_score_mean": positive_score_mean.detach(),
        "positive_score_max": positive_score_max.detach(),
        "positive_score_std": positive_score_std.detach(),
        "best_positive_local_index": best_positive_local_index.detach(),
        "best_split_index": best_split_index.detach(),
        "best_pre_similarity": best_pre_similarity.detach(),
        "best_post_similarity": best_post_similarity.detach(),
        "best_fork_drop": best_fork_drop.detach(),
        "best_post_divergence": best_post_divergence.detach(),
        "early_cosine": best_early_cosine.detach(),
        "middle1_cosine": best_middle1_cosine.detach(),
        "middle2_cosine": best_middle2_cosine.detach(),
        "aggregated_middle_cosine": best_aggregated_middle_cosine.detach(),
        "early_alignment": best_early_alignment.detach(),
        "plateau_drop": best_plateau_drop.detach(),
        "plateau_divergence": best_plateau_divergence.detach(),
        "first_span_cosine": best_first_span_cosine.detach(),
        "second_span_cosine": best_second_span_cosine.detach(),
        "pair_utility": best_pair_utility.detach(),
        "interval_entropy": chosen_interval_entropy.detach(),
        "trajectory_entropy_mean": trajectory_entropy_mean.detach(),
        "unweighted_trajectory_utility": unweighted_trajectory_utility.detach(),
        "trajectory_departure_utility": boundary_utility.detach(),
        "global_entropy_mean": uwtd_global_entropy.detach(),
        "normalized_teaching_value": normalized_teaching_value.detach(),
        "normalized_teaching_value_within_prompt": normalized_value_within_prompt.detach(),
        "normalized_cost": normalized_response_length.detach(),
        "valid_state_count": l21_state_entropy_valid.sum(dim=-1).detach(),
        "valid_transition_count": negative_valid.sum(dim=-1).detach(),
        "valid_departure_interval_count": valid_departure_interval_count.detach(),
        "best_stage_similarity": best_stage_similarity.detach(),
        "best_stage_valid_mask": best_stage_valid.detach(),
        "valid_split_count": valid_split_count.detach(),
        "valid_candidate_mask": valid_candidate_mask.detach(),
        "boundary_degenerate_to_cost_only": torch.tensor(
            degenerate and score_mode not in UWTD_SCORE_MODES, device=cosine.device
        ),
        "cost_only_local_index": torch.argmin(teacher_cost).detach(),
        "normalized_response_length": normalized_response_length.detach(),
        "length_efficiency": (
            cost_efficiency.detach()
            if score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES
            else torch.zeros_like(cost_efficiency)
        ),
        "rollout_count": torch.tensor(num_negative + num_positive, device=cosine.device),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": negative_rollout_ids.detach().to(device=cosine.device)[selected_local_index],
        "score_mode": score_mode,
        "boundary_search_enabled": score_mode == "legacy" or score_mode in ADAPTIVE_EARLY_FORK_SCORE_MODES,
        "window_ratio": window_ratio,
    }


def compute_frontier_weight(num_positive: int, num_rollouts: int) -> float:
    """L15 frontier weight G_x = 4 p_x (1 - p_x) with p_x = |P_x| / K.

    Peaks at the 2P2N decision boundary (G=1), halves for 3P1N/1P3N
    (G=0.75), and vanishes for saturated prompts (4P0N/0P4N, G=0).
    """
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    if not 0 <= num_positive <= num_rollouts:
        raise ValueError("num_positive must be within [0, num_rollouts]")
    p_x = num_positive / float(num_rollouts)
    return 4.0 * p_x * (1.0 - p_x)


def _l31_dtw_path(cost: torch.Tensor, band_ratio: float) -> tuple[list[tuple[int, int]], float]:
    """Return a deterministic banded-DTW path and its mean local cost."""

    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError("L31 DTW cost must be a non-empty matrix")
    
    # Sanitize cost matrix: replace NaN with 0, clamp Inf to [0, 1], ensure non-negative
    sanitized_cost = cost.clone()
    sanitized_cost = torch.nan_to_num(sanitized_cost, nan=0.0, posinf=1.0, neginf=0.0)
    sanitized_cost = sanitized_cost.clamp(0.0, 1.0)
    
    if not torch.isfinite(sanitized_cost).all():
        raise ValueError("L31 DTW cost contains non-finite values after sanitization")
    rows, columns = sanitized_cost.shape
    band = max(abs(rows - columns), int(math.ceil(max(rows, columns) * band_ratio)))
    inf = float("inf")
    accumulated = [[inf] * (columns + 1) for _ in range(rows + 1)]
    predecessor: list[list[tuple[int, int] | None]] = [
        [None] * (columns + 1) for _ in range(rows + 1)
    ]
    accumulated[0][0] = 0.0
    for row in range(1, rows + 1):
        lower = max(1, row - band)
        upper = min(columns, row + band)
        for column in range(lower, upper + 1):
            # Diagonal is the preferred tie-break, followed by advancing the
            # failed trajectory and then the successful trajectory.
            choices = (
                (accumulated[row - 1][column - 1], row - 1, column - 1, 0),
                (accumulated[row - 1][column], row - 1, column, 1),
                (accumulated[row][column - 1], row, column - 1, 2),
            )
            best_value, prev_row, prev_column, _ = min(choices, key=lambda item: (item[0], item[3]))
            if math.isfinite(best_value):
                accumulated[row][column] = best_value + float(sanitized_cost[row - 1, column - 1].item())
                predecessor[row][column] = (prev_row, prev_column)
    if not math.isfinite(accumulated[rows][columns]):
        raise ValueError("L31 DTW band produced no legal alignment path")
    path: list[tuple[int, int]] = []
    row, column = rows, columns
    while row > 0 or column > 0:
        path.append((row - 1, column - 1))
        previous = predecessor[row][column]
        if previous is None:
            raise RuntimeError("L31 DTW predecessor chain is incomplete")
        row, column = previous
    path.reverse()
    return path, accumulated[rows][columns] / len(path)


def l31_response_token_position(state_index: int, response_length: int, num_states: int = 16) -> int:
    """Map a whole-response L31 state slot to a zero-based response token."""

    if not 0 <= state_index < num_states:
        raise ValueError("L31 state_index is out of range")
    if response_length <= 0:
        raise ValueError("L31 response_length must be positive")
    return int(round(state_index * (response_length - 1) / (num_states - 1)))


def compute_l31_positive_sibling_structure(
    states: torch.Tensor,
    state_valid_mask: torch.Tensor,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    rollout_ids: Sequence[int],
    response_lengths: Sequence[int],
    *,
    band_ratio: float = 0.20,
    fork_threshold: float = 0.35,
    fork_persistence: int = 2,
) -> list[dict[str, Any]]:
    """Match each failed rollout to its nearest successful Student sibling.

    The local DTW cost is 0.75 state cosine distance plus 0.25 normalized
    transition cosine distance.  All inputs are Student-side tensors.
    """

    if states.ndim != 3 or states.shape[1] != 16:
        raise ValueError("L31 states must have shape [K, 16, D]")
    if state_valid_mask.shape != states.shape[:2]:
        raise ValueError("L31 state_valid_mask must have shape [K, 16]")
    if len(rollout_ids) != states.shape[0] or len(response_lengths) != states.shape[0]:
        raise ValueError("L31 rollout metadata must align with states")
    if not positive_indices or not negative_indices:
        raise ValueError("L31 requires at least one positive and one negative rollout")
    normalized = F.normalize(states.detach().float(), dim=-1, eps=1.0e-6)
    results: list[dict[str, Any]] = []
    for negative_index in negative_indices:
        candidates: list[dict[str, Any]] = []
        negative_valid_slots = state_valid_mask[negative_index].bool().nonzero(as_tuple=False).flatten()
        if negative_valid_slots.numel() < 2:
            continue
        negative_states = normalized[negative_index].index_select(0, negative_valid_slots)
        for positive_index in positive_indices:
            positive_valid_slots = state_valid_mask[positive_index].bool().nonzero(as_tuple=False).flatten()
            if positive_valid_slots.numel() < 2:
                continue
            positive_states = normalized[positive_index].index_select(0, positive_valid_slots)
            state_cost = 0.5 * (1.0 - negative_states @ positive_states.transpose(0, 1))
            negative_raw_delta = negative_states[1:] - negative_states[:-1]
            positive_raw_delta = positive_states[1:] - positive_states[:-1]
            negative_delta = F.normalize(negative_raw_delta, dim=-1, eps=1.0e-6)
            positive_delta = F.normalize(positive_raw_delta, dim=-1, eps=1.0e-6)
            transition_cost = torch.zeros_like(state_cost)
            transition_similarity = negative_delta @ positive_delta.transpose(0, 1)
            both_stationary = (negative_raw_delta.norm(dim=-1) <= 1.0e-6).unsqueeze(1) & (
                positive_raw_delta.norm(dim=-1) <= 1.0e-6
            ).unsqueeze(0)
            transition_similarity = torch.where(
                both_stationary,
                torch.ones_like(transition_similarity),
                transition_similarity,
            )
            transition_cost[1:, 1:] = 0.5 * (1.0 - transition_similarity)
            # Sanitize transition_cost to handle numerical instability
            transition_cost = torch.nan_to_num(transition_cost, nan=0.0, posinf=1.0, neginf=0.0)
            transition_cost = transition_cost.clamp(0.0, 1.0)
            
            local_cost = (0.75 * state_cost + 0.25 * transition_cost).clamp(0.0, 1.0)
            # A path cell without two preceding states has no well-defined
            # transition comparison, so it uses the full state cost rather
            # than silently shrinking it by the 0.75 state weight.
            local_cost[0, :] = state_cost[0, :]
            local_cost[:, 0] = state_cost[:, 0]
            
            # Final sanitization to ensure local_cost is valid
            local_cost = torch.nan_to_num(local_cost, nan=0.0, posinf=1.0, neginf=0.0)
            local_cost = local_cost.clamp(0.0, 1.0)
            path, distance = _l31_dtw_path(local_cost, band_ratio)
            path_costs = [float(local_cost[row, column].item()) for row, column in path]
            fork_path_index = -1
            for path_index in range(0, len(path_costs) - fork_persistence + 1):
                sustained = path_costs[path_index : path_index + fork_persistence]
                if all(value > fork_threshold for value in sustained):
                    fork_path_index = path_index
                    break
            fork_fallback = fork_path_index < 0
            if fork_fallback:
                jumps = [path_costs[index] - path_costs[index - 1] for index in range(1, len(path_costs))]
                fork_path_index = 1 + max(range(len(jumps)), key=lambda index: (jumps[index], -index))
            negative_path_slot, positive_path_slot = path[fork_path_index]
            negative_state_index = int(negative_valid_slots[negative_path_slot].item())
            positive_state_index = int(positive_valid_slots[positive_path_slot].item())
            candidates.append(
                {
                    "positive_index": int(positive_index),
                    "positive_rollout_id": int(rollout_ids[positive_index]),
                    "distance": float(distance),
                    "negative_state_index": negative_state_index,
                    "positive_state_index": positive_state_index,
                    "negative_fork_token": l31_response_token_position(
                        negative_state_index, int(response_lengths[negative_index])
                    ),
                    "positive_fork_token": l31_response_token_position(
                        positive_state_index, int(response_lengths[positive_index])
                    ),
                    "fork_fallback": int(fork_fallback),
                    "alignment_path_length": len(path),
                }
            )
        if candidates:
            best = min(
                candidates,
                key=lambda item: (
                    item["distance"],
                    abs(
                        int(response_lengths[negative_index])
                        - int(response_lengths[item["positive_index"]])
                    ),
                    item["positive_rollout_id"],
                ),
            )
            results.append({"negative_index": int(negative_index), **best})
    return results


def select_l31_candidate(
    candidates: Sequence[dict[str, Any]],
    *,
    distance_min: float,
    distance_max: float,
    reachability_threshold: float,
) -> tuple[int, str]:
    """Select one L31 candidate using the preregistered fallback cascade."""

    if not candidates:
        raise ValueError("L31 requires at least one candidate")
    valid = [candidate for candidate in candidates if candidate.get("valid", True)]
    if not valid:
        cheapest = min(
            enumerate(candidates),
            key=lambda item: (float(item[1]["teacher_cost"]), int(item[1]["rollout_id"])),
        )
        return cheapest[0], "cost-fallback"

    def indexed(rows: Sequence[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        identities = {id(row) for row in rows}
        return [(index, row) for index, row in enumerate(candidates) if id(row) in identities]

    in_distance = [row for row in valid if distance_min <= float(row["distance"]) <= distance_max]
    strict = [row for row in in_distance if float(row["reachability"]) >= reachability_threshold]
    center = 0.5 * (distance_min + distance_max)
    if strict:
        pool, route = indexed(strict), "strict"
        key = lambda item: (
            -float(item[1]["misranking"]),
            -float(item[1]["reachability"]),
            abs(float(item[1]["distance"]) - center),
            float(item[1]["teacher_cost"]),
            int(item[1]["rollout_id"]),
        )
    elif in_distance:
        pool, route = indexed(in_distance), "relax-C"
        key = lambda item: (
            -float(item[1]["reachability"]),
            -float(item[1]["misranking"]),
            float(item[1]["teacher_cost"]),
            int(item[1]["rollout_id"]),
        )
    else:
        reachable = [row for row in valid if float(row["reachability"]) >= reachability_threshold]
        if reachable:
            pool, route = indexed(reachable), "relax-D"
            key = lambda item: (
                max(distance_min - float(item[1]["distance"]), 0.0)
                + max(float(item[1]["distance"]) - distance_max, 0.0),
                -float(item[1]["misranking"]),
                float(item[1]["teacher_cost"]),
                int(item[1]["rollout_id"]),
            )
        else:
            pool, route = indexed(valid), "all-valid"
            key = lambda item: (
                -float(item[1]["misranking"]),
                -float(item[1]["reachability"]),
                float(item[1]["distance"]),
                float(item[1]["teacher_cost"]),
                int(item[1]["rollout_id"]),
            )
    return min(pool, key=key)[0], route


def l15_query_budget(num_prompts: int, num_rollouts: int, ratio: float = L15_TEACHER_QUERY_RATIO) -> int:
    """L15 Teacher-query budget B = floor(r * N_prompt * K).

    Full OPD issues one Teacher query per (prompt, rollout) pair, so the
    natural cost baseline is B_full = N_prompt * K.  L15 fixes its
    Teacher-query ratio at ``ratio`` (10%): with the training configuration
    N=64, K=4 the budget is floor(0.1 * 256) = 25 queries per step, i.e. at
    most 9.77% of Full OPD.  Both batch dimensions come from the actual
    step, never hardcoded, so other configurations (e.g. N=128, K=8) need
    no code change.
    """
    if num_prompts < 0 or num_rollouts < 0:
        raise ValueError("num_prompts and num_rollouts must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return int(ratio * num_prompts * num_rollouts)


def allocate_l15_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    query_budget: int,
    max_queries_per_prompt: int = L15_MAX_QUERIES_PER_PROMPT,
) -> tuple[list[int], dict[str, float]]:
    """Global Top-B L15 Teacher-query allocation over the candidate pool.

    Each candidate dict must carry the L15 candidate schema: ``prompt_id``,
    ``rollout_id``, ``teacher_cost`` (prompt + response tokens; reported for
    the token-cost tables, not part of the budget), ``bc_utility`` (U_BC),
    ``cost_efficiency`` (C_i), ``l8_score`` (S_BC = U_BC * C_i),
    ``frontier_weight`` (G_x), ``l15_score`` (G_x * U_BC * C_i),
    ``num_positive`` and ``is_l8_winner``.

    Every Teacher query costs exactly one budget unit, so the constrained
    problem max sum S_L15 z subject to sum z <= B and at most
    ``max_queries_per_prompt`` queries per prompt is solved by a global
    Top-B pick in descending ``l15_score`` with a fixed tie-break
    (l15_score desc -> l8_score desc -> prompt_id asc -> rollout_id asc).
    Zero-score candidates (saturated 4P0N/0P4N prompts or degenerate
    zero-utility groups) are never selected: the budget is an upper bound,
    not a fill target.

    Returns the selected candidate positions (indices into ``candidates``)
    and a flat stats dict for step-level metric logging.
    """
    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    if max_queries_per_prompt < 1:
        raise ValueError("max_queries_per_prompt must be positive")

    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates[position]["l15_score"]),
            -float(candidates[position]["l8_score"]),
            str(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )

    selected: list[int] = []
    per_prompt_count: dict[str, int] = {}
    for position in order:
        if len(selected) >= query_budget:
            break
        candidate = candidates[position]
        if float(candidate["l15_score"]) <= 0.0:
            continue
        prompt_id = str(candidate["prompt_id"])
        if per_prompt_count.get(prompt_id, 0) >= max_queries_per_prompt:
            continue
        selected.append(position)
        per_prompt_count[prompt_id] = per_prompt_count.get(prompt_id, 0) + 1

    eligible_prompts = {str(candidate["prompt_id"]) for candidate in candidates}
    selected_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for position in selected:
        selected_by_prompt.setdefault(str(candidates[position]["prompt_id"]), []).append(candidates[position])
    # Agreement with the L8 winner set is measured on (prompt_id, rollout_id)
    # pairs: rollout ids are only unique within a prompt.
    l8_winner_keys = {
        (str(candidate["prompt_id"]), int(candidate["rollout_id"]))
        for candidate in candidates
        if candidate.get("is_l8_winner")
    }
    selected_keys = {
        (str(candidates[position]["prompt_id"]), int(candidates[position]["rollout_id"]))
        for position in selected
    }
    agreement = (
        len(selected_keys & l8_winner_keys) / len(l8_winner_keys) if l8_winner_keys else 0.0
    )
    query_group_counts = {"3p1n": 0, "2p2n": 0, "1p3n": 0}
    for position in selected:
        num_positive = int(candidates[position]["num_positive"])
        if num_positive == 3:
            query_group_counts["3p1n"] += 1
        elif num_positive == 2:
            query_group_counts["2p2n"] += 1
        elif num_positive == 1:
            query_group_counts["1p3n"] += 1

    selected_scores = [float(candidates[position]["l15_score"]) for position in selected]
    stats = {
        "query_budget": float(query_budget),
        "num_eligible_prompts": float(len(eligible_prompts)),
        "num_selected_rollouts": float(len(selected)),
        "query_budget_utilization": (len(selected) / query_budget if query_budget > 0 else 0.0),
        "selected_teacher_tokens": sum(float(candidates[position]["teacher_cost"]) for position in selected),
        "num_zero_query_prompts": float(
            sum(1 for prompt_id in eligible_prompts if not selected_by_prompt.get(prompt_id))
        ),
        "num_one_query_prompts": float(
            sum(1 for rows in selected_by_prompt.values() if len(rows) == 1)
        ),
        "num_two_query_prompts": float(
            sum(1 for rows in selected_by_prompt.values() if len(rows) == 2)
        ),
        "query_3p1n": float(query_group_counts["3p1n"]),
        "query_2p2n": float(query_group_counts["2p2n"]),
        "query_1p3n": float(query_group_counts["1p3n"]),
        "l8_query_agreement": agreement,
        # Selected rollouts outside the L8 winner set: winner replacements
        # plus the extra second rollout on doubled prompts.
        "second_rollouts_added": float(len(selected_keys - l8_winner_keys)),
        "selected_l15_score_mean": (sum(selected_scores) / len(selected_scores) if selected_scores else 0.0),
    }
    return selected, stats


def l22_token_budget(full_teacher_tokens: int, ratio: float = L22_TEACHER_TOKEN_RATIO) -> int:
    """L22 Teacher-token budget = floor(ratio * Full-OPD token total).

    Full OPD forwards every (prompt, rollout) pair, so the natural cost
    baseline is the sum of ``prompt_len + response_len`` over all rollouts of
    the step (including saturated 4P0N/0P4N groups, which Full OPD also
    scores).  L22 fixes its token ratio at ``ratio`` (10%); the realized
    ``selected_teacher_tokens / full_teacher_tokens`` is directly comparable
    with the Teacher Tokens column of the method tables.
    """
    if full_teacher_tokens < 0:
        raise ValueError("full_teacher_tokens must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return int(ratio * full_teacher_tokens)


def allocate_l22_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    token_budget: int,
    max_queries_per_prompt: int = L22_MAX_QUERIES_PER_PROMPT,
) -> tuple[list[int], dict[str, float]]:
    """Cost-aware knapsack L22 Teacher-query allocation over the candidate pool.

    The candidate schema is identical to L15 (``l15_score`` = G_x * U_BC * C_i
    is reused verbatim as the acquisition score), so any scoring difference
    between an L15 and an L22 run is impossible by construction.  Only the
    allocator changes:

    * the budget unit is Teacher input tokens (``teacher_cost`` = prompt +
      response tokens), not query count;
    * the greedy order is unit-token utility ``l15_score / teacher_cost``
      descending (with ``l15_score`` desc and the L15 id tie-break behind it),
      i.e. the classic cost-greedy heuristic for the 0/1 knapsack
      ``max sum S_c z_c  s.t.  sum cost_c z_c <= token_budget``;
    * zero-score candidates are never selected and the budget is an upper
      bound, exactly as in L15.

    Returns the selected candidate positions and a flat stats dict.  The
    fractional-knapsack upper bound on the selected-prefix score mass is
    reported as ``knapsack_bound_score_mass`` for optimality-gap auditing.
    """
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if max_queries_per_prompt < 1:
        raise ValueError("max_queries_per_prompt must be positive")
    for candidate in candidates:
        cost = float(candidate["teacher_cost"])
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("L22 candidate teacher_cost must be finite and positive")
        score = float(candidate["l15_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("L22 candidate l15_score must be finite and non-negative")

    def unit_utility(position: int) -> float:
        return float(candidates[position]["l15_score"]) / float(candidates[position]["teacher_cost"])

    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -unit_utility(position),
            -float(candidates[position]["l15_score"]),
            str(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )

    selected: list[int] = []
    per_prompt_count: dict[str, int] = {}
    spent_tokens = 0.0
    for position in order:
        if spent_tokens >= token_budget:
            break
        candidate = candidates[position]
        if float(candidate["l15_score"]) <= 0.0:
            continue
        prompt_id = str(candidate["prompt_id"])
        if per_prompt_count.get(prompt_id, 0) >= max_queries_per_prompt:
            continue
        cost = float(candidate["teacher_cost"])
        if spent_tokens + cost > token_budget:
            continue
        selected.append(position)
        per_prompt_count[prompt_id] = per_prompt_count.get(prompt_id, 0) + 1
        spent_tokens += cost

    eligible_prompts = {str(candidate["prompt_id"]) for candidate in candidates}
    selected_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for position in selected:
        selected_by_prompt.setdefault(str(candidates[position]["prompt_id"]), []).append(candidates[position])
    l8_winner_keys = {
        (str(candidate["prompt_id"]), int(candidate["rollout_id"]))
        for candidate in candidates
        if candidate.get("is_l8_winner")
    }
    selected_keys = {
        (str(candidates[position]["prompt_id"]), int(candidates[position]["rollout_id"]))
        for position in selected
    }
    agreement = (
        len(selected_keys & l8_winner_keys) / len(l8_winner_keys) if l8_winner_keys else 0.0
    )
    query_group_counts = {"3p1n": 0, "2p2n": 0, "1p3n": 0}
    for position in selected:
        num_positive = int(candidates[position]["num_positive"])
        if num_positive == 3:
            query_group_counts["3p1n"] += 1
        elif num_positive == 2:
            query_group_counts["2p2n"] += 1
        elif num_positive == 1:
            query_group_counts["1p3n"] += 1

    positive_score_mass = sum(float(candidate["l15_score"]) for candidate in candidates)
    selected_score_mass = sum(float(candidates[position]["l15_score"]) for position in selected)
    # Fractional-knapsack upper bound over the same greedy order (per-prompt
    # caps ignored; used only as an optimality-gap diagnostic).
    bound_mass = 0.0
    remaining = float(token_budget)
    for position in order:
        if remaining <= 0.0:
            break
        cost = float(candidates[position]["teacher_cost"])
        score = float(candidates[position]["l15_score"])
        take = min(1.0, remaining / cost) if cost > 0.0 else 1.0
        bound_mass += take * score
        remaining -= take * cost

    stats = {
        "token_budget": float(token_budget),
        "num_eligible_prompts": float(len(eligible_prompts)),
        "num_selected_rollouts": float(len(selected)),
        "token_budget_utilization": (spent_tokens / token_budget if token_budget > 0 else 0.0),
        "selected_teacher_tokens": spent_tokens,
        "selected_score_mass": selected_score_mass,
        "score_mass_ratio": (selected_score_mass / positive_score_mass if positive_score_mass > 0.0 else 0.0),
        "knapsack_bound_score_mass": bound_mass,
        "knapsack_gap_ratio": (
            (bound_mass - selected_score_mass) / bound_mass if bound_mass > 0.0 else 0.0
        ),
        "num_zero_query_prompts": float(
            sum(1 for prompt_id in eligible_prompts if not selected_by_prompt.get(prompt_id))
        ),
        "num_one_query_prompts": float(
            sum(1 for rows in selected_by_prompt.values() if len(rows) == 1)
        ),
        "num_two_query_prompts": float(
            sum(1 for rows in selected_by_prompt.values() if len(rows) == 2)
        ),
        "query_3p1n": float(query_group_counts["3p1n"]),
        "query_2p2n": float(query_group_counts["2p2n"]),
        "query_1p3n": float(query_group_counts["1p3n"]),
        "l8_query_agreement": agreement,
        "second_rollouts_added": float(len(selected_keys - l8_winner_keys)),
        "selected_l22_score_mean": (
            selected_score_mass / len(selected) if selected else 0.0
        ),
        "selected_unit_utility_mean": (
            sum(unit_utility(position) for position in selected) / len(selected) if selected else 0.0
        ),
    }
    return selected, stats




def l23_fork_token_position(response_length: int, best_split_index: int) -> int:
    """Convert an L8 1-based state index to its 1-based response-token position.

    Mirrors :func:`build_l8_adaptive_early_indices`: the 16 states are placed
    at ``round(j * early_end / 15)`` (0-based response-token ordinals) inside
    the first ``L/8`` window, where ``early_end = min(L, 2 * (L // 16))``.
    ``best_split_index`` is the 1-based L8 fork state (the pair (j, j+1) is
    compared at transition j), so its token position is the ordinal of state
    ``best_split_index``.  Returns a value in ``[1, response_length]``.
    """
    if response_length <= 0:
        raise ValueError("response_length must be positive")
    if not 1 <= best_split_index <= 15:
        raise ValueError("best_split_index must be a 1-based L8 state in [1, 15]")
    window_tokens = max(1, response_length // 16)
    early_end = min(response_length, 2 * window_tokens)
    zero_based = round((best_split_index - 1) * early_end / 15)
    return min(max(zero_based + 1, 1), response_length)


def l23_span_bounds(
    response_length: int,
    best_split_index: int,
    *,
    buffer_tokens: int = L23_PRE_FORK_BUFFER_TOKENS,
    min_span_tokens: int = L23_MIN_SPAN_TOKENS,
) -> tuple[int, int, bool]:
    """Return ``(span_start, span_end, fallback_full)`` for L23 supervision.

    ``span_start``/``span_end`` are 1-based inclusive response-token bounds of
    the supervised post-fork span.  ``fallback_full`` is True when the span
    would be shorter than ``min_span_tokens``; callers then supervise the full
    response (the fork location carries no usable span information).
    """
    if response_length <= 0:
        raise ValueError("response_length must be positive")
    if buffer_tokens < 0 or min_span_tokens < 0:
        raise ValueError("buffer_tokens and min_span_tokens must be non-negative")
    fork_position = l23_fork_token_position(response_length, best_split_index)
    span_start = max(fork_position - buffer_tokens, 1)
    if response_length - span_start + 1 < min_span_tokens:
        return 1, response_length, True
    return span_start, response_length, False


def l16_query_budget(
    num_prompts: int,
    num_rollouts: int,
    num_mixed_prompts: int,
    ratio: float = L16_TEACHER_QUERY_RATIO,
) -> int:
    """L16 query budget capped by the number of mixed prompts."""

    if num_prompts < 0 or num_rollouts < 0 or num_mixed_prompts < 0:
        raise ValueError("prompt, rollout, and mixed-prompt counts must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return min(int(ratio * num_prompts * num_rollouts), num_mixed_prompts)


def _l16_prompt_order(value: Any) -> tuple[int, int, str]:
    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def allocate_l16_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    query_budget: int,
) -> tuple[list[int], dict[str, float]]:
    """Select raw-score global Top-B from one L16 winner per mixed prompt."""

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    required_fields = {
        "prompt_id",
        "rollout_id",
        "num_positive",
        "num_negative",
        "response_len",
        "teacher_cost",
        "l16_score",
    }
    for candidate in candidates:
        missing = required_fields - candidate.keys()
        if missing:
            raise ValueError(f"L16 candidate is missing required fields: {sorted(missing)}")
        if not str(candidate["prompt_id"]):
            raise ValueError("L16 prompt_id must be non-empty")
        if int(candidate["rollout_id"]) < 0:
            raise ValueError("L16 rollout_id must be non-negative")
        num_positive = int(candidate["num_positive"])
        num_negative = int(candidate["num_negative"])
        if num_positive not in {1, 2, 3} or num_negative != 4 - num_positive:
            raise ValueError("L16 candidates must have a complete mixed K=4 composition")
        score = float(candidate["l16_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("L16 candidate scores must be finite and non-negative")
        response_len = float(candidate["response_len"])
        teacher_cost = float(candidate["teacher_cost"])
        if not math.isfinite(response_len) or response_len < 0.0:
            raise ValueError("L16 response lengths must be finite and non-negative")
        if not math.isfinite(teacher_cost) or teacher_cost <= 0.0:
            raise ValueError("L16 Teacher costs must be finite and positive")

    prompt_ids = [str(candidate["prompt_id"]) for candidate in candidates]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("L16 requires exactly one candidate per prompt")
    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates[position]["l16_score"]),
            _l16_prompt_order(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )
    selected = order[: min(query_budget, len(order))]
    composition = {1: 0, 2: 0, 3: 0}
    for position in selected:
        num_positive = int(candidates[position]["num_positive"])
        if num_positive in composition:
            composition[num_positive] += 1
    selected_scores = [float(candidates[position]["l16_score"]) for position in selected]
    selected_response_lengths = [float(candidates[position]["response_len"]) for position in selected]
    selected_teacher_costs = [float(candidates[position]["teacher_cost"]) for position in selected]
    stats = {
        "query_budget": float(query_budget),
        "num_eligible_prompts": float(len(candidates)),
        "num_selected_prompts": float(len(selected)),
        "query_budget_utilization": len(selected) / query_budget if query_budget else 0.0,
        "query_1p3n": float(composition[1]),
        "query_2p2n": float(composition[2]),
        "query_3p1n": float(composition[3]),
        "selected_teacher_tokens": sum(selected_teacher_costs),
        "selected_response_len_mean": (
            sum(selected_response_lengths) / len(selected_response_lengths) if selected_response_lengths else 0.0
        ),
        "selected_teacher_cost_mean": (
            sum(selected_teacher_costs) / len(selected_teacher_costs) if selected_teacher_costs else 0.0
        ),
        "selected_l16_score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
    }
    return selected, stats


def compute_l18_success_conditioned_scores_from_cosine(
    negative_positive_cosine: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    negative_response_lengths: torch.Tensor,
    negative_rollout_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute L18 success-conditioned early-change scores for one mixed K=4 prompt."""

    if negative_positive_cosine.ndim != 3 or negative_positive_cosine.shape[-1] != 16:
        raise ValueError("L18 cosine must have shape [N_neg, N_pos, 16]")
    num_negative, num_positive, _ = negative_positive_cosine.shape
    if num_negative < 1 or num_positive < 1:
        raise ValueError("L18 requires at least one negative and one positive rollout")
    if negative_valid_mask.shape != (num_negative, 16) or positive_valid_mask.shape != (num_positive, 16):
        raise ValueError("L18 validity masks must align with 16 states")
    if not bool(negative_valid_mask.bool().all()) or not bool(positive_valid_mask.bool().all()):
        raise ValueError("L18 requires 16 valid states for every mixed-prompt rollout")
    if negative_response_lengths.shape != (num_negative,) or negative_rollout_ids.shape != (num_negative,):
        raise ValueError("L18 negative lengths and rollout ids must have shape [N_neg]")

    cosine = negative_positive_cosine.detach().float()
    lengths = negative_response_lengths.detach().to(device=cosine.device, dtype=torch.float32)
    rollout_ids = negative_rollout_ids.detach().to(device=cosine.device, dtype=torch.long)
    if not torch.isfinite(cosine).all():
        raise FloatingPointError("L18 cosine contains NaN or Inf")
    if not torch.isfinite(lengths).all() or bool((lengths <= 0).any()):
        raise ValueError("L18 response lengths must be finite and positive")
    if bool((rollout_ids < 0).any()) or int(torch.unique(rollout_ids).numel()) != num_negative:
        raise ValueError("L18 rollout ids must be unique non-negative integers")
    cosine = cosine.clamp(-1.0, 1.0)

    pair_split_changes = []
    for tau in range(4, 13):
        pre = cosine[..., :tau].mean(dim=-1)
        post = cosine[..., tau:].mean(dim=-1)
        weight = math.sqrt(tau * (16 - tau) / 16.0)
        pair_split_changes.append(weight * torch.relu(pre - post))
    all_split_changes = torch.stack(pair_split_changes, dim=-1)
    pair_early_change = all_split_changes.mean(dim=-1)
    early_change_score, best_positive_local_index = pair_early_change.max(dim=1)
    candidate_index = torch.arange(num_negative, device=cosine.device)
    split_changes = all_split_changes[candidate_index, best_positive_local_index]
    similarity_trajectory = cosine[candidate_index, best_positive_local_index]

    normalized_length = (lengths - lengths.min()) / (lengths.max() - lengths.min() + L18_LENGTH_EPSILON)
    length_efficiency = 0.25 + 0.75 * (1.0 - normalized_length)
    l18_score = early_change_score * length_efficiency
    best_score = l18_score.max()
    tied = (l18_score == best_score).nonzero(as_tuple=False).flatten()
    selected_local_index = tied[rollout_ids.index_select(0, tied).argmin()]
    return {
        "pair_early_change": pair_early_change.detach(),
        "all_split_changes": all_split_changes.detach(),
        "early_change_score": early_change_score.detach(),
        "best_positive_local_index": best_positive_local_index.detach(),
        "similarity_trajectory": similarity_trajectory.detach(),
        "split_changes": split_changes.detach(),
        "normalized_length": normalized_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "l18_score": l18_score.detach(),
        "selected_local_index": selected_local_index.detach(),
        "selected_rollout_id": rollout_ids[selected_local_index].detach(),
    }


def l18_query_budget(
    num_prompts: int,
    num_rollouts: int,
    num_mixed_prompts: int,
    ratio: float = L18_TEACHER_QUERY_RATIO,
) -> int:
    """L18 query budget capped by the number of complete mixed prompts."""

    if num_prompts < 0 or num_rollouts < 0 or num_mixed_prompts < 0:
        raise ValueError("prompt, rollout, and mixed-prompt counts must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return min(int(ratio * num_prompts * num_rollouts), num_mixed_prompts)


def allocate_l18_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    query_budget: int,
) -> tuple[list[int], dict[str, float]]:
    """Select one L18 winner per prompt by raw-score global Top-B."""

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    required = {
        "prompt_id",
        "rollout_id",
        "num_positive",
        "num_negative",
        "response_len",
        "teacher_cost",
        "l18_score",
    }
    for candidate in candidates:
        missing = required - candidate.keys()
        if missing:
            raise ValueError(f"L18 candidate is missing required fields: {sorted(missing)}")
        num_positive = int(candidate["num_positive"])
        if num_positive not in {1, 2, 3} or int(candidate["num_negative"]) != 4 - num_positive:
            raise ValueError("L18 candidates must have a complete mixed K=4 composition")
        if not str(candidate["prompt_id"]) or int(candidate["rollout_id"]) < 0:
            raise ValueError("L18 prompt and rollout ids must be valid")
        if not math.isfinite(float(candidate["l18_score"])) or float(candidate["l18_score"]) < 0:
            raise ValueError("L18 scores must be finite and non-negative")
        if float(candidate["response_len"]) <= 0 or float(candidate["teacher_cost"]) <= 0:
            raise ValueError("L18 lengths and Teacher costs must be positive")
    prompt_ids = [str(candidate["prompt_id"]) for candidate in candidates]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("L18 requires exactly one candidate per prompt")
    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates[position]["l18_score"]),
            _l16_prompt_order(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )
    selected = order[: min(query_budget, len(order))]
    composition = {1: 0, 2: 0, 3: 0}
    for position in selected:
        composition[int(candidates[position]["num_positive"])] += 1
    scores = [float(candidates[position]["l18_score"]) for position in selected]
    response_lengths = [float(candidates[position]["response_len"]) for position in selected]
    teacher_costs = [float(candidates[position]["teacher_cost"]) for position in selected]
    return selected, {
        "query_budget": float(query_budget),
        "num_eligible_prompts": float(len(candidates)),
        "num_selected_prompts": float(len(selected)),
        "query_budget_utilization": len(selected) / query_budget if query_budget else 0.0,
        "query_1p3n": float(composition[1]),
        "query_2p2n": float(composition[2]),
        "query_3p1n": float(composition[3]),
        "selected_teacher_tokens": sum(teacher_costs),
        "selected_response_len_mean": (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        ),
        "selected_teacher_cost_mean": sum(teacher_costs) / len(teacher_costs) if teacher_costs else 0.0,
        "selected_l18_score_mean": sum(scores) / len(scores) if scores else 0.0,
    }


def compute_l19_scores(
    path_norms: torch.Tensor,
    response_lengths: torch.Tensor,
    rollout_ids: torch.Tensor,
    max_response_length: float,
) -> dict[str, torch.Tensor]:
    """Compute per-negative L19 Latent-Path-Efficiency x length-cost scores.

    ``path_norms`` has shape ``[N_neg, M]`` where the first ``M - 1`` columns are
    the adjacent centered LM-head movements ``||z_{m+1} - z_m||`` and the final
    column is the net movement ``||z_M - z_1||``.  The Latent Path Efficiency is
    ``E = D_net / (D_path + eps)`` with ``D_path = sum ||z_{m+1} - z_m||``; by the
    triangle inequality ``0 <= E <= 1``.  The final score multiplies ``E`` by the
    normalized Teacher-cost efficiency ``1 - L_resp / L_max``.  Tie-breaking is
    deterministic (spec section 16): higher score, then higher LPE, then shorter
    response, then smaller rollout id.
    """
    if path_norms.ndim != 2 or path_norms.shape[-1] != L19_NUM_STATES:
        raise ValueError(f"L19 path norms must have shape [N_neg, {L19_NUM_STATES}]")
    num_negative = int(path_norms.shape[0])
    if num_negative < 1:
        raise ValueError("L19 requires at least one negative rollout")
    if response_lengths.shape != (num_negative,) or rollout_ids.shape != (num_negative,):
        raise ValueError("L19 response lengths and rollout ids must have shape [N_neg]")
    if not math.isfinite(float(max_response_length)) or float(max_response_length) <= 0.0:
        raise ValueError("L19 max_response_length must be finite and positive")

    norms = path_norms.detach().float()
    lengths = response_lengths.detach().to(device=norms.device, dtype=torch.float64)
    ids = rollout_ids.detach().to(device=norms.device, dtype=torch.long)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("L19 path norms contain NaN or Inf")
    if bool((norms < 0).any()):
        raise ValueError("L19 path norms must be non-negative")
    if not torch.isfinite(lengths).all() or bool((lengths <= 0).any()):
        raise ValueError("L19 response lengths must be finite and positive")
    if bool((ids < 0).any()) or int(torch.unique(ids).numel()) != num_negative:
        raise ValueError("L19 rollout ids must be unique non-negative integers")

    step_norms = norms[:, : L19_NUM_STATES - 1]
    net_norm = norms[:, L19_NUM_STATES - 1]
    path_length = step_norms.sum(dim=-1)
    net_progress = net_norm
    latent_path_efficiency = (net_progress / (path_length + L19_PATH_EPSILON)).clamp(0.0, 1.0)
    normalized_length = (lengths.to(torch.float32) / float(max_response_length)).clamp(0.0, 1.0)
    length_efficiency = 1.0 - normalized_length
    l19_score = latent_path_efficiency * length_efficiency

    selected_local_index = min(
        range(num_negative),
        key=lambda index: (
            -float(l19_score[index]),
            -float(latent_path_efficiency[index]),
            float(lengths[index]),
            int(ids[index]),
        ),
    )
    return {
        "path_length": path_length.detach(),
        "net_progress": net_progress.detach(),
        "latent_path_efficiency": latent_path_efficiency.detach(),
        "normalized_length": normalized_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "l19_score": l19_score.detach(),
        "selected_local_index": torch.tensor(selected_local_index, device=norms.device),
        "selected_rollout_id": ids[selected_local_index].detach(),
    }


def l19_query_budget(
    num_prompts: int,
    num_rollouts: int,
    num_mixed_prompts: int,
    ratio: float = L19_TEACHER_QUERY_RATIO,
) -> int:
    """L19 query budget capped by the number of eligible mixed prompts."""

    if num_prompts < 0 or num_rollouts < 0 or num_mixed_prompts < 0:
        raise ValueError("prompt, rollout, and mixed-prompt counts must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return min(int(ratio * num_prompts * num_rollouts), num_mixed_prompts)


def allocate_l19_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    query_budget: int,
) -> tuple[list[int], dict[str, float]]:
    """Select one L19 winner per prompt by raw-score global Top-B."""

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    required = {
        "prompt_id",
        "rollout_id",
        "num_positive",
        "num_negative",
        "response_len",
        "teacher_cost",
        "l19_score",
    }
    for candidate in candidates:
        missing = required - candidate.keys()
        if missing:
            raise ValueError(f"L19 candidate is missing required fields: {sorted(missing)}")
        num_positive = int(candidate["num_positive"])
        # L19 is K-agnostic: any mixed group (>=1 correct and >=1 wrong) is a
        # valid Student competence-frontier prompt, so K may be 4, 8, or any K>=2.
        if num_positive < 1 or int(candidate["num_negative"]) < 1:
            raise ValueError("L19 candidates must be a mixed group (>=1 correct and >=1 wrong)")
        if not str(candidate["prompt_id"]) or int(candidate["rollout_id"]) < 0:
            raise ValueError("L19 prompt and rollout ids must be valid")
        if not math.isfinite(float(candidate["l19_score"])) or float(candidate["l19_score"]) < 0:
            raise ValueError("L19 scores must be finite and non-negative")
        if float(candidate["response_len"]) <= 0 or float(candidate["teacher_cost"]) <= 0:
            raise ValueError("L19 lengths and Teacher costs must be positive")
    prompt_ids = [str(candidate["prompt_id"]) for candidate in candidates]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("L19 requires exactly one candidate per prompt")
    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates[position]["l19_score"]),
            _l16_prompt_order(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )
    selected = order[: min(query_budget, len(order))]
    composition: dict[int, int] = {}
    for position in selected:
        key = int(candidates[position]["num_positive"])
        composition[key] = composition.get(key, 0) + 1
    scores = [float(candidates[position]["l19_score"]) for position in selected]
    response_lengths = [float(candidates[position]["response_len"]) for position in selected]
    teacher_costs = [float(candidates[position]["teacher_cost"]) for position in selected]
    return selected, {
        "query_budget": float(query_budget),
        "num_eligible_prompts": float(len(candidates)),
        "num_selected_prompts": float(len(selected)),
        "query_budget_utilization": len(selected) / query_budget if query_budget else 0.0,
        "query_1p3n": float(composition.get(1, 0)),
        "query_2p2n": float(composition.get(2, 0)),
        "query_3p1n": float(composition.get(3, 0)),
        "selected_teacher_tokens": sum(teacher_costs),
        "selected_response_len_mean": (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        ),
        "selected_teacher_cost_mean": sum(teacher_costs) / len(teacher_costs) if teacher_costs else 0.0,
        "selected_l19_score_mean": sum(scores) / len(scores) if scores else 0.0,
    }


def compute_l17_trajectory_nll(
    sampled_token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute L17 length-normalized trajectory NLL from sampled tokens."""

    if sampled_token_log_probs.shape != response_mask.shape:
        raise ValueError("L17 sampled-token log probabilities and response mask must have equal shape")
    if sampled_token_log_probs.ndim != 2:
        raise ValueError("L17 trajectory tensors must have shape [rollout, response_token]")
    mask = response_mask.to(dtype=torch.bool, device=sampled_token_log_probs.device)
    token_counts = mask.sum(dim=-1)
    if bool((token_counts == 0).any()):
        raise ValueError("every L17 trajectory must contain at least one response token")
    log_probs = sampled_token_log_probs.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(log_probs[mask]).all()):
        raise ValueError("masked L17 sampled-token log probabilities must be finite")
    masked_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return -masked_log_probs.sum(dim=-1) / token_counts.to(dtype=torch.float64)


def compute_l17_oci_scores(
    trajectory_nll: torch.Tensor,
    verifier_correct: torch.Tensor,
    response_lengths: torch.Tensor,
    rollout_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute exact per-negative L17 confidence-inversion utilities for K=4."""

    tensors = (trajectory_nll, verifier_correct, response_lengths, rollout_ids)
    if any(tensor.ndim != 1 or tensor.numel() != 4 for tensor in tensors):
        raise ValueError("L17 requires four aligned one-dimensional rollout values")
    device = trajectory_nll.device
    nll = trajectory_nll.detach().to(dtype=torch.float64)
    correct = verifier_correct.detach().to(dtype=torch.bool, device=device)
    lengths = response_lengths.detach().to(dtype=torch.float64, device=device)
    ids = rollout_ids.detach().to(dtype=torch.long, device=device)
    if not bool(torch.isfinite(nll).all()):
        raise ValueError("L17 trajectory NLL values must be finite")
    if not bool(torch.isfinite(lengths).all()) or bool((lengths <= 0).any()):
        raise ValueError("L17 response lengths must be finite and positive")
    if bool((ids < 0).any()) or ids.unique().numel() != 4:
        raise ValueError("L17 rollout ids must be unique non-negative integers")
    num_positive = int(correct.sum().item())
    if num_positive not in {1, 2, 3}:
        raise ValueError("L17 scores require a complete mixed K=4 composition")

    positive_nll = nll[correct]
    failed_indices = (~correct).nonzero(as_tuple=False).flatten()
    failed_nll = nll[failed_indices]
    failed_lengths = lengths[failed_indices]
    failed_ids = ids[failed_indices]
    successful_nll_mean = positive_nll.mean()
    confidence_inversion = torch.sigmoid(successful_nll_mean - failed_nll)
    normalized_length = (failed_lengths - failed_lengths.min()) / (
        failed_lengths.max() - failed_lengths.min() + L17_LENGTH_EPSILON
    )
    length_efficiency = 0.25 + 0.75 * (1.0 - normalized_length)
    score = confidence_inversion * length_efficiency
    selected_local_index = min(
        range(score.numel()),
        key=lambda index: (-float(score[index]), int(failed_ids[index])),
    )
    return {
        "failed_indices": failed_indices.detach(),
        "failed_rollout_ids": failed_ids.detach(),
        "successful_nll_mean": successful_nll_mean.detach(),
        "failed_trajectory_nll": failed_nll.detach(),
        "confidence_inversion": confidence_inversion.detach(),
        "normalized_length": normalized_length.detach(),
        "length_efficiency": length_efficiency.detach(),
        "score": score.detach(),
        "selected_local_index": torch.tensor(selected_local_index, device=device),
        "selected_rollout_id": failed_ids[selected_local_index].detach(),
    }


def l17_query_budget(
    num_prompts: int,
    num_rollouts: int,
    num_mixed_prompts: int,
    ratio: float = L17_TEACHER_QUERY_RATIO,
) -> int:
    """L17 query budget capped by the number of eligible mixed prompts."""

    if num_prompts < 0 or num_rollouts < 0 or num_mixed_prompts < 0:
        raise ValueError("prompt, rollout, and mixed-prompt counts must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return min(int(ratio * num_prompts * num_rollouts), num_mixed_prompts)


def allocate_l17_teacher_queries(
    candidates: Sequence[dict[str, Any]],
    query_budget: int,
) -> tuple[list[int], dict[str, float]]:
    """Select one-candidate-per-prompt raw-score global Top-B for L17."""

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    required_fields = {
        "prompt_id",
        "rollout_id",
        "num_positive",
        "num_negative",
        "response_len",
        "teacher_cost",
        "l17_score",
    }
    for candidate in candidates:
        missing = required_fields - candidate.keys()
        if missing:
            raise ValueError(f"L17 candidate is missing required fields: {sorted(missing)}")
        if not str(candidate["prompt_id"]):
            raise ValueError("L17 prompt_id must be non-empty")
        if int(candidate["rollout_id"]) < 0:
            raise ValueError("L17 rollout_id must be non-negative")
        num_positive = int(candidate["num_positive"])
        num_negative = int(candidate["num_negative"])
        if num_positive not in {1, 2, 3} or num_negative != 4 - num_positive:
            raise ValueError("L17 candidates must have a complete mixed K=4 composition")
        score = float(candidate["l17_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("L17 candidate scores must be finite and non-negative")
        response_len = float(candidate["response_len"])
        teacher_cost = float(candidate["teacher_cost"])
        if not math.isfinite(response_len) or response_len <= 0.0:
            raise ValueError("L17 response lengths must be finite and positive")
        if not math.isfinite(teacher_cost) or teacher_cost <= 0.0:
            raise ValueError("L17 Teacher costs must be finite and positive")

    prompt_ids = [str(candidate["prompt_id"]) for candidate in candidates]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("L17 requires exactly one candidate per prompt")
    order = sorted(
        range(len(candidates)),
        key=lambda position: (
            -float(candidates[position]["l17_score"]),
            _l16_prompt_order(candidates[position]["prompt_id"]),
            int(candidates[position]["rollout_id"]),
        ),
    )
    selected = order[: min(query_budget, len(order))]
    composition = {1: 0, 2: 0, 3: 0}
    for position in selected:
        composition[int(candidates[position]["num_positive"])] += 1
    selected_scores = [float(candidates[position]["l17_score"]) for position in selected]
    selected_response_lengths = [float(candidates[position]["response_len"]) for position in selected]
    selected_teacher_costs = [float(candidates[position]["teacher_cost"]) for position in selected]
    stats = {
        "query_budget": float(query_budget),
        "num_eligible_prompts": float(len(candidates)),
        "num_selected_prompts": float(len(selected)),
        "query_budget_utilization": len(selected) / query_budget if query_budget else 0.0,
        "query_1p3n": float(composition[1]),
        "query_2p2n": float(composition[2]),
        "query_3p1n": float(composition[3]),
        "selected_teacher_tokens": sum(selected_teacher_costs),
        "selected_response_len_mean": (
            sum(selected_response_lengths) / len(selected_response_lengths) if selected_response_lengths else 0.0
        ),
        "selected_teacher_cost_mean": (
            sum(selected_teacher_costs) / len(selected_teacher_costs) if selected_teacher_costs else 0.0
        ),
        "selected_l17_score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
    }
    return selected, stats


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    cursor = 0
    while cursor < values.numel():
        end = cursor + 1
        while end < values.numel() and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * (cursor + end - 1)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    return ranks


def spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Small-tensor Spearman rho with average ranks for ties."""

    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    valid = torch.isfinite(left) & torch.isfinite(right)
    if int(valid.sum()) < 2:
        return math.nan
    left_rank = _average_ranks(left[valid])
    right_rank = _average_ranks(right[valid])
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator <= 0:
        return math.nan
    return float((left_centered * right_centered).sum() / denominator)


def stable_group_sibling_indices(
    prompt_ids: Sequence[Any],
    rollout_ids: Sequence[Any],
    *,
    expected_siblings: int,
) -> list[tuple[str, list[int]]]:
    """Stable ``prompt_id + rollout_id`` regroup for Boundary selector inputs."""

    if len(prompt_ids) != len(rollout_ids):
        raise ValueError("prompt_ids and rollout_ids must have equal length")
    entries = sorted(
        (
            str(prompt_id),
            int(rollout_id),
            original_index,
        )
        for original_index, (prompt_id, rollout_id) in enumerate(zip(prompt_ids, rollout_ids, strict=True))
    )
    grouped: list[tuple[str, list[int]]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for prompt_id, rollout_id, original_index in entries:
        pair = (prompt_id, rollout_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate Boundary sibling identity: {pair}")
        seen_pairs.add(pair)
        if not grouped or grouped[-1][0] != prompt_id:
            grouped.append((prompt_id, []))
        grouped[-1][1].append(original_index)
    for prompt_id, indices in grouped:
        if len(indices) != expected_siblings:
            raise ValueError(
                f"Boundary-OPD prompt {prompt_id} has {len(indices)} siblings; expected {expected_siblings}"
            )
    return grouped


@dataclass(frozen=True)
class BoundaryCaptureTarget:
    """Resolved pre-LM-head module and hook type."""

    module_path: str
    capture_method: str
    module_type: str
    module: nn.Module = field(compare=False, repr=False)
    related_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryLMHeadTarget:
    """Resolved current Student output projection."""

    module_path: str
    module_type: str
    module: nn.Module = field(compare=False, repr=False)


def _unwrap_model(module: nn.Module) -> nn.Module:
    current = module
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        next_module = None
        for attribute in ("_fsdp_wrapped_module", "module"):
            candidate = getattr(current, attribute, None)
            if isinstance(candidate, nn.Module) and candidate is not current:
                next_module = candidate
                break
        if next_module is None:
            break
        current = next_module
    return current


def _module_hidden_width(module: nn.Module) -> int | None:
    in_features = getattr(module, "in_features", None)
    if in_features is not None:
        return int(in_features)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[-1])
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape:
        return int(normalized_shape[-1])
    return None


def resolve_boundary_lm_head(actor_module: nn.Module) -> BoundaryLMHeadTarget:
    """Resolve the output projection owned by the current Student Actor."""

    model = _unwrap_model(actor_module)
    named_modules = list(model.named_modules())
    path_by_id = {id(module): name or "<root>" for name, module in named_modules}
    output_head = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output_head = getter()
    if not isinstance(output_head, nn.Module):
        head_candidates = [
            (name, module)
            for name, module in named_modules
            if name.split(".")[-1] in {"lm_head", "output_layer", "embed_out"}
        ]
        if head_candidates:
            output_head = head_candidates[-1][1]
    if not isinstance(output_head, nn.Module):
        raise RuntimeError("Boundary-OPD could not discover the Student Actor LM head")
    weight = getattr(output_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("Boundary-OPD Student Actor LM head must expose a rank-2 weight")
    return BoundaryLMHeadTarget(
        module_path=path_by_id.get(id(output_head), type(output_head).__name__),
        module_type=type(output_head).__name__,
        module=output_head,
    )


def resolve_boundary_capture_target(
    actor_module: nn.Module,
    *,
    capture_method: str = "auto",
) -> BoundaryCaptureTarget:
    """Discover the final norm or LM head without assuming a fixed model class."""

    if capture_method not in _BOUNDARY_CAPTURE_METHODS:
        raise ValueError(f"unknown Boundary hidden capture method: {capture_method!r}")
    model = _unwrap_model(actor_module)
    named_modules = list(model.named_modules())
    path_by_id = {id(module): name or "<root>" for name, module in named_modules}

    output_head = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output_head = getter()
    if not isinstance(output_head, nn.Module):
        head_candidates = [
            (name, module)
            for name, module in named_modules
            if name.split(".")[-1] in {"lm_head", "output_layer", "embed_out"}
        ]
        if head_candidates:
            output_head = head_candidates[-1][1]
    output_head_path = path_by_id.get(id(output_head)) if isinstance(output_head, nn.Module) else None
    output_width = _module_hidden_width(output_head) if isinstance(output_head, nn.Module) else None

    final_markers = {"norm", "final_layernorm", "final_norm", "ln_f"}
    norm_candidates: list[tuple[int, str, nn.Module]] = []
    for name, module in named_modules:
        if not name or "norm" not in type(module).__name__.lower():
            continue
        segments = name.split(".")
        if segments[-1] not in final_markers or any(segment.isdigit() for segment in segments):
            continue
        candidate_width = _module_hidden_width(module)
        if output_width is not None and candidate_width not in {None, output_width}:
            continue
        score = 100
        if "vision" in name.lower() or "visual" in name.lower():
            score -= 100
        if "language" in name.lower() or name.endswith("model.norm"):
            score += 20
        score += len(segments)
        norm_candidates.append((score, name, module))
    norm_candidates.sort(key=lambda item: (-item[0], item[1]))

    related = tuple(
        f"{name or '<root>'}:{type(module).__name__}"
        for name, module in named_modules
        if (
            name.split(".")[-1] in final_markers | {"lm_head", "output_layer", "embed_out"}
            and not any(segment.isdigit() for segment in name.split("."))
        )
    )
    if capture_method in {"auto", "final_norm_hook"} and norm_candidates:
        _, name, module = norm_candidates[0]
        return BoundaryCaptureTarget(
            module_path=name,
            capture_method="final_norm_forward_hook",
            module_type=type(module).__name__,
            module=module,
            related_modules=related,
        )
    if capture_method == "final_norm_hook":
        raise RuntimeError(f"Boundary-OPD could not discover an unambiguous final norm; related modules={related}")
    if isinstance(output_head, nn.Module):
        return BoundaryCaptureTarget(
            module_path=output_head_path or type(output_head).__name__,
            capture_method="lm_head_forward_pre_hook",
            module_type=type(output_head).__name__,
            module=output_head,
            related_modules=related,
        )
    raise RuntimeError(
        f"Boundary-OPD could not discover either the final norm or output embedding; related modules={related}"
    )


def extract_hidden_tensor(value: Any) -> torch.Tensor | None:
    """Extract the first tensor from a hook input/output container."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = extract_hidden_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = extract_hidden_tensor(item)
            if tensor is not None:
                return tensor
    return None


def gather_boundary_states_from_hidden(
    captured_hidden: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    hidden_storage_dtype: torch.dtype,
    unpadded_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather only ``M + 1`` states per rollout from a captured final hidden tensor."""

    hidden = captured_hidden.detach()
    if unpadded_indices is None:
        if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, sequence_length):
            raise ValueError(
                "padded Boundary hidden must have shape "
                f"[{batch_size}, {sequence_length}, D], got {tuple(hidden.shape)}"
            )
        gather_index = (
            state_indices.to(hidden.device)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                hidden.shape[-1],
            )
        )
        boundary_states = hidden.gather(dim=1, index=gather_index)
    else:
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        elif hidden.ndim == 3 and hidden.shape[1] == 1:
            hidden = hidden.squeeze(1)
        if hidden.ndim != 2:
            raise ValueError(
                f"remove-padding Boundary hidden must have shape [total_nnz, D], got {tuple(hidden.shape)}"
            )
        flat_valid_indices = unpadded_indices.to(device=hidden.device, dtype=torch.long).flatten()
        if hidden.shape[0] != flat_valid_indices.numel():
            raise ValueError(
                f"captured hidden and unpadded index counts differ: {hidden.shape[0]} vs {flat_valid_indices.numel()}"
            )
        flat_targets = (
            torch.arange(batch_size, device=hidden.device, dtype=torch.long).unsqueeze(1) * sequence_length
            + state_indices.to(device=hidden.device, dtype=torch.long)
        ).flatten()
        positions = torch.searchsorted(flat_valid_indices, flat_targets)
        positions = positions.clamp(max=max(flat_valid_indices.numel() - 1, 0))
        if flat_valid_indices.numel() == 0 or not torch.equal(
            flat_valid_indices[positions],
            flat_targets,
        ):
            raise ValueError("a Boundary state index points to a padded token")
        boundary_states = hidden[positions].reshape(
            batch_size,
            state_indices.shape[1],
            hidden.shape[-1],
        )

    boundary_states = boundary_states.to(dtype=hidden_storage_dtype).detach()
    if boundary_states.requires_grad:
        raise AssertionError("Boundary-OPD hidden states must be detached")
    if not torch.isfinite(boundary_states.float()).all():
        raise FloatingPointError("Boundary-OPD hidden states contain NaN or Inf")
    return boundary_states


def gather_response_hidden_from_hidden(
    captured_hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    hidden_storage_dtype: torch.dtype,
    unpadded_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize response-token hidden only for span-distribution metrics."""

    hidden = captured_hidden.detach()
    if response_mask.ndim != 2 or response_mask.shape[0] != batch_size:
        raise ValueError("response_mask must have shape [B, response_length]")
    response_mask = response_mask.to(device=hidden.device, dtype=torch.bool)
    response_length = response_mask.shape[1]
    prompt_width = sequence_length - response_length
    if prompt_width < 0:
        raise ValueError("response length exceeds sequence length")
    if unpadded_indices is None:
        if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, sequence_length):
            raise ValueError(
                f"padded span hidden must have shape [{batch_size}, {sequence_length}, D], got {tuple(hidden.shape)}"
            )
        response_hidden = hidden[:, prompt_width:, :]
    else:
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        elif hidden.ndim == 3 and hidden.shape[1] == 1:
            hidden = hidden.squeeze(1)
        if hidden.ndim != 2:
            raise ValueError(f"remove-padding span hidden must have shape [total_nnz, D], got {tuple(hidden.shape)}")
        flat_valid_indices = unpadded_indices.to(device=hidden.device, dtype=torch.long).flatten()
        if hidden.shape[0] != flat_valid_indices.numel():
            raise ValueError("captured span hidden and unpadded index counts differ")
        response_hidden = torch.zeros(
            batch_size,
            response_length,
            hidden.shape[-1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        batch_positions, response_positions = response_mask.nonzero(as_tuple=True)
        if batch_positions.numel():
            flat_targets = batch_positions * sequence_length + prompt_width + response_positions
            positions = torch.searchsorted(flat_valid_indices, flat_targets)
            positions = positions.clamp(max=max(flat_valid_indices.numel() - 1, 0))
            if flat_valid_indices.numel() == 0 or not torch.equal(flat_valid_indices[positions], flat_targets):
                raise ValueError("a valid response token points to padded hidden storage")
            response_hidden[batch_positions, response_positions] = hidden[positions]
    response_hidden = response_hidden.to(dtype=hidden_storage_dtype)
    response_hidden = response_hidden.masked_fill(~response_mask.unsqueeze(-1), 0.0).detach()
    if response_hidden.requires_grad:
        raise AssertionError("Boundary span hidden must be detached")
    if not torch.isfinite(response_hidden.float()).all():
        raise FloatingPointError("Boundary span hidden contains NaN or Inf")
    return response_hidden


def gather_boundary_states_from_sequence_shard(
    captured_hidden: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    unpadded_indices: torch.Tensor,
    sequence_padding_size: int,
    sequence_parallel_size: int,
    sequence_parallel_rank: int,
    process_group: dist.ProcessGroup,
    hidden_storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Gather sparse boundary states from a contiguous Ulysses sequence shard.

    Each rank materializes only its owned boundary slots and an all-reduce
    combines ``[B, M + 1, D]`` values. The full ``[total_nnz, D]`` hidden
    sequence is never gathered.
    """

    hidden = captured_hidden.detach()
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    elif hidden.ndim == 3 and hidden.shape[1] == 1:
        hidden = hidden.squeeze(1)
    if hidden.ndim != 2:
        raise ValueError(f"sequence-parallel Boundary hidden must have shape [local_nnz, D], got {tuple(hidden.shape)}")
    if sequence_parallel_size < 2:
        raise ValueError("sequence_parallel_size must be at least two")
    flat_valid_indices = unpadded_indices.to(device=hidden.device, dtype=torch.long).flatten()
    padded_count = flat_valid_indices.numel() + int(sequence_padding_size)
    if padded_count % sequence_parallel_size:
        raise ValueError("padded Ulysses sequence length is not divisible by SP size")
    local_count = padded_count // sequence_parallel_size
    if hidden.shape[0] != local_count:
        raise ValueError(
            f"captured local hidden length does not match the Ulysses shard: {hidden.shape[0]} vs {local_count}"
        )

    flat_targets = (
        torch.arange(batch_size, device=hidden.device, dtype=torch.long).unsqueeze(1) * sequence_length
        + state_indices.to(device=hidden.device, dtype=torch.long)
    ).flatten()
    target_ordinals = torch.searchsorted(flat_valid_indices, flat_targets)
    target_ordinals = target_ordinals.clamp(max=max(flat_valid_indices.numel() - 1, 0))
    if flat_valid_indices.numel() == 0 or not torch.equal(
        flat_valid_indices[target_ordinals],
        flat_targets,
    ):
        raise ValueError("a Boundary state index points to a padded token")

    shard_start = int(sequence_parallel_rank) * local_count
    shard_end = shard_start + local_count
    owned = (target_ordinals >= shard_start) & (target_ordinals < shard_end)
    local_states = torch.zeros(
        flat_targets.numel(),
        hidden.shape[-1],
        dtype=hidden_storage_dtype,
        device=hidden.device,
    )
    if owned.any():
        local_positions = target_ordinals[owned] - shard_start
        local_states[owned] = hidden[local_positions].to(hidden_storage_dtype)
    ownership = owned.to(dtype=torch.int32)
    dist.all_reduce(local_states, op=dist.ReduceOp.SUM, group=process_group)
    dist.all_reduce(ownership, op=dist.ReduceOp.SUM, group=process_group)
    if not torch.equal(ownership, torch.ones_like(ownership)):
        raise AssertionError("every Boundary state must be owned by exactly one Ulysses rank")

    boundary_states = local_states.reshape(
        batch_size,
        state_indices.shape[1],
        hidden.shape[-1],
    ).detach()
    if boundary_states.requires_grad:
        raise AssertionError("Boundary-OPD hidden states must be detached")
    if not torch.isfinite(boundary_states.float()).all():
        raise FloatingPointError("Boundary-OPD hidden states contain NaN or Inf")
    return boundary_states


def gather_response_hidden_from_sequence_shard(
    captured_hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    unpadded_indices: torch.Tensor,
    sequence_padding_size: int,
    sequence_parallel_size: int,
    sequence_parallel_rank: int,
    process_group: dist.ProcessGroup,
    hidden_storage_dtype: torch.dtype,
) -> torch.Tensor:
    """All-reduce the response-token hidden required only by span metrics."""

    hidden = captured_hidden.detach()
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    elif hidden.ndim == 3 and hidden.shape[1] == 1:
        hidden = hidden.squeeze(1)
    if hidden.ndim != 2:
        raise ValueError(f"sequence-parallel span hidden must have shape [local_nnz, D], got {tuple(hidden.shape)}")
    if response_mask.ndim != 2 or response_mask.shape[0] != batch_size:
        raise ValueError("response_mask must have shape [B, response_length]")
    response_mask = response_mask.to(device=hidden.device, dtype=torch.bool)
    response_length = response_mask.shape[1]
    prompt_width = sequence_length - response_length
    flat_valid_indices = unpadded_indices.to(device=hidden.device, dtype=torch.long).flatten()
    padded_count = flat_valid_indices.numel() + int(sequence_padding_size)
    if padded_count % sequence_parallel_size:
        raise ValueError("padded Ulysses sequence length is not divisible by SP size")
    local_count = padded_count // sequence_parallel_size
    if hidden.shape[0] != local_count:
        raise ValueError(
            f"captured local span hidden length differs from Ulysses shard: {hidden.shape[0]} vs {local_count}"
        )
    batch_positions, response_positions = response_mask.nonzero(as_tuple=True)
    flat_targets = batch_positions * sequence_length + prompt_width + response_positions
    target_ordinals = torch.searchsorted(flat_valid_indices, flat_targets)
    target_ordinals = target_ordinals.clamp(max=max(flat_valid_indices.numel() - 1, 0))
    if flat_targets.numel() and (
        flat_valid_indices.numel() == 0 or not torch.equal(flat_valid_indices[target_ordinals], flat_targets)
    ):
        raise ValueError("a valid response token points to padded Ulysses storage")
    shard_start = int(sequence_parallel_rank) * local_count
    shard_end = shard_start + local_count
    owned = (target_ordinals >= shard_start) & (target_ordinals < shard_end)
    response_hidden = torch.zeros(
        batch_size,
        response_length,
        hidden.shape[-1],
        dtype=hidden_storage_dtype,
        device=hidden.device,
    )
    if owned.any():
        local_positions = target_ordinals[owned] - shard_start
        response_hidden[batch_positions[owned], response_positions[owned]] = hidden[local_positions].to(
            hidden_storage_dtype
        )
    ownership = torch.zeros(batch_size, response_length, dtype=torch.int32, device=hidden.device)
    ownership[batch_positions[owned], response_positions[owned]] = 1
    dist.all_reduce(response_hidden, op=dist.ReduceOp.SUM, group=process_group)
    dist.all_reduce(ownership, op=dist.ReduceOp.SUM, group=process_group)
    if not torch.equal(ownership[response_mask], torch.ones_like(ownership[response_mask])):
        raise AssertionError("every valid response token must be owned by exactly one Ulysses rank")
    if bool(ownership[~response_mask].any()):
        raise AssertionError("padding response tokens must not have Ulysses ownership")
    response_hidden = response_hidden.detach()
    if not torch.isfinite(response_hidden.float()).all():
        raise FloatingPointError("Boundary span hidden contains NaN or Inf")
    return response_hidden
