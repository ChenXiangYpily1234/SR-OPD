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

"""FF-OPD routing and unified rollout-selector ablations."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

from verl.utils.boundary_calibration import BoundaryCalibrationArtifact, load_boundary_calibration
from verl.utils.boundary_opd import (
    BOUNDARY_CALIBRATION_ARTIFACT_METRICS,
    PDA_SCORE_MODES,
    BoundaryOPDSettings,
    boundary_state_count,
    boundary_transition_count,
    build_centered_hidden_states,
    compute_persistent_departure_area_scores_from_cosine,
    is_boundary_selector_mode,
    spearman_correlation,
)
from verl.utils.frontier_selector import (
    FF_SELECTOR_MODES,
    FRONTIER_SELECTOR,
    SELECTOR_FORMULA_VERSION,
    FrontierSelectionResult,
    FrontierTrajectorySelector,
    select_ff_rollouts,
)

_FRONTIER_SELECTION_MODES = (
    FRONTIER_SELECTOR,
    "nearest_only",
)

LOSS_TYPE = "reverse_kl"
TARGET_MODE = "sampled_token"
BASE_EPOCHS = 1
LEGACY_TEACHER_QUERY_RATIO = 0.25
K_ROLLOUTS = 4
MAX_TEACHER_CANDIDATES_PER_PROMPT = 1
MAX_NO_SUCCESS_RETRIES = 0
TEACHER_STOP_GRADIENT = True
USE_HT = False
USE_IPW = False


class FFBucket(str, Enum):
    FRESH = "fresh"
    ALL_CORRECT = "all_correct"
    FRONTIER = "frontier"
    NO_SUCCESS = "no_success"
    INVALID = "invalid"


class FFCandidateType(str, Enum):
    NONE = "none"
    FRONTIER_INCORRECT = "frontier_incorrect"


class FFFinalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    RETRY_PENDING = "retry_pending"
    EXHAUSTED_NO_SUCCESS = "exhausted_no_success"


class FFRejectedReason(str, Enum):
    NONE = "none"
    ALL_CORRECT = "all_correct"
    QUERY_CAP_REACHED = "query_cap_reached"
    NO_VALID_INCORRECT = "no_valid_incorrect"
    NO_VALID_ROLLOUT = "no_valid_rollout"


@dataclass
class FFOPDConfig:
    enable: bool = False
    base_epochs: int = BASE_EPOCHS
    k_rollouts: int = K_ROLLOUTS
    # Zero runs the complete fresh dataset pass. Positive values are reserved
    # for explicit engineering smoke tests.
    fresh_step_limit: int = 0
    frontier_only: bool = True
    # Historical cap denominator. With K=4, 0.25 permits at most one Teacher
    # query per prompt attempt; it is not the realized query rate.
    legacy_teacher_query_ratio: float = LEGACY_TEACHER_QUERY_RATIO
    max_no_success_retries: int = MAX_NO_SUCCESS_RETRIES
    seed: int = 42
    csv_path: str = "metrics/ff.csv"
    csv_flush_interval: int = 1
    debug_assertions: bool = True
    save_queue_state: bool = True
    # Full confidence profiles are far too large for ff.csv, so only a
    # deterministic sample of Frontier attempts is mirrored into this JSONL.
    profile_jsonl_path: str = "metrics/ff_profiles.jsonl"
    profile_audit_sample_rate: float = 0.02
    # Per-token reverse-KL profiles of the trained (selected) samples, written
    # after the Teacher forward: one JSONL row per selected sample carrying the
    # masked token log-ratios r_t = logp_student - logp_teacher. Scalar ff.csv
    # aggregates cannot resolve where along the response the training signal
    # sits; this sidecar can. Diagnostic-only, so it defaults to off.
    log_kl_profiles: bool = False
    kl_profile_jsonl_path: str = "metrics/ff_kl_profiles.jsonl"
    # Cost-aware selector (FF-Cost): the sibling score is
    #   J_j = (d_j + eps) * c_j^alpha
    #   log S_j = -log(d_j + eps) - alpha * log(c_j)
    # with d_j the raw nearest-positive distance and c_j the raw Teacher cost
    # (prompt + response tokens). alpha is fixed at 0.5 by the recipe.
    # selector_cost_aware is the single-variable ablation switch.
    selector_cost_aware: bool = True
    selector_cost_alpha: float = 0.5
    selector_score_eps: float = 1e-8
    selector_mode: str = "boundary_opd"
    boundary_opd: Optional[BoundaryOPDSettings] = None

    def __post_init__(self) -> None:
        # Boundary selector settings are normally injected by the trainer from
        # algorithm.boundary_opd. Resolve them here as well so a directly
        # constructed config uses the documented defaults instead of crashing
        # inside the selector.
        if self.boundary_opd is not None and not isinstance(self.boundary_opd, BoundaryOPDSettings):
            self.boundary_opd = BoundaryOPDSettings.from_mapping(self.boundary_opd)
        if is_boundary_selector_mode(self.selector_mode) and self.boundary_opd is None:
            self.boundary_opd = BoundaryOPDSettings()

    @classmethod
    def from_mapping(cls, values: Any) -> FFOPDConfig:
        if values is None:
            return cls()
        data = {}
        for name, config_field in cls.__dataclass_fields__.items():
            data[name] = values.get(name, config_field.default)
        return cls(**data)

    def validate(self, *, rollout_n: Optional[int] = None, opd_target_mode: str = TARGET_MODE) -> None:
        if self.base_epochs != BASE_EPOCHS:
            raise ValueError("FF-OPD fixes ff_opd.base_epochs=1")
        if not self.frontier_only:
            raise ValueError("FF-OPD requires ff_opd.frontier_only=true")
        if self.k_rollouts < 2:
            raise ValueError("FF-OPD requires ff_opd.k_rollouts >= 2")
        if self.fresh_step_limit < 0:
            raise ValueError("ff_opd.fresh_step_limit must be >= 0")
        if self.legacy_teacher_query_ratio != LEGACY_TEACHER_QUERY_RATIO:
            raise ValueError("FF-OPD fixes ff_opd.legacy_teacher_query_ratio=0.25")
        if self.max_no_success_retries not in {0, 1, 2}:
            raise ValueError("ff_opd.max_no_success_retries must be one of {0, 1, 2}")
        if self.selector_cost_alpha != 0.5:
            raise ValueError("FF-OPD requires ff_opd.selector_cost_alpha == 0.5")
        if self.selector_score_eps <= 0:
            raise ValueError("ff_opd.selector_score_eps must be positive")
        if self.selector_mode not in FF_SELECTOR_MODES:
            raise ValueError(
                f"algorithm.ff_selector_mode must be one of {FF_SELECTOR_MODES}, got {self.selector_mode!r}"
            )
        if is_boundary_selector_mode(self.selector_mode):
            if self.boundary_opd is None:
                raise AssertionError("Boundary selector settings must be resolved by FFOPDConfig.__post_init__")
            self.boundary_opd.validate()
            if self.boundary_opd.score_mode in PDA_SCORE_MODES:
                if self.max_no_success_retries != 0:
                    raise ValueError("PDA score modes require max_no_success_retries=0")
        elif self.boundary_opd is not None:
            raise ValueError(
                "algorithm.boundary_opd is only valid for a Boundary selector mode, "
                f"got algorithm.ff_selector_mode={self.selector_mode!r}"
            )
        FrontierTrajectorySelector(
            seed=self.seed,
            cost_aware=self.selector_cost_aware,
            cost_alpha=self.selector_cost_alpha,
            score_eps=self.selector_score_eps,
        )
        if self.csv_flush_interval < 1:
            raise ValueError("ff_opd.csv_flush_interval must be >= 1")
        if self.seed < 0:
            raise ValueError("ff_opd.seed must be >= 0")
        if not 0.0 <= self.profile_audit_sample_rate <= 1.0:
            raise ValueError("ff_opd.profile_audit_sample_rate must be within [0, 1]")
        if rollout_n is not None and int(rollout_n) != self.k_rollouts:
            raise ValueError("actor_rollout_ref.rollout.n must equal ff_opd.k_rollouts")
        if opd_target_mode != TARGET_MODE:
            raise ValueError("FF-OPD requires opd_target_mode=sampled_token")


@dataclass
class FFPromptState:
    prompt_uid: str
    source_index: str
    current_bucket: str = FFBucket.FRESH.value
    previous_bucket: str = FFBucket.FRESH.value
    attempt_count: int = 0
    retry_count: int = 0
    retries_left: int = MAX_NO_SUCCESS_RETRIES
    retries_left_before: int = MAX_NO_SUCCESS_RETRIES
    last_correct_count: int = 0
    total_teacher_queries: int = 0
    cumulative_teacher_processing_tokens: int = 0
    cumulative_teacher_supervised_tokens: int = 0
    last_optimizer_step: int = 0
    last_student_model_version: str = "0"
    last_attempt_model_version: str = "0"
    final_status: str = FFFinalStatus.ACTIVE.value


@dataclass
class FFCandidate:
    prompt_uid: str
    source_index: str
    rollout_index: int
    teacher_processing_tokens: int
    response_tokens: int
    frontier_selection: FrontierSelectionResult
    selected_for_teacher: bool = False
    rejected_reason: str = FFRejectedReason.NONE.value
    query_cap_before: int = 0
    query_cap_after: int = 0
    used_queries_before: int = 0
    used_queries_after: int = 0
    candidate_type: str = FFCandidateType.FRONTIER_INCORRECT.value
    bucket: str = FFBucket.FRONTIER.value
    verifier_correct: bool = False
    rollout_valid: bool = True
    query_probability: float = 1.0
    inverse_probability_weight: float = 1.0


@dataclass
class FFRouteResult:
    candidates: list[FFCandidate]
    records: list[dict[str, Any]]
    selected_indices: list[int]
    metrics: dict[str, float]
    available_query_budget: int
    used_queries: int
    profile_records: list[dict[str, Any]] = field(default_factory=list)
    boundary_csv_rows: list[dict[str, Any]] = field(default_factory=list)
    representation_dynamics_rows: list[dict[str, Any]] = field(default_factory=list)
    ht_enabled: bool = False
    ht_full_valid_response_tokens: int = 0
    ht_full_candidate_count: int = 0
    # L23 post-fork span supervision: prompt_uid -> {rollout_id, fork_token_pos,
    # span_start, span_end, span_fallback_full, response_length} for the
    # selected rollout of each mixed prompt.  Empty unless the L23 score mode
    # is active.
    l23_fork_positions: dict[str, dict[str, float]] = field(default_factory=dict)


def stable_int_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def select_shortest_wrong_rollouts(
    *,
    valid_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    rollout_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the shortest verified-wrong rollout from mixed prompts only.

    Lengths are the existing valid-response-token counts supplied by the
    routing batch. Equal lengths are resolved by rollout id, then by the
    original candidate position. All-correct and all-wrong prompts are left
    unselected so this selector has exactly the BC-OPD mixed-prompt coverage.
    """
    if valid_mask.ndim != 2:
        raise ValueError("shortest_wrong inputs must have shape [B, K]")
    if any(tensor.shape != valid_mask.shape for tensor in (correct_mask, response_lengths, rollout_ids)):
        raise ValueError("shortest_wrong inputs must have identical shapes")

    valid = valid_mask.detach().bool().cpu()
    correct = correct_mask.detach().bool().cpu()
    lengths = response_lengths.detach().to(dtype=torch.long, device="cpu")
    ids = rollout_ids.detach().to(dtype=torch.long, device="cpu")
    if torch.any(valid & (lengths < 0)):
        raise ValueError("valid response lengths must be non-negative")

    wrong = valid & ~correct
    mixed = (valid & correct).any(dim=1) & wrong.any(dim=1)
    selected = torch.zeros_like(valid)
    for prompt_index in mixed.nonzero(as_tuple=False).flatten().tolist():
        candidates = wrong[prompt_index].nonzero(as_tuple=False).flatten().tolist()
        chosen = min(
            candidates,
            key=lambda index: (int(lengths[prompt_index, index]), int(ids[prompt_index, index]), index),
        )
        selected[prompt_index, chosen] = True
    return selected.to(device=valid_mask.device)


def select_frontier_tlr_rollouts(
    *,
    valid_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    student_entropy_mean: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Select the highest TLR-score verified-wrong rollout from mixed prompts only.

    Frontier-TLR is a matched-support selector baseline for Frontier-PDA:
    identical mixed-only gating, wrong-only candidate set, and one Teacher
    query per eligible prompt — only the selector is replaced with the
    original TLR score  Score_LH = (1 - L̂) * (1 - Ĥ)  where L̂ and Ĥ are
    min-max normalized response length and mean Student token entropy,
    normalised over the wrong candidates of each prompt (not over all K
    siblings as in the original BoN-4 TLR, because here only wrong rollouts
    are eligible).

    All-correct and all-wrong prompts are skipped (no retry, no fallback),
    matching the Frontier-PDA / Frontier-Shortest / Frontier-Random support.
    """
    if valid_mask.ndim != 2:
        raise ValueError("frontier_tlr inputs must have shape [B, K]")
    if any(
        tensor.shape != valid_mask.shape
        for tensor in (correct_mask, response_lengths, student_entropy_mean)
    ):
        raise ValueError("frontier_tlr inputs must have identical shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")

    valid = valid_mask.detach().bool().cpu()
    correct = correct_mask.detach().bool().cpu()
    lengths = response_lengths.detach().to(dtype=torch.float32, device="cpu")
    entropies = student_entropy_mean.detach().to(dtype=torch.float32, device="cpu")

    wrong = valid & ~correct
    mixed = (valid & correct).any(dim=1) & wrong.any(dim=1)
    selected = torch.zeros_like(valid)

    for prompt_index in mixed.nonzero(as_tuple=False).flatten().tolist():
        candidates = wrong[prompt_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        cand_lengths = lengths[prompt_index, candidates]
        cand_entropies = entropies[prompt_index, candidates]
        # Min-max normalise over the wrong candidates of this prompt only,
        # matching the TLR paper's within-group normalisation convention.
        norm_length = (cand_lengths - cand_lengths.min()) / (
            cand_lengths.max() - cand_lengths.min() + eps
        )
        norm_entropy = (cand_entropies - cand_entropies.min()) / (
            cand_entropies.max() - cand_entropies.min() + eps
        )
        scores = (1.0 - norm_length) * (1.0 - norm_entropy)
        chosen = candidates[int(torch.argmax(scores).item())]
        selected[prompt_index, chosen] = True

    return selected.to(device=valid_mask.device)




def _phase_of(queue_source: Any) -> str:
    """Collapse the queue label into the coarse fresh/retry phase."""
    return "retry" if str(queue_source).startswith("retry") else "fresh"


def _json_compact(value: Any) -> str:
    """Serialize small structured columns so one CSV cell stays one value."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def normalize_prompt_text(text: str) -> str:
    return " ".join(str(text).split())


def make_prompt_uid(dataset_name: str, source_index: Any, prompt_text: str, existing_id: Any = None) -> str:
    if existing_id not in (None, ""):
        return str(existing_id)
    material = f"{dataset_name}\n{source_index}\n{normalize_prompt_text(prompt_text)}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def sampled_reverse_kl_statistics(
    student_log_probs: Any, teacher_log_probs: Any, response_mask: Any
) -> dict[str, float]:
    if teacher_log_probs.requires_grad:
        raise AssertionError("Teacher sampled log-prob entering FF metrics must be detached")
    mask = response_mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return {
            "sampled_reverse_kl_sum": math.nan,
            "sampled_reverse_kl_token_mean": math.nan,
            "sampled_reverse_kl_valid_token_count": 0,
            "student_sampled_logprob_mean": math.nan,
            "teacher_sampled_logprob_mean": math.nan,
            "sampled_logratio_mean": math.nan,
        }
    ratio = student_log_probs - teacher_log_probs
    return {
        "sampled_reverse_kl_sum": float(ratio[mask].sum().item()),
        "sampled_reverse_kl_token_mean": float(ratio[mask].mean().item()),
        "sampled_reverse_kl_valid_token_count": count,
        "student_sampled_logprob_mean": float(student_log_probs[mask].mean().item()),
        "teacher_sampled_logprob_mean": float(teacher_log_probs[mask].mean().item()),
        "sampled_logratio_mean": float(ratio[mask].mean().item()),
    }


def teacher_dummy_padding_size(real_count: int, teacher_dp_world_size: int) -> int:
    if real_count < 0 or teacher_dp_world_size < 1:
        raise ValueError("real_count must be non-negative and teacher_dp_world_size must be positive")
    return (-real_count) % teacher_dp_world_size


class FFCSVWriter:
    """Rank-0 append-only per-prompt CSV with resume de-duplication."""

    def __init__(self, path: str, flush_interval: int = 1):
        self.path = Path(os.path.expandvars(path))
        self.flush_interval = flush_interval
        self.written_keys: set[str] = set()
        self._file = None
        self._writer = None
        self._pending = 0
        if self.path.exists():
            with self.path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    self.written_keys.add(self.row_key(row))

    @staticmethod
    def row_key(row: dict[str, Any]) -> str:
        base = (
            f"{row.get('run_name', '')}:{row['prompt_uid']}:{row['attempt_id']}:{row.get('selected_rollout_idx', '')}"
        )
        # Multi-epoch FF-OPD (trainer.total_epochs > 1) re-attempts the same
        # prompt in later epochs, so attempt_id restarts at 1 each epoch. Add the
        # epoch suffix only for epoch >= 2 so single-epoch keys stay byte-identical.
        base_epoch = int(row.get("base_epoch", 1) or 1)
        return base if base_epoch <= 1 else f"{base}:e{base_epoch}"

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            key = self.row_key(record)
            if key in self.written_keys:
                continue
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                existed = self.path.exists() and self.path.stat().st_size > 0
                self._file = self.path.open("a", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=list(record), extrasaction="ignore")
                if not existed:
                    self._writer.writeheader()
            self._writer.writerow(record)
            self.written_keys.add(key)
            self._pending += 1
        if self._pending >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
        self._pending = 0

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None


def boundary_contrast_csv_fieldnames(num_boundaries: int, score_mode: str | None = None) -> list[str]:
    """Fixed column schema of the per-candidate Boundary-Contrast CSV."""
    fieldnames = (
        ["prompt_id", "rollout_id", "is_selected"]
        + [f"sim_{index + 1:02d}" for index in range(num_boundaries)]
        + [f"valid_{index + 1:02d}" for index in range(num_boundaries)]
        + ["boundary_score", "best_split_idx"]
        + [
            "score_mode",
            "boundary_search_enabled",
            "window_ratio",
            "early_cosine",
            "middle1_cosine",
            "middle2_cosine",
            "aggregated_middle_cosine",
            "early_alignment",
            "plateau_drop",
            "plateau_divergence",
            "pair_utility",
            "first_span_cosine",
            "second_span_cosine",
            "length_efficiency",
            "rollout_count",
            "boundary_utility",
            "best_positive_idx",
            "best_positive_rollout_id",
            "early_similarity",
            "middle_similarity",
            "middle_divergence",
            "early_middle_drop",
            "quality_score",
            "teacher_cost",
            "cost_efficiency",
            "normalized_response_length",
            "final_score",
            "selected",
            "selected_positive_sibling_id",
            "negative_rollout_id",
            "prompt_instance_id",
            "global_step",
            "response_length",
            "valid_response_tokens",
            "finite_score",
        ]
    )
    if score_mode in PDA_SCORE_MODES:
        fieldnames += (
            [f"raw_cosine_{index + 1:02d}" for index in range(num_boundaries)]
            + [f"similarity_s_{index + 1:02d}" for index in range(num_boundaries)]
            + [f"running_max_{index + 1:02d}" for index in range(num_boundaries)]
            + [f"departure_{index + 1:02d}" for index in range(num_boundaries)]
            + [
                "prompt_length",
                "teacher_input_len",
                "group_min_teacher_input_len",
                "cost_ratio",
                "num_positive_rollouts",
                "num_negative_rollouts",
                "pda_area",
                "best_positive_area",
                "num_positive_siblings",
                "max_departure",
                "peak_departure_index",
                "peak_departure_fraction",
                "mean_departure",
                "first_similarity",
                "max_similarity",
                "final_similarity",
                "uses_hidden_difference",
                "representation_domain",
            ]
        )
    return fieldnames


class BoundaryContrastCSVWriter:
    """Rank-0 append-only CSV for per-candidate Boundary-Contrast profiles."""

    def __init__(self, path: str, num_boundaries: int):
        self.path = Path(os.path.expandvars(path))
        self.num_boundaries = num_boundaries
        self.fieldnames = boundary_contrast_csv_fieldnames(num_boundaries)
        self.written_keys: set[str] = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.written_keys.add(self.row_key(row))

    @staticmethod
    def row_key(row: dict[str, Any]) -> str:
        return ":".join(str(row.get(name, "")) for name in ("global_step", "prompt_instance_id", "negative_rollout_id"))

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if any(row.get("score_mode") in PDA_SCORE_MODES for row in rows):
            score_mode = next(row.get("score_mode") for row in rows if row.get("score_mode") in PDA_SCORE_MODES)
            self.fieldnames = boundary_contrast_csv_fieldnames(self.num_boundaries, score_mode)
        rows = [row for row in rows if self.row_key(row) not in self.written_keys]
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        if not write_header:
            with self.path.open(newline="", encoding="utf-8") as stream:
                existing_fields = next(csv.reader(stream), [])
            if existing_fields != self.fieldnames:
                raise RuntimeError(
                    f"{self.path} has an incompatible CSV header. "
                    "Start a fresh metrics directory before changing the Boundary-Contrast schema; "
                    f"existing={existing_fields}, expected={self.fieldnames}"
                )
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            self.written_keys.update(self.row_key(row) for row in rows)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        # No persistent handle: append() already flushes and fsyncs per call.
        return None


class FFProfileJSONLWriter:
    """Rank-0 append-only audit sink for full confidence profiles."""

    def __init__(self, path: str, flush_interval: int = 1):
        self.path = Path(os.path.expandvars(path))
        self.flush_interval = max(1, int(flush_interval))
        self.written_keys: set[str] = set()
        self._file = None
        self._pending = 0
        if self.path.exists():
            with self.path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.written_keys.add(self.row_key(json.loads(line)))
                    except (ValueError, KeyError):
                        continue

    @staticmethod
    def row_key(row: dict[str, Any]) -> str:
        base = f"{row.get('run_name', '')}:{row['prompt_uid']}:{row['attempt_id']}"
        # See FFCSVWriter.row_key: disambiguate re-attempts across FF-OPD epochs.
        base_epoch = int(row.get("base_epoch", 1) or 1)
        return base if base_epoch <= 1 else f"{base}:e{base_epoch}"

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            key = self.row_key(record)
            if key in self.written_keys:
                continue
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a")
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.written_keys.add(key)
            self._pending += 1
        if self._pending >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
        self._pending = 0

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None




class FFOPDQueueManager:
    """Cumulative query-cap router with bounded, round-based NO_SUCCESS retries."""

    def __init__(self, config: FFOPDConfig, run_name: str = "", tokenizer: Any = None):
        self.config = config
        self.run_name = run_name
        # Trainer tokenizer; required by DTW-PDA smoke diagnostics, which decode
        # local token windows for the blind text-alignment export.
        self.tokenizer = tokenizer
        self.prompt_states: dict[str, FFPromptState] = {}
        self.current_queue: deque[str] = deque()
        self.next_queue: deque[str] = deque()
        self.retry_round = 0
        self.total_full_opd_queries = 0
        self.used_queries = 0
        self.total_prompt_attempts = 0
        self.total_student_rollouts = 0
        # Cumulative counters behind the four headline FF-OPD analyses.
        self.total_selector_attempts = 0
        self.total_selector_fallbacks = 0
        self.total_no_success_retry_attempts = 0
        self.total_no_success_to_frontier = 0
        self.total_no_success_to_all_correct = 0
        # Cumulative counters behind the cost-induced switch-rate analysis.
        self.total_cost_comparable = 0
        self.total_cost_switches = 0
        self.student_model_version = "0"
        # Step-level L15 allocation stats, refreshed by _prepare_boundary_selector
        # whenever score_mode is an L15 mode (empty otherwise).
        self._l15_step_stats: dict[str, float] = {}
        self._l16_step_stats: dict[str, float] = {}
        self._l17_step_stats: dict[str, float] = {}
        self._l18_step_stats: dict[str, float] = {}
        self._l19_step_stats: dict[str, float] = {}
        self._sasb_step_stats: dict[str, float] = {}
        self.sasb_tau_comp: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.sasb_tau_initialized: dict[int, bool] = {1: False, 2: False, 3: False}
        self.sasb_sigma_3 = 0.0
        self.sasb_sigma_3_initialized = False
        self.boundary_calibration: BoundaryCalibrationArtifact | None = None
        if config.boundary_opd is not None and (
            config.boundary_opd.similarity_metric in BOUNDARY_CALIBRATION_ARTIFACT_METRICS
        ):
            self.boundary_calibration = load_boundary_calibration(config.boundary_opd)
        self.frontier_selector = FrontierTrajectorySelector(
            seed=config.seed,
            cost_aware=config.selector_cost_aware,
            cost_alpha=config.selector_cost_alpha,
            score_eps=config.selector_score_eps,
        )

    @property
    def fallback_rate(self) -> float:
        """Share of Frontier selections that had to fall back to uniform choice."""
        return self.total_selector_fallbacks / max(self.total_selector_attempts, 1)

    @property
    def no_success_to_frontier_rate(self) -> float:
        """Share of retried NO_SUCCESS attempts that recovered into FRONTIER."""
        return self.total_no_success_to_frontier / max(self.total_no_success_retry_attempts, 1)

    @property
    def cost_switch_rate(self) -> float:
        """Share of cost-ranked Frontier picks differing from the nearest-only pick."""
        return self.total_cost_switches / max(self.total_cost_comparable, 1)

    def should_audit_profile(self, prompt_uid: str, attempt_id: int) -> bool:
        """Deterministically sample a few attempts for full-profile auditing."""
        rate = float(self.config.profile_audit_sample_rate)
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        material = f"{self.config.seed}:{prompt_uid}:{attempt_id}:profile_audit"
        return (stable_int_hash(material) % 1_000_000) < rate * 1_000_000

    @property
    def no_success_retry_queue(self) -> list[str]:
        """Compatibility view of all prompts still pending a retry."""
        return list(self.current_queue) + list(self.next_queue)

    def advance_retry_round(self) -> bool:
        """Finish the current round and expose only its failures to the next round."""
        if self.current_queue:
            raise RuntimeError("cannot advance retry round before current_queue is drained")
        self.current_queue, self.next_queue = self.next_queue, deque()
        if not self.current_queue:
            return False
        self.retry_round += 1
        return True

    def begin_retry_round(self) -> bool:
        """Start round 1 after fresh, or advance from a drained retry round."""
        if self.retry_round == 0:
            if not self.current_queue:
                return False
            self.retry_round = 1
            return True
        return self.advance_retry_round()

    def begin_new_epoch(self) -> None:
        """Reset per-epoch fresh-pass state for a new dataset epoch.

        Multi-epoch FF-OPD (trainer.total_epochs > 1) re-runs the fresh pass with
        the current student. Prompt classification and the retry queues are
        per-epoch and must start clean so the fresh-coverage/disjointness
        assertions hold for each pass. Cumulative run-level counters, the Teacher
        query accounting, and student_model_version are intentionally preserved
        so cross-epoch analysis metrics stay continuous.
        """
        self.prompt_states = {}
        self.current_queue = deque()
        self.next_queue = deque()
        self.retry_round = 0

    def classify_prompt(
        self,
        verifier_correct: Sequence[int],
        verifier_failed: bool = False,
        rollout_valid: Optional[Sequence[bool]] = None,
    ) -> FFBucket:
        if len(verifier_correct) != self.config.k_rollouts:
            raise ValueError(f"expected {self.config.k_rollouts} sibling verifier results, got {len(verifier_correct)}")
        valid = (
            [not verifier_failed] * len(verifier_correct) if rollout_valid is None else [bool(v) for v in rollout_valid]
        )
        if len(valid) != len(verifier_correct):
            raise ValueError("rollout_valid must align with verifier_correct")
        correct_count = sum(v and bool(c) for v, c in zip(valid, verifier_correct, strict=True))
        incorrect_count = sum(v and not bool(c) for v, c in zip(valid, verifier_correct, strict=True))
        invalid_count = len(valid) - correct_count - incorrect_count
        if correct_count == self.config.k_rollouts and invalid_count == 0:
            return FFBucket.ALL_CORRECT
        if incorrect_count == self.config.k_rollouts and invalid_count == 0:
            return FFBucket.NO_SUCCESS
        if correct_count > 0 and incorrect_count > 0:
            return FFBucket.FRONTIER
        return FFBucket.INVALID

    def _build_candidate(
        self,
        prompt: dict[str, Any],
        state: FFPromptState,
    ) -> tuple[Optional[FFCandidate], Optional[FrontierSelectionResult], str]:
        selection = self.frontier_selector.select(
            response_masks=prompt.get("response_masks"),
            verifier_correct=prompt["verifier_correct"],
            rollout_valid=prompt["rollout_valid"],
            student_sampled_log_probs=prompt.get("sampled_token_log_probs"),
            prompt_uid=state.prompt_uid,
            attempt_id=state.attempt_count,
            collect_profiles=self.should_audit_profile(state.prompt_uid, state.attempt_count),
            # Teacher cost c_{p,j} = prompt + response tokens of each sibling.
            candidate_costs=prompt.get("processing_tokens"),
        )
        self.total_selector_attempts += 1
        self.total_selector_fallbacks += int(bool(selection.fallback_used))
        if selection.cost_aware and selection.selection_mode in _FRONTIER_SELECTION_MODES:
            self.total_cost_comparable += 1
            self.total_cost_switches += int(selection.cost_switch)
        sibling = selection.selected_negative_idx
        if sibling is None:
            return None, selection, FFRejectedReason.NO_VALID_INCORRECT.value
        if prompt["verifier_correct"][sibling] or not prompt["rollout_valid"][sibling]:
            raise AssertionError("Frontier candidate must be a valid incorrect rollout")
        return (
            FFCandidate(
                prompt_uid=state.prompt_uid,
                source_index=state.source_index,
                rollout_index=int(prompt["rollout_indices"][sibling]),
                teacher_processing_tokens=int(prompt["processing_tokens"][sibling]),
                response_tokens=int(prompt["response_tokens"][sibling]),
                frontier_selection=selection,
            ),
            selection,
            FFRejectedReason.NONE.value,
        )

    @staticmethod
    def _boundary_rank_map(
        sibling_indices: Sequence[int],
        values: Sequence[float],
        rollout_ids: Sequence[int],
    ) -> dict[int, int]:
        ordered = sorted(
            zip(sibling_indices, values, rollout_ids, strict=True),
            key=lambda item: (float(item[1]), int(item[2])),
        )
        return {int(sibling): rank for rank, (sibling, _, _) in enumerate(ordered, start=1)}




    def _prepare_boundary_selector(
        self,
        prompts: Sequence[dict[str, Any]],
        valid_mask: torch.Tensor,
        correct_mask: torch.Tensor,
        confidence_distance: torch.Tensor,
        teacher_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Boundary distances while retaining confidence FF-Cost for fallback/control."""

        settings = self.config.boundary_opd
        if settings is None:
            raise RuntimeError("Boundary selector settings were not initialized")
        mode = self.config.selector_mode
        selector_distance = torch.full_like(confidence_distance, float("nan"))
        fallback_mask = torch.zeros(len(prompts), dtype=torch.bool)
        contrast_selected_mask = torch.zeros_like(valid_mask)
        # L15 defers the per-prompt pick: candidates are scored exactly like
        # L8, then a batch-global Top-B allocator spends the fixed
        # Teacher-query budget B = floor(0.1 * N_prompt * K) defined against
        # the Full-OPD query count of this exact step.
        is_pda = settings.score_mode in PDA_SCORE_MODES
        error_names = {
            0: "none",
            1: "capture_target_resolution_failed",
            2: "capture_hook_not_invoked",
            3: "capture_or_sequence_gather_failed",
        }

        for prompt_index, prompt in enumerate(prompts):
            uid = str(prompt["prompt_uid"])
            valid_row = valid_mask[prompt_index]
            correct_row = correct_mask[prompt_index]
            positive_indices = (valid_row & correct_row).nonzero(as_tuple=False).flatten().tolist()
            negative_indices = (valid_row & ~correct_row).nonzero(as_tuple=False).flatten().tolist()
            rollout_ids = [int(value) for value in prompt.get("rollout_ids", range(self.config.k_rollouts))]
            capture_success = [
                bool(value)
                for value in prompt.get(
                    "boundary_hidden_capture_success",
                    [False] * self.config.k_rollouts,
                )
            ]
            capture_error_codes = [
                int(value)
                for value in prompt.get(
                    "boundary_hidden_capture_error_code",
                    [0] * self.config.k_rollouts,
                )
            ]
            capture_times = [
                float(value)
                for value in prompt.get(
                    "boundary_hidden_capture_time_ms",
                    [0.0] * self.config.k_rollouts,
                )
            ]
            communication_times = [
                float(value)
                for value in prompt.get(
                    "boundary_communication_time_ms",
                    [0.0] * self.config.k_rollouts,
                )
            ]
            extra_memory = [
                float(value)
                for value in prompt.get(
                    "boundary_extra_peak_memory_mb",
                    [0.0] * self.config.k_rollouts,
                )
            ]
            metric_compute_times = [
                float(value)
                for value in prompt.get(
                    "boundary_metric_compute_time_ms",
                    [0.0] * self.config.k_rollouts,
                )
            ]
            sinkhorn_residuals = [
                float(value)
                for value in prompt.get(
                    "boundary_sinkhorn_max_residual",
                    [0.0] * self.config.k_rollouts,
                )
            ]
            sinkhorn_iterations = [
                int(value)
                for value in prompt.get(
                    "boundary_sinkhorn_max_iterations",
                    [0] * self.config.k_rollouts,
                )
            ]
            sinkhorn_unconverged = [
                int(value)
                for value in prompt.get(
                    "boundary_sinkhorn_unconverged",
                    [0] * self.config.k_rollouts,
                )
            ]
            relevant_indices = positive_indices + negative_indices
            diagnostics: dict[str, Any] = {
                "ff_selector_mode": mode,
                "num_boundaries": settings.num_boundaries,
                "score_mode": settings.score_mode,
                "boundary_search_enabled": 0,
                "boundary_similarity_metric": settings.similarity_metric,
                "hidden_capture_module": prompt.get("boundary_hidden_capture_module", ""),
                "hidden_capture_method": prompt.get("boundary_hidden_capture_method", ""),
                "hidden_capture_success": int(
                    bool(relevant_indices) and all(capture_success[index] for index in relevant_indices)
                ),
                "hidden_dim": int(prompt.get("boundary_hidden_dim", 0) or 0),
                "hidden_shape": prompt.get("boundary_hidden_shape", ()),
                "sequence_parallel_size": int(prompt.get("boundary_sequence_parallel_size", 1) or 1),
                "sequence_parallel_sharded": int(bool(prompt.get("boundary_sequence_parallel_sharded", False))),
                "sequence_parallel_gathered": int(bool(prompt.get("boundary_sequence_parallel_gathered", True))),
                "tensor_parallel_sharded": int(bool(prompt.get("boundary_tensor_parallel_sharded", False))),
                "lm_head_module": prompt.get("boundary_lm_head_module", ""),
                "lm_head_vocab_size": int(prompt.get("boundary_lm_head_vocab_size", 0) or 0),
                "lm_head_weight_current": int(bool(prompt.get("boundary_lm_head_weight_current", False))),
                "lm_head_vocab_sharded": int(bool(prompt.get("boundary_lm_head_vocab_sharded", False))),
                "weight_mode": "pre_post_split",
                "boundary_transition_distance": "",
                "boundary_nearest_positive_rollout_id": "",
                "boundary_teacher_cost": "",
                "boundary_objective": "",
                "boundary_selected_rollout_id": "",
                "boundary_nearest_rollout_id": "",
                "boundary_cost_only_rollout_id": "",
                "boundary_random_wrong_rollout_id": "",
                "boundary_cost_switch": "",
                "boundary_hidden_available": 0,
                "boundary_fallback_used": 0,
                "boundary_fallback_reason": "",
                "boundary_degenerate_to_cost_only": 0,
                "boundary_valid_transition_count": 0,
                "num_positive_rollouts": len(positive_indices),
                "num_negative_rollouts": len(negative_indices),
                "boundary_candidates_json": [],
                "_boundary_distance_by_sibling": {},
                "_boundary_nearest_positive_by_sibling": {},
                "_boundary_objective_by_sibling": {},
                "_boundary_transition_build_time_ms": 0.0,
                "_boundary_distance_compute_time_ms": 0.0,
                "_boundary_metric_compute_time_ms": (
                    sum(metric_compute_times[index] for index in relevant_indices) / max(len(relevant_indices), 1)
                ),
                "_boundary_sinkhorn_max_residual": max(
                    (sinkhorn_residuals[index] for index in relevant_indices),
                    default=0.0,
                ),
                "_boundary_sinkhorn_max_iterations": max(
                    (sinkhorn_iterations[index] for index in relevant_indices),
                    default=0,
                ),
                "_boundary_sinkhorn_unconverged_rate": (
                    sum(sinkhorn_unconverged[index] for index in relevant_indices) / max(len(relevant_indices), 1)
                ),
                "_boundary_hidden_capture_time_ms": (
                    sum(capture_times[index] for index in relevant_indices) / max(len(relevant_indices), 1)
                ),
                "_boundary_communication_time_ms": (
                    sum(communication_times[index] for index in relevant_indices) / max(len(relevant_indices), 1)
                ),
                "_boundary_extra_peak_memory_mb": max(
                    (extra_memory[index] for index in relevant_indices),
                    default=0.0,
                ),
                "_boundary_valid_transition_total": (len(relevant_indices) * boundary_transition_count(settings)),
            }
            prompt["_boundary_diagnostics"] = diagnostics


            # Boundary selection is defined only for Frontier prompts. The
            # classifier and retry state above remain the sole authority.
            if not positive_indices or not negative_indices:
                continue



            states = prompt.get("boundary_states")
            transition_valid = prompt.get("boundary_transition_valid_mask")
            state_shape = tuple(states.shape) if isinstance(states, torch.Tensor) else None
            valid_shape = tuple(transition_valid.shape) if isinstance(transition_valid, torch.Tensor) else None
            try:
                expected_state_shape = (
                    self.config.k_rollouts,
                    boundary_state_count(settings),
                )
                expected_valid_shape = (
                    self.config.k_rollouts,
                    boundary_transition_count(settings),
                )
                if not isinstance(states, torch.Tensor):
                    raise ValueError("missing boundary_states")
                if not isinstance(transition_valid, torch.Tensor):
                    raise ValueError("missing boundary_transition_valid_mask")
                if states.ndim != 3 or tuple(states.shape[:2]) != expected_state_shape:
                    raise ValueError(f"boundary_states must start with shape {expected_state_shape}")
                if tuple(transition_valid.shape) != expected_valid_shape:
                    raise ValueError(f"boundary_transition_valid_mask must have shape {expected_valid_shape}")
                if states.requires_grad:
                    raise ValueError("boundary_states.requires_grad must be False")
                if not all(capture_success[index] for index in relevant_indices):
                    failed_codes = sorted(
                        {capture_error_codes[index] for index in relevant_indices if not capture_success[index]}
                    )
                    failed_names = [error_names.get(code, f"unknown_capture_error_{code}") for code in failed_codes]
                    raise ValueError("hidden capture failed: " + ",".join(failed_names))
                if not torch.isfinite(states[relevant_indices].float()).all():
                    raise ValueError("boundary_states contains NaN or Inf")

                states = states.detach()
                transition_valid = transition_valid.detach().bool()
                diagnostics["boundary_hidden_available"] = 1
                diagnostics["hidden_dim"] = int(states.shape[-1])
                diagnostics["boundary_valid_transition_count"] = int(transition_valid[relevant_indices].sum().item())

                negative_index_tensor = torch.as_tensor(
                    negative_indices,
                    dtype=torch.long,
                    device=states.device,
                )
                positive_index_tensor = torch.as_tensor(
                    positive_indices,
                    dtype=torch.long,
                    device=states.device,
                )
                build_start = time.perf_counter()

                distance_start = time.perf_counter()
                response_lengths = torch.as_tensor(
                    [prompt["response_tokens"][index] for index in negative_indices],
                    dtype=torch.long,
                    device=states.device,
                )
                prompt_length = int(prompt["processing_tokens"][0]) - int(prompt["response_tokens"][0])
                if prompt_length <= 0:
                    raise ValueError("effective prompt length must be positive")
                negative_rollout_id_tensor = torch.as_tensor(
                    [rollout_ids[index] for index in negative_indices],
                    dtype=torch.long,
                    device=states.device,
                )
                if settings.score_mode != "persistent_departure_area":
                    raise ValueError(
                        "This release implements the PDA routing rule only "
                        f"(score_mode='persistent_departure_area'), got {settings.score_mode!r}"
                    )
                if is_pda:
                    if self.boundary_calibration is None:
                        raise RuntimeError("missing frozen direct-hidden calibration for PDA")
                    centered_states = build_centered_hidden_states(
                        states,
                        transition_valid,
                        mean=self.boundary_calibration.mean,
                        eps=settings.hidden_norm_epsilon,
                    )
                    negative_states = centered_states.index_select(0, negative_index_tensor)
                    positive_states = centered_states.index_select(0, positive_index_tensor)
                    direct_cosine = torch.einsum("nmd,pmd->npm", negative_states, positive_states)
                    contrast = compute_persistent_departure_area_scores_from_cosine(
                        direct_cosine,
                        prompt_length,
                        response_lengths,
                        negative_rollout_id_tensor,
                        positive_aggregation="max",
                    )
                    del centered_states, negative_states, positive_states, direct_cosine
                diagnostics["_boundary_transition_build_time_ms"] = (time.perf_counter() - build_start) * 1000.0
                diagnostics["_boundary_distance_compute_time_ms"] = (time.perf_counter() - distance_start) * 1000.0
                selected_local = int(contrast["selected_local_index"].item())
                contrast_selected_mask[prompt_index, negative_indices[selected_local]] = True
                diagnostics["boundary_degenerate_to_cost_only"] = int(
                    contrast["boundary_degenerate_to_cost_only"].item()
                )
                boundary_utility = contrast["boundary_utility"].cpu()
                negative_cost = contrast["teacher_cost"].cpu()
                objective = contrast["final_score"].cpu()
                nearest_distance = (1.0 - boundary_utility).cpu()
                nearest_positive_local = contrast["best_positive_local_index"].cpu()

                distance_by_sibling = {
                    int(sibling): float(nearest_distance[local]) for local, sibling in enumerate(negative_indices)
                }
                objective_by_sibling = {
                    int(sibling): float(objective[local]) for local, sibling in enumerate(negative_indices)
                }
                nearest_positive_by_sibling = {
                    int(sibling): (
                        int(rollout_ids[positive_indices[int(nearest_positive_local[local].item())]])
                        if int(nearest_positive_local[local].item()) >= 0
                        else None
                    )
                    for local, sibling in enumerate(negative_indices)
                }
                diagnostics["_boundary_distance_by_sibling"] = distance_by_sibling
                diagnostics["_boundary_objective_by_sibling"] = objective_by_sibling
                diagnostics["_boundary_nearest_positive_by_sibling"] = nearest_positive_by_sibling
                for sibling, value in distance_by_sibling.items():
                    selector_distance[prompt_index, sibling] = value

                confidence_values = [float(confidence_distance[prompt_index, sibling]) for sibling in negative_indices]
                # The confidence profile can be unavailable for a whole prompt
                # (token alignment failure). Do not invent a ranking for it.
                confidence_ranked = all(math.isfinite(value) for value in confidence_values)
                transition_values = [distance_by_sibling[sibling] for sibling in negative_indices]
                objective_values = [objective_by_sibling[sibling] for sibling in negative_indices]
                transition_ranks = self._boundary_rank_map(
                    negative_indices, transition_values, [rollout_ids[i] for i in negative_indices]
                )
                confidence_ranks = (
                    self._boundary_rank_map(
                        negative_indices,
                        confidence_values,
                        [rollout_ids[i] for i in negative_indices],
                    )
                    if confidence_ranked
                    else {}
                )
                objective_ranks = self._boundary_rank_map(
                    negative_indices, objective_values, [rollout_ids[i] for i in negative_indices]
                )
                diagnostics["boundary_candidates_json"] = []
                for local, sibling in enumerate(negative_indices):
                    candidate_record = {
                        "prompt_id": uid,
                        "rollout_id": rollout_ids[sibling],
                        "is_positive": 0,
                        "is_negative": 1,
                        "is_selected": 0,
                        "is_frontier": 1,
                        "prompt_length": int(prompt["processing_tokens"][0]) - int(prompt["response_tokens"][0]),
                        "response_length": int(prompt["response_tokens"][sibling]),
                        "transition_distance": distance_by_sibling[sibling],
                        "confidence_distance": (
                            float(confidence_distance[prompt_index, sibling]) if confidence_ranked else None
                        ),
                        "teacher_cost": float(negative_cost[local]),
                        "boundary_objective": objective_by_sibling[sibling],
                        "rank_transition": transition_ranks[sibling],
                        "rank_confidence": confidence_ranks.get(sibling),
                        "rank_objective": objective_ranks[sibling],
                        "selected": 0,
                        "has_eos": int(bool(prompt["rollout_valid"][sibling])),
                        "is_truncated": 0,
                        "retry_round": self.retry_round,
                        "num_positive_rollouts": len(positive_indices),
                        "num_negative_rollouts": len(negative_indices),
                    }
                    best_positive_local = int(contrast["best_positive_local_index"][local].item())
                    candidate_record.update(
                        {
                            "score_mode": settings.score_mode,
                            "boundary_search_enabled": int(contrast["boundary_search_enabled"]),
                            "window_ratio": settings.window_ratio,
                            "boundary_utility": float(contrast["boundary_utility"][local]),
                            "early_cosine": float(contrast["early_cosine"][local]),
                            "middle1_cosine": float(contrast["middle1_cosine"][local]),
                            "middle2_cosine": float(contrast["middle2_cosine"][local]),
                            "aggregated_middle_cosine": float(contrast["aggregated_middle_cosine"][local]),
                            "early_alignment": float(contrast["early_alignment"][local]),
                            "plateau_drop": float(contrast["plateau_drop"][local]),
                            "plateau_divergence": float(contrast["plateau_divergence"][local]),
                            "pair_utility": float(contrast["pair_utility"][local]),
                            "first_span_cosine": float(contrast["first_span_cosine"][local]),
                            "second_span_cosine": float(contrast["second_span_cosine"][local]),
                            "cost_efficiency": float(contrast["cost_efficiency"][local]),
                            "teacher_input_len": float(contrast["teacher_input_len"][local]),
                            "group_min_response_len": float(contrast["group_min_response_len"]),
                            "group_min_teacher_input_len": float(contrast["group_min_teacher_input_len"]),
                            "cost_ratio": float(contrast["cost_ratio"][local]),
                            "normalized_response_length": float(contrast["normalized_response_length"][local]),
                            "length_efficiency": float(contrast["length_efficiency"][local]),
                            "rollout_count": int(contrast["rollout_count"].item()),
                            "boundary_final_score": float(contrast["final_score"][local]),
                            "best_positive_rollout_id": (
                                rollout_ids[positive_indices[best_positive_local]] if best_positive_local >= 0 else None
                            ),
                            "best_positive_idx": best_positive_local,
                            "best_split_index": int(contrast["best_split_index"][local]),
                            "best_split_fraction": (
                                float(contrast["best_split_index"][local])
                                / settings.num_boundaries
                                if int(contrast["best_split_index"][local]) >= 0
                                else -1.0
                            ),
                            "pre_similarity": float(contrast["best_pre_similarity"][local]),
                            "post_similarity": float(contrast["best_post_similarity"][local]),
                            "fork_drop": float(contrast["best_fork_drop"][local]),
                            "post_divergence": float(contrast["best_post_divergence"][local]),
                            "early_similarity": float(contrast["best_pre_similarity"][local]),
                            "middle_similarity": float(contrast["best_post_similarity"][local]),
                            "middle_divergence": float(contrast["best_post_divergence"][local]),
                            "early_middle_drop": float(contrast["best_fork_drop"][local]),
                            "quality_score": float(contrast["boundary_utility"][local]),
                            "finite_score": int(torch.isfinite(contrast["final_score"][local]).item()),
                            "valid_split_count": int(contrast["valid_split_count"][local]),
                            "valid_candidate": int(contrast["valid_candidate_mask"][local]),
                            "stage_similarities": [
                                round(float(value), 6) for value in contrast["best_stage_similarity"][local].tolist()
                            ],
                            "stage_valid_mask": [
                                int(value) for value in contrast["best_stage_valid_mask"][local].tolist()
                            ],
                            "boundary_degenerate_to_cost_only": diagnostics["boundary_degenerate_to_cost_only"],
                            "selection_probability": 1.0,
                            "inverse_probability_weight": 1.0,
                        }
                    )
                    if is_pda:
                        valid_state_count = int(contrast["best_stage_valid_mask"][local].sum().item())
                        valid_similarity = contrast["best_stage_similarity"][local, :valid_state_count]
                        valid_running = contrast["best_running_max"][local, :valid_state_count]
                        valid_departure = contrast["best_departure"][local, :valid_state_count]
                        valid_raw_cosine = contrast["best_raw_cosine"][local, :valid_state_count]
                        candidate_record.update(
                            {
                                "pda_area": float(contrast["pda_area"][local]),
                                "best_positive_area": float(contrast["pda_area"][local]),
                                "num_positive_siblings": len(positive_indices),
                                "max_departure": float(contrast["max_departure"][local]),
                                "peak_departure_index": int(contrast["peak_departure_index"][local]),
                                "peak_departure_fraction": float(contrast["peak_departure_index"][local])
                                / (settings.num_boundaries - 1),
                                "mean_departure": float(contrast["mean_departure"][local]),
                                "first_similarity": float(valid_similarity[0]),
                                "max_similarity": float(valid_similarity.max()),
                                "final_similarity": float(valid_similarity[-1]),
                                "running_max": [float(v) for v in valid_running.tolist()],
                                "departure": [float(v) for v in valid_departure.tolist()],
                                "raw_cosine": [float(v) for v in valid_raw_cosine.tolist()],
                                "normalized_similarity": [
                                    float(v) for v in valid_similarity.tolist()
                                ],
                                "uses_hidden_difference": False,
                                "representation_domain": "direct_hidden_state",
                            }
                        )
                    diagnostics["boundary_candidates_json"].append(candidate_record)
                diagnostics["_boundary_confidence_spearman"] = spearman_correlation(
                    nearest_distance,
                    torch.as_tensor(confidence_values, dtype=torch.float32),
                )
            except Exception as error:
                reason = str(error)
                diagnostics["boundary_fallback_reason"] = reason
                if not settings.fallback_to_ff_cost:
                    raise RuntimeError(
                        "Boundary-OPD failed with fallback disabled: "
                        f"prompt_id={uid} boundary_states_shape={state_shape} "
                        f"transition_valid_mask_shape={valid_shape} reason={reason}"
                    ) from error
                fallback_distances = confidence_distance[prompt_index, negative_indices]
                if not torch.isfinite(fallback_distances).all():
                    raise RuntimeError(
                        "Boundary-OPD FF-Cost fallback is unavailable: "
                        f"prompt_id={uid} boundary_states_shape={state_shape} "
                        f"transition_valid_mask_shape={valid_shape} reason={reason}; "
                        "confidence-profile distance is non-finite"
                    ) from error
                fallback_mask[prompt_index] = True
                selector_distance[prompt_index, negative_indices] = fallback_distances
                diagnostics["boundary_fallback_used"] = 1
                diagnostics["boundary_candidates_json"] = [
                    {
                        "rollout_id": rollout_ids[sibling],
                        "transition_distance": None,
                        "confidence_distance": float(confidence_distance[prompt_index, sibling]),
                        "teacher_cost": float(teacher_cost[prompt_index, sibling]),
                        "boundary_objective": None,
                        "rank_transition": None,
                        "rank_confidence": rank,
                        "rank_objective": None,
                        "selected": 0,
                    }
                    for rank, sibling in enumerate(
                        sorted(
                            negative_indices,
                            key=lambda index: (
                                float(confidence_distance[prompt_index, index]),
                                rollout_ids[index],
                            ),
                        ),
                        start=1,
                    )
                ]
        return selector_distance, fallback_mask, contrast_selected_mask

    def _pack_query_cap(self, candidates: Sequence[FFCandidate]) -> tuple[list[FFCandidate], int, int]:
        cumulative_cap = math.floor(self.config.legacy_teacher_query_ratio * self.total_full_opd_queries)
        available = max(0, cumulative_cap - self.used_queries)
        ordered = sorted(candidates, key=lambda candidate: (candidate.prompt_uid, candidate.rollout_index))
        for position, candidate in enumerate(ordered):
            candidate.query_cap_before = max(0, available - position)
            candidate.used_queries_before = self.used_queries
            if position < available:
                candidate.selected_for_teacher = True
                self.used_queries += 1
            else:
                candidate.rejected_reason = FFRejectedReason.QUERY_CAP_REACHED.value
            candidate.used_queries_after = self.used_queries
            candidate.query_cap_after = max(0, available - position - int(candidate.selected_for_teacher))
        return ordered, available, sum(candidate.selected_for_teacher for candidate in ordered)

    def route_prompt_attempts(self, prompts: Sequence[dict[str, Any]]) -> FFRouteResult:
        analyses: list[Optional[FrontierSelectionResult]] = []
        records: list[dict[str, Any]] = []
        bucket_counts = {bucket.value: 0 for bucket in FFBucket}
        used_before_step = self.used_queries
        pending: list[tuple[dict[str, Any], FFPromptState, FFBucket, str]] = []

        for prompt in prompts:
            uid = str(prompt["prompt_uid"])
            state = self.prompt_states.get(uid)
            if state is None:
                state = FFPromptState(
                    uid,
                    str(prompt["source_index"]),
                    retries_left=self.config.max_no_success_retries,
                    retries_left_before=self.config.max_no_success_retries,
                )
                self.prompt_states[uid] = state
                state.retries_left_before = state.retries_left
                # The 25% Teacher budget is based only on the one-pass fresh
                # Full-OPD baseline. Retry rollouts never enlarge it.
                self.total_full_opd_queries += self.config.k_rollouts
            else:
                if self.retry_round == 0:
                    raise RuntimeError(f"prompt {uid} is not eligible for retry before retry round 1")
                if uid not in self.current_queue:
                    raise RuntimeError(f"prompt {uid} is not eligible for retry")
                if state.retries_left <= 0:
                    raise RuntimeError(f"prompt {uid} exhausted its retry limit")
                self.current_queue.remove(uid)
                state.retry_count += 1
                state.retries_left_before = state.retries_left
                state.retries_left -= 1
            correct = [int(value) for value in prompt["verifier_correct"]]
            valid = [bool(value) for value in prompt["rollout_valid"]]
            bucket = self.classify_prompt(correct, bool(prompt.get("verifier_failed", False)), valid)
            previous_bucket = state.current_bucket
            state.attempt_count += 1
            state.previous_bucket = previous_bucket
            state.current_bucket = bucket.value
            state.last_correct_count = sum(v and bool(c) for v, c in zip(valid, correct, strict=True))
            state.last_attempt_model_version = self.student_model_version
            if previous_bucket == FFBucket.NO_SUCCESS.value:
                # A retried NO_SUCCESS prompt: track whether it recovered.
                self.total_no_success_retry_attempts += 1
                self.total_no_success_to_frontier += int(bucket == FFBucket.FRONTIER)
                self.total_no_success_to_all_correct += int(bucket == FFBucket.ALL_CORRECT)
            if bucket == FFBucket.NO_SUCCESS:
                if state.attempt_count == 1 and state.retries_left > 0:
                    state.final_status = FFFinalStatus.RETRY_PENDING.value
                    self.current_queue.append(uid)
                elif state.retries_left > 0:
                    state.final_status = FFFinalStatus.RETRY_PENDING.value
                    self.next_queue.append(uid)
                else:
                    state.final_status = FFFinalStatus.EXHAUSTED_NO_SUCCESS.value
            else:
                state.final_status = FFFinalStatus.COMPLETE.value
            self.total_prompt_attempts += 1
            self.total_student_rollouts += len(correct)
            bucket_counts[bucket.value] += 1
            if bucket == FFBucket.FRONTIER:
                if self.config.selector_mode in {"shortest_wrong", "all_wrong", "frontier_tlr"}:
                    # These modes bypass the legacy confidence-trajectory
                    # analyzer; their dedicated masks below are authoritative.
                    selection = FrontierSelectionResult(
                        selector_type=self.config.selector_mode,
                        selection_mode=self.config.selector_mode,
                        positive_indices=tuple(
                            index for index, (is_valid, is_correct) in enumerate(zip(valid, correct, strict=True))
                            if is_valid and bool(is_correct)
                        ),
                        negative_indices=tuple(
                            index for index, (is_valid, is_correct) in enumerate(zip(valid, correct, strict=True))
                            if is_valid and not bool(is_correct)
                        ),
                    )
                    reason = FFRejectedReason.NONE.value
                else:
                    _, selection, reason = self._build_candidate(prompt, state)
            else:
                selection = None
                reason = (
                    FFRejectedReason.ALL_CORRECT.value
                    if bucket == FFBucket.ALL_CORRECT
                    else FFRejectedReason.NO_VALID_ROLLOUT.value
                )
            analyses.append(selection)
            pending.append((prompt, state, bucket, reason))

        batch_size = len(prompts)
        k_rollouts = self.config.k_rollouts
        valid_mask = torch.tensor([prompt["rollout_valid"] for prompt in prompts], dtype=torch.bool)
        correct_mask = torch.tensor([prompt["verifier_correct"] for prompt in prompts], dtype=torch.bool)
        teacher_cost = torch.tensor([prompt["processing_tokens"] for prompt in prompts], dtype=torch.float64)
        # Confidence-profile distance is always computed, including in Boundary
        # modes, because it remains the exact FF-Cost fallback and audit control.
        distance = torch.full((batch_size, k_rollouts), float("nan"), dtype=torch.float64)
        for prompt_index, selection in enumerate(analyses):
            if selection is None:
                continue
            wrong_indices = (valid_mask[prompt_index] & ~correct_mask[prompt_index]).nonzero(as_tuple=False).flatten()
            for sibling in wrong_indices.tolist():
                candidate_distance = selection.candidate_nearest_positive_distances.get(sibling)
                if candidate_distance is not None:
                    distance[prompt_index, sibling] = candidate_distance

        selector_distance = distance
        boundary_fallback_mask = None
        global_step = int(prompts[0].get("global_step", 0)) if prompts else 0
        response_lengths = torch.tensor([prompt["response_tokens"] for prompt in prompts], dtype=torch.long)
        rollout_id_tensor = torch.tensor(
            [prompt.get("rollout_ids", range(k_rollouts)) for prompt in prompts], dtype=torch.long
        )
        if is_boundary_selector_mode(self.config.selector_mode):
            selector_distance, boundary_fallback_mask, contrast_selected_mask = self._prepare_boundary_selector(
                prompts,
                valid_mask,
                correct_mask,
                distance,
                teacher_cost,
            )
            selected_mask = contrast_selected_mask
            frontier_mask = correct_mask.any(dim=1) & (valid_mask & ~correct_mask).any(dim=1)
            selected_count = int(selected_mask.sum().item())
            target_prompt_mask = frontier_mask
            selector_stats = {
                "frontier_prompt_count": float(frontier_mask.sum().item()),
                "target_query_count": float(target_prompt_mask.sum().item()),
                "actual_query_count": float(selected_count),
                "selected_valid_count": float((selected_mask & valid_mask).sum().item()),
                "selected_correct_count": float((selected_mask & correct_mask).sum().item()),
                "selected_wrong_count": float((selected_mask & ~correct_mask).sum().item()),
                "selected_frontier_count": float((selected_mask & frontier_mask[:, None]).sum().item()),
                "selected_nonfrontier_count": float((selected_mask & ~frontier_mask[:, None]).sum().item()),
                "selected_distance_mean": 0.0,
                "selected_cost_mean": (
                    float(teacher_cost[selected_mask].float().mean().item()) if selected_count else 0.0
                ),
                "selected_objective_mean": 0.0,
                "all_wrong_distance_mean": 0.0,
                "all_wrong_cost_mean": float(teacher_cost[valid_mask & ~correct_mask].float().mean().item()),
                "tie_count": 0.0,
                "zero_frontier_step": float(not frontier_mask.any()),
                "boundary_fallback_prompt_count": 0.0,
            }
            # Controls are evaluated with the same seeded tie-breaking as their
            # real selector modes. They never alter the Teacher query mask.
            nearest_control_mask, _ = select_ff_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                distance=selector_distance,
                teacher_cost=teacher_cost,
                mode="nearest_only",
                base_seed=self.config.seed,
                global_step=global_step,
                eps=self.config.selector_score_eps,
            )
            cost_control_mask, _ = select_ff_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                distance=selector_distance,
                teacher_cost=teacher_cost,
                mode="cost_only",
                base_seed=self.config.seed,
                global_step=global_step,
                eps=self.config.selector_score_eps,
            )
            random_wrong_control_mask, _ = select_ff_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                distance=selector_distance,
                teacher_cost=teacher_cost,
                mode="random_wrong",
                base_seed=self.config.seed,
                global_step=global_step,
                eps=self.config.selector_score_eps,
            )
        elif self.config.selector_mode == "shortest_wrong":
            selected_mask = select_shortest_wrong_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                response_lengths=response_lengths,
                rollout_ids=rollout_id_tensor,
            )
            wrong_mask = valid_mask & ~correct_mask
            frontier_mask = correct_mask.any(dim=1) & wrong_mask.any(dim=1)
            selected_count = int(selected_mask.sum().item())
            selector_stats = {
                "frontier_prompt_count": float(frontier_mask.sum().item()),
                "target_query_count": float(frontier_mask.sum().item()),
                "actual_query_count": float(selected_count),
                "selected_valid_count": float((selected_mask & valid_mask).sum().item()),
                "selected_correct_count": 0.0,
                "selected_wrong_count": float(selected_count),
                "selected_frontier_count": float(selected_count),
                "selected_nonfrontier_count": 0.0,
                "selected_distance_mean": 0.0,
                "selected_cost_mean": (
                    float(teacher_cost[selected_mask].float().mean().item()) if selected_count else 0.0
                ),
                "selected_objective_mean": 0.0,
                "all_wrong_distance_mean": 0.0,
                "all_wrong_cost_mean": (
                    float(teacher_cost[wrong_mask].float().mean().item()) if wrong_mask.any() else 0.0
                ),
                "tie_count": 0.0,
                "zero_frontier_step": float(not frontier_mask.any()),
                "boundary_fallback_prompt_count": 0.0,
            }
        elif self.config.selector_mode == "frontier_tlr":
            # Frontier-TLR: matched-support baseline for Frontier-PDA.
            # Mixed-only gating and wrong-only candidate set are identical to
            # Frontier-PDA / Frontier-Shortest; only the selector is replaced
            # with the TLR score  Score_LH = (1 - L̂) * (1 - Ĥ).
            #
            # Compute per-rollout mean Student entropy from the token-level
            # entropy tensors stored in each prompt dict.  Fall back to zeros
            # when entropies are unavailable (e.g. LOG_PROB_TOP_K=0 path).
            entropy_rows: list[list[float]] = []
            for prompt in prompts:
                token_entropies = prompt.get("student_token_entropies")
                resp_masks = prompt.get("response_masks")
                if token_entropies is not None and resp_masks is not None:
                    # token_entropies: [K, T], resp_masks: [K, T]
                    ent = token_entropies.detach().float().cpu()
                    mask = resp_masks.detach().bool().cpu()
                    valid_counts = mask.sum(dim=-1).clamp_min(1).float()
                    means = (ent * mask).sum(dim=-1) / valid_counts
                    entropy_rows.append(means.tolist())
                else:
                    entropy_rows.append([0.0] * k_rollouts)
            student_entropy_mean = torch.tensor(entropy_rows, dtype=torch.float32)
            selected_mask = select_frontier_tlr_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                response_lengths=response_lengths.float(),
                student_entropy_mean=student_entropy_mean,
            )
            wrong_mask = valid_mask & ~correct_mask
            frontier_mask = correct_mask.any(dim=1) & wrong_mask.any(dim=1)
            selected_count = int(selected_mask.sum().item())
            selector_stats = {
                "frontier_prompt_count": float(frontier_mask.sum().item()),
                "target_query_count": float(frontier_mask.sum().item()),
                "actual_query_count": float(selected_count),
                "selected_valid_count": float((selected_mask & valid_mask).sum().item()),
                "selected_correct_count": 0.0,
                "selected_wrong_count": float(selected_count),
                "selected_frontier_count": float(selected_count),
                "selected_nonfrontier_count": 0.0,
                "selected_distance_mean": 0.0,
                "selected_cost_mean": (
                    float(teacher_cost[selected_mask].float().mean().item()) if selected_count else 0.0
                ),
                "selected_objective_mean": 0.0,
                "all_wrong_distance_mean": 0.0,
                "all_wrong_cost_mean": (
                    float(teacher_cost[wrong_mask].float().mean().item()) if wrong_mask.any() else 0.0
                ),
                "tie_count": 0.0,
                "zero_frontier_step": float(not frontier_mask.any()),
                "boundary_fallback_prompt_count": 0.0,
            }
        else:
            # Keep the legacy call shape and parameters unchanged for all six
            # existing modes.
            selected_mask, selector_stats = select_ff_rollouts(
                valid_mask=valid_mask,
                correct_mask=correct_mask,
                distance=distance,
                teacher_cost=teacher_cost,
                mode=self.config.selector_mode,
                base_seed=self.config.seed,
                global_step=global_step,
                eps=self.config.selector_score_eps,
            )
        selected_pairs = selected_mask.nonzero(as_tuple=False).tolist()
        ht_enabled = False
        ht_full_valid_response_tokens = 0
        ht_full_candidate_count = 0

        if is_boundary_selector_mode(self.config.selector_mode):
            for prompt_index, sibling in selected_pairs:
                prompt = prompts[prompt_index]
                diagnostics = prompt["_boundary_diagnostics"]
                rollout_ids = [int(value) for value in prompt.get("rollout_ids", range(self.config.k_rollouts))]

                def selected_sibling(control_mask: torch.Tensor, row: int = prompt_index) -> Optional[int]:
                    selected_local = control_mask[row].nonzero(as_tuple=False).flatten()
                    return int(selected_local[0].item()) if selected_local.numel() else None

                nearest_sibling = selected_sibling(nearest_control_mask)
                cost_sibling = selected_sibling(cost_control_mask)
                random_sibling = selected_sibling(random_wrong_control_mask)
                selected_rollout_id = rollout_ids[sibling]
                diagnostics["boundary_selected_rollout_id"] = selected_rollout_id
                diagnostics["boundary_nearest_rollout_id"] = (
                    rollout_ids[nearest_sibling] if nearest_sibling is not None else ""
                )
                diagnostics["boundary_cost_only_rollout_id"] = (
                    rollout_ids[cost_sibling] if cost_sibling is not None else ""
                )
                diagnostics["boundary_random_wrong_rollout_id"] = (
                    rollout_ids[random_sibling] if random_sibling is not None else ""
                )
                diagnostics["boundary_cost_switch"] = int(nearest_sibling is not None and sibling != nearest_sibling)
                diagnostics["boundary_teacher_cost"] = float(teacher_cost[prompt_index, sibling])
                if not diagnostics["boundary_fallback_used"]:
                    distance_by_sibling = diagnostics["_boundary_distance_by_sibling"]
                    objective_by_sibling = diagnostics["_boundary_objective_by_sibling"]
                    diagnostics["boundary_transition_distance"] = distance_by_sibling[sibling]
                    diagnostics["boundary_nearest_positive_rollout_id"] = diagnostics[
                        "_boundary_nearest_positive_by_sibling"
                    ][sibling]
                    diagnostics["boundary_objective"] = objective_by_sibling[sibling]
                for candidate_record in diagnostics["boundary_candidates_json"]:
                    candidate_record["selected"] = int(int(candidate_record["rollout_id"]) == selected_rollout_id)
                    if candidate_record["selected"]:
                        diagnostics.update(
                            {
                                "boundary_utility": candidate_record["boundary_utility"],
                                "boundary_final_score": candidate_record["boundary_final_score"],
                                "boundary_best_positive_rollout_id": candidate_record["best_positive_rollout_id"],
                                "boundary_best_split_index": candidate_record["best_split_index"],
                                "boundary_best_split_fraction": candidate_record["best_split_fraction"],
                                "boundary_pre_similarity": candidate_record["pre_similarity"],
                                "boundary_post_similarity": candidate_record["post_similarity"],
                                "boundary_fork_drop": candidate_record["fork_drop"],
                                "boundary_post_divergence": candidate_record["post_divergence"],
                                "score_mode": candidate_record["score_mode"],
                                "boundary_search_enabled": candidate_record["boundary_search_enabled"],
                                "early_similarity": candidate_record["early_similarity"],
                                "middle_similarity": candidate_record["middle_similarity"],
                                "middle_divergence": candidate_record["middle_divergence"],
                                "early_middle_drop": candidate_record["early_middle_drop"],
                                "early_cosine": candidate_record["early_cosine"],
                                "middle1_cosine": candidate_record["middle1_cosine"],
                                "middle2_cosine": candidate_record["middle2_cosine"],
                                "aggregated_middle_cosine": candidate_record["aggregated_middle_cosine"],
                                "early_alignment": candidate_record["early_alignment"],
                                "plateau_drop": candidate_record["plateau_drop"],
                                "plateau_divergence": candidate_record["plateau_divergence"],
                                "pair_utility": candidate_record["pair_utility"],
                                "first_span_cosine": candidate_record["first_span_cosine"],
                                "second_span_cosine": candidate_record["second_span_cosine"],
                                "normalized_response_length": candidate_record["normalized_response_length"],
                                "length_efficiency": candidate_record["length_efficiency"],
                                "rollout_count": candidate_record["rollout_count"],
                                "best_positive_idx": candidate_record["best_positive_idx"],
                                "utility_definition": "legacy",
                                "boundary_valid_split_count": candidate_record["valid_split_count"],
                                "boundary_cost_efficiency": candidate_record["cost_efficiency"],
                                "boundary_selected_is_cost_only": diagnostics["boundary_degenerate_to_cost_only"],
                                "boundary_cost_only_agreement": int(
                                    cost_sibling is not None and sibling == cost_sibling
                                ),
                                "l11_score": candidate_record.get("l11_score", ""),
                                "l11_anchor_similarity": candidate_record.get("anchor_similarity", ""),
                                "l11_body_similarity": candidate_record.get("body_similarity", ""),
                                "l11_persistent_drop": candidate_record.get("persistent_drop", ""),
                                "l11_reference_positive_index": candidate_record.get(
                                    "reference_positive_index", ""
                                ),
                                "l11_unique_state_count": candidate_record.get("unique_state_count", ""),
                                "l11_duplicate_state_fraction": candidate_record.get(
                                    "duplicate_state_fraction", ""
                                ),
                                "l11_score_margin": candidate_record.get("score_margin", ""),
                                "l11_score_margin_normalized": candidate_record.get(
                                    "score_margin_normalized", ""
                                ),
                            }
                        )
                        diagnostics["boundary_teacher_cost"] = candidate_record["teacher_cost"]
        selected_indices = [
            int(prompts[prompt_index]["rollout_indices"][sibling]) for prompt_index, sibling in selected_pairs
        ]
        used_this_step = len(selected_indices)
        available = int(selector_stats["target_query_count"])
        self.used_queries += used_this_step
        cumulative_cap = self.used_queries

        candidates_by_prompt: dict[int, list[FFCandidate]] = {}
        packed: list[FFCandidate] = []
        for offset, (prompt_index, sibling) in enumerate(selected_pairs):
            prompt, state, bucket, _ = pending[prompt_index]
            selection = copy.deepcopy(analyses[prompt_index]) or FrontierSelectionResult()
            previously_selected = selection.selected_negative_idx
            selection.selected_negative_idx = int(sibling)
            selection.selector_type = self.config.selector_mode
            selection.selection_mode = self.config.selector_mode
            selection.cost_aware = self.config.selector_cost_aware
            selection.selected_cost = float(teacher_cost[prompt_index, sibling])
            selected_distance = float(distance[prompt_index, sibling])
            if math.isfinite(selected_distance):
                candidate_distances = {
                    int(index): float(distance[prompt_index, index])
                    for index in (valid_mask[prompt_index] & ~correct_mask[prompt_index])
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                    if math.isfinite(float(distance[prompt_index, index]))
                }
                candidate_costs = {
                    int(index): float(teacher_cost[prompt_index, index]) for index in candidate_distances
                }
                selection.candidate_nearest_positive_distances = candidate_distances
                selection.candidate_costs = candidate_costs
                selection.candidate_objectives = {
                    index: (value + self.config.selector_score_eps) * math.sqrt(candidate_costs[index])
                    for index, value in candidate_distances.items()
                }
                selection.candidate_log_scores = {
                    index: -math.log(value + self.config.selector_score_eps) - 0.5 * math.log(candidate_costs[index])
                    for index, value in candidate_distances.items()
                }
                selection.nearest_positive_profile_distance = selected_distance
                selection.selected_objective = selection.candidate_objectives[int(sibling)]
                selection.selected_log_score = selection.candidate_log_scores[int(sibling)]
                ranked_distances = sorted(candidate_distances.values())
                selection.distance_min = ranked_distances[0]
                selection.distance_second_min = ranked_distances[1] if len(ranked_distances) > 1 else None
                selection.distance_gap = (
                    selection.distance_second_min - selection.distance_min
                    if selection.distance_second_min is not None
                    else None
                )
                nearest_distance = min(candidate_distances.values())
                nearest_indices = [index for index, value in candidate_distances.items() if value == nearest_distance]
                # This control field is audit-only. Selector ties themselves
                # are resolved by select_ff_rollouts using the step seed.
                nearest_index = min(nearest_indices)
                lowest_cost = min(
                    candidate_costs,
                    key=lambda index: (candidate_costs[index], index),
                )
                selection.nearest_only_selected_idx = nearest_index
                selection.lowest_cost_selected_idx = lowest_cost
                selection.nearest_only_cost = candidate_costs[nearest_index]
                selection.cost_switch = bool(self.config.selector_cost_aware and sibling != nearest_index)
                selection.distance_regret = selected_distance - nearest_distance
                selection.relative_cost_reduction = (
                    1.0 - selection.selected_cost / max(selection.nearest_only_cost, 1.0)
                    if selection.cost_switch
                    else 0.0
                )
            else:
                selection.nearest_positive_profile_distance = None
                selection.selected_objective = None
                selection.selected_log_score = None
                selection.cost_switch = False
                selection.distance_regret = 0.0
                selection.relative_cost_reduction = 0.0
            if previously_selected != sibling:
                # The legacy analyzer retained profiles only for its own pick.
                # Do not attach those profiles to a different ablation pick.
                selection.matched_nearest_positive_idx = None
                selection.positive_length = None
                selection.negative_length = None
                selection.positive_profile = None
                selection.negative_profile = None
                selection.positive_profile_resampled = None
                selection.negative_profile_resampled = None
            candidate = FFCandidate(
                prompt_uid=state.prompt_uid,
                source_index=state.source_index,
                rollout_index=int(prompt["rollout_indices"][sibling]),
                teacher_processing_tokens=int(prompt["processing_tokens"][sibling]),
                response_tokens=int(prompt["response_tokens"][sibling]),
                frontier_selection=selection,
                selected_for_teacher=True,
                query_cap_before=max(0, available - offset),
                query_cap_after=max(0, available - offset - 1),
                used_queries_before=used_before_step + offset,
                used_queries_after=used_before_step + offset + 1,
                bucket=bucket.value,
                verifier_correct=bool(prompt["verifier_correct"][sibling]),
                rollout_valid=bool(prompt["rollout_valid"][sibling]),
                query_probability=(
                    next(
                        float(record["selection_probability"])
                        for record in prompt["_boundary_diagnostics"]["boundary_candidates_json"]
                        if int(record["rollout_id"]) == int(prompt["rollout_ids"][sibling])
                    )
                    if ht_enabled
                    else 1.0
                ),
                inverse_probability_weight=(
                    next(
                        float(record["inverse_probability_weight"])
                        for record in prompt["_boundary_diagnostics"]["boundary_candidates_json"]
                        if int(record["rollout_id"]) == int(prompt["rollout_ids"][sibling])
                    )
                    if ht_enabled
                    else 1.0
                ),
            )
            packed.append(candidate)
            candidates_by_prompt.setdefault(prompt_index, []).append(candidate)

        profile_records: list[dict[str, Any]] = []

        for prompt_index, (prompt, state, bucket, reason) in enumerate(pending):
            prompt_candidates = candidates_by_prompt.get(prompt_index, [])
            record_candidates: list[Optional[FFCandidate]] = prompt_candidates or [None]
            for candidate in record_candidates:
                selection = candidate.frontier_selection if candidate is not None else analyses[prompt_index]
                record = self._record(
                    prompt,
                    state,
                    bucket,
                    candidate,
                    selection,
                    reason if candidate is None else FFRejectedReason.NONE.value,
                    cumulative_cap,
                    available,
                    used_before_step,
                )
                records.append(record)
            audit = self._profile_record(
                prompt,
                state,
                bucket,
                prompt_candidates[0] if prompt_candidates else None,
                analyses[prompt_index],
            )
            if audit is not None:
                records[-1]["profile_audited"] = 1
                profile_records.append(audit)

        response_length_values = torch.as_tensor(
            [length for prompt in prompts for length in prompt["response_tokens"]], dtype=torch.float32
        )
        generated_rollout_count = int(response_length_values.numel())
        truncated_count = sum(int(prompt.get("truncated_rollout_count", 0)) for prompt in prompts)

        def response_quantile(q: float) -> float:
            return float(torch.quantile(response_length_values, q)) if generated_rollout_count else 0.0

        metrics = {
            "ff/all_correct_prompts": float(bucket_counts[FFBucket.ALL_CORRECT.value]),
            "ff/frontier_prompts": float(bucket_counts[FFBucket.FRONTIER.value]),
            "ff/no_success_prompts": float(bucket_counts[FFBucket.NO_SUCCESS.value]),
            "ff/invalid_ambiguous_prompts": float(bucket_counts[FFBucket.INVALID.value]),
            "ff/frontier_candidates": float(bucket_counts[FFBucket.FRONTIER.value]),
            "ff/frontier_selected": selector_stats["selected_frontier_count"],
            "ff/no_success_selected": float(sum(candidate.bucket == FFBucket.NO_SUCCESS.value for candidate in packed)),
            "ff/all_correct_selected": float(
                sum(candidate.bucket == FFBucket.ALL_CORRECT.value for candidate in packed)
            ),
            "ff/teacher_real_rollouts": float(used_this_step),
            "ff/teacher_dummy_rollouts": 0.0,
            "ff/teacher_padded_batch_size": float(used_this_step),
            "ff/query_cap": float(cumulative_cap),
            "ff/available_query_budget": float(available),
            "ff/used_queries_this_step": float(used_this_step),
            "ff/used_queries": float(self.used_queries),
            "ff/query_ratio_actual": self.used_queries / max(self.total_full_opd_queries, 1),
            "ff/total_full_opd_queries": float(self.total_full_opd_queries),
            "ff/no_success_retry_queue_size": float(len(self.current_queue) + len(self.next_queue)),
            "ff/retry_round": float(self.retry_round),
            "ff/no_success_requeued": float(
                sum(record["final_status"] == FFFinalStatus.RETRY_PENDING.value for record in records)
            ),
            "ff/no_success_exhausted": float(
                sum(record["final_status"] == FFFinalStatus.EXHAUSTED_NO_SUCCESS.value for record in records)
            ),
            "ff/all_pair_count": float(
                sum(selection.all_pair_count for selection in analyses if selection is not None)
            ),
            "ff/single_negative_direct_prompts": float(
                sum(
                    selection.selection_mode == "single_negative_direct"
                    for selection in analyses
                    if selection is not None
                )
            ),
            "ff/nearest_positive_selected_prompts": float(
                sum(record["selection_mode"] == "nearest_only" for record in records)
            ),
            # Distance-uninformative uniform fallback was removed; the new
            # formula resolves equal distances by Teacher cost automatically.
            "ff/uninformative_fallback_prompts": 0.0,
            "ff/teacher_drop_dp_alignment": 0.0,
            "opd_query/queried_rollout_count": float(used_this_step),
            # Full-OPD would query every generated sibling, including a
            # sibling later rejected by validity checks.
            "opd_query/full_rollout_count": float(valid_mask.numel()),
            "ff/generated_rollouts": float(valid_mask.numel()),
            "ff/realized_query_rate": float(used_this_step) / max(valid_mask.numel(), 1),
            "ff/realized_query_rate_cumulative": float(self.used_queries) / max(self.total_student_rollouts, 1),
            "ff/fresh_prompt_attempts": float(sum(not record["no_success_retry_attempt"] for record in records)),
            "ff/retry_prompt_attempts": float(sum(record["no_success_retry_attempt"] for record in records)),
            "ff/frontier_prompt_count": float(bucket_counts[FFBucket.FRONTIER.value]),
            "ff/all_success_prompt_count": float(bucket_counts[FFBucket.ALL_CORRECT.value]),
            "ff/no_success_prompt_count": float(bucket_counts[FFBucket.NO_SUCCESS.value]),
            "ff/invalid_prompt_count": float(bucket_counts[FFBucket.INVALID.value]),
            "ff/queried_rollout_count": float(used_this_step),
            "ff/actual_query_ratio": float(used_this_step) / max(generated_rollout_count, 1),
            "train/fresh_prompt_count": float(
                sum(prompt.get("queue_source", "fresh") == "fresh" for prompt in prompts)
            ),
            "train/retry_prompt_count": float(
                sum(prompt.get("queue_source", "fresh") != "fresh" for prompt in prompts)
            ),
            "train/generated_rollout_count": float(generated_rollout_count),
            "teacher/input_tokens": float(sum(candidate.teacher_processing_tokens for candidate in packed)),
            "teacher/scored_tokens": float(sum(candidate.response_tokens for candidate in packed)),
            "student/generated_tokens": float(response_length_values.sum()),
            "generation/truncated_count": float(truncated_count),
            "generation/truncated_rate": float(truncated_count) / max(generated_rollout_count, 1),
            "generation/eos_rate": float(generated_rollout_count - truncated_count) / max(generated_rollout_count, 1),
            "generation/response_length_p50": response_quantile(0.50),
            "generation/response_length_p90": response_quantile(0.90),
            "generation/response_length_p95": response_quantile(0.95),
            "generation/response_length_p99": response_quantile(0.99),
        }
        metrics.update({f"ff_ablation/{key}": value for key, value in selector_stats.items()})
        fully_valid = valid_mask.all(dim=1)
        correct_counts = (correct_mask & valid_mask).sum(dim=1)
        composition_counts = {
            count: int((fully_valid & (correct_counts == count)).sum().item()) for count in range(k_rollouts + 1)
        }

        def mean_or_zero(values: Sequence[float] | torch.Tensor) -> float:
            if isinstance(values, torch.Tensor):
                return float(values.mean().item()) if values.numel() else 0.0
            return sum(values) / len(values) if values else 0.0

        if self.config.selector_mode == "shortest_wrong":
            wrong_mask = valid_mask & ~correct_mask
            eligible = correct_mask.any(dim=1) & wrong_mask.any(dim=1)
            selected_lengths = response_lengths[selected_mask].to(torch.float32)
            unselected_wrong_lengths = response_lengths[wrong_mask & ~selected_mask].to(torch.float32)
            minimum_lengths: list[float] = []
            maximum_lengths: list[float] = []
            margins: list[float] = []
            shortest_hits = 0
            for prompt_index in eligible.nonzero(as_tuple=False).flatten().tolist():
                candidate_lengths = sorted(
                    int(response_lengths[prompt_index, sibling])
                    for sibling in wrong_mask[prompt_index].nonzero(as_tuple=False).flatten().tolist()
                )
                minimum_lengths.append(float(candidate_lengths[0]))
                maximum_lengths.append(float(candidate_lengths[-1]))
                margins.append(float(candidate_lengths[1] - candidate_lengths[0]) if len(candidate_lengths) > 1 else 0.0)
                chosen = selected_mask[prompt_index].nonzero(as_tuple=False).flatten()
                shortest_hits += int(
                    chosen.numel() == 1
                    and int(response_lengths[prompt_index, int(chosen[0])]) == candidate_lengths[0]
                )

            metrics.update(
                {
                    "shortest_wrong/eligible_prompt_count": float(eligible.sum().item()),
                    "shortest_wrong/selected_count": float(selected_mask.sum().item()),
                    "shortest_wrong/composition_4p0n": float(composition_counts[4]),
                    "shortest_wrong/composition_3p1n": float(composition_counts[3]),
                    "shortest_wrong/composition_2p2n": float(composition_counts[2]),
                    "shortest_wrong/composition_1p3n": float(composition_counts[1]),
                    "shortest_wrong/composition_0p4n": float(composition_counts[0]),
                    "shortest_wrong/selected_length_mean": mean_or_zero(selected_lengths),
                    "shortest_wrong/unselected_wrong_length_mean": mean_or_zero(unselected_wrong_lengths),
                    "shortest_wrong/min_wrong_length_mean": mean_or_zero(minimum_lengths),
                    "shortest_wrong/max_wrong_length_mean": mean_or_zero(maximum_lengths),
                    "shortest_wrong/selected_is_global_shortest_fraction": (
                        float(shortest_hits) / max(int(eligible.sum().item()), 1)
                    ),
                    "shortest_wrong/length_margin_mean": mean_or_zero(margins),
                }
            )
        elif self.config.selector_mode == "frontier_tlr":
            wrong_mask = valid_mask & ~correct_mask
            eligible = correct_mask.any(dim=1) & wrong_mask.any(dim=1)
            selected_lengths = response_lengths[selected_mask].to(torch.float32)
            unselected_wrong_lengths = response_lengths[wrong_mask & ~selected_mask].to(torch.float32)
            # Recompute per-rollout entropy mean for metrics (same logic as selector).
            entropy_rows_m: list[list[float]] = []
            for prompt in prompts:
                token_entropies = prompt.get("student_token_entropies")
                resp_masks = prompt.get("response_masks")
                if token_entropies is not None and resp_masks is not None:
                    ent = token_entropies.detach().float().cpu()
                    mask = resp_masks.detach().bool().cpu()
                    valid_counts = mask.sum(dim=-1).clamp_min(1).float()
                    means = (ent * mask).sum(dim=-1) / valid_counts
                    entropy_rows_m.append(means.tolist())
                else:
                    entropy_rows_m.append([0.0] * k_rollouts)
            entropy_mean_m = torch.tensor(entropy_rows_m, dtype=torch.float32)
            selected_entropy = entropy_mean_m[selected_mask]
            unselected_wrong_entropy = entropy_mean_m[wrong_mask & ~selected_mask]

            # Compute per-prompt TLR score for the selected rollout (for audit).
            selected_scores: list[float] = []
            eps_tlr = 1.0e-8
            for prompt_index in eligible.nonzero(as_tuple=False).flatten().tolist():
                candidates = wrong_mask[prompt_index].nonzero(as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                cand_lengths = response_lengths[prompt_index, candidates].float()
                cand_entropies = entropy_mean_m[prompt_index, candidates]
                norm_l = (cand_lengths - cand_lengths.min()) / (cand_lengths.max() - cand_lengths.min() + eps_tlr)
                norm_h = (cand_entropies - cand_entropies.min()) / (cand_entropies.max() - cand_entropies.min() + eps_tlr)
                scores = (1.0 - norm_l) * (1.0 - norm_h)
                chosen_local = int(torch.argmax(scores).item())
                selected_scores.append(float(scores[chosen_local].item()))

            metrics.update(
                {
                    "frontier_tlr/eligible_prompt_count": float(eligible.sum().item()),
                    "frontier_tlr/selected_count": float(selected_mask.sum().item()),
                    "frontier_tlr/composition_4p0n": float(composition_counts[4]),
                    "frontier_tlr/composition_3p1n": float(composition_counts[3]),
                    "frontier_tlr/composition_2p2n": float(composition_counts[2]),
                    "frontier_tlr/composition_1p3n": float(composition_counts[1]),
                    "frontier_tlr/composition_0p4n": float(composition_counts[0]),
                    "frontier_tlr/selected_length_mean": mean_or_zero(selected_lengths),
                    "frontier_tlr/unselected_wrong_length_mean": mean_or_zero(unselected_wrong_lengths),
                    "frontier_tlr/selected_entropy_mean": mean_or_zero(selected_entropy),
                    "frontier_tlr/unselected_wrong_entropy_mean": mean_or_zero(unselected_wrong_entropy),
                    "frontier_tlr/selected_score_lh_mean": mean_or_zero(selected_scores),
                }
            )
        # The four headline analyses: selection power (distance gap), selector
        # health (fallback rate), retry recovery, and realized Teacher spend.
        step_gaps = [float(record["distance_gap"]) for record in records if record["distance_gap"] not in ("", None)]
        step_selector_attempts = sum(record["selector_ran"] for record in records)
        step_fallbacks = sum(record["fallback_used"] for record in records)
        step_no_success_retries = sum(record["no_success_retry_attempt"] for record in records)
        step_no_success_to_frontier = sum(record["no_success_to_frontier"] for record in records)
        # Cost-aware selection diagnostics: the switch rate is measured only
        # over multi-negative prompts where the cost score actually ranked.
        step_cost_comparable = sum(
            1
            for record in records
            if record["selector_cost_aware"] and record["selection_mode"] in _FRONTIER_SELECTION_MODES
        )
        step_cost_switches = sum(int(record["cost_switch"]) for record in records)
        selected_candidate_costs = [
            float(record["selected_cost"])
            for record in records
            if record["selected_for_teacher"] and record["selected_cost"] not in ("", None)
        ]
        frontier_cost_records = [
            record
            for record in records
            if record["selection_mode"] in _FRONTIER_SELECTION_MODES
            and record["selected_cost"] not in ("", None)
            and record["nearest_only_cost"] not in ("", None)
        ]
        selected_total_cost = sum(float(record["selected_cost"]) for record in frontier_cost_records)
        nearest_total_cost = sum(float(record["nearest_only_cost"]) for record in frontier_cost_records)
        switched_records = [
            record
            for record in records
            if record["cost_switch"]
            and record["distance_regret"] not in ("", None)
            and record["relative_cost_reduction"] not in ("", None)
        ]
        metrics.update(
            {
                "ff/distance_gap_mean": (sum(step_gaps) / len(step_gaps)) if step_gaps else 0.0,
                "ff/distance_gap_min": min(step_gaps) if step_gaps else 0.0,
                "ff/distance_gap_reported_prompts": float(len(step_gaps)),
                "ff/fallback_used": float(step_fallbacks),
                "ff/fallback_rate": float(step_fallbacks) / max(step_selector_attempts, 1),
                "ff/fallback_rate_cumulative": self.fallback_rate,
                "ff/no_success_retry_attempts": float(step_no_success_retries),
                "ff/no_success_to_frontier": float(step_no_success_to_frontier),
                "ff/no_success_to_frontier_rate": float(step_no_success_to_frontier) / max(step_no_success_retries, 1),
                "ff/no_success_to_frontier_rate_cumulative": self.no_success_to_frontier_rate,
                "ff/no_success_to_all_correct_cumulative": float(self.total_no_success_to_all_correct),
                "ff/teacher_queried": float(used_this_step),
                "ff/profile_audit_records": float(len(profile_records)),
                "ff/cost_comparable_prompts": float(step_cost_comparable),
                "ff/cost_switches": float(step_cost_switches),
                "ff/cost_switch_rate": float(step_cost_switches) / max(step_cost_comparable, 1),
                "ff/cost_switch_rate_cumulative": self.cost_switch_rate,
                "ff/cost_saving_vs_nearest": 1.0 - selected_total_cost / max(nearest_total_cost, 1.0),
                "ff/distance_regret_mean": (
                    sum(float(record["distance_regret"]) for record in switched_records) / len(switched_records)
                    if switched_records
                    else 0.0
                ),
                "ff/relative_cost_reduction_mean": (
                    sum(float(record["relative_cost_reduction"]) for record in switched_records) / len(switched_records)
                    if switched_records
                    else 0.0
                ),
                "ff/selected_candidate_cost_mean": (
                    sum(selected_candidate_costs) / len(selected_candidate_costs) if selected_candidate_costs else 0.0
                ),
            }
        )
        if is_boundary_selector_mode(self.config.selector_mode):
            boundary_diagnostics = [prompt["_boundary_diagnostics"] for prompt in prompts]
            frontier_boundary = [
                item
                for item in boundary_diagnostics
                if item["num_positive_rollouts"] > 0 and item["num_negative_rollouts"] > 0
            ]
            selected_boundary = [item for item in frontier_boundary if item["boundary_selected_rollout_id"] != ""]
            selected_with_distance = [item for item in selected_boundary if item["boundary_transition_distance"] != ""]
            selected_with_objective = [item for item in selected_boundary if item["boundary_objective"] != ""]
            spearman_values = [
                float(item["_boundary_confidence_spearman"])
                for item in frontier_boundary
                if math.isfinite(float(item.get("_boundary_confidence_spearman", math.nan)))
            ]
            valid_transition_count = sum(int(item["boundary_valid_transition_count"]) for item in frontier_boundary)
            valid_transition_total = sum(int(item["_boundary_valid_transition_total"]) for item in frontier_boundary)
            metrics.update(
                {
                    "boundary/selected_distance_mean": (
                        sum(float(item["boundary_transition_distance"]) for item in selected_with_distance)
                        / len(selected_with_distance)
                        if selected_with_distance
                        else 0.0
                    ),
                    "boundary/selected_cost_mean": (
                        sum(float(item["boundary_teacher_cost"]) for item in selected_boundary) / len(selected_boundary)
                        if selected_boundary
                        else 0.0
                    ),
                    "boundary/selected_objective_mean": (
                        sum(float(item["boundary_objective"]) for item in selected_with_objective)
                        / len(selected_with_objective)
                        if selected_with_objective
                        else 0.0
                    ),
                    "boundary/cost_switch_rate": (
                        sum(int(item["boundary_cost_switch"]) for item in selected_boundary) / len(selected_boundary)
                        if selected_boundary
                        else 0.0
                    ),
                    "boundary/confidence_distance_spearman": (
                        sum(spearman_values) / len(spearman_values) if spearman_values else 0.0
                    ),
                    "boundary/fallback_rate": (
                        sum(int(item["boundary_fallback_used"]) for item in frontier_boundary) / len(frontier_boundary)
                        if frontier_boundary
                        else 0.0
                    ),
                    "boundary/valid_transition_ratio": (
                        valid_transition_count / valid_transition_total if valid_transition_total else 0.0
                    ),
                    "boundary/hidden_capture_time_ms": (
                        sum(float(item["_boundary_hidden_capture_time_ms"]) for item in boundary_diagnostics)
                        / len(boundary_diagnostics)
                        if boundary_diagnostics
                        else 0.0
                    ),
                    "boundary/transition_build_time_ms": sum(
                        float(item["_boundary_transition_build_time_ms"]) for item in frontier_boundary
                    ),
                    "boundary/distance_compute_time_ms": sum(
                        float(item["_boundary_distance_compute_time_ms"]) for item in frontier_boundary
                    ),
                    "boundary/metric_compute_time_ms": (
                        sum(float(item["_boundary_metric_compute_time_ms"]) for item in boundary_diagnostics)
                        / len(boundary_diagnostics)
                        if boundary_diagnostics
                        else 0.0
                    ),
                    "boundary/sinkhorn_max_residual": max(
                        (float(item["_boundary_sinkhorn_max_residual"]) for item in boundary_diagnostics),
                        default=0.0,
                    ),
                    "boundary/sinkhorn_max_iterations": max(
                        (float(item["_boundary_sinkhorn_max_iterations"]) for item in boundary_diagnostics),
                        default=0.0,
                    ),
                    "boundary/sinkhorn_unconverged_rate": (
                        sum(float(item["_boundary_sinkhorn_unconverged_rate"]) for item in boundary_diagnostics)
                        / len(boundary_diagnostics)
                        if boundary_diagnostics
                        else 0.0
                    ),
                    "boundary/communication_time_ms": (
                        sum(float(item["_boundary_communication_time_ms"]) for item in boundary_diagnostics)
                        / len(boundary_diagnostics)
                        if boundary_diagnostics
                        else 0.0
                    ),
                    "boundary/extra_peak_memory_mb": max(
                        (float(item["_boundary_extra_peak_memory_mb"]) for item in boundary_diagnostics),
                        default=0.0,
                    ),
                }
            )

            def selected_mean(key: str) -> float:
                values = [float(item.get(key, 0.0)) for item in selected_boundary]
                return sum(values) / len(values) if values else 0.0

            all_candidates = [
                candidate for item in frontier_boundary for candidate in item.get("boundary_candidates_json", [])
            ]
            if self.config.boundary_opd.score_mode in PDA_SCORE_MODES:
                pda_selected = [candidate for candidate in all_candidates if int(candidate.get("selected", 0))]

                def pda_values(key: str) -> list[float]:
                    return [float(candidate[key]) for candidate in pda_selected if candidate.get(key) not in (None, "")]

                def pda_mean(key: str) -> float:
                    values = pda_values(key)
                    return sum(values) / len(values) if values else 0.0

                areas = pda_values("pda_area")
                area_mean = sum(areas) / len(areas) if areas else 0.0
                zero_area_prompts = sum(int(item["boundary_degenerate_to_cost_only"]) for item in frontier_boundary)
                metrics.update(
                    {
                        "pda/eligible_mixed_prompts": float(len(frontier_boundary)),
                        "pda/selected_queries": float(len(pda_selected)),
                        "pda/selected_area_mean": area_mean,
                        "pda/selected_area_std": (
                            math.sqrt(sum((value - area_mean) ** 2 for value in areas) / len(areas)) if areas else 0.0
                        ),
                        "pda/selected_cost_ratio_mean": pda_mean("cost_ratio"),
                        "pda/selected_score_mean": pda_mean("boundary_final_score"),
                        "pda/selected_max_departure_mean": pda_mean("max_departure"),
                        "pda/selected_peak_fraction_mean": pda_mean("peak_departure_fraction"),
                        "pda/degenerate_zero_area_rate": zero_area_prompts / max(len(frontier_boundary), 1),
                        "pda/fallback_cost_only_rate": zero_area_prompts / max(len(frontier_boundary), 1),
                        "pda/selected_teacher_tokens": sum(
                            float(candidate.get("teacher_cost", 0.0)) for candidate in pda_selected
                        ),
                        "pda/mean_num_positive_siblings": (
                            sum(int(item["num_positive_rollouts"]) for item in frontier_boundary)
                            / max(len(frontier_boundary), 1)
                        ),
                    }
                )
            metrics.update(
                {
                    "boundary/selected_utility_mean": selected_mean("boundary_utility"),
                    "boundary/selected_score_mean": selected_mean("boundary_final_score"),
                    "boundary/selected_pre_similarity_mean": selected_mean("boundary_pre_similarity"),
                    "boundary/selected_post_similarity_mean": selected_mean("boundary_post_similarity"),
                    "boundary/selected_fork_drop_mean": selected_mean("boundary_fork_drop"),
                    "boundary/selected_split_fraction_mean": selected_mean("boundary_best_split_fraction"),
                    "boundary/cost_only_agreement": selected_mean("boundary_cost_only_agreement"),
                    "boundary/degenerate_to_cost_only_rate": selected_mean("boundary_degenerate_to_cost_only"),
                    "boundary/valid_candidate_rate": (
                        sum(int(candidate.get("valid_candidate", 0)) for candidate in all_candidates)
                        / len(all_candidates)
                        if all_candidates
                        else 0.0
                    ),
                    "boundary/valid_split_mean": (
                        sum(int(candidate.get("valid_split_count", 0)) for candidate in all_candidates)
                        / len(all_candidates)
                        if all_candidates
                        else 0.0
                    ),
                    "boundary/score_compute_time_ms": sum(
                        float(item["_boundary_distance_compute_time_ms"]) for item in frontier_boundary
                    ),
                    "boundary/fallback_rate": 0.0,
                }
            )
        representation_dynamics_rows: list[dict[str, Any]] = []
        boundary_csv_rows: list[dict[str, Any]] = []
        if is_boundary_selector_mode(self.config.selector_mode):
            num_boundaries = boundary_transition_count(self.config.boundary_opd)
            for prompt in prompts:
                diagnostics = prompt.get("_boundary_diagnostics", {})
                for candidate_record in diagnostics.get("boundary_candidates_json", []):
                    stage_similarities = candidate_record.get("stage_similarities") or []
                    stage_valid_mask = candidate_record.get("stage_valid_mask") or []
                    row = {
                        "prompt_id": str(prompt["prompt_uid"]),
                        "rollout_id": candidate_record["rollout_id"],
                        "is_selected": int(candidate_record.get("selected", 0)),
                        "boundary_score": candidate_record.get("boundary_final_score", ""),
                        "best_split_idx": candidate_record.get("best_split_index", ""),
                        "score_mode": candidate_record.get("score_mode", "legacy"),
                        "boundary_search_enabled": candidate_record.get("boundary_search_enabled", 1),
                        "window_ratio": candidate_record.get("window_ratio", ""),
                        "early_cosine": candidate_record.get("early_cosine", ""),
                        "middle1_cosine": candidate_record.get("middle1_cosine", ""),
                        "middle2_cosine": candidate_record.get("middle2_cosine", ""),
                        "aggregated_middle_cosine": candidate_record.get("aggregated_middle_cosine", ""),
                        "early_alignment": candidate_record.get("early_alignment", ""),
                        "plateau_drop": candidate_record.get("plateau_drop", ""),
                        "plateau_divergence": candidate_record.get("plateau_divergence", ""),
                        "pair_utility": candidate_record.get("pair_utility", ""),
                        "first_span_cosine": candidate_record.get("first_span_cosine", ""),
                        "second_span_cosine": candidate_record.get("second_span_cosine", ""),
                        "boundary_utility": candidate_record.get("boundary_utility", ""),
                        "best_positive_idx": candidate_record.get("best_positive_idx", ""),
                        "best_positive_rollout_id": candidate_record.get("best_positive_rollout_id", ""),
                        "early_similarity": candidate_record.get("early_similarity", ""),
                        "middle_similarity": candidate_record.get("middle_similarity", ""),
                        "middle_divergence": candidate_record.get("middle_divergence", ""),
                        "early_middle_drop": candidate_record.get("early_middle_drop", ""),
                        "quality_score": candidate_record.get("quality_score", ""),
                        "teacher_cost": candidate_record.get("teacher_cost", ""),
                        "teacher_input_len": candidate_record.get("teacher_input_len", ""),
                        "group_min_response_len": candidate_record.get("group_min_response_len", ""),
                        "group_min_teacher_input_len": candidate_record.get("group_min_teacher_input_len", ""),
                        "cost_ratio": candidate_record.get("cost_ratio", ""),
                        "cost_efficiency": candidate_record.get("cost_efficiency", ""),
                        "normalized_response_length": candidate_record.get("normalized_response_length", ""),
                        "length_efficiency": candidate_record.get("length_efficiency", ""),
                        "rollout_count": candidate_record.get("rollout_count", ""),
                        "final_score": candidate_record.get("boundary_final_score", ""),
                        "selected": int(candidate_record.get("selected", 0)),
                        "selected_positive_sibling_id": candidate_record.get("best_positive_rollout_id", ""),
                        "negative_rollout_id": candidate_record.get("rollout_id", ""),
                        "prompt_instance_id": str(prompt["prompt_uid"]),
                        "global_step": prompt.get("global_step", 0),
                        "prompt_length": candidate_record.get("prompt_length", ""),
                        "response_length": candidate_record.get("response_length", ""),
                        "valid_response_tokens": candidate_record.get("response_length", ""),
                        "num_positive_rollouts": candidate_record.get("num_positive_rollouts", ""),
                        "num_negative_rollouts": candidate_record.get("num_negative_rollouts", ""),
                        "finite_score": candidate_record.get("finite_score", ""),
                        "selection_probability": candidate_record.get("selection_probability", ""),
                        "inverse_probability_weight": candidate_record.get(
                            "inverse_probability_weight", ""
                        ),
                        "l11_score": candidate_record.get("l11_score", ""),
                        "anchor_similarity": candidate_record.get("anchor_similarity", ""),
                        "body_similarity": candidate_record.get("body_similarity", ""),
                        "persistent_drop": candidate_record.get("persistent_drop", ""),
                        "reference_positive_index": candidate_record.get("reference_positive_index", ""),
                        "unique_state_count": candidate_record.get("unique_state_count", ""),
                        "duplicate_state_fraction": candidate_record.get("duplicate_state_fraction", ""),
                        "score_margin": candidate_record.get("score_margin", ""),
                        "score_margin_normalized": candidate_record.get("score_margin_normalized", ""),
                        "equivalence_error": candidate_record.get("equivalence_error", ""),
                        "l12_geometry": candidate_record.get("l12_geometry", ""),
                        "l12_objective": candidate_record.get("l12_objective", ""),
                        "l12_cost_factor": candidate_record.get("l12_cost_factor", ""),
                        "l12_reference_rollout_id": candidate_record.get("l12_reference_rollout_id", ""),
                        "l12_route": candidate_record.get("l12_route", ""),
                        "l14_pre_margin": candidate_record.get("l14_pre_margin", ""),
                        "l14_post_margin": candidate_record.get("l14_post_margin", ""),
                        "l14_fork_drop": candidate_record.get("l14_fork_drop", ""),
                        "l14_post_divergence": candidate_record.get("l14_post_divergence", ""),
                        "l14_persistence_horizon": candidate_record.get("l14_persistence_horizon", ""),
                        "l14_peer_negative_idx": candidate_record.get("l14_peer_negative_idx", ""),
                        "l14_peer_negative_rollout_id": candidate_record.get(
                            "l14_peer_negative_rollout_id", ""
                        ),
                        "l14_score_margin": candidate_record.get("l14_score_margin", ""),
                        "safr_alignment_quality": candidate_record.get("safr_alignment_quality", ""),
                        "safr_fork_index": candidate_record.get("safr_fork_index", ""),
                        "safr_fork_score": candidate_record.get("safr_fork_score", ""),
                        "safr_span_count": candidate_record.get("safr_span_count", ""),
                        "safr_prompt_raw_score": candidate_record.get("safr_prompt_raw_score", ""),
                        "safr_composition_percentile": candidate_record.get(
                            "safr_composition_percentile", ""
                        ),
                        "frontier_weight": candidate_record.get("frontier_weight", ""),
                        "l8_score": candidate_record.get("l8_score", ""),
                        "l15_score": candidate_record.get("l15_score", ""),
                        "is_l8_winner": candidate_record.get("is_l8_winner", ""),
                        "fork_area_value": candidate_record.get("fork_area_value", ""),
                        "unit_token_value": candidate_record.get("unit_token_value", ""),
                        "is_fork_area_winner": candidate_record.get("is_fork_area_winner", ""),
                        "pda_area": candidate_record.get("pda_area", ""),
                        "best_positive_area": candidate_record.get("best_positive_area", ""),
                        "num_positive_siblings": candidate_record.get("num_positive_siblings", ""),
                        "max_departure": candidate_record.get("max_departure", ""),
                        "peak_departure_index": candidate_record.get("peak_departure_index", ""),
                        "peak_departure_fraction": candidate_record.get("peak_departure_fraction", ""),
                        "mean_departure": candidate_record.get("mean_departure", ""),
                        "first_similarity": candidate_record.get("first_similarity", ""),
                        "max_similarity": candidate_record.get("max_similarity", ""),
                        "final_similarity": candidate_record.get("final_similarity", ""),
                        "uses_hidden_difference": candidate_record.get("uses_hidden_difference", ""),
                        "representation_domain": candidate_record.get("representation_domain", ""),
                        "dtw_cost": candidate_record.get("dtw_cost", ""),
                        "diag_mean_similarity": candidate_record.get("diag_mean_similarity", ""),
                        "dtw_mean_similarity": candidate_record.get("dtw_mean_similarity", ""),
                        "dtw_similarity_gain": candidate_record.get("dtw_similarity_gain", ""),
                        "pda_diag": candidate_record.get("pda_diag", ""),
                        "pda_dtw": candidate_record.get("pda_dtw", ""),
                        "dtw_mean_warp": candidate_record.get("dtw_mean_warp", ""),
                        "dtw_max_warp": candidate_record.get("dtw_max_warp", ""),
                        "dtw_diagonal_step_ratio": candidate_record.get("dtw_diagonal_step_ratio", ""),
                        "dtw_vertical_step_ratio": candidate_record.get("dtw_vertical_step_ratio", ""),
                        "dtw_horizontal_step_ratio": candidate_record.get("dtw_horizontal_step_ratio", ""),
                        "dtw_gap_penalty": candidate_record.get("dtw_gap_penalty", ""),
                        "dtw_gap_step_count": candidate_record.get("dtw_gap_step_count", ""),
                        "dtw_gap_penalty_cost": candidate_record.get("dtw_gap_penalty_cost", ""),
                        "dtw_path_json": candidate_record.get("dtw_path_json", ""),
                        "dtw_cosine_matrix_json": candidate_record.get("dtw_cosine_matrix_json", ""),
                        "dtw_all_pairs_json": candidate_record.get("dtw_all_pairs_json", ""),
                        "dtw_alignment_text_json": candidate_record.get("dtw_alignment_text_json", ""),
                        "dtw_alignment_key_json": candidate_record.get("dtw_alignment_key_json", ""),
                        "source_index": candidate_record.get("source_index", prompt.get("source_index", "")),
                        "optimizer_step": candidate_record.get("optimizer_step", prompt.get("optimizer_step", 0)),
                        "student_model_version": candidate_record.get("student_model_version", ""),
                        "verifier_outcomes": candidate_record.get("verifier_outcomes", ""),
                        "positive_pair_rollout_ids": candidate_record.get("positive_pair_rollout_ids", ""),
                        "positive_pair_scores": candidate_record.get("positive_pair_scores", ""),
                        "positive_score_mean": candidate_record.get("positive_score_mean", ""),
                        "positive_score_max": candidate_record.get("positive_score_max", ""),
                        "positive_score_std": candidate_record.get("positive_score_std", ""),
                        "l8_max_score": candidate_record.get("l8_max_score", ""),
                        "l16_score": candidate_record.get("l16_score", ""),
                        "is_l16_prompt_winner": candidate_record.get("is_l16_prompt_winner", ""),
                        "is_shortest_wrong": candidate_record.get("is_shortest_wrong", ""),
                        "trajectory_nll": candidate_record.get("trajectory_nll", ""),
                        "successful_nll_mean": candidate_record.get("successful_nll_mean", ""),
                        "confidence_inversion": candidate_record.get("confidence_inversion", ""),
                        "normalized_length": candidate_record.get("normalized_length", ""),
                        "l17_score": candidate_record.get("l17_score", ""),
                        "is_l17_prompt_winner": candidate_record.get("is_l17_prompt_winner", ""),
                        "similarity_trajectory": candidate_record.get("similarity_trajectory", ""),
                        "split_changes": candidate_record.get("split_changes", ""),
                        "early_change_mean": candidate_record.get("early_change_mean", ""),
                        "early_change_score": candidate_record.get("early_change_score", ""),
                        "l18_score": candidate_record.get("l18_score", ""),
                        "is_l18_prompt_winner": candidate_record.get("is_l18_prompt_winner", ""),
                        "best_positive_id": candidate_record.get("best_positive_id", ""),
                        "interval_entropy": candidate_record.get("interval_entropy", ""),
                        "trajectory_entropy_mean": candidate_record.get(
                            "trajectory_entropy_mean", ""
                        ),
                        "unweighted_trajectory_utility": candidate_record.get(
                            "unweighted_trajectory_utility", ""
                        ),
                        "trajectory_departure_utility": candidate_record.get(
                            "trajectory_departure_utility", ""
                        ),
                        "global_entropy_mean": candidate_record.get("global_entropy_mean", ""),
                        "normalized_teaching_value": candidate_record.get(
                            "normalized_teaching_value", ""
                        ),
                        "normalized_cost": candidate_record.get("normalized_cost", ""),
                        "best_interval_idx": candidate_record.get("best_interval_idx", ""),
                        "valid_state_count": candidate_record.get("valid_state_count", ""),
                        "valid_transition_count": candidate_record.get("valid_transition_count", ""),
                        "valid_departure_interval_count": candidate_record.get(
                            "valid_departure_interval_count", ""
                        ),
                        "l31_distance": candidate_record.get("l31_distance", ""),
                        "l31_reachability": candidate_record.get("l31_reachability", ""),
                        "l31_q_minus": candidate_record.get("l31_q_minus", ""),
                        "l31_q_plus": candidate_record.get("l31_q_plus", ""),
                        "l31_misranking": candidate_record.get("l31_misranking", ""),
                        "l31_route": candidate_record.get("l31_route", ""),
                        "l31_negative_fork_token": candidate_record.get(
                            "l31_negative_fork_token", ""
                        ),
                        "l31_positive_fork_token": candidate_record.get(
                            "l31_positive_fork_token", ""
                        ),
                        "l31_branch_length": candidate_record.get("l31_branch_length", ""),
                        "l31_fork_fallback": candidate_record.get("l31_fork_fallback", ""),
                        "l31_alignment_path_length": candidate_record.get(
                            "l31_alignment_path_length", ""
                        ),
                        "l31_student_probe_time_ms": candidate_record.get(
                            "l31_student_probe_time_ms", ""
                        ),
                        "sasb_success_distance": candidate_record.get("sasb_success_distance", ""),
                        "sasb_tau_used": candidate_record.get("sasb_tau_used", ""),
                        "sasb_z": candidate_record.get("sasb_z", ""),
                        "sasb_boundary_q": candidate_record.get("sasb_boundary_q", ""),
                        "sasb_medoid_r": candidate_record.get("sasb_medoid_r", ""),
                        "sasb_utility": candidate_record.get("sasb_utility", ""),
                        "sasb_cost_penalty": candidate_record.get("sasb_cost_penalty", ""),
                        "sasb_gain": candidate_record.get("sasb_gain", ""),
                        "sasb_geometry_valid": candidate_record.get("sasb_geometry_valid", ""),
                        "sasb_is_shortest": candidate_record.get("sasb_is_shortest", ""),
                        "sasb_override_shortest": candidate_record.get("sasb_override_shortest", ""),
                        "sasb_medoid_enabled": candidate_record.get("sasb_medoid_enabled", ""),
                    }
                    a_plus_profile = candidate_record.get("a_plus_profile") or []
                    a_minus_profile = candidate_record.get("a_minus_profile") or []
                    margin_profile = candidate_record.get("relative_margin_profile") or []
                    running_max_profile = candidate_record.get("running_max") or []
                    departure_profile = candidate_record.get("departure") or []
                    raw_cosine_profile = candidate_record.get("raw_cosine") or []
                    normalized_similarity_profile = candidate_record.get("normalized_similarity") or []
                    for stage in range(num_boundaries):
                        row[f"sim_{stage + 1:02d}"] = (
                            stage_similarities[stage] if stage < len(stage_similarities) else ""
                        )
                        row[f"valid_{stage + 1:02d}"] = stage_valid_mask[stage] if stage < len(stage_valid_mask) else ""
                        row[f"similarity_{stage + 1:02d}"] = (
                            stage_similarities[stage] if stage < len(stage_similarities) else ""
                        )
                        row[f"a_plus_{stage + 1:02d}"] = (
                            a_plus_profile[stage] if stage < len(a_plus_profile) else ""
                        )
                        row[f"a_minus_{stage + 1:02d}"] = (
                            a_minus_profile[stage] if stage < len(a_minus_profile) else ""
                        )
                        row[f"margin_{stage + 1:02d}"] = (
                            margin_profile[stage] if stage < len(margin_profile) else ""
                        )
                        row[f"running_max_{stage + 1:02d}"] = (
                            running_max_profile[stage] if stage < len(running_max_profile) else ""
                        )
                        row[f"departure_{stage + 1:02d}"] = (
                            departure_profile[stage] if stage < len(departure_profile) else ""
                        )
                        row[f"raw_cosine_{stage + 1:02d}"] = (
                            raw_cosine_profile[stage] if stage < len(raw_cosine_profile) else ""
                        )
                        row[f"similarity_s_{stage + 1:02d}"] = (
                            normalized_similarity_profile[stage]
                            if stage < len(normalized_similarity_profile)
                            else ""
                        )
                    split_changes = candidate_record.get("split_changes") or []
                    for split_index, tau in enumerate(range(4, 13)):
                        row[f"split_change_tau{tau}"] = (
                            split_changes[split_index] if split_index < len(split_changes) else ""
                        )
                    boundary_csv_rows.append(row)
        if self.config.debug_assertions:
            assert len(selected_indices) == int(selector_stats["target_query_count"])
            assert all(candidate.rollout_valid for candidate in packed)
            if self.config.selector_mode == "random_correct":
                assert all(
                    candidate.verifier_correct and candidate.bucket == FFBucket.FRONTIER.value for candidate in packed
                )
            elif self.config.selector_mode != "global_random":
                assert all(
                    not candidate.verifier_correct and candidate.bucket == FFBucket.FRONTIER.value
                    for candidate in packed
                )
        return FFRouteResult(
            packed,
            records,
            selected_indices,
            metrics,
            available,
            used_this_step,
            profile_records,
            boundary_csv_rows,
            representation_dynamics_rows,
            ht_enabled,
            ht_full_valid_response_tokens,
            ht_full_candidate_count,
            l23_fork_positions=dict(getattr(self, "_l23_fork_positions", {})),
        )

    def _profile_record(
        self,
        prompt: dict[str, Any],
        state: FFPromptState,
        bucket: FFBucket,
        candidate: Optional[FFCandidate],
        selection: Optional[FrontierSelectionResult],
    ) -> Optional[dict[str, Any]]:
        """Build the audit-only JSONL row holding the full confidence profiles."""
        if candidate is not None:
            selection = candidate.frontier_selection
        if selection is None or selection.positive_profile is None:
            return None
        return {
            "run_name": self.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_step": prompt.get("global_step", 0),
            "base_epoch": prompt.get("base_epoch", 1),
            "phase": _phase_of(prompt.get("queue_source", "fresh")),
            "retry_round": self.retry_round,
            "prompt_uid": state.prompt_uid,
            "source_index": state.source_index,
            "attempt_id": state.attempt_count,
            "student_model_version": self.student_model_version,
            "bucket": bucket.value,
            "selection_mode": selection.selection_mode,
            "selected_negative_idx": selection.selected_negative_idx,
            "matched_nearest_positive_idx": selection.matched_nearest_positive_idx,
            "positive_indices": list(selection.positive_indices),
            "negative_indices": list(selection.negative_indices),
            "candidate_mean_positive_distances": {
                str(key): value for key, value in selection.candidate_mean_positive_distances.items()
            },
            "cost_aware": selection.cost_aware,
            "candidate_nearest_positive_distances": {
                str(key): value for key, value in selection.candidate_nearest_positive_distances.items()
            },
            "candidate_costs": {str(key): value for key, value in selection.candidate_costs.items()},
            "candidate_objectives": {str(key): value for key, value in selection.candidate_objectives.items()},
            "candidate_log_scores": {str(key): value for key, value in selection.candidate_log_scores.items()},
            "selected_objective": selection.selected_objective,
            "selected_log_score": selection.selected_log_score,
            "nearest_only_selected_idx": selection.nearest_only_selected_idx,
            "lowest_cost_selected_idx": selection.lowest_cost_selected_idx,
            "cost_switch": selection.cost_switch,
            "selected_cost": selection.selected_cost,
            "nearest_only_cost": selection.nearest_only_cost,
            "distance_regret": selection.distance_regret,
            "relative_cost_reduction": selection.relative_cost_reduction,
            "nearest_positive_profile_distance": selection.nearest_positive_profile_distance,
            "distance_min": selection.distance_min,
            "distance_second_min": selection.distance_second_min,
            "distance_gap": selection.distance_gap,
            "common_profile_length": selection.common_profile_length,
            "teacher_queried": int(bool(candidate and candidate.selected_for_teacher)),
            "positive_profile": selection.positive_profile,
            "negative_profile": selection.negative_profile,
            "positive_profile_resampled": selection.positive_profile_resampled,
            "negative_profile_resampled": selection.negative_profile_resampled,
        }

    def _record(
        self,
        prompt: dict[str, Any],
        state: FFPromptState,
        bucket: FFBucket,
        candidate: Optional[FFCandidate],
        selection: Optional[FrontierSelectionResult],
        rejected_reason: str,
        cumulative_cap: int,
        available_query_budget: int,
        used_queries_before_step: int = 0,
    ) -> dict[str, Any]:
        correct = [int(value) for value in prompt["verifier_correct"]]
        valid = [bool(value) for value in prompt["rollout_valid"]]
        selected_sibling: int | str = ""
        if candidate is not None:
            selected_sibling = list(prompt["rollout_indices"]).index(candidate.rollout_index)
            selection = candidate.frontier_selection

        def selected(name: str, default: Any = "") -> Any:
            if selection is None:
                return default
            value = getattr(selection, name)
            return default if value is None else value

        valid_correct_count = sum(v and bool(c) for v, c in zip(valid, correct, strict=True))
        valid_incorrect_count = sum(v and not bool(c) for v, c in zip(valid, correct, strict=True))
        matched_nearest_positive = selected("matched_nearest_positive_idx")
        teacher_queried = int(bool(candidate is not None and candidate.selected_for_teacher))
        no_success_retry_attempt = int(state.previous_bucket == FFBucket.NO_SUCCESS.value)
        boundary = prompt.get("_boundary_diagnostics", {})

        def boundary_value(name: str, default: Any = "") -> Any:
            return boundary.get(name, default)

        ht_enabled = False

        return {
            "run_name": self.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": self.config.seed,
            "base_epochs": BASE_EPOCHS,
            "k_rollouts": self.config.k_rollouts,
            "legacy_teacher_query_ratio": self.config.legacy_teacher_query_ratio,
            "query_cap": cumulative_cap,
            "available_query_budget": available_query_budget,
            "available_queries_before": candidate.query_cap_before if candidate else available_query_budget,
            "query_cap_before": candidate.query_cap_before if candidate else "",
            "query_cap_after": candidate.query_cap_after if candidate else "",
            "used_queries_before": candidate.used_queries_before if candidate else used_queries_before_step,
            "used_queries_after": candidate.used_queries_after if candidate else used_queries_before_step,
            "used_queries": self.used_queries,
            "query_ratio_actual": self.used_queries / max(self.total_full_opd_queries, 1),
            "total_full_opd_queries": self.total_full_opd_queries,
            # Only the fresh pass grows the Full-OPD baseline; retries never do.
            "fresh_full_opd_queries": self.total_full_opd_queries,
            "loss_type": LOSS_TYPE,
            "target_mode": TARGET_MODE,
            "teacher_stop_gradient": int(TEACHER_STOP_GRADIENT),
            "use_ht": int(ht_enabled),
            "use_ipw": int(ht_enabled),
            "query_probability": candidate.query_probability if candidate else "",
            "inverse_probability_weight": candidate.inverse_probability_weight if candidate else "",
            "global_step": prompt.get("global_step", 0),
            "optimizer_step": prompt.get("optimizer_step", 0),
            "fresh_step": prompt.get("fresh_step", ""),
            "retry_step": prompt.get("retry_step", ""),
            "phase": _phase_of(prompt.get("queue_source", "fresh")),
            "retry_round": self.retry_round,
            "queue_source": prompt.get("queue_source", "fresh"),
            "current_queue_size": len(self.current_queue),
            "next_queue_size": len(self.next_queue),
            "retry_queue_size": len(self.current_queue) + len(self.next_queue),
            "base_epoch": prompt.get("base_epoch", 1),
            "attempt_id": state.attempt_count,
            "retry_count": state.retry_count,
            "max_no_success_retries": self.config.max_no_success_retries,
            "retries_left": state.retries_left,
            "retries_left_before": state.retries_left_before,
            "retries_left_after": state.retries_left,
            "prompt_uid": state.prompt_uid,
            "source_index": state.source_index,
            "bucket": bucket.value,
            "current_bucket": bucket.value,
            "previous_bucket": state.previous_bucket,
            "no_success_retry_attempt": no_success_retry_attempt,
            "no_success_to_frontier": int(no_success_retry_attempt and bucket == FFBucket.FRONTIER),
            "no_success_to_all_correct": int(no_success_retry_attempt and bucket == FFBucket.ALL_CORRECT),
            "no_success_to_frontier_rate_cumulative": self.no_success_to_frontier_rate,
            "final_status": state.final_status,
            "num_rollouts": len(correct),
            "correct_count": valid_correct_count,
            "incorrect_count": valid_incorrect_count,
            "valid_rollout_count": sum(valid),
            "valid_correct_count": valid_correct_count,
            "valid_incorrect_count": valid_incorrect_count,
            "invalid_count": len(valid) - valid_correct_count - valid_incorrect_count,
            "truncated_count": int(prompt.get("truncated_rollout_count", 0)),
            "verifier_timeout_count": int(prompt.get("verifier_timeout_count", 0)),
            "verifier_parse_error_count": int(prompt.get("verifier_parse_error_count", 0)),
            "verifier_error_count": int(prompt.get("verifier_error_count", 0)),
            "candidate_type": candidate.candidate_type if candidate else FFCandidateType.NONE.value,
            "candidate_created": int(candidate is not None),
            "teacher_candidate": int(candidate is not None),
            "positive_indices": _json_compact(list(selected("positive_indices", ()))),
            "negative_indices": _json_compact(list(selected("negative_indices", ()))),
            "selected_rollout_idx": selected_sibling,
            "selected_rollout_correct": correct[selected_sibling] if selected_sibling != "" else "",
            "selected_rollout_valid": int(valid[selected_sibling]) if selected_sibling != "" else "",
            "frontier_selector": FRONTIER_SELECTOR if bucket == FFBucket.FRONTIER else "",
            "selector_formula_version": SELECTOR_FORMULA_VERSION,
            "selector_name": selected("selector_type"),
            "selection_mode": selected("selection_mode", "none"),
            "ff_selector_mode": self.config.selector_mode,
            "num_boundaries": boundary_value("num_boundaries"),
            "boundary_similarity_metric": boundary_value("boundary_similarity_metric"),
            "score_mode": boundary_value("score_mode", "legacy"),
            "boundary_search_enabled": boundary_value("boundary_search_enabled", 1),
            "hidden_capture_module": boundary_value("hidden_capture_module"),
            "hidden_capture_method": boundary_value("hidden_capture_method"),
            "hidden_capture_success": boundary_value("hidden_capture_success"),
            "hidden_dim": boundary_value("hidden_dim"),
            "hidden_shape": _json_compact(boundary_value("hidden_shape", ())),
            "sequence_parallel_size": boundary_value("sequence_parallel_size", 1),
            "sequence_parallel_sharded": boundary_value("sequence_parallel_sharded", 0),
            "sequence_parallel_gathered": boundary_value("sequence_parallel_gathered", 1),
            "tensor_parallel_sharded": boundary_value("tensor_parallel_sharded", 0),
            "lm_head_module": boundary_value("lm_head_module"),
            "lm_head_vocab_size": boundary_value("lm_head_vocab_size", 0),
            "lm_head_weight_current": boundary_value("lm_head_weight_current", 0),
            "lm_head_vocab_sharded": boundary_value("lm_head_vocab_sharded", 0),
            "weight_mode": boundary_value("weight_mode"),
            "boundary_transition_distance": boundary_value("boundary_transition_distance"),
            "boundary_nearest_positive_rollout_id": boundary_value("boundary_nearest_positive_rollout_id"),
            "boundary_teacher_cost": boundary_value("boundary_teacher_cost"),
            "boundary_objective": boundary_value("boundary_objective"),
            "boundary_utility": boundary_value("boundary_utility"),
            "boundary_final_score": boundary_value("boundary_final_score"),
            "boundary_best_positive_rollout_id": boundary_value("boundary_best_positive_rollout_id"),
            "boundary_best_split_index": boundary_value("boundary_best_split_index"),
            "boundary_best_split_fraction": boundary_value("boundary_best_split_fraction"),
            "boundary_pre_similarity": boundary_value("boundary_pre_similarity"),
            "boundary_post_similarity": boundary_value("boundary_post_similarity"),
            "boundary_fork_drop": boundary_value("boundary_fork_drop"),
            "boundary_post_divergence": boundary_value("boundary_post_divergence"),
            "early_similarity": boundary_value("early_similarity"),
            "middle_similarity": boundary_value("middle_similarity"),
            "middle_divergence": boundary_value("middle_divergence"),
            "early_middle_drop": boundary_value("early_middle_drop"),
            "early_cosine": boundary_value("early_cosine"),
            "middle1_cosine": boundary_value("middle1_cosine"),
            "middle2_cosine": boundary_value("middle2_cosine"),
            "aggregated_middle_cosine": boundary_value("aggregated_middle_cosine"),
            "early_alignment": boundary_value("early_alignment"),
            "plateau_drop": boundary_value("plateau_drop"),
            "plateau_divergence": boundary_value("plateau_divergence"),
            "pair_utility": boundary_value("pair_utility"),
            "first_span_cosine": boundary_value("first_span_cosine"),
            "second_span_cosine": boundary_value("second_span_cosine"),
            "best_positive_idx": boundary_value("best_positive_idx"),
            "cost_efficiency": boundary_value("boundary_cost_efficiency"),
            "normalized_response_length": boundary_value("normalized_response_length"),
            "length_efficiency": boundary_value("length_efficiency"),
            "rollout_count": boundary_value("rollout_count"),
            "final_score": boundary_value("boundary_final_score"),
            "selected_positive_sibling_id": boundary_value("boundary_best_positive_rollout_id"),
            "best_positive_rollout_id": boundary_value("boundary_best_positive_rollout_id"),
            "excluded_late_stage_count": boundary_value("excluded_late_stage_count"),
            "utility_definition": boundary_value("utility_definition"),
            "boundary_valid_split_count": boundary_value("boundary_valid_split_count"),
            "boundary_cost_efficiency": boundary_value("boundary_cost_efficiency"),
            "boundary_degenerate_to_cost_only": boundary_value("boundary_degenerate_to_cost_only", 0),
            "boundary_selected_is_cost_only": boundary_value("boundary_selected_is_cost_only", 0),
            "boundary_selected_rollout_id": boundary_value("boundary_selected_rollout_id"),
            "boundary_nearest_rollout_id": boundary_value("boundary_nearest_rollout_id"),
            "boundary_cost_only_rollout_id": boundary_value("boundary_cost_only_rollout_id"),
            "boundary_random_wrong_rollout_id": boundary_value("boundary_random_wrong_rollout_id"),
            "boundary_cost_switch": boundary_value("boundary_cost_switch"),
            "boundary_hidden_available": boundary_value("boundary_hidden_available"),
            "boundary_fallback_used": boundary_value("boundary_fallback_used"),
            "boundary_fallback_reason": boundary_value("boundary_fallback_reason"),
            "boundary_valid_transition_count": boundary_value("boundary_valid_transition_count"),
            "num_positive_rollouts": boundary_value("num_positive_rollouts", valid_correct_count),
            "num_negative_rollouts": boundary_value("num_negative_rollouts", valid_incorrect_count),
            "l11_score": boundary_value("l11_score"),
            "l11_anchor_similarity": boundary_value("l11_anchor_similarity"),
            "l11_body_similarity": boundary_value("l11_body_similarity"),
            "l11_persistent_drop": boundary_value("l11_persistent_drop"),
            "l11_reference_positive_index": boundary_value("l11_reference_positive_index"),
            "l11_unique_state_count": boundary_value("l11_unique_state_count"),
            "l11_duplicate_state_fraction": boundary_value("l11_duplicate_state_fraction"),
            "l11_score_margin": boundary_value("l11_score_margin"),
            "l11_score_margin_normalized": boundary_value("l11_score_margin_normalized"),
            "boundary_candidates_json": _json_compact(boundary_value("boundary_candidates_json", [])),
            "single_negative_direct": int(selected("selection_mode") == "single_negative_direct"),
            "all_pair_count": selected("all_pair_count", 0),
            "matched_nearest_positive_idx": matched_nearest_positive,
            "matched_nearest_positive_correct": (
                correct[int(matched_nearest_positive)] if matched_nearest_positive != "" else ""
            ),
            "selected_negative_idx": selected_sibling if bucket == FFBucket.FRONTIER else "",
            "selected_negative_correct": correct[selected_sibling] if selected_sibling != "" else "",
            "positive_length": selected("positive_length"),
            "negative_length": selected("negative_length"),
            "common_profile_length": selected("common_profile_length"),
            "nearest_positive_profile_distance": selected("nearest_positive_profile_distance"),
            "mean_positive_profile_distance": selected("mean_positive_profile_distance"),
            "distance_min": selected("distance_min"),
            "distance_second_min": selected("distance_second_min"),
            # distance_second_min - distance_min: the selection power of the
            # confidence trajectories. Values near 0 mean the selector cannot
            # meaningfully discriminate between negatives.
            "distance_gap": selected("distance_gap"),
            "candidate_mean_positive_distances_json": _json_compact(
                {
                    str(key): round(float(value), 6)
                    for key, value in selected("candidate_mean_positive_distances", {}).items()
                }
            ),
            # Cost-aware (FF-Cost) per-sibling diagnostics. All dict columns
            # are keyed by the rollout index inside the sibling group. The
            # objective is the raw J_j = (d_j + eps) * sqrt(c_j).
            "selector_cost_aware": int(bool(selected("cost_aware", False))),
            "selector_cost_alpha": self.config.selector_cost_alpha,
            "selector_score_eps": self.config.selector_score_eps,
            "candidate_nearest_positive_distances_json": _json_compact(
                {
                    str(key): round(float(value), 6)
                    for key, value in selected("candidate_nearest_positive_distances", {}).items()
                }
            ),
            "candidate_costs_json": _json_compact(
                {str(key): round(float(value), 3) for key, value in selected("candidate_costs", {}).items()}
            ),
            "candidate_objectives_json": _json_compact(
                {str(key): round(float(value), 6) for key, value in selected("candidate_objectives", {}).items()}
            ),
            "selector_log_scores_json": _json_compact(
                {str(key): round(float(value), 6) for key, value in selected("candidate_log_scores", {}).items()}
            ),
            "selected_objective": selected("selected_objective"),
            "selected_log_score": selected("selected_log_score"),
            "nearest_only_selected_idx": selected("nearest_only_selected_idx"),
            "lowest_cost_selected_idx": selected("lowest_cost_selected_idx"),
            "cost_switch": int(bool(selected("cost_switch", False))),
            "nearest_only_cost": selected("nearest_only_cost"),
            "selected_cost": selected("selected_cost"),
            "distance_regret": selected("distance_regret", 0.0),
            "relative_cost_reduction": selected("relative_cost_reduction", 0.0),
            "cost_switch_rate_cumulative": self.cost_switch_rate,
            "selector_ran": int(selection is not None and bucket == FFBucket.FRONTIER),
            "fallback_used": int(bool(selected("fallback_used", False))),
            "selector_fallback_used": int(bool(selected("fallback_used", False))),
            "selector_fallback_reason": selected("fallback_reason"),
            "fallback_reason": selected("fallback_reason"),
            "fallback_rate_cumulative": self.fallback_rate,
            "selected_for_teacher": teacher_queried,
            "teacher_queried": teacher_queried,
            "rejected_by_budget": int(
                candidate is not None and candidate.rejected_reason == FFRejectedReason.QUERY_CAP_REACHED.value
            ),
            "rejected_reason": rejected_reason,
            "candidate_processing_tokens": candidate.teacher_processing_tokens if candidate else 0,
            "selected_response_tokens": (
                candidate.response_tokens if candidate is not None and candidate.selected_for_teacher else 0
            ),
            "teacher_query_count_this_attempt": teacher_queried,
            "teacher_query_count_total": state.total_teacher_queries,
            "teacher_processing_tokens": 0,
            "teacher_scored_tokens": 0,
            "teacher_supervised_tokens": 0,
            "teacher_forward_seconds": "",
            "teacher_drop_overlength": 0,
            "teacher_drop_truncated": 0,
            "teacher_drop_truncated_after_selection": 0,
            "teacher_drop_invalid": 0,
            "teacher_drop_dp_alignment": 0,
            "teacher_dummy_rollouts": 0,
            "teacher_real_rollout_count": 0,
            "teacher_dummy_rollout_count": 0,
            "teacher_padded_batch_size": 0,
            "sampled_reverse_kl_sum": "",
            "sampled_reverse_kl_token_mean": "",
            "sampled_reverse_kl_valid_token_count": "",
            "reverse_kl_mean": "",
            "student_sampled_logprob_mean": "",
            "teacher_sampled_logprob_mean": "",
            "sampled_logratio_mean": "",
            "response_mask_token_count": 0,
            "opd_loss": "",
            "opd_loss_contributed": 0,
            "optimizer_updated": 0,
            "profile_audited": 0,
            "no_success_selected": 0,
            "all_correct_selected": 0,
            "student_model_version": self.student_model_version,
            "last_attempt_model_version": state.last_attempt_model_version,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "prompt_states": {uid: asdict(state) for uid, state in self.prompt_states.items()},
            "current_queue": list(self.current_queue),
            "next_queue": list(self.next_queue),
            "retry_round": self.retry_round,
            "total_full_opd_queries": self.total_full_opd_queries,
            "used_queries": self.used_queries,
            "student_model_version": self.student_model_version,
            "total_prompt_attempts": self.total_prompt_attempts,
            "total_student_rollouts": self.total_student_rollouts,
            "total_selector_attempts": self.total_selector_attempts,
            "total_selector_fallbacks": self.total_selector_fallbacks,
            "total_no_success_retry_attempts": self.total_no_success_retry_attempts,
            "total_no_success_to_frontier": self.total_no_success_to_frontier,
            "total_no_success_to_all_correct": self.total_no_success_to_all_correct,
            "total_cost_comparable": self.total_cost_comparable,
            "total_cost_switches": self.total_cost_switches,
            "sasb_tau_comp": dict(self.sasb_tau_comp),
            "sasb_tau_initialized": dict(self.sasb_tau_initialized),
            "sasb_sigma_3": self.sasb_sigma_3,
            "sasb_sigma_3_initialized": self.sasb_sigma_3_initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        known_fields = set(FFPromptState.__dataclass_fields__)
        self.prompt_states = {
            uid: FFPromptState(**{key: value for key, value in values.items() if key in known_fields})
            for uid, values in state.get("prompt_states", {}).items()
        }
        legacy_queue = state.get("no_success_retry_queue", [])
        self.current_queue = deque(state.get("current_queue", legacy_queue))
        self.next_queue = deque(state.get("next_queue", []))
        self.retry_round = int(state.get("retry_round", 0))
        self.total_full_opd_queries = int(state.get("total_full_opd_queries", 0))
        self.used_queries = int(state.get("used_queries", 0))
        self.student_model_version = str(state.get("student_model_version", "0"))
        self.total_prompt_attempts = int(state.get("total_prompt_attempts", 0))
        self.total_student_rollouts = int(state.get("total_student_rollouts", 0))
        self.total_selector_attempts = int(state.get("total_selector_attempts", 0))
        self.total_selector_fallbacks = int(state.get("total_selector_fallbacks", 0))
        self.total_no_success_retry_attempts = int(state.get("total_no_success_retry_attempts", 0))
        self.total_no_success_to_frontier = int(state.get("total_no_success_to_frontier", 0))
        self.total_no_success_to_all_correct = int(state.get("total_no_success_to_all_correct", 0))
        self.total_cost_comparable = int(state.get("total_cost_comparable", 0))
        self.total_cost_switches = int(state.get("total_cost_switches", 0))
        tau_values = state.get("sasb_tau_comp", {})
        tau_flags = state.get("sasb_tau_initialized", {})
        self.sasb_tau_comp = {
            composition: float(tau_values.get(composition, tau_values.get(str(composition), 0.0)))
            for composition in (1, 2, 3)
        }
        self.sasb_tau_initialized = {
            composition: bool(tau_flags.get(composition, tau_flags.get(str(composition), False)))
            for composition in (1, 2, 3)
        }
        self.sasb_sigma_3 = float(state.get("sasb_sigma_3", 0.0))
        self.sasb_sigma_3_initialized = bool(state.get("sasb_sigma_3_initialized", False))
        if self.used_queries < 0:
            raise ValueError("invalid restored FF-OPD query accounting")
