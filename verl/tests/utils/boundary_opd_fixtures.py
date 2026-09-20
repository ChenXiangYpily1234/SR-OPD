# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared Boundary-OPD fixtures for the FF-OPD queue-manager tests."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.utils.test_boundary_opd import HIDDEN_DIM, states_from_transitions, unit
from verl.utils.boundary_opd import BOUNDARY_SELECTOR_MODES, BoundaryOPDSettings
from verl.utils.ff_opd import FFOPDConfig, FFOPDQueueManager

K_ROLLOUTS = 4
NUM_BOUNDARIES = 8
LEGACY_MODES = (
    "global_random",
    "random_wrong",
    "nearest_only",
    "cost_only",
    "farthest_only",
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def direction_plan(plan) -> torch.Tensor:
    """Build ``[K, M, D]`` step directions from per-rollout basis indices."""

    return torch.stack([torch.stack([unit(index) for index in row]) for row in plan])


def uniform_plan() -> torch.Tensor:
    return direction_plan([[0] * NUM_BOUNDARIES] * K_ROLLOUTS)


def distinct_plan() -> torch.Tensor:
    return direction_plan(
        [
            [0] * NUM_BOUNDARIES,
            [1] * NUM_BOUNDARIES,
            [2] * NUM_BOUNDARIES,
            [3] * NUM_BOUNDARIES,
        ]
    )


def early_late_plan() -> torch.Tensor:
    """Positive reference, early divergence, late divergence, full divergence."""

    return direction_plan(
        [
            [0] * NUM_BOUNDARIES,
            [1] + [0] * (NUM_BOUNDARIES - 1),
            [0] * (NUM_BOUNDARIES - 1) + [1],
            [1] * NUM_BOUNDARIES,
        ]
    )


def make_manager(
    mode: str,
    tmp_path,
    *,
    seed: int = 42,
    max_no_success_retries: int = 0,
    **boundary_overrides,
) -> FFOPDQueueManager:
    boundary = None
    if mode in BOUNDARY_SELECTOR_MODES:
        boundary = BoundaryOPDSettings(**{"num_boundaries": NUM_BOUNDARIES, **boundary_overrides})
    elif boundary_overrides:
        raise ValueError("Boundary settings are only valid for Boundary selector modes")
    config = FFOPDConfig(
        k_rollouts=K_ROLLOUTS,
        selector_mode=mode,
        seed=seed,
        max_no_success_retries=max_no_success_retries,
        csv_path=str(Path(tmp_path) / f"{mode}.csv"),
        profile_jsonl_path=str(Path(tmp_path) / f"{mode}.jsonl"),
        profile_audit_sample_rate=0.0,
        boundary_opd=boundary,
    )
    return FFOPDQueueManager(config, run_name=mode)


def make_prompt(
    uid: str,
    correct,
    *,
    directions: torch.Tensor | None = None,
    costs=(120, 120, 120, 120),
    response_tokens=(20, 20, 20, 20),
    valid=(True, True, True, True),
    rollout_offset: int = 0,
    rollout_ids=(0, 1, 2, 3),
    capture_success=None,
    with_boundary: bool = True,
    log_profiles=None,
    global_step: int = 0,
    queue_source: str = "fresh",
) -> dict:
    """Build one FF-OPD sibling group in the shape the trainer passes in."""

    if log_profiles is None:
        log_profiles = [[-0.10, -0.20], [-0.11, -0.21], [-0.50, -0.60], [-0.90, -1.00]]
    prompt = {
        "prompt_uid": uid,
        "source_index": uid,
        "rollout_indices": [rollout_offset + index for index in range(K_ROLLOUTS)],
        "verifier_correct": [int(bool(value)) for value in correct],
        "rollout_valid": [bool(value) for value in valid],
        "processing_tokens": list(costs),
        "response_tokens": list(response_tokens),
        "response_masks": torch.ones(K_ROLLOUTS, len(log_profiles[0]), dtype=torch.bool),
        "sampled_token_log_probs": torch.tensor(log_profiles, dtype=torch.float32),
        "global_step": global_step,
        "queue_source": queue_source,
        "rollout_ids": list(rollout_ids),
    }
    if not with_boundary:
        prompt["boundary_hidden_capture_success"] = [False] * K_ROLLOUTS
        prompt["boundary_hidden_capture_error_code"] = [3] * K_ROLLOUTS
        return prompt
    if directions is None:
        directions = uniform_plan()
    prompt.update(
        {
            "boundary_states": states_from_transitions(directions).to(torch.float16),
            "boundary_transition_valid_mask": torch.ones(
                K_ROLLOUTS, directions.shape[1], dtype=torch.bool
            ),
            "boundary_hidden_capture_success": (
                [True] * K_ROLLOUTS if capture_success is None else list(capture_success)
            ),
            "boundary_hidden_capture_error_code": [0] * K_ROLLOUTS,
            "boundary_hidden_capture_time_ms": [0.4] * K_ROLLOUTS,
            "boundary_communication_time_ms": [0.0] * K_ROLLOUTS,
            "boundary_extra_peak_memory_mb": [0.07] * K_ROLLOUTS,
            "boundary_hidden_capture_module": "model.norm",
            "boundary_hidden_capture_method": "final_norm_forward_hook",
            "boundary_hidden_dim": HIDDEN_DIM,
        }
    )
    return prompt


def diagnostics_of(result, uid: str) -> dict:
    return next(record for record in result.records if record["prompt_uid"] == uid)
def candidates_of(record) -> dict:
    import json

    return {item["rollout_id"]: item for item in json.loads(record["boundary_candidates_json"])}


# ---------------------------------------------------------------------------
# PDA helpers: this release ships the persistent_departure_area routing rule
# only, so the boundary-scoring tests run against PDA settings with a frozen
# synthetic direct-hidden-state calibration artifact.
# ---------------------------------------------------------------------------
PDA_SIMILARITY_METRIC = "centered_hidden_state_cosine"
PDA_SCORE_MODE = "persistent_departure_area"


def build_pda_calibration(tmp_path) -> dict:
    """Freeze a direct-hidden-state calibration artifact over synthetic states."""

    from verl.utils.boundary_calibration import BoundaryCalibrationAccumulator

    manifest_path = Path(tmp_path) / "calibration" / "manifest.json"
    settings = BoundaryOPDSettings(
        num_boundaries=NUM_BOUNDARIES,
        similarity_metric=PDA_SIMILARITY_METRIC,
        score_mode=PDA_SCORE_MODE,
        calibration_collect=True,
        calibration_artifact_path=str(manifest_path),
        calibration_num_prompts=2,
        calibration_model_hash="3" * 64,
        calibration_data_hash="4" * 64,
    )
    accumulator = BoundaryCalibrationAccumulator(settings, k_rollouts=K_ROLLOUTS)
    generator = torch.Generator().manual_seed(101)
    states = torch.randn(2 * K_ROLLOUTS, NUM_BOUNDARIES, HIDDEN_DIM, generator=generator)
    valid = torch.ones(2 * K_ROLLOUTS, NUM_BOUNDARIES, dtype=torch.bool)
    accumulator.update(
        states,
        valid,
        [f"calibration-{index // K_ROLLOUTS}" for index in range(2 * K_ROLLOUTS)],
        [index % K_ROLLOUTS for index in range(2 * K_ROLLOUTS)],
    )
    artifact = accumulator.finalize()
    return {
        "path": str(manifest_path),
        "sha256": artifact.manifest_sha256,
        "model_hash": "3" * 64,
        "data_hash": "4" * 64,
        "num_prompts": 2,
    }


def make_pda_manager(tmp_path, mode: str = "boundary_opd", **overrides) -> FFOPDQueueManager:
    """Build a queue manager wired for the released PDA routing rule."""

    calibration = build_pda_calibration(tmp_path)
    pda_overrides = {
        "similarity_metric": PDA_SIMILARITY_METRIC,
        "score_mode": PDA_SCORE_MODE,
        "calibration_artifact_path": calibration["path"],
        "calibration_artifact_sha256": calibration["sha256"],
        "calibration_model_hash": calibration["model_hash"],
        "calibration_data_hash": calibration["data_hash"],
        "calibration_num_prompts": calibration["num_prompts"],
        **overrides,
    }
    return make_manager(mode, Path(tmp_path) / "run", **pda_overrides)


def with_direct_states(prompt: dict, states: torch.Tensor) -> dict:
    """Overwrite the fixture prompt with ``[K, M, D]`` direct hidden states.

    PDA consumes M position-normalized states per rollout (no transitions), so
    the legacy transition-derived state layout does not apply.
    """

    prompt["boundary_states"] = states.to(torch.float16)
    prompt["boundary_transition_valid_mask"] = torch.ones(
        K_ROLLOUTS, states.shape[1], dtype=torch.bool
    )
    return prompt
