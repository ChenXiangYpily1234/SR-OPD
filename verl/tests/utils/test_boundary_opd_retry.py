# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Boundary-OPD retry, regroup and logging contract tests (spec 14, 16-17)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.utils.boundary_opd_fixtures import (
    K_ROLLOUTS,
    NUM_BOUNDARIES,
    candidates_of,
    diagnostics_of,
    distinct_plan,
    early_late_plan,
    make_manager,
    make_pda_manager,
    make_prompt,
    with_direct_states,
)
from tests.utils.test_boundary_opd import HIDDEN_DIM
from verl.utils.boundary_opd import (
    BOUNDARY_SELECTOR_MODES,
    gather_boundary_states_from_hidden,
    stable_group_sibling_indices,
)
from verl.utils.ff_opd import FFBucket
from verl.utils.frontier_selector import FF_SELECTOR_MODES


# ---------------------------------------------------------------------------
# Spec test 14: retry behaviour is selector independent
# ---------------------------------------------------------------------------
def _run_retry_schedule(mode: str, tmp_path) -> dict:
    manager = make_manager(mode, tmp_path, max_no_success_retries=2)
    verifier_by_round = [
        {"p0": (1, 1, 1, 1), "p1": (0, 0, 0, 0), "p2": (0, 0, 0, 0), "p3": (1, 0, 0, 0)},
        {"p1": (0, 0, 0, 0), "p2": (1, 0, 0, 0)},
        {"p1": (0, 0, 0, 0)},
    ]
    trace: dict = {"queues": [], "buckets": [], "rounds": [], "queries": []}
    for round_index, verifier in enumerate(verifier_by_round):
        if round_index and not manager.begin_retry_round():
            break
        prompts = [
            make_prompt(
                uid,
                correct,
                directions=distinct_plan(),
                rollout_offset=index * K_ROLLOUTS,
                global_step=round_index,
                queue_source="fresh" if round_index == 0 else "retry",
            )
            for index, (uid, correct) in enumerate(sorted(verifier.items()))
        ]
        result = manager.route_prompt_attempts(prompts)
        trace["buckets"].append(sorted({(record["prompt_uid"], record["bucket"]) for record in result.records}))
        trace["queues"].append(list(manager.no_success_retry_queue))
        trace["rounds"].append(manager.retry_round)
        trace["queries"].append(len(result.selected_indices))
    trace["final_status"] = sorted(
        (uid, state.final_status, state.attempt_count, state.retry_count)
        for uid, state in manager.prompt_states.items()
    )
    return trace


# The release supports the PDA routing rule only, and PDA score modes require
# max_no_success_retries=0, so the retry schedule is exercised over the
# non-boundary selectors (the boundary selector never sees NO_SUCCESS retries).
_NON_BOUNDARY_SELECTOR_MODES = tuple(
    mode for mode in FF_SELECTOR_MODES if mode not in BOUNDARY_SELECTOR_MODES
)


@pytest.mark.parametrize("mode", _NON_BOUNDARY_SELECTOR_MODES)
def test_14_retry_queue_is_identical_across_selector_modes(mode, tmp_path):
    reference = _run_retry_schedule("nearest_only", tmp_path / "reference")
    observed = _run_retry_schedule(mode, tmp_path / mode)

    assert observed["queues"] == reference["queues"]
    assert observed["buckets"] == reference["buckets"]
    assert observed["rounds"] == reference["rounds"]
    if mode == "all_wrong":
        assert observed["queries"] == [3, 3, 0]
    else:
        assert observed["queries"] == reference["queries"]
    assert observed["final_status"] == reference["final_status"]
    # One fresh pass plus exactly two NO_SUCCESS retry rounds.
    assert reference["rounds"] == [0, 1, 2]
    assert reference["queues"][0] == ["p1", "p2"]
    assert dict((uid, status) for uid, status, _, _ in reference["final_status"])["p1"] == ("exhausted_no_success")
    assert dict((uid, retries) for uid, _, _, retries in reference["final_status"])["p1"] == 2


@pytest.mark.parametrize("mode", BOUNDARY_SELECTOR_MODES)
def test_14b_boundary_modes_do_not_change_bucket_classification(mode, tmp_path):
    manager = make_manager(mode, tmp_path)

    assert manager.classify_prompt((1, 1, 1, 1)) is FFBucket.ALL_CORRECT
    assert manager.classify_prompt((0, 0, 0, 0)) is FFBucket.NO_SUCCESS
    assert manager.classify_prompt((1, 0, 0, 0)) is FFBucket.FRONTIER
    assert manager.classify_prompt((1, 1, 1, 0)) is FFBucket.FRONTIER
    assert manager.classify_prompt((0, 0, 0, 0), rollout_valid=(True, True, True, False)) is FFBucket.INVALID
    assert manager.config.max_no_success_retries == 0
    assert manager.config.base_epochs == 1
    assert manager.config.k_rollouts == 4


def test_14c_exhausted_prompts_cannot_be_retried_a_third_time(tmp_path):
    manager = make_manager("boundary_opd", tmp_path, max_no_success_retries=2)

    for round_index in range(4):
        if round_index and not manager.begin_retry_round():
            break
        manager.route_prompt_attempts(
            [make_prompt("p0", (0, 0, 0, 0), directions=distinct_plan(), global_step=round_index)]
        )

    assert manager.retry_round == 2
    assert manager.prompt_states["p0"].attempt_count == 3
    assert manager.prompt_states["p0"].retry_count == 2
    assert manager.no_success_retry_queue == []
    assert manager.prompt_states["p0"].final_status == "exhausted_no_success"
    assert manager.begin_retry_round() is False


def test_14d_no_success_prompts_never_reach_the_boundary_selector(tmp_path):
    manager = make_manager("boundary_opd", tmp_path)

    result = manager.route_prompt_attempts([make_prompt("p0", (0, 0, 0, 0), directions=distinct_plan())])
    record = diagnostics_of(result, "p0")

    assert result.selected_indices == []
    assert record["boundary_hidden_available"] == 0
    assert record["boundary_selected_rollout_id"] == ""
    assert record["boundary_fallback_used"] == 0
    assert result.metrics["boundary/fallback_rate"] == 0.0
    assert manager.no_success_retry_queue == []
    assert manager.prompt_states["p0"].final_status == "exhausted_no_success"


# ---------------------------------------------------------------------------
# Spec test 16: distributed sibling regroup consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]])
def test_16_scattered_siblings_regroup_to_the_same_selection(tmp_path, permutation):
    """Sibling rows may arrive in any rank order; the pick must not move."""

    plan = early_late_plan()
    order = torch.tensor(permutation)
    correct = torch.tensor([1, 0, 0, 0])[order]
    costs = torch.tensor([120, 900, 120, 130])[order]
    prompt = make_prompt(
        "p0",
        tuple(int(value) for value in correct),
        directions=plan[order],
        costs=tuple(int(value) for value in costs),
        rollout_ids=tuple(permutation),
    )
    with_direct_states(prompt, plan[order])

    result = make_pda_manager(tmp_path).route_prompt_attempts([prompt])
    record = diagnostics_of(result, "p0")

    assert len(result.selected_indices) == 1
    assert record["boundary_selected_rollout_id"] == 2
    assert record["boundary_nearest_positive_rollout_id"] == 0
    assert candidates_of(record)[2]["selected"] == 1


def test_16b_stable_group_sibling_indices_is_order_independent():
    prompt_ids = ["b", "a", "b", "a", "a", "b", "a", "b"]
    rollout_ids = [3, 1, 0, 0, 3, 2, 2, 1]

    grouped = stable_group_sibling_indices(prompt_ids, rollout_ids, expected_siblings=4)

    assert [key for key, _ in grouped] == ["a", "b"]
    assert [[rollout_ids[index] for index in indices] for _, indices in grouped] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    shuffled = torch.randperm(len(prompt_ids), generator=torch.Generator().manual_seed(0)).tolist()
    reshuffled = stable_group_sibling_indices(
        [prompt_ids[index] for index in shuffled],
        [rollout_ids[index] for index in shuffled],
        expected_siblings=4,
    )
    assert [key for key, _ in reshuffled] == ["a", "b"]
    assert [[rollout_ids[shuffled[index]] for index in indices] for _, indices in reshuffled] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    with pytest.raises(ValueError, match="duplicate"):
        stable_group_sibling_indices(["a", "a"], [0, 0], expected_siblings=2)
    with pytest.raises(ValueError, match="expected 4"):
        stable_group_sibling_indices(["a", "a"], [0, 1], expected_siblings=4)


def _sequence_shard_inputs():
    generator = torch.Generator().manual_seed(1234)
    batch_size, sequence_length = 2, 6
    hidden = torch.randn(batch_size * sequence_length, HIDDEN_DIM, generator=generator)
    unpadded_indices = torch.arange(batch_size * sequence_length)
    state_indices = torch.tensor([[1, 3, 5], [0, 2, 4]])
    return hidden, unpadded_indices, state_indices, batch_size, sequence_length


def _sequence_shard_reference() -> torch.Tensor:
    hidden, unpadded_indices, state_indices, batch_size, sequence_length = _sequence_shard_inputs()
    return gather_boundary_states_from_hidden(
        hidden.unsqueeze(0),
        state_indices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_storage_dtype=torch.float16,
        unpadded_indices=unpadded_indices,
    )


def _sequence_shard_worker(rank: int, world_size: int, rendezvous: str, queue) -> None:  # pragma: no cover
    import torch.distributed as dist

    from verl.utils.boundary_opd import gather_boundary_states_from_sequence_shard

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=world_size,
        )
        hidden, unpadded_indices, state_indices, batch_size, sequence_length = _sequence_shard_inputs()
        shard = hidden.chunk(world_size, dim=0)[rank]
        states = gather_boundary_states_from_sequence_shard(
            shard,
            state_indices,
            batch_size=batch_size,
            sequence_length=sequence_length,
            unpadded_indices=unpadded_indices,
            sequence_padding_size=0,
            sequence_parallel_size=world_size,
            sequence_parallel_rank=rank,
            process_group=dist.group.WORLD,
            hidden_storage_dtype=torch.float16,
        )
        queue.put((rank, "ok", states.float().tolist()))
    except Exception as error:
        queue.put((rank, "error", f"{type(error).__name__}: {error}"))
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pragma: no cover - teardown best effort
            pass


def test_16c_sequence_parallel_shard_gather_matches_single_rank(tmp_path):
    """Two gloo ranks must reconstruct exactly the single-rank boundary states."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():  # pragma: no cover - env dependent
        pytest.skip("gloo is unavailable")
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    rendezvous = str(Path(tmp_path) / "sp_rendezvous")
    processes = [context.Process(target=_sequence_shard_worker, args=(rank, 2, rendezvous, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = [queue.get() for _ in processes]
    for process in processes:
        process.join(timeout=180)

    payloads = {}
    for rank, status, payload in results:
        assert status == "ok", payload
        payloads[rank] = torch.tensor(payload)
    assert torch.equal(payloads[0], payloads[1])
    assert torch.equal(payloads[0].half(), _sequence_shard_reference())


# ---------------------------------------------------------------------------
# Spec sections 16 and 17: CSV rows, step metrics and alpha*
# ---------------------------------------------------------------------------
def test_csv_row_exposes_every_boundary_field(tmp_path):
    prompt = make_prompt("p0", (1, 0, 0, 0), directions=early_late_plan(), costs=(100, 4000, 150, 200))
    result = make_pda_manager(tmp_path).route_prompt_attempts(
        [with_direct_states(prompt, early_late_plan())]
    )
    record = diagnostics_of(result, "p0")

    for field in (
        "ff_selector_mode",
        "num_boundaries",
        "hidden_capture_module",
        "hidden_capture_method",
        "hidden_capture_success",
        "hidden_dim",
        "weight_mode",
        "boundary_transition_distance",
        "boundary_nearest_positive_rollout_id",
        "boundary_teacher_cost",
        "boundary_objective",
        "boundary_selected_rollout_id",
        "boundary_nearest_rollout_id",
        "boundary_cost_only_rollout_id",
        "boundary_random_wrong_rollout_id",
        "boundary_cost_switch",
        "boundary_hidden_available",
        "boundary_fallback_used",
        "boundary_fallback_reason",
        "boundary_valid_transition_count",
        "num_positive_rollouts",
        "num_negative_rollouts",
        "boundary_utility",
        "boundary_final_score",
        "boundary_best_positive_rollout_id",
        "boundary_best_split_index",
        "boundary_pre_similarity",
        "boundary_post_similarity",
        "boundary_fork_drop",
        "boundary_post_divergence",
        "boundary_degenerate_to_cost_only",
        "boundary_candidates_json",
    ):
        assert field in record, field

    assert record["ff_selector_mode"] == "boundary_opd"
    assert record["num_boundaries"] == NUM_BOUNDARIES
    assert record["weight_mode"] == "pre_post_split"
    assert record["hidden_capture_module"] == "model.norm"
    assert record["hidden_capture_method"] == "final_norm_forward_hook"
    assert record["hidden_capture_success"] == 1
    assert record["hidden_dim"] == HIDDEN_DIM
    assert record["boundary_cost_switch"] == int(
        record["boundary_selected_rollout_id"] != record["boundary_nearest_rollout_id"]
    )
    assert float(record["boundary_teacher_cost"]) > 0
    assert float(record["boundary_objective"]) > 0

    candidates = candidates_of(record)
    assert len(candidates) == 3
    for candidate in candidates.values():
        assert {
            "rollout_id",
            "transition_distance",
            "confidence_distance",
            "teacher_cost",
            "boundary_objective",
            "rank_transition",
            "rank_confidence",
            "rank_objective",
            "selected",
            "boundary_utility",
            "cost_efficiency",
            "boundary_final_score",
            "best_positive_rollout_id",
            "best_split_index",
            "pre_similarity",
            "post_similarity",
            "fork_drop",
            "post_divergence",
            "valid_split_count",
            "valid_candidate",
        } <= set(candidate)
    assert sorted(item["rank_objective"] for item in candidates.values()) == [1, 2, 3]
    assert sorted(item["rank_transition"] for item in candidates.values()) == [1, 2, 3]
    assert sorted(item["rank_confidence"] for item in candidates.values()) == [1, 2, 3]

    # The README-documented per-candidate PDA routing trace.
    for candidate in candidates.values():
        assert {
            "raw_cosine",
            "stage_similarities",
            "running_max",
            "departure",
            "pda_area",
            "max_departure",
            "mean_departure",
        } <= set(candidate)
        assert candidate["representation_domain"] == "direct_hidden_state"
        assert int(candidate["uses_hidden_difference"]) == 0
        assert candidate["score_mode"] == "persistent_departure_area"


def test_step_metrics_report_boundary_and_legacy_accounting(tmp_path):
    plan = early_late_plan()
    prompts = [
        make_prompt("p0", (1, 0, 0, 0), directions=plan, costs=(100, 4000, 150, 200)),
        make_prompt("p1", (1, 1, 0, 0), directions=distinct_plan(), rollout_offset=4),
        make_prompt("p2", (0, 0, 0, 0), directions=distinct_plan(), rollout_offset=8),
        make_prompt("p3", (1, 1, 1, 1), directions=distinct_plan(), rollout_offset=12),
    ]
    manager = make_pda_manager(tmp_path)
    result = manager.route_prompt_attempts(
        [with_direct_states(prompt, plan if prompt["prompt_uid"] == "p0" else distinct_plan()) for prompt in prompts]
    )

    for metric in (
        "boundary/selected_distance_mean",
        "boundary/selected_cost_mean",
        "boundary/selected_objective_mean",
        "boundary/cost_switch_rate",
        "boundary/confidence_distance_spearman",
        "boundary/fallback_rate",
        "boundary/valid_transition_ratio",
        "boundary/hidden_capture_time_ms",
        "boundary/transition_build_time_ms",
        "boundary/distance_compute_time_ms",
        "boundary/communication_time_ms",
        "boundary/extra_peak_memory_mb",
        "boundary/selected_utility_mean",
        "boundary/selected_score_mean",
        "boundary/selected_pre_similarity_mean",
        "boundary/selected_post_similarity_mean",
        "boundary/selected_fork_drop_mean",
        "boundary/selected_split_fraction_mean",
        "boundary/cost_only_agreement",
        "boundary/degenerate_to_cost_only_rate",
        "boundary/valid_candidate_rate",
        "boundary/valid_split_mean",
        "boundary/score_compute_time_ms",
    ):
        assert metric in result.metrics, metric
        assert math.isfinite(float(result.metrics[metric])), metric

    # The unchanged FF-OPD accounting is still reported next to it.
    assert result.metrics["ff/frontier_prompts"] == 2
    assert result.metrics["ff/no_success_prompts"] == 1
    assert result.metrics["ff/all_correct_prompts"] == 1
    assert result.metrics["ff/retry_round"] == 0
    assert result.metrics["opd_query/queried_rollout_count"] == 2
    assert result.metrics["opd_query/full_rollout_count"] == 16
    assert result.metrics["boundary/valid_transition_ratio"] == 1.0
    assert result.metrics["boundary/fallback_rate"] == 0.0
    assert 0.0 <= result.metrics["boundary/cost_switch_rate"] <= 1.0
    assert result.metrics["boundary/extra_peak_memory_mb"] > 0.0


def test_old_boundary_cost_formula_is_absent_from_executable_code():
    root = Path(__file__).resolve().parents[2]
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (root / "verl").rglob("*.py"))
    assert "cost_power_alpha" not in sources
    assert "select_boundary_opd_rollout" not in sources
