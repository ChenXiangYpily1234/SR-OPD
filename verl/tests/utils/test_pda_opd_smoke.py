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

"""PDA-OPD end-to-end CPU smoke test.

Runs the real ``FFOPDQueueManager`` with ``score_mode=persistent_departure_area``
through three optimizer steps on a tiny model and checks the invariants that
make the paper's method what it is:

1. mixed-outcome gating: exactly one Teacher query per mixed prompt and none for
   all-correct / all-incorrect prompts;
2. Teacher-free pre-query routing: no Teacher forward happens before the query
   decision, and no extra Student forward is added;
3. the queried wrong sibling is the one the PDA score ranks first, verified by
   recomputing the score independently from the captured hidden states;
4. the routing diagnostics record the direct-hidden-state representation domain.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.utils.boundary_opd_fixtures import K_ROLLOUTS, NUM_BOUNDARIES
from verl.utils.boundary_calibration import BoundaryCalibrationAccumulator, load_boundary_calibration
from verl.utils.boundary_opd import (
    BoundaryOPDSettings,
    build_centered_hidden_states,
    build_unique_uniform_response_state_indices,
    compute_persistent_departure_area_scores_from_cosine,
    extract_hidden_tensor,
    gather_boundary_states_from_hidden,
    resolve_boundary_capture_target,
)
from verl.utils.ff_opd import FFOPDConfig, FFOPDQueueManager, sampled_reverse_kl_statistics

PROMPT_WIDTH = 4
RESPONSE_WIDTH = 12
VOCAB_SIZE = 11
HIDDEN_SIZE = 8
SCORE_MODE = "persistent_departure_area"
SIMILARITY_METRIC = "centered_hidden_state_cosine"

# Three fresh batches, each mixing 3P1N / 2P2N / 1P3N with all-correct and
# all-incorrect prompts. Only the mixed prompts may trigger a Teacher query.
SCHEDULE = [
    {
        "one-positive": (1, 0, 0, 0),
        "two-positive": (1, 1, 0, 0),
        "three-positive": (1, 1, 1, 0),
        "no-success": (0, 0, 0, 0),
        "all-correct": (1, 1, 1, 1),
    },
    {"step-two": (1, 0, 0, 0)},
    {"step-three": (1, 1, 0, 0)},
]
RESPONSE_LENGTHS = {
    "one-positive": (6, 11, 3, 9),
    "two-positive": (5, 7, 12, 4),
    "three-positive": (6, 6, 6, 10),
    "no-success": (5, 5, 5, 5),
    "all-correct": (7, 4, 8, 6),
    "step-two": (6, 8, 5, 7),
    "step-three": (4, 5, 9, 6),
}


class _Student(nn.Module):
    """Minimal decoder-shaped module: embedding -> body -> final norm -> LM head."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.model.layers = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(2)])
        self.model.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.forward_count = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = torch.tanh(layer(hidden))
        hidden = self.model.norm(hidden)
        return self.lm_head(hidden)


class _Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.forward_count = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        return self.head(torch.tanh(self.body(self.embed(input_ids))))


def _sampled_log_probs(logits: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits[:, PROMPT_WIDTH - 1 : -1, :].float(), dim=-1)
    return log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)


def _build_step_batch(uids, generator):
    rows = []
    for uid in uids:
        for rollout_id in range(K_ROLLOUTS):
            rows.append((uid, rollout_id, RESPONSE_LENGTHS[uid][rollout_id]))
    batch_size = len(rows)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, PROMPT_WIDTH + RESPONSE_WIDTH), generator=generator)
    prompt_mask = torch.ones(batch_size, PROMPT_WIDTH, dtype=torch.bool)
    response_mask = torch.zeros(batch_size, RESPONSE_WIDTH, dtype=torch.bool)
    for index, (_, _, length) in enumerate(rows):
        response_mask[index, :length] = True
    return input_ids, prompt_mask, response_mask


def _build_direct_state_calibration(tmp_path) -> dict:
    """Freeze a direct-hidden-state calibration artifact from synthetic states."""
    manifest_path = tmp_path / "calibration" / "manifest.json"
    settings = BoundaryOPDSettings(
        num_boundaries=NUM_BOUNDARIES,
        similarity_metric=SIMILARITY_METRIC,
        score_mode=SCORE_MODE,
        calibration_collect=True,
        calibration_artifact_path=str(manifest_path),
        calibration_num_prompts=2,
        calibration_model_hash="3" * 64,
        calibration_data_hash="4" * 64,
        whitening_regularization=1.0e-4,
    )
    accumulator = BoundaryCalibrationAccumulator(settings, k_rollouts=K_ROLLOUTS)
    generator = torch.Generator().manual_seed(101)
    # PDA consumes M direct hidden states per rollout (no transitions).
    states = torch.randn(2 * K_ROLLOUTS, NUM_BOUNDARIES, HIDDEN_SIZE, generator=generator)
    valid = torch.ones(2 * K_ROLLOUTS, NUM_BOUNDARIES, dtype=torch.bool)
    prompt_ids = [f"calibration-{index // K_ROLLOUTS}" for index in range(2 * K_ROLLOUTS)]
    rollout_ids = [index % K_ROLLOUTS for index in range(2 * K_ROLLOUTS)]
    accumulator.update(states, valid, prompt_ids, rollout_ids)
    artifact = accumulator.finalize()
    return {
        "path": str(manifest_path),
        "sha256": artifact.manifest_sha256,
        "model_hash": "3" * 64,
        "data_hash": "4" * 64,
        "num_prompts": 2,
    }


def _expected_pda_sibling(prompt: dict, mean: torch.Tensor) -> int:
    """Independently recompute which wrong sibling PDE ranks first."""
    states = prompt["boundary_states"].float()
    correct = prompt["verifier_correct"]
    positive_indices = [i for i, is_correct in enumerate(correct) if is_correct]
    negative_indices = [i for i, is_correct in enumerate(correct) if not is_correct]

    centered = build_centered_hidden_states(
        states,
        prompt["boundary_transition_valid_mask"],
        mean=mean,
        eps=1.0e-6,
    )
    negative = centered.index_select(0, torch.tensor(negative_indices, dtype=torch.long))
    positive = centered.index_select(0, torch.tensor(positive_indices, dtype=torch.long))
    cosine = torch.einsum("nmd,pmd->npm", negative, positive)
    scores = compute_persistent_departure_area_scores_from_cosine(
        cosine,
        PROMPT_WIDTH,
        torch.tensor([prompt["response_tokens"][index] for index in negative_indices], dtype=torch.long),
        torch.tensor(negative_indices, dtype=torch.long),
    )
    return negative_indices[int(scores["selected_local_index"].item())]


def test_pda_opd_end_to_end(tmp_path):
    calibration = _build_direct_state_calibration(tmp_path)
    settings = BoundaryOPDSettings(
        num_boundaries=NUM_BOUNDARIES,
        similarity_metric=SIMILARITY_METRIC,
        score_mode=SCORE_MODE,
        calibration_artifact_path=calibration["path"],
        calibration_artifact_sha256=calibration["sha256"],
        calibration_model_hash=calibration["model_hash"],
        calibration_data_hash=calibration["data_hash"],
        calibration_num_prompts=calibration["num_prompts"],
    )
    mean = load_boundary_calibration(settings).mean

    torch.manual_seed(7)
    student = _Student()
    teacher = _Teacher()
    manager = FFOPDQueueManager(
        FFOPDConfig(
            k_rollouts=K_ROLLOUTS,
            selector_mode="boundary_opd",
            seed=42,
            max_no_success_retries=0,
            csv_path=str(tmp_path / "pda.csv"),
            profile_jsonl_path=str(tmp_path / "pda.jsonl"),
            profile_audit_sample_rate=0.0,
            boundary_opd=settings,
        ),
        run_name="pda-opd-smoke",
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-4)
    generator = torch.Generator().manual_seed(11)
    capture_target = resolve_boundary_capture_target(student)

    total_queries = 0
    total_mixed = 0

    for round_index, verdicts in enumerate(SCHEDULE):
        uids = sorted(verdicts)
        input_ids, prompt_mask, response_mask = _build_step_batch(uids, generator)
        state_indices, state_valid = build_unique_uniform_response_state_indices(
            response_mask, prompt_mask, NUM_BOUNDARIES
        )

        # Exactly one Student forward this step, and the Teacher is untouched.
        captured: dict[str, torch.Tensor] = {}

        def _capture(_module, _inputs, output, captured=captured):
            captured["hidden"] = extract_hidden_tensor(output)

        handle = capture_target.module.register_forward_hook(_capture)
        try:
            student_logits = student(input_ids)
        finally:
            handle.remove()
        assert student.forward_count == round_index + 1, "PDA must not add a Student forward"
        assert teacher.forward_count == round_index, "PDA must select without any Teacher forward"

        boundary_states = gather_boundary_states_from_hidden(
            captured["hidden"],
            state_indices,
            batch_size=input_ids.shape[0],
            sequence_length=input_ids.shape[1],
            hidden_storage_dtype=torch.float16,
        )
        assert boundary_states.requires_grad is False
        response_ids = input_ids[:, PROMPT_WIDTH:]
        student_log_probs = _sampled_log_probs(student_logits, response_ids)

        prompts = []
        for prompt_index, uid in enumerate(uids):
            local = slice(prompt_index * K_ROLLOUTS, (prompt_index + 1) * K_ROLLOUTS)
            lengths = [int(value) for value in response_mask[local].sum(dim=-1)]
            prompts.append(
                {
                    "prompt_uid": uid,
                    "source_index": uid,
                    "rollout_indices": list(range(prompt_index * K_ROLLOUTS, (prompt_index + 1) * K_ROLLOUTS)),
                    "verifier_correct": list(verdicts[uid]),
                    "rollout_valid": [True] * K_ROLLOUTS,
                    "processing_tokens": [PROMPT_WIDTH + length for length in lengths],
                    "response_tokens": lengths,
                    "response_masks": response_mask[local],
                    "sampled_token_log_probs": student_log_probs[local].detach(),
                    "global_step": round_index,
                    "queue_source": "fresh",
                    "rollout_ids": list(range(K_ROLLOUTS)),
                    "boundary_states": boundary_states[local],
                    "boundary_transition_valid_mask": state_valid[local],
                    "boundary_hidden_capture_success": [True] * K_ROLLOUTS,
                    "boundary_hidden_capture_error_code": [0] * K_ROLLOUTS,
                    "boundary_hidden_capture_module": capture_target.module_path,
                    "boundary_hidden_capture_method": capture_target.capture_method,
                    "boundary_hidden_dim": HIDDEN_SIZE,
                }
            )

        result = manager.route_prompt_attempts(prompts)
        selected = set(result.selected_indices)

        for prompt_index, prompt in enumerate(prompts):
            correct = prompt["verifier_correct"]
            is_mixed = any(correct) and not all(correct)
            # Gating: a query exists iff the prompt is mixed.
            queried = [index for index in range(K_ROLLOUTS) if prompt_index * K_ROLLOUTS + index in selected]
            if not is_mixed:
                assert queried == [], f"non-mixed prompt {prompt['prompt_uid']} must not be queried"
                continue
            total_mixed += 1
            assert len(queried) == 1, f"mixed prompt {prompt['prompt_uid']} must query exactly one sibling"
            sibling = queried[0]
            assert not correct[sibling], "PDA must query an incorrect sibling"

            expected = _expected_pda_sibling(prompt, mean)
            assert sibling == expected, (
                f"prompt {prompt['prompt_uid']}: queried sibling {sibling} != PDA argmax {expected}"
            )

            diagnostics = prompt["_boundary_diagnostics"]
            candidates = diagnostics["boundary_candidates_json"]
            assert candidates, "PDA must emit per-candidate diagnostics"
            for candidate in candidates:
                assert candidate["representation_domain"] == "direct_hidden_state"
                assert candidate["uses_hidden_difference"] is False
            assert diagnostics["boundary_fallback_used"] == 0

        # ---- Teacher forward on the selected rollouts only -----------------
        if selected:
            selected_tensor = torch.tensor(sorted(selected), dtype=torch.long)
            teacher_logits = teacher(input_ids[selected_tensor])
            assert teacher.forward_count == round_index + 1, "exactly one Teacher forward per querying step"
            teacher_log_probs = _sampled_log_probs(teacher_logits, response_ids[selected_tensor]).detach()
            sampled_reverse_kl_statistics(
                student_log_probs[selected_tensor].detach(), teacher_log_probs, response_mask[selected_tensor]
            )
            optimizer.zero_grad(set_to_none=True)
            selected_mask = response_mask[selected_tensor].float()
            loss = ((student_log_probs[selected_tensor] - teacher_log_probs) * selected_mask).sum()
            loss = loss / selected_mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
        else:
            assert teacher.forward_count == round_index, "no query, no Teacher forward"

        total_queries += len(selected)

    assert total_mixed > 0, "the schedule must contain mixed prompts"
    assert total_queries == total_mixed