# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Boundary-OPD hidden-trajectory unit tests (spec tests 1-8, 15, 17)."""

import math

import pytest
import torch
from torch import nn

from verl.utils.boundary_opd import (
    BoundaryOPDSettings,
    build_boundary_indices,
    build_normalized_hidden_transitions,
    compute_boundary_contrast_scores,
    extract_hidden_tensor,
    gather_boundary_states_from_hidden,
    resolve_boundary_capture_target,
    spearman_correlation,
)

HIDDEN_DIM = 6


def unit(index: int, dim: int = HIDDEN_DIM) -> torch.Tensor:
    """Return the ``index``-th canonical basis vector."""

    vector = torch.zeros(dim, dtype=torch.float32)
    vector[index] = 1.0
    return vector


def states_from_transitions(directions: torch.Tensor, origin: torch.Tensor | None = None) -> torch.Tensor:
    """Build ``[N, M + 1, D]`` boundary states whose deltas are ``directions``."""

    if directions.ndim != 3:
        raise ValueError("directions must have shape [N, M, D]")
    start = (
        torch.zeros(directions.shape[0], 1, directions.shape[-1], dtype=directions.dtype)
        if origin is None
        else origin.reshape(1, 1, -1).expand(directions.shape[0], 1, -1).clone()
    )
    return torch.cat([start, start + directions.cumsum(dim=1)], dim=1)


# ---------------------------------------------------------------------------
# Spec test 1: boundary indices for response_len=10, M=4
# ---------------------------------------------------------------------------
def test_1_boundary_indices_response_len_10_m_4():
    prompt_mask = torch.ones(1, 5, dtype=torch.bool)
    response_mask = torch.ones(1, 10, dtype=torch.bool)

    state_indices, transition_valid = build_boundary_indices(response_mask, prompt_mask, 4)

    assert state_indices.shape == (1, 5)
    assert transition_valid.shape == (1, 4)
    # Column zero is the last valid prompt token.
    assert int(state_indices[0, 0]) == 4
    # ceil(m * 10 / 4) - 1 -> 2, 4, 7, 9 in response-local ordinals.
    assert state_indices[0, 1:].tolist() == [7, 9, 12, 14]
    response_boundaries = state_indices[0, 1:]
    assert torch.all(response_boundaries[1:] >= response_boundaries[:-1])
    # The final boundary must be the final valid response token.
    assert int(response_boundaries[-1]) == 5 + 9
    assert int(state_indices.min()) >= 0
    assert int(state_indices.max()) < 15
    assert bool(transition_valid.all())


def test_1b_boundary_indices_handle_left_padded_prompts_and_variable_lengths():
    # verl pads prompts on the left and responses on the right.
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.bool)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    state_indices, transition_valid = build_boundary_indices(response_mask, prompt_mask, 4)

    assert state_indices[:, 0].tolist() == [4, 4]
    # row0: T=6 -> ordinals 1,2,4,5 ; row1: T=4 -> ordinals 0,1,2,3
    assert state_indices[0, 1:].tolist() == [6, 7, 9, 10]
    assert state_indices[1, 1:].tolist() == [5, 6, 7, 8]
    assert bool(transition_valid.all())
    assert int(state_indices.max()) < prompt_mask.shape[1] + response_mask.shape[1]


# ---------------------------------------------------------------------------
# Spec test 2: short response, response_len=3 with M=16
# ---------------------------------------------------------------------------
def test_2_short_response_allows_repeated_boundaries():
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    response_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

    state_indices, transition_valid = build_boundary_indices(response_mask, prompt_mask, 16)

    assert state_indices.shape == (1, 17)
    assert transition_valid.shape == (1, 16)
    assert int(state_indices[0, 0]) == 4
    assert state_indices[0, 1:].tolist() == [5] * 5 + [6] * 5 + [7] * 6
    # Only the three transitions that actually move are valid.
    assert transition_valid[0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert int(transition_valid.sum()) == 3
    assert int(state_indices.min()) >= 0
    assert int(state_indices.max()) < 12

    states = torch.randn(1, 17, HIDDEN_DIM)
    transitions = build_normalized_hidden_transitions(states, transition_valid)
    assert torch.isfinite(transitions).all()
    assert torch.equal(
        transitions[0, ~transition_valid[0]],
        torch.zeros(int((~transition_valid[0]).sum()), HIDDEN_DIM),
    )


def test_2b_empty_response_is_invalid_everywhere():
    prompt_mask = torch.ones(2, 3, dtype=torch.bool)
    response_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)

    state_indices, transition_valid = build_boundary_indices(response_mask, prompt_mask, 8)

    assert not bool(transition_valid[0].any())
    assert bool(transition_valid[1].any())
    # An empty response collapses onto the prompt end and stays in range.
    assert state_indices[0].tolist() == [2] * 9
    assert int(state_indices.min()) >= 0
    assert int(state_indices.max()) < 7


# ---------------------------------------------------------------------------
# Spec test 15: no gradient leakage
# ---------------------------------------------------------------------------
class _TinyLanguageModel(nn.Module):
    """Minimal decoder-shaped module: embedding -> body -> final norm -> LM head."""

    def __init__(self, vocab_size: int = 11, hidden_size: int = HIDDEN_DIM):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(2)])
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
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


def _capture_boundary_states(model: _TinyLanguageModel, input_ids: torch.Tensor, state_indices: torch.Tensor):
    target = resolve_boundary_capture_target(model, capture_method="auto")
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = extract_hidden_tensor(output)

    handle = target.module.register_forward_hook(hook)
    try:
        logits = model(input_ids)
    finally:
        handle.remove()
    states = gather_boundary_states_from_hidden(
        captured["hidden"],
        state_indices,
        batch_size=input_ids.shape[0],
        sequence_length=input_ids.shape[1],
        hidden_storage_dtype=torch.float16,
    )
    return logits, states, target


def test_15_boundary_capture_does_not_leak_gradients():
    torch.manual_seed(0)
    model = _TinyLanguageModel()
    input_ids = torch.randint(0, 11, (4, 12))
    prompt_mask = torch.ones(4, 4, dtype=torch.bool)
    response_mask = torch.ones(4, 8, dtype=torch.bool)
    state_indices, transition_valid = build_boundary_indices(response_mask, prompt_mask, 4)

    logits, states, _ = _capture_boundary_states(model, input_ids, state_indices)
    transitions = build_normalized_hidden_transitions(states, transition_valid)

    assert states.requires_grad is False
    assert transitions.requires_grad is False
    assert states.grad_fn is None and transitions.grad_fn is None
    assert states.shape == (4, 5, HIDDEN_DIM)
    assert states.dtype == torch.float16

    # Consuming the boundary payload must not change actor gradients at all.
    loss = logits.float().pow(2).mean()
    scores = compute_boundary_contrast_scores(
        transitions[:2],
        transitions[2:],
        transition_valid[:2],
        transition_valid[2:],
        prompt_length=4,
        negative_response_lengths=torch.full((2,), 8),
        negative_rollout_ids=torch.arange(2),
    )
    assert scores["boundary_utility"].requires_grad is False
    assert scores["final_score"].requires_grad is False
    loss.backward()
    with_boundary = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}

    baseline = _TinyLanguageModel()
    baseline.load_state_dict(model.state_dict())
    baseline(input_ids).float().pow(2).mean().backward()
    for name, parameter in baseline.named_parameters():
        assert torch.equal(parameter.grad, with_boundary[name])


def test_15b_capture_target_is_pre_lm_head_and_needs_one_forward():
    torch.manual_seed(0)
    model = _TinyLanguageModel()
    input_ids = torch.randint(0, 11, (2, 6))
    state_indices = torch.tensor([[1, 3, 5], [0, 2, 4]])

    _, states, target = _capture_boundary_states(model, input_ids, state_indices)

    assert target.module_path == "model.norm"
    assert target.capture_method == "final_norm_forward_hook"
    assert model.forward_count == 1
    assert states.shape == (2, 3, HIDDEN_DIM)

    # The captured tensor is exactly the LM-head input, i.e. post final norm.
    lm_head_inputs = []
    handle = model.lm_head.register_forward_pre_hook(lambda _m, inputs: lm_head_inputs.append(inputs[0]))
    try:
        model(input_ids)
    finally:
        handle.remove()
    expected = gather_boundary_states_from_hidden(
        lm_head_inputs[0],
        state_indices,
        batch_size=2,
        sequence_length=6,
        hidden_storage_dtype=torch.float16,
    )
    assert torch.equal(expected, states)


def test_15c_lm_head_pre_hook_capture_matches_final_norm_capture():
    torch.manual_seed(0)
    model = _TinyLanguageModel()
    input_ids = torch.randint(0, 11, (2, 6))
    state_indices = torch.tensor([[1, 3, 5], [0, 2, 4]])

    target = resolve_boundary_capture_target(model, capture_method="lm_head_pre_hook")
    assert target.capture_method == "lm_head_forward_pre_hook"
    assert target.module_path == "lm_head"

    captured = {}
    handle = target.module.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("hidden", extract_hidden_tensor(inputs))
    )
    try:
        model(input_ids)
    finally:
        handle.remove()
    from_pre_hook = gather_boundary_states_from_hidden(
        captured["hidden"],
        state_indices,
        batch_size=2,
        sequence_length=6,
        hidden_storage_dtype=torch.float16,
    )

    _, from_norm_hook, _ = _capture_boundary_states(model, input_ids, state_indices)
    assert torch.equal(from_pre_hook, from_norm_hook)


def test_15d_remove_padding_gather_matches_padded_gather():
    torch.manual_seed(0)
    batch_size, sequence_length = 3, 7
    hidden = torch.randn(batch_size, sequence_length, HIDDEN_DIM)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    state_indices = torch.tensor([[1, 3, 4], [0, 3, 6], [2, 4, 5]])
    padded = gather_boundary_states_from_hidden(
        hidden,
        state_indices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_storage_dtype=torch.float16,
    )

    flat_indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    unpadded_hidden = hidden.reshape(-1, HIDDEN_DIM)[flat_indices].unsqueeze(0)
    unpadded = gather_boundary_states_from_hidden(
        unpadded_hidden,
        state_indices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_storage_dtype=torch.float16,
        unpadded_indices=flat_indices,
    )

    assert torch.equal(padded, unpadded)

    # A boundary that points at a padded token must be rejected loudly.
    with pytest.raises(ValueError, match="padded token"):
        gather_boundary_states_from_hidden(
            unpadded_hidden,
            torch.tensor([[0, 3, 4], [0, 3, 6], [2, 4, 5]]),
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_storage_dtype=torch.float16,
            unpadded_indices=flat_indices,
        )


# ---------------------------------------------------------------------------
# Spec test 17: numerical stability
# ---------------------------------------------------------------------------
def test_17e_spearman_guard_degenerate_inputs():
    assert math.isnan(spearman_correlation(torch.tensor([1.0]), torch.tensor([2.0])))
    assert spearman_correlation(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 9.0])) == pytest.approx(1.0)
    assert spearman_correlation(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([9.0, 4.0, 2.0])) == pytest.approx(-1.0)


def test_settings_validation_rejects_unsupported_choices():
    assert BoundaryOPDSettings().num_boundaries == 16
    assert BoundaryOPDSettings.from_mapping({"num_boundaries": 8}).num_boundaries == 8
    for values in (
        {"num_boundaries": 12},
        {"capture_location": "post_lm_head"},
        {"detach_hidden": False},
        {"hidden_storage_dtype": "float32"},
        {"similarity_metric": "unknown"},
        {"fallback_to_ff_cost": True},
    ):
        with pytest.raises(ValueError):
            BoundaryOPDSettings.from_mapping(values)
