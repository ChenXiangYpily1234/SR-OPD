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

"""Shared categorical sampling and Horvitz--Thompson loss utilities."""

from __future__ import annotations

import torch


def build_softmax_sampling_probabilities(
    scores: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> torch.Tensor:
    """Return strictly positive softmax probabilities on eligible candidates."""
    if scores.ndim != 1 or eligible_mask.shape != scores.shape:
        raise ValueError("scores and eligible_mask must be same-length 1-D tensors")
    eligible = eligible_mask.detach().to(device=scores.device, dtype=torch.bool)
    if not bool(eligible.any()):
        raise ValueError("categorical sampling requires at least one eligible candidate")
    finite_scores = scores.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(finite_scores[eligible]).all()):
        raise ValueError("eligible categorical scores must be finite")
    logits = finite_scores.masked_fill(~eligible, -float("inf"))
    probabilities = torch.softmax(logits, dim=0)
    if not bool(torch.all(probabilities[eligible] > 0.0)):
        raise RuntimeError("eligible categorical probabilities must be strictly positive")
    if not torch.isclose(
        probabilities.sum(),
        probabilities.new_tensor(1.0),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("categorical probabilities must sum to one")
    return probabilities


def sample_categorical_index(probabilities: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw one candidate from a normalized probability vector."""
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty 1-D tensor")
    probs = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(probs).all()) or bool((probs < 0.0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    if not torch.isclose(probs.sum(), probs.new_tensor(1.0), rtol=0.0, atol=1.0e-12):
        raise ValueError("probabilities must sum to one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    sampled = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
    return sampled[0].to(device=probabilities.device)


def aggregate_ht_opd_numerator(
    per_token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    inverse_probability_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the detached inverse-probability-weighted token-loss numerator."""
    if per_token_loss.shape != response_mask.shape:
        raise ValueError("per_token_loss and response_mask must have matching [B, T] shapes")
    if inverse_probability_weight.shape != (per_token_loss.shape[0],):
        raise ValueError("inverse_probability_weight must have shape [B]")
    weights = inverse_probability_weight.detach().to(per_token_loss.dtype).unsqueeze(-1)
    mask = response_mask.to(per_token_loss.dtype)
    return (per_token_loss * mask * weights).sum()
