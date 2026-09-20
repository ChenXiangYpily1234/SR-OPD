import pytest
import torch

from verl.utils.ht_opd import (
    aggregate_ht_opd_numerator,
    build_softmax_sampling_probabilities,
    sample_categorical_index,
)


def test_softmax_probabilities_are_positive_normalized_and_mask_invalid_candidates():
    probabilities = build_softmax_sampling_probabilities(
        torch.tensor([0.0, 2.0, 8.0, float("nan")]),
        torch.tensor([True, True, True, False]),
    )
    assert probabilities[:3].tolist() == pytest.approx(torch.softmax(torch.tensor([0.0, 2.0, 8.0]), 0).tolist())
    assert probabilities[3].item() == 0.0
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_categorical_sampling_is_seed_deterministic():
    probabilities = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    assert sample_categorical_index(probabilities, 42).item() == sample_categorical_index(
        probabilities, 42
    ).item()


def test_ht_numerator_is_unbiased_for_one_categorical_draw():
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    losses = torch.tensor([2.0, 5.0, 11.0], dtype=torch.float64)
    expected = torch.zeros((), dtype=torch.float64)
    for index, probability in enumerate(probabilities):
        numerator = aggregate_ht_opd_numerator(
            per_token_loss=losses[index].reshape(1, 1),
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            inverse_probability_weight=(1.0 / probability).reshape(1),
        )
        expected += probability * numerator
    assert expected.item() == pytest.approx(losses.sum().item())
