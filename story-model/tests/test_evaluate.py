import math

import pytest
import torch

from story_model.evaluate import (
    calculate_bits_per_byte,
    evaluate_token_stream,
)
from story_model.models.bigram import BigramLanguageModel


def make_data() -> torch.Tensor:
    return torch.tensor(
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
        dtype=torch.long,
    )


def test_evaluator_returns_finite_loss_for_every_target():
    model = BigramLanguageModel(vocabulary_size=5)
    data = make_data()

    loss, token_count = evaluate_token_stream(
        model=model,
        data=data,
        block_size=4,
        batch_size=2,
        device="cpu",
    )

    assert math.isfinite(loss)
    assert token_count == len(data) - 1


def test_evaluator_is_deterministic():
    model = BigramLanguageModel(vocabulary_size=5)
    data = make_data()

    first_loss, first_count = evaluate_token_stream(
        model=model,
        data=data,
        block_size=4,
        batch_size=2,
        device="cpu",
    )
    second_loss, second_count = evaluate_token_stream(
        model=model,
        data=data,
        block_size=4,
        batch_size=2,
        device="cpu",
    )

    assert first_loss == pytest.approx(second_loss)
    assert first_count == second_count


def test_evaluator_rejects_too_short_stream():
    model = BigramLanguageModel(vocabulary_size=5)
    data = torch.tensor([0], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="at least two tokens",
    ):
        evaluate_token_stream(
            model=model,
            data=data,
            block_size=4,
            batch_size=2,
            device="cpu",
        )


def test_bits_per_byte_normalizes_token_loss():
    result = calculate_bits_per_byte(
        mean_loss=math.log(2.0),
        token_count=10,
        byte_count=20,
    )

    assert result == pytest.approx(0.5)


def test_bits_per_byte_rejects_empty_byte_count():
    with pytest.raises(
        ValueError,
        match="byte_count must be positive",
    ):
        calculate_bits_per_byte(
            mean_loss=1.0,
            token_count=10,
            byte_count=0,
        )