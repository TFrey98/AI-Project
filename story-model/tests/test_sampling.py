import pytest
import torch
from torch import nn

from story_model.sampling import (
    generate_tokens,
    sample_next_token,
)


def test_greedy_selects_highest_logit():
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [4.0, 0.0, 2.0],
        ]
    )

    selected = sample_next_token(
        logits,
        greedy=True,
    )

    assert torch.equal(
        selected,
        torch.tensor([[1], [0]]),
    )


def test_top_k_one_matches_greedy():
    logits = torch.tensor([[0.0, 3.0, 1.0]])

    top_k_result = sample_next_token(
        logits,
        top_k=1,
    )
    greedy_result = sample_next_token(
        logits,
        greedy=True,
    )

    assert torch.equal(top_k_result, greedy_result)


def test_temperature_must_be_positive():
    logits = torch.zeros((1, 3))

    with pytest.raises(
        ValueError,
        match="temperature must be positive",
    ):
        sample_next_token(logits, temperature=0.0)


@pytest.mark.parametrize("top_k", [0, 4])
def test_top_k_must_be_within_vocabulary(top_k):
    logits = torch.zeros((1, 3))

    with pytest.raises(
        ValueError,
        match="top_k must be between",
    ):
        sample_next_token(logits, top_k=top_k)


def test_seeded_sampling_is_reproducible():
    logits = torch.tensor([[0.0, 0.5, 1.0]])

    torch.manual_seed(123)
    first = sample_next_token(logits)

    torch.manual_seed(123)
    second = sample_next_token(logits)

    assert torch.equal(first, second)


def test_generation_crops_visible_context():
    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.longest_sequence = 0

        def forward(self, tokens, targets=None):
            self.longest_sequence = max(
                self.longest_sequence,
                tokens.shape[1],
            )

            logits = torch.zeros(
                (*tokens.shape, 3)
            )
            return logits, None

    model = RecordingModel()
    starting_tokens = torch.zeros(
        (1, 7),
        dtype=torch.long,
    )

    output = generate_tokens(
        model=model,
        starting_tokens=starting_tokens,
        max_new_tokens=3,
        block_size=4,
        greedy=True,
    )

    assert output.shape == (1, 10)
    assert model.longest_sequence == 4