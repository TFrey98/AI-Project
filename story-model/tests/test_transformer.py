import pytest
import torch

from story_model.models.transformer import TransformerLanguageModel
from story_model.models import build_model


def make_model(**overrides) -> TransformerLanguageModel:
    settings = {
        "vocabulary_size": 10,
        "block_size": 8,
        "embedding_dim": 16,
        "attention_heads": 4,
        "layers": 2,
        "feed_forward_dim": 32,
        "dropout": 0.0,
    }

    settings.update(overrides)
    return TransformerLanguageModel(**settings)

def test_factory_builds_transformer():
    model = build_model(
        {
            "name": "transformer",
            "embedding_dim": 16,
            "attention_heads": 4,
            "layers": 2,
            "feed_forward_dim": 32,
            "dropout": 0.0,
        },
        vocabulary_size=10,
        block_size=8,
    )

    assert isinstance(
        model,
        TransformerLanguageModel,
    )

def test_transformer_forward_shapes_and_loss():
    model = make_model()

    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_transformer_cannot_see_future_tokens():
    torch.manual_seed(7)

    model = make_model(block_size=5)
    model.eval()

    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 8, 9]])

    first_logits, _ = model(first)
    second_logits, _ = model(second)

    assert torch.allclose(
        first_logits[:, :3],
        second_logits[:, :3],
        atol=1e-6,
    )


def test_transformer_gradients_are_finite():
    model = make_model()

    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    _, loss = model(tokens, targets)

    assert loss is not None
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"{name} did not receive a gradient"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"{name} received a non-finite gradient"
        )


def test_transformer_generation_extends_sequence():
    model = make_model()

    tokens = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=5)

    assert output.shape == (1, 6)


def test_generation_crops_context_to_block_size():
    model = make_model(block_size=8)

    # The initial sequence is already longer than the model's context.
    tokens = torch.zeros((1, 12), dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=3)

    assert output.shape == (1, 15)


def test_invalid_head_dimensions_are_rejected():
    with pytest.raises(
        ValueError,
        match="embedding_dim must be divisible",
    ):
        make_model(
            embedding_dim=18,
            attention_heads=4,
        )


def test_invalid_layer_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="layers must be at least 1",
    ):
        make_model(layers=0)


def test_forward_rejects_excessive_sequence_length():
    model = make_model(block_size=8)
    tokens = torch.zeros((1, 9), dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="exceeds block size",
    ):
        model(tokens)

def test_generation_is_reproducible_with_seed():
    model = make_model()
    model.eval()

    starting_tokens = torch.zeros(
        (1, 1),
        dtype=torch.long,
    )

    torch.manual_seed(42)
    first = model.generate(
        starting_tokens.clone(),
        max_new_tokens=20,
    )

    torch.manual_seed(42)
    second = model.generate(
        starting_tokens.clone(),
        max_new_tokens=20,
    )

    assert torch.equal(first, second)