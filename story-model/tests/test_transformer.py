import pytest
import torch
from torch.nn import functional as F

from story_model.models.transformer import (
    FeedForward,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLUFeedForward,
    TransformerLanguageModel,
    parameter_matched_swiglu_dim,
)
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


def test_transformer_loss_ignores_masked_targets():
    model = make_model()
    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.full_like(tokens, -100)
    targets[:, -1] = torch.tensor([3, 4])

    logits, loss = model(tokens, targets)
    expected = F.cross_entropy(
        logits[:, -1, :],
        targets[:, -1],
    )

    assert loss is not None
    assert torch.allclose(loss, expected)


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


def test_rotary_embedding_preserves_pair_norms():
    rotary = RotaryPositionEmbedding(
        head_dim=4,
        block_size=3,
    )
    vectors = torch.ones((1, 1, 3, 4))

    rotated = rotary(vectors)

    assert torch.allclose(
        rotated[:, :, 0],
        vectors[:, :, 0],
    )

    original_norms = vectors.reshape(
        1,
        1,
        3,
        2,
        2,
    ).norm(dim=-1)
    rotated_norms = rotated.reshape(
        1,
        1,
        3,
        2,
        2,
    ).norm(dim=-1)

    assert torch.allclose(
        original_norms,
        rotated_norms,
        atol=1e-6,
    )


def test_factory_builds_rope_transformer():
    model = build_model(
        {
            "name": "transformer",
            "embedding_dim": 16,
            "attention_heads": 4,
            "layers": 2,
            "feed_forward_dim": 32,
            "dropout": 0.0,
            "position_encoding": "rope",
        },
        vocabulary_size=10,
        block_size=8,
    )

    assert model.position_encoding == "rope"
    assert model.position_embeddings is None


def test_position_variants_share_initial_weights_and_rng():
    torch.manual_seed(123)
    learned = make_model(position_encoding="learned")
    learned_rng = torch.get_rng_state()

    torch.manual_seed(123)
    rope = make_model(position_encoding="rope")
    rope_rng = torch.get_rng_state()

    learned_parameters = dict(
        learned.named_parameters()
    )

    for name, parameter in rope.named_parameters():
        assert torch.equal(
            parameter,
            learned_parameters[name],
        )

    assert torch.equal(learned_rng, rope_rng)


def test_rope_transformer_forward_shapes_and_loss():
    model = make_model(position_encoding="rope")
    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_rope_transformer_cannot_see_future_tokens():
    torch.manual_seed(7)

    model = make_model(
        block_size=5,
        position_encoding="rope",
    )
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


def test_rope_transformer_gradients_are_finite():
    model = make_model(position_encoding="rope")
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


def test_invalid_position_encoding_is_rejected():
    with pytest.raises(
        ValueError,
        match="position_encoding must be",
    ):
        make_model(position_encoding="absolute-ish")


def test_rope_rejects_odd_head_dimension():
    with pytest.raises(
        ValueError,
        match="even attention head dimension",
    ):
        make_model(
            embedding_dim=12,
            attention_heads=4,
            position_encoding="rope",
        )


def test_rmsnorm_scales_to_unit_root_mean_square():
    norm = RMSNorm(dim=4, eps=0.0)
    vectors = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

    normalized = norm(vectors)

    root_mean_square = normalized.pow(2).mean(
        dim=-1
    ).sqrt()

    assert torch.allclose(
        root_mean_square,
        torch.ones_like(root_mean_square),
        atol=1e-6,
    )


def test_factory_builds_rmsnorm_transformer():
    model = build_model(
        {
            "name": "transformer",
            "embedding_dim": 16,
            "attention_heads": 4,
            "layers": 2,
            "feed_forward_dim": 32,
            "dropout": 0.0,
            "normalization": "rmsnorm",
        },
        vocabulary_size=10,
        block_size=8,
    )

    assert isinstance(
        model.final_norm,
        RMSNorm,
    )


def test_rmsnorm_transformer_forward_shapes_and_loss():
    model = make_model(normalization="rmsnorm")
    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_rmsnorm_transformer_cannot_see_future_tokens():
    torch.manual_seed(7)

    model = make_model(
        block_size=5,
        normalization="rmsnorm",
    )
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


def test_rmsnorm_transformer_gradients_are_finite():
    model = make_model(normalization="rmsnorm")
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


def test_rmsnorm_transformer_generation_extends_sequence():
    model = make_model(normalization="rmsnorm")

    tokens = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=5)

    assert output.shape == (1, 6)


def test_invalid_normalization_is_rejected():
    with pytest.raises(
        ValueError,
        match="normalization must be",
    ):
        make_model(normalization="batchnorm")


def test_rmsnorm_has_no_bias_parameter():
    norm = RMSNorm(dim=4)

    parameter_names = {
        name for name, _ in norm.named_parameters()
    }

    assert parameter_names == {"weight"}


def test_swiglu_width_is_parameter_matched():
    hidden_dim = parameter_matched_swiglu_dim(
        embedding_dim=128,
        gelu_hidden_dim=512,
    )

    gelu = FeedForward(
        embedding_dim=128,
        hidden_dim=512,
        dropout=0.0,
    )
    swiglu = SwiGLUFeedForward(
        embedding_dim=128,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    gelu_parameters = sum(
        parameter.numel()
        for parameter in gelu.parameters()
    )
    swiglu_parameters = sum(
        parameter.numel()
        for parameter in swiglu.parameters()
    )

    assert hidden_dim == 341
    assert swiglu_parameters - gelu_parameters == 42


def test_swiglu_matches_manual_gating():
    module = SwiGLUFeedForward(
        embedding_dim=2,
        hidden_dim=2,
        dropout=0.0,
    )

    with torch.no_grad():
        identity = torch.eye(2)

        module.gate_projection.weight.copy_(identity)
        module.gate_projection.bias.zero_()
        module.value_projection.weight.copy_(identity)
        module.value_projection.bias.zero_()
        module.output_projection.weight.copy_(identity)
        module.output_projection.bias.zero_()

    values = torch.tensor([[1.0, -2.0]])

    output = module(values)
    expected = F.silu(values) * values

    assert torch.allclose(
        output,
        expected,
        atol=1e-6,
    )


def test_swiglu_preserves_shape_and_finite_values():
    module = SwiGLUFeedForward(
        embedding_dim=8,
        hidden_dim=6,
        dropout=0.0,
    )
    values = torch.randn((2, 3, 8))

    output = module(values)

    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def test_factory_builds_swiglu_transformer():
    model = build_model(
        {
            "name": "transformer",
            "position_encoding": "rope",
            "normalization": "layernorm",
            "feed_forward_activation": "swiglu",
            "embedding_dim": 16,
            "attention_heads": 4,
            "layers": 2,
            "feed_forward_dim": 32,
            "dropout": 0.0,
        },
        vocabulary_size=10,
        block_size=8,
    )

    assert model.feed_forward_activation == "swiglu"
    assert isinstance(
        model.blocks[0].feed_forward,
        SwiGLUFeedForward,
    )


def test_swiglu_transformer_forward_shapes_and_loss():
    model = make_model(
        position_encoding="rope",
        normalization="layernorm",
        feed_forward_activation="swiglu",
    )
    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_swiglu_transformer_cannot_see_future_tokens():
    torch.manual_seed(7)

    model = make_model(
        block_size=5,
        position_encoding="rope",
        normalization="layernorm",
        feed_forward_activation="swiglu",
    )
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


def test_swiglu_transformer_gradients_are_finite():
    model = make_model(
        position_encoding="rope",
        normalization="layernorm",
        feed_forward_activation="swiglu",
    )
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


def test_invalid_feed_forward_activation_is_rejected():
    with pytest.raises(
        ValueError,
        match="feed_forward_activation must be",
    ):
        make_model(feed_forward_activation="relu")


def test_phase13_medium_model_parameter_count():
    model = TransformerLanguageModel(
        vocabulary_size=512,
        block_size=256,
        embedding_dim=192,
        attention_heads=6,
        layers=6,
        feed_forward_dim=768,
        dropout=0.1,
        position_encoding="rope",
        normalization="layernorm",
        feed_forward_activation="swiglu",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 2_863_616
