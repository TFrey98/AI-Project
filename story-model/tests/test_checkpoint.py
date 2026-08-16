import pytest
import torch

from story_model.checkpoint import (
    load_checkpoint,
    load_model_warm_start,
    read_checkpoint,
    save_checkpoint,
)
from story_model.models.bigram import BigramLanguageModel
from story_model.models.transformer import TransformerLanguageModel


def test_checkpoint_contains_version_and_rng(tmp_path):
    model = BigramLanguageModel(vocabulary_size=10)

    path = tmp_path / "versioned.pt"

    save_checkpoint(
        path,
        model,
        step=12,
    )

    checkpoint = read_checkpoint(
        path,
        map_location="cpu",
    )

    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["step"] == 12
    assert "cpu" in checkpoint["rng_state"]


def test_checkpoint_restores_cpu_rng(tmp_path):
    torch.manual_seed(1234)

    model = BigramLanguageModel(vocabulary_size=10)
    path = tmp_path / "rng.pt"

    save_checkpoint(
        path,
        model,
        step=1,
    )

    expected_random_values = torch.rand(5)

    # Deliberately disturb the random generator.
    torch.manual_seed(9999)
    torch.rand(20)

    restored_model = BigramLanguageModel(
        vocabulary_size=10
    )

    load_checkpoint(
        path,
        restored_model,
        map_location="cpu",
        restore_rng=True,
    )

    actual_random_values = torch.rand(5)

    assert torch.equal(
        expected_random_values,
        actual_random_values,
    )
def test_save_and_load_roundtrip(tmp_path):
    model = BigramLanguageModel(vocabulary_size=10)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    path = tmp_path / "checkpoints" / "test.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        step=42,
        extra={"note": "test"},
    )

    assert path.exists()

    loaded_model = BigramLanguageModel(
        vocabulary_size=10
    )
    loaded_optimizer = torch.optim.AdamW(
        loaded_model.parameters(),
        lr=1e-3,
    )

    checkpoint = load_checkpoint(
        path,
        loaded_model,
        loaded_optimizer,
        map_location="cpu",
    )

    assert checkpoint["step"] == 42
    assert checkpoint["extra"] == {"note": "test"}

    for original, loaded in zip(
        model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.equal(original, loaded)


def build_warm_start_transformer(
    vocabulary_size,
    block_size,
):
    return TransformerLanguageModel(
        vocabulary_size=vocabulary_size,
        block_size=block_size,
        embedding_dim=16,
        attention_heads=4,
        layers=1,
        feed_forward_dim=32,
        dropout=0.0,
        position_encoding="rope",
    )


def test_warm_start_preserves_old_rows_and_new_initialization():
    torch.manual_seed(1)
    source_model = build_warm_start_transformer(10, 8)
    checkpoint = {
        "model_state_dict": source_model.state_dict(),
    }

    torch.manual_seed(2)
    destination_model = build_warm_start_transformer(13, 16)
    initial_destination = {
        name: tensor.clone()
        for name, tensor in destination_model.state_dict().items()
    }

    expanded = load_model_warm_start(
        model=destination_model,
        checkpoint=checkpoint,
        source_vocabulary_size=10,
        destination_vocabulary_size=13,
    )
    loaded = destination_model.state_dict()

    assert set(expanded) == {
        "token_embeddings.weight",
        "output_projection.weight",
        "output_projection.bias",
    }

    for name in expanded:
        assert torch.equal(
            loaded[name][:10],
            checkpoint["model_state_dict"][name],
        )
        assert torch.equal(
            loaded[name][10:],
            initial_destination[name][10:],
        )

    for name, source_tensor in checkpoint[
        "model_state_dict"
    ].items():
        if name not in expanded:
            assert torch.equal(loaded[name], source_tensor)


def test_warm_start_rejects_vocabulary_shrink():
    model = build_warm_start_transformer(9, 8)
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    with pytest.raises(ValueError, match="cannot shrink"):
        load_model_warm_start(
            model=model,
            checkpoint=checkpoint,
            source_vocabulary_size=10,
            destination_vocabulary_size=9,
        )
