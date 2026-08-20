import math

import pytest
import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer
from story_model.train import (
    early_stopping_reached,
    learning_rate_for_step,
    save_best_validation_checkpoint,
    tokenizer_for_warm_start,
    validation_loss_improved,
)


def test_early_stopping_is_disabled_without_patience():
    assert not early_stopping_reached(100, None)


def test_early_stopping_reaches_configured_patience():
    assert not early_stopping_reached(2, 3)
    assert early_stopping_reached(3, 3)
    assert early_stopping_reached(4, 3)


def test_early_stopping_rejects_invalid_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        early_stopping_reached(-1, 3)

    with pytest.raises(ValueError, match="must be positive"):
        early_stopping_reached(0, 0)


def test_learning_rate_warmup():
    rate = learning_rate_for_step(
        step=0,
        max_steps=100,
        maximum_rate=3.0e-4,
        minimum_rate=3.0e-5,
        warmup_steps=10,
    )

    assert math.isclose(
        rate,
        3.0e-5,
        rel_tol=1.0e-6,
    )


def test_learning_rate_reaches_maximum():
    rate = learning_rate_for_step(
        step=10,
        max_steps=100,
        maximum_rate=3.0e-4,
        minimum_rate=3.0e-5,
        warmup_steps=10,
    )

    assert math.isclose(
        rate,
        3.0e-4,
        rel_tol=1.0e-6,
    )


def test_learning_rate_reaches_minimum():
    rate = learning_rate_for_step(
        step=99,
        max_steps=100,
        maximum_rate=3.0e-4,
        minimum_rate=3.0e-5,
        warmup_steps=10,
    )

    assert math.isclose(
        rate,
        3.0e-5,
        rel_tol=1.0e-6,
    )


def test_invalid_warmup_is_rejected():
    with pytest.raises(
        ValueError,
        match="smaller than max_steps",
    ):
        learning_rate_for_step(
            step=0,
            max_steps=100,
            maximum_rate=3.0e-4,
            minimum_rate=3.0e-5,
            warmup_steps=100,
        )


def test_validation_improvement_is_strict():
    assert validation_loss_improved(1.9, None)
    assert validation_loss_improved(1.9, 2.0)
    assert not validation_loss_improved(2.0, 2.0)
    assert not validation_loss_improved(2.1, 2.0)


def test_non_finite_validation_loss_is_not_an_improvement():
    assert not validation_loss_improved(float("nan"), None)
    assert not validation_loss_improved(float("inf"), 2.0)


def test_best_checkpoint_records_evaluated_step(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "best.pt"
    metadata = {"config": {"name": "test"}}

    save_best_validation_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=500,
        checkpoint_metadata=metadata,
        validation_loss=1.75,
    )

    checkpoint = read_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["step"] == 500
    assert checkpoint["extra"]["best_step"] == 500
    assert checkpoint["extra"]["best_validation_loss"] == 1.75
    assert checkpoint["extra"]["evaluation_completed_step"] == 500
    assert "evaluation_completed_step" not in metadata


def test_warm_start_tokenizer_preserves_existing_ids():
    source = ByteBPETokenizer(
        merges=[(ord("a"), ord("b"))],
        special_tokens=("<|existing|>",),
    )
    checkpoint = {
        "extra": {"tokenizer": source.to_dict()}
    }

    restored, extended = tokenizer_for_warm_start(
        checkpoint,
        {
            "type": "byte_bpe",
            "vocab_size": source.base_vocab_size,
            "special_tokens": (
                "<|existing|>",
                "<|new|>",
            ),
        },
    )

    assert restored.special_token_ids["<|existing|>"] == 257
    assert extended.special_token_ids["<|existing|>"] == 257
    assert extended.special_token_ids["<|new|>"] == 258


def test_warm_start_tokenizer_rejects_special_token_reorder():
    source = ByteBPETokenizer(
        merges=[],
        special_tokens=("<|first|>", "<|second|>"),
    )
    checkpoint = {
        "extra": {"tokenizer": source.to_dict()}
    }

    with pytest.raises(ValueError, match="preserve existing"):
        tokenizer_for_warm_start(
            checkpoint,
            {
                "type": "byte_bpe",
                "vocab_size": 256,
                "special_tokens": (
                    "<|second|>",
                    "<|first|>",
                ),
            },
        )


def test_foundation_warm_start_rejects_character_special_tokens():
    source = ByteBPETokenizer(
        merges=[],
        special_tokens=("<|character|>", "<|assistant|>"),
    )
    checkpoint = {
        "extra": {"tokenizer": source.to_dict()}
    }

    with pytest.raises(ValueError, match="preserve existing"):
        tokenizer_for_warm_start(
            checkpoint,
            {
                "type": "byte_bpe",
                "vocab_size": 256,
                "special_tokens": [],
            },
        )
