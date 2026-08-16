import math

import pytest
import torch

from story_model.checkpoint import read_checkpoint
from story_model.train import (
    learning_rate_for_step,
    save_best_validation_checkpoint,
    validation_loss_improved,
)


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
