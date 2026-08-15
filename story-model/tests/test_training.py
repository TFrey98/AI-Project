import math

import pytest

from story_model.train import learning_rate_for_step


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