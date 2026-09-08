from pathlib import Path

import yaml

from scripts.train_explicit_offset_candidate_proposer import (
    _combined_selection,
    _counterbalance_slots,
    _load_config,
)


ROOT = Path(__file__).parents[1]


def _metrics(value=1.0, loss=0.1):
    return {
        "exact_span_precision": value,
        "exact_span_recall": value,
        "exact_span_f1": value,
        "answer_candidate_recall": value,
        "boundary_type_accuracy": value,
        "offset_validity_rate": 1.0,
        "proposal_overflow_rate": 0.0,
        "loss": loss,
    }


def test_phase35_configs_change_data_composition_not_architecture():
    baseline = yaml.safe_load(
        (ROOT / "configs/explicit_offset_boundary_supervision_pilot.yaml")
        .read_text(encoding="utf-8")
    )
    candidate = _load_config(
        ROOT / "configs/explicit_offset_boundary_counterbalance_pilot.yaml"
    )

    assert candidate["model"] == baseline["model"]
    assert candidate["tokenizer"] == baseline["tokenizer"]
    assert candidate["train"]["boundary_loss_weight"] == 1.0
    assert candidate["train"]["boundary_counterbalance_version"] == 2
    for forbidden in (
        "token_width_geometry_version",
        "token_end_geometry_version",
        "factorized_boundary_type_version",
    ):
        assert forbidden not in candidate["train"]
    assert candidate["train"]["max_steps"] == baseline["train"]["max_steps"]
    assert candidate["train"]["gradient_accumulation_steps"] == 4
    assert candidate["data"]["batch_size"] == baseline["data"]["batch_size"]


def test_counterbalance_schedule_preserves_one_of_four_microbatches():
    slots = [_counterbalance_slots(step, 4, 1) for step in range(1, 5)]

    assert slots == [(0,), (1,), (2,), (3,)]


def test_checkpoint_selection_requires_retained_and_counterbalance_panels():
    eligible, _ = _combined_selection(
        _metrics(), _metrics(), 0.98, 0.98, 0.995, 0.99
    )
    failed, _ = _combined_selection(
        _metrics(), _metrics(value=0.90), 0.98, 0.98, 0.995, 0.99
    )

    assert eligible is True
    assert failed is False
