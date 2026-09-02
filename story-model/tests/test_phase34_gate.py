from copy import deepcopy

from scripts.gate_explicit_offset_candidate_proposer import gate_failures


SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


def _phase33_metrics(skill: str, widths: tuple[str, ...]) -> dict:
    base = {
        "end_to_end_resolve_accuracy": 1.0,
        "counterfactual_pair_resolve_accuracy": 1.0,
        "real_candidate_top1_accuracy": 1.0,
        "clarify_accuracy": 1.0,
        "clarify_sentinel_accuracy": 1.0,
        "generate_accuracy": 1.0,
        "mode_accuracy": 1.0,
        "value_missing_rate": 0.0,
        "wrong_alternative_rate": 0.0,
        "no_support_false_positive_rate": 0.0,
        "multi_candidate_action_examples": 0,
        "multi_candidate_action_accuracy": 1.0,
    }
    return {
        **base,
        "per_skill": {skill: dict(base)},
        "per_candidate_width": {width: dict(base) for width in widths},
        "per_skill_candidate_width": {
            skill: {width: dict(base) for width in widths}
        },
    }


def _phase33_summary() -> dict:
    return {
        "phase31_regression": {
            "splits": {
                split: _phase33_metrics("scene_route", ("2",))
                for split in SPLITS
            }
        },
        "phase32": {
            "splits": {
                split: _phase33_metrics(
                    "multi_turn_memory", ("2", "3", "4")
                )
                for split in SPLITS
            }
        },
    }


def _phase34_metrics(skill: str) -> dict:
    proposer = {
        "exact_span_precision": 1.0,
        "exact_span_recall": 1.0,
        "answer_candidate_recall": 1.0,
        "boundary_type_accuracy": 1.0,
        "offset_validity_rate": 1.0,
        "proposal_overflow_rate": 0.0,
    }
    base = {
        "candidate_inventory_recall": 1.0,
        "end_to_end_resolve_accuracy": 1.0,
        "counterfactual_pair_resolve_accuracy": 1.0,
        "real_candidate_top1_accuracy": 1.0,
        "clarify_accuracy": 1.0,
        "clarify_sentinel_accuracy": 1.0,
        "generate_accuracy": 1.0,
        "mode_accuracy": 1.0,
        "value_missing_rate": 0.0,
        "wrong_alternative_rate": 0.0,
        "no_support_false_positive_rate": 0.0,
        "proposer": proposer,
    }
    return {**base, "per_skill": {skill: deepcopy(base)}}


def _phase34_summary() -> dict:
    return {
        "checkpoint_eligible": True,
        "excluded_cases": ["wrong_type"],
        "phase31_regression": {
            "excluded_rows": 10,
            "splits": {
                split: _phase34_metrics("scene_route") for split in SPLITS
            },
        },
        "phase32": {
            "excluded_rows": 10,
            "splits": {
                split: _phase34_metrics("multi_turn_memory")
                for split in SPLITS
            },
        },
    }


def test_phase34_gate_accepts_proposal_end_to_end_and_oracle_passes():
    assert gate_failures(_phase34_summary(), _phase33_summary()) == ()


def test_phase34_gate_rejects_hidden_lexical_proposal_regression():
    phase34 = _phase34_summary()
    phase34["phase32"]["splits"]["lexical"]["proposer"][
        "answer_candidate_recall"
    ] = 0.99

    failures = gate_failures(phase34, _phase33_summary())

    assert any(
        "phase32/lexical" in failure
        and "answer_candidate_recall" in failure
        for failure in failures
    )


def test_phase34_gate_rejects_oracle_candidate_regression():
    phase33 = _phase33_summary()
    phase33["phase31_regression"]["splits"]["val"][
        "end_to_end_resolve_accuracy"
    ] = 0.0

    failures = gate_failures(_phase34_summary(), phase33)

    assert any("Phase 33c oracle regression" in failure for failure in failures)
