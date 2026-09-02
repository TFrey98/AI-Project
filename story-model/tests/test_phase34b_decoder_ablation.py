from copy import deepcopy
from dataclasses import dataclass

from scripts.phase34b_decoder_decision import (
    decoder_ablation_decision,
    false_positive_boundary_displacements,
)


EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


@dataclass(frozen=True)
class _Span:
    byte_start: int
    byte_end: int
    text: str
    value_type: str


def _metrics(precision=1.0, recall=1.0, answer=1.0):
    skill = {
        "exact_span_precision": precision,
        "exact_span_recall": recall,
        "answer_candidate_recall": answer,
        "boundary_type_accuracy": 1.0,
        "offset_validity_rate": 1.0,
        "proposal_overflow_rate": 0.0,
        "per_skill": {},
    }
    return {**skill, "per_skill": {"multi_turn_memory": skill}}


def _summary(strict_metrics, permissive_metrics):
    split = {
        "policies": {
            "permissive": deepcopy(permissive_metrics),
            "strict": deepcopy(strict_metrics),
        }
    }
    return {
        "overall": {
            "permissive": deepcopy(permissive_metrics),
            "strict": deepcopy(strict_metrics),
        },
        "datasets": {
            dataset: {
                "splits": {
                    name: deepcopy(split) for name in EXPANDED_SPLITS
                }
            }
            for dataset in ("phase31_regression", "phase32")
        },
    }


def test_false_positive_displacement_uses_nearest_same_type_gold():
    gold = (_Span(10, 20, "abcdefghij", "container"),)
    predicted = (
        _Span(9, 20, "zabcdefghij", "container"),
        _Span(30, 35, "other", "person"),
    )

    displacements = false_positive_boundary_displacements(gold, predicted)

    assert displacements == (
        "start=-1,end=+0",
        "no_same_type_gold",
    )


def test_decision_authorizes_decoder_only_rescore_only_at_full_floor():
    summary = _summary(_metrics(), _metrics(0.35, 0.54, 0.90))

    decision = decoder_ablation_decision(summary)

    assert decision["branch"] == "decoder_only_pass"
    assert decision["strict_gate_failures"] == []


def test_decision_marks_large_safe_precision_gain_as_helpful_not_passed():
    summary = _summary(
        _metrics(0.60, 0.53, 0.90),
        _metrics(0.35, 0.54, 0.90),
    )

    decision = decoder_ablation_decision(summary)

    assert decision["branch"] == "strict_helpful_but_insufficient"
    assert decision["strict_gate_failures"]


def test_decision_rejects_strict_decoder_when_recall_falls_too_far():
    summary = _summary(
        _metrics(0.70, 0.40, 0.85),
        _metrics(0.35, 0.54, 0.90),
    )

    decision = decoder_ablation_decision(summary)

    assert decision["branch"] == "strict_decoder_rejected"
