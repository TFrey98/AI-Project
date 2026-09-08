from copy import deepcopy

from scripts.audit_boundary_counterbalance import summarize_focus_predictions
from scripts.phase35_boundary_counterbalance_decision import (
    boundary_counterbalance_decision,
)


def _metrics():
    return {
        "gold_begin_as_inside_rate": 0.0,
        "gold_outside_as_inside_rate": 0.0,
        "end_spill_rate": 0.0,
        "positive_type_accuracy": 1.0,
        "exact_span_precision": 1.0,
        "exact_span_recall": 1.0,
        "answer_candidate_recall": 1.0,
        "boundary_type_accuracy": 1.0,
        "offset_validity_rate": 1.0,
        "proposal_overflow_rate": 0.0,
    }


def _tag_audit(candidate=False):
    value = {
        "tag_confusion_audit_version": 1,
        "datasets": {
            dataset: {
                "splits": {
                    split: {
                        "metrics": _metrics(),
                        "per_skill": {"scene_route": _metrics()},
                    }
                    for split in (
                        "train",
                        "val",
                        "lexical",
                        "paraphrase",
                        "transfer",
                    )
                }
            }
            for dataset in ("phase31_regression", "phase32")
        },
    }
    if candidate:
        value.update(
            {
                "checkpoint_sha256": "candidate-sha",
                "checkpoint_step": 900,
                "checkpoint_eligible": True,
                "checkpoint_tokenizer_sha256": "tokenizer-sha",
                "checkpoint_counterbalance_manifest_sha256": "manifest-sha",
            }
        )
    return value


def _focus_metrics():
    return {
        "gold_begin_count": 25,
        "gold_inside_count": 25,
        "begin_as_inside_rate": 0.0,
        "inside_as_begin_rate": 0.0,
    }


def _counterbalance():
    heldout_ids = [400, 401, 402, 403]
    return {
        "boundary_counterbalance_audit_version": 1,
        "checkpoint": {
            "sha256": "candidate-sha",
            "step": 900,
            "eligible": True,
            "boundary_objective_version": 1,
            "boundary_loss_weight": 1.0,
            "boundary_counterbalance_version": 2,
            "tokenizer_sha256": "tokenizer-sha",
            "manifest_sha256": "manifest-sha",
        },
        "manifest": {
            "phase34k_premise": {
                "branch": "local_token_state_dominant",
                "focus_token_ids": {"or": 270, "ro": 357},
            }
        },
        "focus_token_ids": {
            "train": [300, 301, 302, 303, 304, 305, 306, 307],
            "heldout": heldout_ids,
        },
        "splits": {
            split: {
                "focus_pool": "heldout",
                "metrics": _metrics(),
                "focus": {
                    "overall": _focus_metrics(),
                    "per_token": {
                        str(token_id): _focus_metrics()
                        for token_id in heldout_ids
                    },
                },
            }
            for split in ("lexical", "transfer")
        },
    }


def test_focus_summary_keeps_bidirectional_boundary_errors_separate():
    summary = summarize_focus_predictions(
        (
            {"token_id": 400, "gold": "B", "predicted": "I"},
            {"token_id": 400, "gold": "I", "predicted": "B"},
            {"token_id": 400, "gold": "B", "predicted": "B"},
            {"token_id": 400, "gold": "I", "predicted": "I"},
        )
    )

    assert summary["overall"]["begin_as_inside_rate"] == 0.5
    assert summary["overall"]["inside_as_begin_rate"] == 0.5


def test_phase35_pass_authorizes_only_the_inherited_full_evaluation():
    decision = boundary_counterbalance_decision(
        _tag_audit(), _tag_audit(candidate=True), _counterbalance()
    )

    assert decision["branch"] == "boundary_counterbalance_diagnostic_pass"
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False


def test_refreshed_token_failure_does_not_hide_behind_aggregate_metrics():
    audit = _counterbalance()
    audit["splits"]["transfer"]["focus"]["per_token"]["401"][
        "begin_as_inside_rate"
    ] = 0.04

    decision = boundary_counterbalance_decision(
        _tag_audit(), _tag_audit(candidate=True), audit
    )

    assert decision["branch"] == (
        "boundary_counterbalance_generalization_insufficient"
    )


def test_retained_regression_rejects_counterbalance_before_full_gate():
    candidate = _tag_audit(candidate=True)
    candidate["datasets"]["phase31_regression"]["splits"]["lexical"][
        "per_skill"
    ]["scene_route"]["gold_begin_as_inside_rate"] = 0.03

    decision = boundary_counterbalance_decision(
        _tag_audit(), candidate, _counterbalance()
    )

    assert decision["branch"] == "boundary_counterbalance_rejected"


def test_checkpoint_or_focus_pool_mismatch_invalidates_comparison():
    audit = _counterbalance()
    audit["checkpoint"]["sha256"] = "wrong"
    audit["focus_token_ids"]["heldout"] = [300]

    decision = boundary_counterbalance_decision(
        _tag_audit(), _tag_audit(candidate=True), audit
    )

    assert decision["branch"] == "invalid_boundary_counterbalance_comparison"
    assert decision["invalid_reasons"]


def test_decision_does_not_mutate_inputs():
    baseline = _tag_audit()
    candidate = _tag_audit(candidate=True)
    audit = _counterbalance()
    originals = deepcopy((baseline, candidate, audit))

    boundary_counterbalance_decision(baseline, candidate, audit)

    assert (baseline, candidate, audit) == originals
