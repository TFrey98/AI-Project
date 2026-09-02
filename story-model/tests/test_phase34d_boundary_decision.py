from copy import deepcopy

from scripts.phase34d_boundary_decision import boundary_supervision_decision


SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


def _metrics(start=0.20, end=0.50, type_accuracy=1.0):
    return {
        "gold_begin_as_inside_rate": start,
        "end_spill_rate": end,
        "positive_type_accuracy": type_accuracy,
    }


def _summary(metrics, boundary=False):
    summary = {
        "tag_confusion_audit_version": 1,
        "datasets": {
            dataset: {
                "splits": {
                    split: {
                        "metrics": deepcopy(metrics),
                        "per_skill": {skill: deepcopy(metrics)},
                    }
                    for split in SPLITS
                }
            }
            for dataset, skill in (
                ("phase31_regression", "scene_route"),
                ("phase32", "multi_turn_memory"),
            )
        },
    }
    if boundary:
        summary["checkpoint_boundary_objective_version"] = 1
        summary["checkpoint_boundary_loss_weight"] = 1.0
        summary["checkpoint_eligible"] = True
    return summary


def test_boundary_decision_passes_only_after_start_end_and_type_gates():
    baseline = _summary(_metrics())
    candidate = _summary(
        _metrics(start=0.02, end=0.25, type_accuracy=0.99),
        boundary=True,
    )

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "boundary_diagnostic_pass"
    assert decision["start_gate_passed"] is True
    assert decision["end_gate_passed"] is True
    assert decision["type_safety_passed"] is True
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False


def test_boundary_decision_rejects_wrong_checkpoint_provenance():
    baseline = _summary(_metrics())
    candidate = _summary(_metrics(start=0.01, end=0.20))

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "invalid_boundary_supervision_comparison"
    assert len(decision["provenance_failures"]) == 2


def test_boundary_decision_stops_when_start_is_fixed_but_end_is_not():
    baseline = _summary(_metrics())
    candidate = _summary(_metrics(start=0.01, end=0.30), boundary=True)

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "start_fixed_end_insufficient"
    assert decision["start_gate_passed"] is True
    assert decision["end_gate_passed"] is False


def test_boundary_decision_stops_when_end_is_fixed_but_start_is_not():
    baseline = _summary(_metrics())
    candidate = _summary(_metrics(start=0.03, end=0.20), boundary=True)

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "end_fixed_start_insufficient"
    assert decision["start_gate_passed"] is False
    assert decision["end_gate_passed"] is True


def test_boundary_decision_rejects_positive_type_regression():
    baseline = _summary(_metrics())
    candidate = _summary(
        _metrics(start=0.01, end=0.20, type_accuracy=0.98),
        boundary=True,
    )

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "boundary_supervision_rejected"
    assert decision["type_safety_passed"] is False


def test_boundary_decision_does_not_gate_an_ineligible_checkpoint():
    baseline = _summary(_metrics())
    candidate = _summary(
        _metrics(start=0.01, end=0.20), boundary=True
    )
    candidate["checkpoint_eligible"] = False

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["branch"] == "boundary_tags_fixed_span_gate_ineligible"
    assert decision["full_phase34_evaluation_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_boundary_decision_checks_hidden_per_skill_cells():
    baseline = _summary(_metrics())
    candidate = _summary(_metrics(start=0.01, end=0.20), boundary=True)
    hidden = candidate["datasets"]["phase31_regression"]["splits"][
        "transfer"
    ]["per_skill"]["scene_route"]
    hidden["gold_begin_as_inside_rate"] = 0.25

    decision = boundary_supervision_decision(baseline, candidate)

    assert decision["start_gate_passed"] is False
    assert any(
        "phase31_regression/transfer/scene_route" in failure
        for failure in decision["target_start_failures"]
    )
