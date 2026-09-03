from copy import deepcopy

from scripts.phase34i_factorized_head_decision import factorized_head_decision


SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


def _metrics(start=0.01, outside=0.001, end=0.001, type_accuracy=1.0):
    return {
        "gold_begin_as_inside_rate": start,
        "gold_outside_as_inside_rate": outside,
        "end_spill_rate": end,
        "positive_type_accuracy": type_accuracy,
    }


def _summary(factorized=False, eligible=True):
    metrics = _metrics()
    summary = {
        "tag_confusion_audit_version": 1,
        "checkpoint_eligible": eligible,
        "checkpoint_boundary_objective_version": 1,
        "checkpoint_boundary_loss_weight": 1.0,
        "checkpoint_token_width_geometry_version": None,
        "checkpoint_token_end_geometry_version": None,
        "checkpoint_factorized_boundary_type_version": (
            1 if factorized else None
        ),
        "datasets": {
            dataset: {
                "splits": {
                    split: {
                        "metrics": deepcopy(metrics),
                        "per_skill": {
                            skill: deepcopy(metrics),
                        },
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
    if factorized:
        summary["target_begin_geometry"] = {
            "dimensions": {
                "token_width": {
                    "2": {
                        "gold_begin_count": 100,
                        "begin_as_inside_count": 1,
                        "begin_as_inside_rate": 0.01,
                    }
                },
                "token_id": {
                    "270": {
                        "gold_begin_count": 40,
                        "begin_as_inside_count": 1,
                        "begin_as_inside_rate": 0.025,
                    },
                    "357": {
                        "gold_begin_count": 60,
                        "begin_as_inside_count": 0,
                        "begin_as_inside_rate": 0.0,
                    },
                },
            }
        }
    else:
        for split in ("lexical", "transfer"):
            target = summary["datasets"]["phase31_regression"]["splits"][
                split
            ]
            target["metrics"]["gold_begin_as_inside_rate"] = 0.18
            target["per_skill"]["scene_route"][
                "gold_begin_as_inside_rate"
            ] = 0.22
    return summary


def test_factorized_head_passes_registered_target_and_safety_gates():
    decision = factorized_head_decision(
        _summary(), _summary(factorized=True)
    )

    assert decision["branch"] == "factorized_head_diagnostic_pass"
    assert decision["target_gate_passed"] is True
    assert decision["safety_gate_passed"] is True
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False
    assert decision["diagnostic_token_identities"]["270"] is not None
    assert decision["diagnostic_token_identities_are_gate_inputs"] is False


def test_factorized_head_requires_eligible_training_panel():
    candidate = _summary(factorized=True, eligible=False)

    decision = factorized_head_decision(_summary(), candidate)

    assert decision["branch"] == "factorized_head_span_fixed_but_ineligible"
    assert decision["target_gate_passed"] is True
    assert decision["full_phase34_evaluation_authorized"] is False


def test_factorized_head_reports_insufficient_target_improvement():
    candidate = _summary(factorized=True)
    target = candidate["datasets"]["phase31_regression"]["splits"][
        "lexical"
    ]
    target["metrics"]["gold_begin_as_inside_rate"] = 0.15
    target["per_skill"]["scene_route"][
        "gold_begin_as_inside_rate"
    ] = 0.20
    candidate["target_begin_geometry"]["dimensions"]["token_width"]["2"][
        "begin_as_inside_rate"
    ] = 0.18

    decision = factorized_head_decision(_summary(), candidate)

    assert decision["branch"] == "factorized_head_insufficient"
    assert decision["target_gate_passed"] is False


def test_factorized_head_rejects_boundary_or_type_regression():
    candidate = _summary(factorized=True)
    metrics = candidate["datasets"]["phase32"]["splits"]["val"][
        "metrics"
    ]
    metrics["end_spill_rate"] = 0.02
    metrics["positive_type_accuracy"] = 0.98

    decision = factorized_head_decision(_summary(), candidate)

    assert decision["branch"] == "factorized_head_rejected"
    assert decision["safety_failures"]


def test_factorized_head_rejects_wrong_provenance():
    candidate = _summary(factorized=True)
    candidate["checkpoint_factorized_boundary_type_version"] = None
    candidate["checkpoint_token_width_geometry_version"] = 1

    decision = factorized_head_decision(_summary(), candidate)

    assert decision["branch"] == "invalid_factorized_head_comparison"
    assert decision["provenance_failures"]


def test_factorized_head_requires_supported_width_two_bucket():
    candidate = _summary(factorized=True)
    del candidate["target_begin_geometry"]["dimensions"]["token_width"]["2"]

    decision = factorized_head_decision(_summary(), candidate)

    assert decision["branch"] == "invalid_factorized_head_comparison"
