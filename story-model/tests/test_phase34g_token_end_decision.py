from copy import deepcopy

from scripts.phase34g_token_end_decision import token_end_geometry_decision


SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


def _metrics(
    start=0.01,
    end=0.001,
    type_accuracy=1.0,
    outside_as_inside=0.001,
):
    return {
        "gold_begin_as_inside_rate": start,
        "gold_outside_as_inside_rate": outside_as_inside,
        "end_spill_rate": end,
        "positive_type_accuracy": type_accuracy,
    }


def _summary(metrics, *, geometry=False, eligible=True):
    summary = {
        "tag_confusion_audit_version": 1,
        "checkpoint_boundary_objective_version": 1,
        "checkpoint_boundary_loss_weight": 1.0,
        "checkpoint_eligible": eligible,
        "checkpoint_token_width_geometry_version": None,
        "checkpoint_token_end_geometry_version": 1 if geometry else None,
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
    if geometry:
        summary["target_begin_geometry"] = {
            "dimensions": {
                name: {
                    bucket: {
                        "gold_begin_count": 100,
                        "begin_as_inside_count": 1,
                        "begin_as_inside_rate": 0.01,
                    }
                }
                for name, bucket in (
                    ("token_width", "2"),
                    ("token_end", "True"),
                )
            }
        }
    return summary


def test_token_end_geometry_passes_registered_target_and_safety_gates():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(_metrics(), geometry=True)

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "token_end_geometry_diagnostic_pass"
    assert decision["target_gate_passed"] is True
    assert decision["safety_gate_passed"] is True
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False


def test_token_end_geometry_requires_training_panel_eligibility():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(_metrics(), geometry=True, eligible=False)

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "token_end_fixed_span_gate_ineligible"
    assert decision["full_phase34_evaluation_authorized"] is False


def test_token_end_bucket_can_fail_when_aggregate_cells_pass():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(_metrics(), geometry=True)
    token_end = candidate["target_begin_geometry"]["dimensions"][
        "token_end"
    ]["True"]
    token_end["begin_as_inside_count"] = 10
    token_end["begin_as_inside_rate"] = 0.10

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "token_end_geometry_insufficient"
    assert any("token_end=True" in item for item in decision["target_failures"])


def test_hidden_scene_route_failure_blocks_token_end_pass():
    baseline = _summary(_metrics(start=0.25))
    candidate = _summary(_metrics(), geometry=True)
    hidden = candidate["datasets"]["phase31_regression"]["splits"][
        "transfer"
    ]["per_skill"]["scene_route"]
    hidden["gold_begin_as_inside_rate"] = 0.23

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "token_end_geometry_insufficient"
    assert any(
        "phase31_regression/transfer/scene_route" in item
        for item in decision["target_failures"]
    )


def test_token_end_geometry_rejects_boundary_or_type_regression():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(
        _metrics(
            end=0.02,
            type_accuracy=0.98,
            outside_as_inside=0.01,
        ),
        geometry=True,
    )

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "token_end_geometry_rejected"
    assert decision["safety_gate_passed"] is False
    assert any("end spill" in item for item in decision["safety_failures"])
    assert any("O->I" in item for item in decision["safety_failures"])
    assert any(
        "positive type accuracy" in item
        for item in decision["safety_failures"]
    )


def test_token_end_geometry_rejects_stacked_width_feature():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(_metrics(), geometry=True)
    candidate["checkpoint_token_width_geometry_version"] = 1

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "invalid_token_end_geometry_comparison"
    assert any("replace" in item for item in decision["provenance_failures"])


def test_token_end_geometry_requires_both_registered_buckets():
    baseline = _summary(_metrics(start=0.20))
    candidate = _summary(_metrics(), geometry=True)
    del candidate["target_begin_geometry"]["dimensions"]["token_end"]

    decision = token_end_geometry_decision(baseline, candidate)

    assert decision["branch"] == "invalid_token_end_geometry_comparison"
    assert decision["target_token_end_begin_as_inside_rate"] is None
