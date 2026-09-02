from copy import deepcopy

from scripts.phase34e_begin_calibration_decision import (
    begin_calibration_decision,
    geometry_signals,
    select_begin_bias,
)


SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")


def _calibration_metrics(
    precision=1.0,
    recall=1.0,
    inside_as_begin=0.01,
    outside_as_begin=0.001,
    type_accuracy=1.0,
):
    return {
        "exact_span_precision": precision,
        "exact_span_recall": recall,
        "gold_inside_as_begin_rate": inside_as_begin,
        "gold_outside_as_begin_rate": outside_as_begin,
        "positive_type_accuracy": type_accuracy,
    }


def _policy(
    begin_as_inside=0.20,
    precision=0.40,
    recall=0.50,
    end_spill=0.001,
    type_accuracy=1.0,
):
    return {
        "gold_begin_as_inside_rate": begin_as_inside,
        "exact_span_precision": precision,
        "exact_span_recall": recall,
        "end_spill_rate": end_spill,
        "positive_type_accuracy": type_accuracy,
    }


def _summary(calibrated=None):
    baseline = _policy()
    calibrated = calibrated or _policy(
        begin_as_inside=0.01, precision=0.41, recall=0.55
    )
    return {
        "checkpoint_boundary_objective_version": 1,
        "checkpoint_boundary_loss_weight": 1.0,
        "calibration": {"selected_bias": 0.25},
        "datasets": {
            dataset: {
                "splits": {
                    split: {
                        "policies": {
                            "baseline": deepcopy(baseline),
                            "calibrated": deepcopy(calibrated),
                        },
                        "per_skill": {
                            skill: {
                                "policies": {
                                    "baseline": deepcopy(baseline),
                                    "calibrated": deepcopy(calibrated),
                                }
                            }
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
        "target_geometry": {
            "dimensions": {
                "token_alignment": {},
                "token_byte_offset": {},
                "token_width": {},
            }
        },
    }


def test_calibration_selects_largest_train_val_safe_bias():
    metrics = {
        "0.00": _calibration_metrics(),
        "0.25": _calibration_metrics(
            precision=0.997,
            recall=0.998,
            inside_as_begin=0.014,
            outside_as_begin=0.0014,
            type_accuracy=0.995,
        ),
        "0.50": _calibration_metrics(inside_as_begin=0.02),
    }

    selection = select_begin_bias(metrics)

    assert selection["selected_bias"] == 0.25
    assert selection["evaluations"]["0.25"]["safe"] is True
    assert selection["evaluations"]["0.50"]["safe"] is False


def test_calibration_rejects_bias_unsafe_in_one_panel_cell():
    metrics = {
        "0.00": _calibration_metrics(),
        "0.25": _calibration_metrics(),
    }
    cell = deepcopy(metrics)
    cell["0.25"] = _calibration_metrics(precision=0.90)

    selection = select_begin_bias(
        metrics, {"phase31_regression/val/scene_route": cell}
    )

    assert selection["selected_bias"] == 0.0
    assert selection["evaluations"]["0.25"]["safe"] is False
    assert "phase31_regression/val/scene_route" in selection[
        "evaluations"
    ]["0.25"]["failures"][0]


def test_calibration_decision_accepts_safe_held_out_improvement():
    decision = begin_calibration_decision(_summary())

    assert decision["branch"] == "global_begin_bias_indicated"
    assert decision["calibration_passed"] is True
    assert decision["runtime_change_applied"] is False
    assert decision["training_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_geometry_signal_requires_capture_rate_ratio_and_support():
    summary = _summary(_policy(begin_as_inside=0.10))
    summary["target_geometry"]["dimensions"]["token_alignment"] = {
        "inside_token": {
            "gold_begin_count": 100,
            "baseline_begin_as_inside_count": 20,
        },
        "at_token_start": {
            "gold_begin_count": 100,
            "baseline_begin_as_inside_count": 0,
        },
    }

    signals = geometry_signals(summary)

    assert len(signals) == 1
    assert signals[0]["dimension"] == "token_alignment"
    assert signals[0]["bucket"] == "inside_token"
    assert signals[0]["error_capture"] == 1.0


def test_geometry_signal_requires_a_supported_comparison_bucket():
    summary = _summary(_policy(begin_as_inside=0.10))
    summary["target_geometry"]["dimensions"]["token_alignment"] = {
        "inside_token": {
            "gold_begin_count": 200,
            "baseline_begin_as_inside_count": 20,
        }
    }

    assert geometry_signals(summary) == []


def test_decision_prefers_bpe_geometry_after_calibration_fails():
    summary = _summary(_policy(begin_as_inside=0.10))
    summary["target_geometry"]["dimensions"]["token_alignment"] = {
        "inside_token": {
            "gold_begin_count": 100,
            "baseline_begin_as_inside_count": 20,
        },
        "at_token_start": {
            "gold_begin_count": 100,
            "baseline_begin_as_inside_count": 0,
        },
    }

    decision = begin_calibration_decision(summary)

    assert decision["branch"] == "bpe_geometry_representation_indicated"
    assert decision["geometry_signals"]


def test_decision_selects_factorized_head_without_safe_bias_or_geometry():
    decision = begin_calibration_decision(
        _summary(_policy(begin_as_inside=0.10))
    )

    assert decision["branch"] == "factorized_boundary_type_head_indicated"
    assert decision["calibration_passed"] is False
    assert decision["geometry_signals"] == []


def test_decision_rejects_wrong_phase_checkpoint():
    summary = _summary()
    summary["checkpoint_boundary_objective_version"] = None

    decision = begin_calibration_decision(summary)

    assert decision["branch"] == "invalid_begin_calibration_ablation"
    assert decision["provenance_failures"]


def test_hidden_per_skill_failure_blocks_global_calibration_branch():
    summary = _summary()
    hidden = summary["datasets"]["phase31_regression"]["splits"][
        "lexical"
    ]["per_skill"]["scene_route"]["policies"]["calibrated"]
    hidden["gold_begin_as_inside_rate"] = 0.25

    decision = begin_calibration_decision(summary)

    assert decision["calibration_passed"] is False
    assert any(
        "phase31_regression/lexical/scene_route" in failure
        for failure in decision["target_failures"]
    )
