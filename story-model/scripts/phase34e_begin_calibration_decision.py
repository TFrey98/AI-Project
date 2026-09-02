"""Dependency-free selection and decisions for Phase 34e."""

from __future__ import annotations


CALIBRATION_PRECISION_LOSS = 0.005
CALIBRATION_RECALL_LOSS = 0.005
CALIBRATION_INSIDE_AS_BEGIN_INCREASE = 0.005
CALIBRATION_OUTSIDE_AS_BEGIN_INCREASE = 0.0005
MAX_TYPE_ACCURACY_LOSS = 0.01
TARGET_BEGIN_AS_INSIDE_CEILING = 0.02
MAX_EXACT_METRIC_LOSS = 0.01
MAX_END_SPILL_RATE = 0.01
GEOMETRY_CAPTURE_FLOOR = 0.50
GEOMETRY_RATE_RATIO_FLOOR = 2.0
GEOMETRY_MINIMUM_BUCKET_FRACTION = 0.10
GEOMETRY_MINIMUM_BUCKET_COUNT = 20
BPE_GEOMETRY_DIMENSIONS = (
    "token_alignment",
    "token_byte_offset",
    "token_width",
)


def select_begin_bias(
    calibration_metrics: dict,
    calibration_cells: dict | None = None,
) -> dict:
    """Select the largest bias safe in every train/val panel cell."""

    by_bias = {
        float(key): value for key, value in calibration_metrics.items()
    }
    if 0.0 not in by_bias:
        raise ValueError("calibration metrics must include zero bias")
    cells = {"overall": by_bias}
    for label, metrics in (calibration_cells or {}).items():
        cell_by_bias = {float(key): value for key, value in metrics.items()}
        if set(cell_by_bias) != set(by_bias):
            raise ValueError(
                f"calibration cell {label} does not contain the bias grid"
            )
        cells[label] = cell_by_bias
    evaluations = {}
    safe_biases = []
    for bias in sorted(by_bias):
        failures = []
        for label, cell_by_bias in cells.items():
            baseline = cell_by_bias[0.0]
            metrics = cell_by_bias[bias]
            checks = (
                (
                    "exact_span_precision",
                    float(metrics["exact_span_precision"]),
                    float(baseline["exact_span_precision"])
                    - CALIBRATION_PRECISION_LOSS,
                    "minimum",
                ),
                (
                    "exact_span_recall",
                    float(metrics["exact_span_recall"]),
                    float(baseline["exact_span_recall"])
                    - CALIBRATION_RECALL_LOSS,
                    "minimum",
                ),
                (
                    "gold_inside_as_begin_rate",
                    float(metrics["gold_inside_as_begin_rate"]),
                    float(baseline["gold_inside_as_begin_rate"])
                    + CALIBRATION_INSIDE_AS_BEGIN_INCREASE,
                    "maximum",
                ),
                (
                    "gold_outside_as_begin_rate",
                    float(metrics["gold_outside_as_begin_rate"]),
                    float(baseline["gold_outside_as_begin_rate"])
                    + CALIBRATION_OUTSIDE_AS_BEGIN_INCREASE,
                    "maximum",
                ),
                (
                    "positive_type_accuracy",
                    float(metrics["positive_type_accuracy"]),
                    float(baseline["positive_type_accuracy"])
                    - MAX_TYPE_ACCURACY_LOSS,
                    "minimum",
                ),
            )
            for name, actual, threshold, direction in checks:
                failed = (
                    actual < threshold
                    if direction == "minimum"
                    else actual > threshold
                )
                if failed:
                    failures.append(
                        f"{label} {name} {actual:.6f} violates "
                        f"{direction} {threshold:.6f}"
                    )
        key = f"{bias:.2f}"
        evaluations[key] = {"safe": not failures, "failures": failures}
        if not failures:
            safe_biases.append(bias)
    if not safe_biases:
        raise RuntimeError("zero bias unexpectedly failed its own safety gate")
    return {
        "selected_bias": max(safe_biases),
        "selection_rule": "largest_safe_bias_in_every_train_val_cell",
        "safety_cells": sorted(cells),
        "evaluations": evaluations,
        "thresholds": {
            "maximum_precision_loss": CALIBRATION_PRECISION_LOSS,
            "maximum_recall_loss": CALIBRATION_RECALL_LOSS,
            "maximum_inside_as_begin_increase": (
                CALIBRATION_INSIDE_AS_BEGIN_INCREASE
            ),
            "maximum_outside_as_begin_increase": (
                CALIBRATION_OUTSIDE_AS_BEGIN_INCREASE
            ),
            "maximum_type_accuracy_loss": MAX_TYPE_ACCURACY_LOSS,
        },
    }


def _policy_cells(summary: dict):
    for dataset_name, dataset in summary["datasets"].items():
        for split, split_summary in dataset["splits"].items():
            yield (
                f"{dataset_name}/{split}",
                dataset_name,
                split,
                split_summary["policies"],
            )
            for skill, skill_summary in split_summary["per_skill"].items():
                yield (
                    f"{dataset_name}/{split}/{skill}",
                    dataset_name,
                    split,
                    skill_summary["policies"],
                )


def geometry_signals(summary: dict) -> list[dict]:
    """Find concentrated Phase 31 scene-route B-to-I geometry buckets."""

    dimensions = summary["target_geometry"]["dimensions"]
    signals = []
    for dimension in BPE_GEOMETRY_DIMENSIONS:
        buckets = dimensions.get(dimension, {})
        total_count = sum(
            int(bucket["gold_begin_count"]) for bucket in buckets.values()
        )
        total_errors = sum(
            int(bucket["baseline_begin_as_inside_count"])
            for bucket in buckets.values()
        )
        for bucket_name, bucket in buckets.items():
            count = int(bucket["gold_begin_count"])
            errors = int(bucket["baseline_begin_as_inside_count"])
            minimum_count = max(
                GEOMETRY_MINIMUM_BUCKET_COUNT,
                int(total_count * GEOMETRY_MINIMUM_BUCKET_FRACTION),
            )
            outside_count = total_count - count
            outside_errors = total_errors - errors
            if (
                count < minimum_count
                or outside_count < minimum_count
                or total_errors == 0
            ):
                continue
            rate = errors / count if count else 0.0
            outside_rate = (
                outside_errors / outside_count if outside_count else 0.0
            )
            if outside_rate == 0.0:
                rate_ratio = 999.0 if rate > 0.0 else 1.0
            else:
                rate_ratio = rate / outside_rate
            capture = errors / total_errors
            if (
                capture >= GEOMETRY_CAPTURE_FLOOR
                and rate_ratio >= GEOMETRY_RATE_RATIO_FLOOR
            ):
                signals.append(
                    {
                        "dimension": dimension,
                        "bucket": bucket_name,
                        "gold_begin_count": count,
                        "error_count": errors,
                        "error_capture": capture,
                        "error_rate": rate,
                        "outside_error_rate": outside_rate,
                        "rate_ratio": rate_ratio,
                    }
                )
    return sorted(
        signals,
        key=lambda signal: (
            -signal["error_capture"],
            -signal["rate_ratio"],
            signal["dimension"],
            signal["bucket"],
        ),
    )


def begin_calibration_decision(summary: dict) -> dict:
    """Choose calibration, BPE geometry, or factorized boundary/type."""

    provenance_failures = []
    if summary.get("checkpoint_boundary_objective_version") != 1:
        provenance_failures.append(
            "checkpoint does not declare Phase 34d boundary objective version 1"
        )
    weight = summary.get("checkpoint_boundary_loss_weight")
    if weight is None or abs(float(weight) - 1.0) > 1.0e-9:
        provenance_failures.append(
            "checkpoint boundary loss weight is not 1.0"
        )
    calibration = summary["calibration"]
    selected_bias = float(calibration["selected_bias"])

    target_failures = []
    safety_failures = []
    comparisons = {}
    for label, dataset, split, policies in _policy_cells(summary):
        baseline = policies["baseline"]
        calibrated = policies["calibrated"]
        comparisons[label] = {
            "baseline_begin_as_inside_rate": baseline[
                "gold_begin_as_inside_rate"
            ],
            "calibrated_begin_as_inside_rate": calibrated[
                "gold_begin_as_inside_rate"
            ],
            "baseline_exact_span_precision": baseline[
                "exact_span_precision"
            ],
            "calibrated_exact_span_precision": calibrated[
                "exact_span_precision"
            ],
            "baseline_exact_span_recall": baseline["exact_span_recall"],
            "calibrated_exact_span_recall": calibrated["exact_span_recall"],
            "baseline_end_spill_rate": baseline["end_spill_rate"],
            "calibrated_end_spill_rate": calibrated["end_spill_rate"],
            "baseline_positive_type_accuracy": baseline[
                "positive_type_accuracy"
            ],
            "calibrated_positive_type_accuracy": calibrated[
                "positive_type_accuracy"
            ],
        }
        target = (
            dataset == "phase31_regression"
            and split in {"lexical", "transfer"}
        )
        if target and float(
            calibrated["gold_begin_as_inside_rate"]
        ) > TARGET_BEGIN_AS_INSIDE_CEILING:
            target_failures.append(
                f"{label} calibrated B->I "
                f"{calibrated['gold_begin_as_inside_rate']:.3f} is above "
                f"{TARGET_BEGIN_AS_INSIDE_CEILING:.3f}"
            )
        for metric in ("exact_span_precision", "exact_span_recall"):
            loss = float(baseline[metric]) - float(calibrated[metric])
            if loss > MAX_EXACT_METRIC_LOSS:
                safety_failures.append(
                    f"{label} {metric} lost {loss:.3f}, above "
                    f"{MAX_EXACT_METRIC_LOSS:.3f}"
                )
        type_loss = float(baseline["positive_type_accuracy"]) - float(
            calibrated["positive_type_accuracy"]
        )
        if type_loss > MAX_TYPE_ACCURACY_LOSS:
            safety_failures.append(
                f"{label} positive type accuracy lost {type_loss:.3f}, "
                f"above {MAX_TYPE_ACCURACY_LOSS:.3f}"
            )
        if float(calibrated["end_spill_rate"]) > MAX_END_SPILL_RATE:
            safety_failures.append(
                f"{label} end spill {calibrated['end_spill_rate']:.3f} "
                f"is above {MAX_END_SPILL_RATE:.3f}"
            )

    signals = geometry_signals(summary)
    calibration_passed = (
        selected_bias > 0.0
        and not target_failures
        and not safety_failures
    )
    if provenance_failures:
        branch = "invalid_begin_calibration_ablation"
        next_action = (
            "Stop and correct the Phase 34d checkpoint provenance; do not "
            "interpret this ablation."
        )
    elif calibration_passed:
        branch = "global_begin_bias_indicated"
        next_action = (
            "Test the frozen train/val-selected B-logit bias as the sole "
            "runtime change. Do not retrain or promote this checkpoint yet."
        )
    elif signals:
        branch = "bpe_geometry_representation_indicated"
        next_action = (
            "A BPE geometry bucket captures the held-out start failures. The "
            "next controlled variable should expose cross-token byte-boundary "
            "features; do not change the decoder or class weights."
        )
    else:
        branch = "factorized_boundary_type_head_indicated"
        next_action = (
            "Calibration and BPE geometry do not explain the failure. Replace "
            "the joint typed-BIO head with separate shared boundary and type "
            "heads while retaining Phase 34d end supervision."
        )
    return {
        "branch": branch,
        "selected_begin_logit_bias": selected_bias,
        "calibration_passed": calibration_passed,
        "target_failures": target_failures,
        "safety_failures": safety_failures,
        "provenance_failures": provenance_failures,
        "geometry_signals": signals,
        "comparisons": comparisons,
        "runtime_change_applied": False,
        "training_authorized": False,
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
