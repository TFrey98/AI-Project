"""Dependency-free decision logic for Phase 34d boundary supervision."""

from __future__ import annotations


EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")
EXPECTED_BOUNDARY_OBJECTIVE_VERSION = 1
EXPECTED_BOUNDARY_LOSS_WEIGHT = 1.0
TARGET_BEGIN_AS_INSIDE_CEILING = 0.02
END_SPILL_REDUCTION_FLOOR = 0.50
MAX_TYPE_ACCURACY_LOSS = 0.01
MAX_BEGIN_AS_INSIDE_INCREASE = 0.02


def _metric_cells(summary: dict):
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary["datasets"][dataset_name]
        for split in EXPANDED_SPLITS:
            split_summary = dataset["splits"][split]
            yield (
                f"{dataset_name}/{split}",
                dataset_name,
                split,
                split_summary["metrics"],
            )
            for skill, metrics in sorted(split_summary["per_skill"].items()):
                yield (
                    f"{dataset_name}/{split}/{skill}",
                    dataset_name,
                    split,
                    metrics,
                )


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else -1.0
    return (baseline - candidate) / baseline


def boundary_supervision_decision(baseline: dict, candidate: dict) -> dict:
    """Evaluate only the Phase 34d diagnostic boundary criteria."""

    provenance_failures = []
    if (
        candidate.get("checkpoint_boundary_objective_version")
        != EXPECTED_BOUNDARY_OBJECTIVE_VERSION
    ):
        provenance_failures.append(
            "candidate checkpoint does not declare boundary objective version 1"
        )
    actual_weight = candidate.get("checkpoint_boundary_loss_weight")
    if actual_weight is None or abs(
        float(actual_weight) - EXPECTED_BOUNDARY_LOSS_WEIGHT
    ) > 1.0e-9:
        provenance_failures.append(
            "candidate checkpoint boundary loss weight is not 1.0"
        )

    baseline_cells = {
        label: (dataset, split, metrics)
        for label, dataset, split, metrics in _metric_cells(baseline)
    }
    candidate_cells = {
        label: (dataset, split, metrics)
        for label, dataset, split, metrics in _metric_cells(candidate)
    }
    missing = sorted(set(baseline_cells) ^ set(candidate_cells))
    if missing:
        provenance_failures.append(
            "baseline and candidate audit cells differ: " + ", ".join(missing)
        )

    target_start_failures = []
    end_spill_failures = []
    type_safety_failures = []
    start_safety_failures = []
    comparisons = {}
    for label in sorted(set(baseline_cells) & set(candidate_cells)):
        dataset, split, baseline_metrics = baseline_cells[label]
        _, _, candidate_metrics = candidate_cells[label]
        baseline_start = float(
            baseline_metrics["gold_begin_as_inside_rate"]
        )
        candidate_start = float(
            candidate_metrics["gold_begin_as_inside_rate"]
        )
        baseline_end = float(baseline_metrics["end_spill_rate"])
        candidate_end = float(candidate_metrics["end_spill_rate"])
        end_reduction = _relative_reduction(baseline_end, candidate_end)
        baseline_type = float(baseline_metrics["positive_type_accuracy"])
        candidate_type = float(candidate_metrics["positive_type_accuracy"])
        comparisons[label] = {
            "baseline_gold_begin_as_inside_rate": baseline_start,
            "candidate_gold_begin_as_inside_rate": candidate_start,
            "gold_begin_as_inside_change": candidate_start - baseline_start,
            "baseline_end_spill_rate": baseline_end,
            "candidate_end_spill_rate": candidate_end,
            "end_spill_relative_reduction": end_reduction,
            "baseline_positive_type_accuracy": baseline_type,
            "candidate_positive_type_accuracy": candidate_type,
            "positive_type_accuracy_change": candidate_type - baseline_type,
        }
        target_cell = (
            dataset == "phase31_regression"
            and split in {"lexical", "transfer"}
        )
        if target_cell and candidate_start > TARGET_BEGIN_AS_INSIDE_CEILING:
            target_start_failures.append(
                f"{label} B->I {candidate_start:.3f} is above "
                f"{TARGET_BEGIN_AS_INSIDE_CEILING:.3f}"
            )
        if end_reduction < END_SPILL_REDUCTION_FLOOR:
            end_spill_failures.append(
                f"{label} end-spill reduction {end_reduction:.3f} is below "
                f"{END_SPILL_REDUCTION_FLOOR:.3f}"
            )
        if candidate_type < baseline_type - MAX_TYPE_ACCURACY_LOSS:
            type_safety_failures.append(
                f"{label} positive type accuracy fell "
                f"{baseline_type - candidate_type:.3f}, above the "
                f"{MAX_TYPE_ACCURACY_LOSS:.3f} tolerance"
            )
        if candidate_start > baseline_start + MAX_BEGIN_AS_INSIDE_INCREASE:
            start_safety_failures.append(
                f"{label} B->I increased "
                f"{candidate_start - baseline_start:.3f}, above the "
                f"{MAX_BEGIN_AS_INSIDE_INCREASE:.3f} tolerance"
            )

    start_pass = not target_start_failures and not start_safety_failures
    end_pass = not end_spill_failures
    type_safe = not type_safety_failures
    checkpoint_eligible = candidate.get("checkpoint_eligible") is True
    if provenance_failures:
        branch = "invalid_boundary_supervision_comparison"
        next_action = (
            "Stop and correct the checkpoint or audit inputs; do not infer a "
            "training result from this comparison."
        )
    elif not type_safe:
        branch = "boundary_supervision_rejected"
        next_action = (
            "Reject this objective because type generalization regressed; "
            "retain the original Phase 34 checkpoint and decoder."
        )
    elif start_pass and end_pass and checkpoint_eligible:
        branch = "boundary_diagnostic_pass"
        next_action = (
            "Run the existing full Phase 34 proposer evaluation and semantic "
            "gate on this checkpoint. Promote it only if those original gates "
            "also pass."
        )
    elif start_pass and end_pass:
        branch = "boundary_tags_fixed_span_gate_ineligible"
        next_action = (
            "The boundary diagnostics pass, but the training panel did not "
            "produce an eligible checkpoint. Do not run the full gate, "
            "promote the checkpoint, or extend training."
        )
    elif start_pass:
        branch = "start_fixed_end_insufficient"
        next_action = (
            "Boundary-start supervision worked, but end spill did not fall "
            "enough. Do not run the full gate or extend training."
        )
    elif end_pass:
        branch = "end_fixed_start_insufficient"
        next_action = (
            "End spill improved, but Phase 31 lexical/transfer B-to-I remains. "
            "Do not run the full gate or extend training."
        )
    else:
        branch = "boundary_supervision_insufficient"
        next_action = (
            "Neither boundary failure cleared its diagnostic gate. Do not run "
            "longer, alter decoding, or promote this checkpoint."
        )
    return {
        "branch": branch,
        "start_gate_passed": start_pass,
        "end_gate_passed": end_pass,
        "type_safety_passed": type_safe,
        "candidate_checkpoint_eligible": checkpoint_eligible,
        "target_gold_begin_as_inside_ceiling": (
            TARGET_BEGIN_AS_INSIDE_CEILING
        ),
        "end_spill_relative_reduction_floor": END_SPILL_REDUCTION_FLOOR,
        "maximum_positive_type_accuracy_loss": MAX_TYPE_ACCURACY_LOSS,
        "maximum_gold_begin_as_inside_increase": (
            MAX_BEGIN_AS_INSIDE_INCREASE
        ),
        "provenance_failures": provenance_failures,
        "target_start_failures": target_start_failures,
        "end_spill_failures": end_spill_failures,
        "type_safety_failures": type_safety_failures,
        "start_safety_failures": start_safety_failures,
        "comparisons": comparisons,
        "full_phase34_evaluation_authorized": (
            branch == "boundary_diagnostic_pass"
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
