"""Dependency-free decision logic for Phase 34f token-width geometry."""

from __future__ import annotations


EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")
EXPECTED_BOUNDARY_OBJECTIVE_VERSION = 1
EXPECTED_BOUNDARY_LOSS_WEIGHT = 1.0
EXPECTED_TOKEN_WIDTH_GEOMETRY_VERSION = 1
TARGET_BEGIN_AS_INSIDE_CEILING = 0.02
MAX_END_SPILL_RATE = 0.01
MAX_TYPE_ACCURACY_LOSS = 0.01
MAX_BEGIN_AS_INSIDE_INCREASE = 0.02
MAX_OUTSIDE_AS_INSIDE_INCREASE = 0.005


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


def token_width_geometry_decision(baseline: dict, candidate: dict) -> dict:
    """Decide whether explicit token width fixes the held-out boundary."""

    provenance_failures = []
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        if (
            summary.get("checkpoint_boundary_objective_version")
            != EXPECTED_BOUNDARY_OBJECTIVE_VERSION
        ):
            provenance_failures.append(
                f"{label} does not declare boundary objective version 1"
            )
        weight = summary.get("checkpoint_boundary_loss_weight")
        if weight is None or abs(
            float(weight) - EXPECTED_BOUNDARY_LOSS_WEIGHT
        ) > 1.0e-9:
            provenance_failures.append(
                f"{label} boundary loss weight is not 1.0"
            )
    if baseline.get("checkpoint_token_width_geometry_version") is not None:
        provenance_failures.append(
            "baseline unexpectedly contains token-width geometry"
        )
    if (
        candidate.get("checkpoint_token_width_geometry_version")
        != EXPECTED_TOKEN_WIDTH_GEOMETRY_VERSION
    ):
        provenance_failures.append(
            "candidate does not declare token-width geometry version 1"
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

    target_failures = []
    safety_failures = []
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
        baseline_type = float(baseline_metrics["positive_type_accuracy"])
        candidate_type = float(candidate_metrics["positive_type_accuracy"])
        candidate_end = float(candidate_metrics["end_spill_rate"])
        baseline_outside = float(
            baseline_metrics["gold_outside_as_inside_rate"]
        )
        candidate_outside = float(
            candidate_metrics["gold_outside_as_inside_rate"]
        )
        comparisons[label] = {
            "baseline_gold_begin_as_inside_rate": baseline_start,
            "candidate_gold_begin_as_inside_rate": candidate_start,
            "gold_begin_as_inside_change": candidate_start - baseline_start,
            "baseline_positive_type_accuracy": baseline_type,
            "candidate_positive_type_accuracy": candidate_type,
            "positive_type_accuracy_change": candidate_type - baseline_type,
            "candidate_end_spill_rate": candidate_end,
            "baseline_gold_outside_as_inside_rate": baseline_outside,
            "candidate_gold_outside_as_inside_rate": candidate_outside,
            "gold_outside_as_inside_change": (
                candidate_outside - baseline_outside
            ),
        }
        target_cell = (
            dataset == "phase31_regression"
            and split in {"lexical", "transfer"}
        )
        if target_cell and candidate_start > TARGET_BEGIN_AS_INSIDE_CEILING:
            target_failures.append(
                f"{label} B->I {candidate_start:.3f} is above "
                f"{TARGET_BEGIN_AS_INSIDE_CEILING:.3f}"
            )
        if candidate_start > baseline_start + MAX_BEGIN_AS_INSIDE_INCREASE:
            safety_failures.append(
                f"{label} B->I increased "
                f"{candidate_start - baseline_start:.3f}, above "
                f"{MAX_BEGIN_AS_INSIDE_INCREASE:.3f}"
            )
        if candidate_type < baseline_type - MAX_TYPE_ACCURACY_LOSS:
            safety_failures.append(
                f"{label} positive type accuracy fell "
                f"{baseline_type - candidate_type:.3f}, above "
                f"{MAX_TYPE_ACCURACY_LOSS:.3f}"
            )
        if candidate_end > MAX_END_SPILL_RATE:
            safety_failures.append(
                f"{label} end spill {candidate_end:.3f} is above "
                f"{MAX_END_SPILL_RATE:.3f}"
            )
        if (
            candidate_outside
            > baseline_outside + MAX_OUTSIDE_AS_INSIDE_INCREASE
        ):
            safety_failures.append(
                f"{label} O->I increased "
                f"{candidate_outside - baseline_outside:.3f}, above "
                f"{MAX_OUTSIDE_AS_INSIDE_INCREASE:.3f}"
            )

    geometry = candidate.get("target_begin_geometry", {})
    width_two = geometry.get("dimensions", {}).get("token_width", {}).get(
        "2"
    )
    if not isinstance(width_two, dict) or int(
        width_two.get("gold_begin_count", 0)
    ) <= 0:
        provenance_failures.append(
            "candidate audit has no supported token-width-2 target bucket"
        )
        width_two_rate = None
    else:
        width_two_rate = float(width_two["begin_as_inside_rate"])
        if width_two_rate > TARGET_BEGIN_AS_INSIDE_CEILING:
            target_failures.append(
                f"target token-width-2 B->I {width_two_rate:.3f} is above "
                f"{TARGET_BEGIN_AS_INSIDE_CEILING:.3f}"
            )

    target_passed = not target_failures
    safety_passed = not safety_failures
    checkpoint_eligible = candidate.get("checkpoint_eligible") is True
    if provenance_failures:
        branch = "invalid_token_width_geometry_comparison"
        next_action = (
            "Stop and correct the checkpoints or audit inputs; do not "
            "interpret this comparison."
        )
    elif not safety_passed:
        branch = "token_width_geometry_rejected"
        next_action = (
            "Reject token-width geometry because a retained boundary or type "
            "cell regressed. Do not run the full gate or extend training."
        )
    elif target_passed and checkpoint_eligible:
        branch = "token_width_geometry_diagnostic_pass"
        next_action = (
            "Run the unchanged full Phase 34 proposer and Phase 33c oracle "
            "gates. Promote only if both pass."
        )
    elif target_passed:
        branch = "token_width_fixed_span_gate_ineligible"
        next_action = (
            "Token-width geometry fixed the registered boundary cells, but "
            "the training panel is ineligible. Stop without a full gate, "
            "checkpoint promotion, or longer training."
        )
    else:
        branch = "token_width_geometry_insufficient"
        next_action = (
            "Token width did not fix the registered held-out boundary. Stop "
            "without changing decoding or loss weights; test an explicit "
            "end-of-token feature next."
        )
    return {
        "branch": branch,
        "target_gate_passed": target_passed,
        "safety_gate_passed": safety_passed,
        "candidate_checkpoint_eligible": checkpoint_eligible,
        "target_token_width_2_begin_as_inside_rate": width_two_rate,
        "target_gold_begin_as_inside_ceiling": (
            TARGET_BEGIN_AS_INSIDE_CEILING
        ),
        "maximum_end_spill_rate": MAX_END_SPILL_RATE,
        "maximum_positive_type_accuracy_loss": MAX_TYPE_ACCURACY_LOSS,
        "maximum_gold_begin_as_inside_increase": (
            MAX_BEGIN_AS_INSIDE_INCREASE
        ),
        "maximum_gold_outside_as_inside_increase": (
            MAX_OUTSIDE_AS_INSIDE_INCREASE
        ),
        "provenance_failures": provenance_failures,
        "target_failures": target_failures,
        "safety_failures": safety_failures,
        "comparisons": comparisons,
        "full_phase34_evaluation_authorized": (
            branch == "token_width_geometry_diagnostic_pass"
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
