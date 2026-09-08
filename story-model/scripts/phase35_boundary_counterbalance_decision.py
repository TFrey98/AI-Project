"""Dependency-free Phase 35 boundary-counterbalance decision logic."""

from __future__ import annotations


EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")
TARGET_SPLITS = ("lexical", "transfer")
EXPECTED_AUDIT_VERSION = 1
EXPECTED_COUNTERBALANCE_VERSION = 2
EXPECTED_BOUNDARY_OBJECTIVE_VERSION = 1
BOUNDARY_ERROR_CEILING = 0.02
END_SPILL_CEILING = 0.01
MAX_BEGIN_AS_INSIDE_INCREASE = 0.02
MAX_OUTSIDE_AS_INSIDE_INCREASE = 0.005
MAX_TYPE_ACCURACY_LOSS = 0.01
MINIMUM_FOCUS_CLASS_SUPPORT = 20
SPAN_PRECISION_FLOOR = 0.98
SPAN_RECALL_FLOOR = 0.98
ANSWER_RECALL_FLOOR = 0.995
TYPE_ACCURACY_FLOOR = 0.99


def _metric_cells(summary: dict) -> dict:
    cells = {}
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary.get("datasets", {}).get(dataset_name, {})
        for split in EXPANDED_SPLITS:
            split_summary = dataset.get("splits", {}).get(split, {})
            if "metrics" in split_summary:
                cells[f"{dataset_name}/{split}"] = split_summary["metrics"]
            for skill, metrics in split_summary.get("per_skill", {}).items():
                cells[f"{dataset_name}/{split}/{skill}"] = metrics
    return cells


def boundary_counterbalance_decision(
    baseline: dict,
    candidate: dict,
    counterbalance: dict,
) -> dict:
    """Gate refreshed-token generalization and retained Phase 34 behavior."""

    invalid = []
    if baseline.get("tag_confusion_audit_version") != EXPECTED_AUDIT_VERSION:
        invalid.append("baseline is not a Phase 34 tag-confusion audit")
    if candidate.get("tag_confusion_audit_version") != EXPECTED_AUDIT_VERSION:
        invalid.append("candidate is not a Phase 34 tag-confusion audit")
    if counterbalance.get("boundary_counterbalance_audit_version") != (
        EXPECTED_AUDIT_VERSION
    ):
        invalid.append("counterbalance audit has the wrong version")

    checkpoint = counterbalance.get("checkpoint", {})
    if candidate.get("checkpoint_sha256") != checkpoint.get("sha256"):
        invalid.append("candidate and counterbalance audits use different checkpoints")
    if candidate.get("checkpoint_step") != checkpoint.get("step"):
        invalid.append("candidate and counterbalance checkpoint steps differ")
    if candidate.get("checkpoint_tokenizer_sha256") != checkpoint.get(
        "tokenizer_sha256"
    ):
        invalid.append("candidate and counterbalance tokenizers differ")
    if candidate.get("checkpoint_counterbalance_manifest_sha256") != (
        checkpoint.get("manifest_sha256")
    ):
        invalid.append("candidate and counterbalance manifests differ")
    if checkpoint.get("boundary_counterbalance_version") != (
        EXPECTED_COUNTERBALANCE_VERSION
    ):
        invalid.append("checkpoint lacks Phase 35 counterbalance provenance")
    if checkpoint.get("boundary_objective_version") != (
        EXPECTED_BOUNDARY_OBJECTIVE_VERSION
    ):
        invalid.append("checkpoint lost the Phase 34d boundary objective")
    weight = checkpoint.get("boundary_loss_weight")
    if weight is None or abs(float(weight) - 1.0) > 1.0e-9:
        invalid.append("checkpoint boundary loss weight is not 1.0")

    premise = counterbalance.get("manifest", {}).get(
        "phase34k_premise", {}
    )
    if premise.get("branch") != "local_token_state_dominant":
        invalid.append("Phase 35 premise is not local-token-state dominance")
    focus_ids = counterbalance.get("focus_token_ids", {})
    train_ids = set(focus_ids.get("train", ()))
    heldout_ids = set(focus_ids.get("heldout", ()))
    if not train_ids or not heldout_ids or train_ids & heldout_ids:
        invalid.append("Phase 35 focus-token pools are empty or overlapping")
    if len(train_ids) != 8 or len(heldout_ids) != 4:
        invalid.append("Phase 35 focus-token pools are not the registered 8/4")
    legacy_ids = set(premise.get("focus_token_ids", {}).values())
    if legacy_ids & (train_ids | heldout_ids):
        invalid.append("Phase 35 reused an or/ro focus token")

    baseline_cells = _metric_cells(baseline)
    candidate_cells = _metric_cells(candidate)
    if set(baseline_cells) != set(candidate_cells):
        invalid.append("baseline and candidate regression cells differ")

    regression_failures = []
    regression_comparisons = {}
    for label in sorted(set(baseline_cells) & set(candidate_cells)):
        before = baseline_cells[label]
        after = candidate_cells[label]
        begin_change = float(after["gold_begin_as_inside_rate"]) - float(
            before["gold_begin_as_inside_rate"]
        )
        outside_change = float(after["gold_outside_as_inside_rate"]) - float(
            before["gold_outside_as_inside_rate"]
        )
        type_change = float(after["positive_type_accuracy"]) - float(
            before["positive_type_accuracy"]
        )
        regression_comparisons[label] = {
            "gold_begin_as_inside_change": begin_change,
            "gold_outside_as_inside_change": outside_change,
            "positive_type_accuracy_change": type_change,
            "candidate_end_spill_rate": float(after["end_spill_rate"]),
        }
        if begin_change > MAX_BEGIN_AS_INSIDE_INCREASE:
            regression_failures.append(
                f"{label} B->I increased {begin_change:.3f}"
            )
        if outside_change > MAX_OUTSIDE_AS_INSIDE_INCREASE:
            regression_failures.append(
                f"{label} O->I increased {outside_change:.3f}"
            )
        if type_change < -MAX_TYPE_ACCURACY_LOSS:
            regression_failures.append(
                f"{label} type accuracy fell {-type_change:.3f}"
            )
        if float(after["end_spill_rate"]) > END_SPILL_CEILING:
            regression_failures.append(
                f"{label} end spill {float(after['end_spill_rate']):.3f} "
                f"exceeds {END_SPILL_CEILING:.3f}"
            )

    for split in TARGET_SPLITS:
        label = f"phase31_regression/{split}/scene_route"
        metrics = candidate_cells.get(label)
        if metrics is None:
            invalid.append(f"candidate is missing legacy target cell {label}")
        elif float(metrics["gold_begin_as_inside_rate"]) > BOUNDARY_ERROR_CEILING:
            regression_failures.append(
                f"{label} B->I {float(metrics['gold_begin_as_inside_rate']):.3f} "
                f"exceeds {BOUNDARY_ERROR_CEILING:.3f}"
            )

    generalization_failures = []
    focus_results = {}
    for split in TARGET_SPLITS:
        split_summary = counterbalance.get("splits", {}).get(split)
        if not isinstance(split_summary, dict):
            invalid.append(f"counterbalance audit is missing {split}")
            continue
        if split_summary.get("focus_pool") != "heldout":
            invalid.append(f"counterbalance {split} does not use heldout tokens")
        metrics = split_summary.get("metrics", {})
        for field, floor in (
            ("exact_span_precision", SPAN_PRECISION_FLOOR),
            ("exact_span_recall", SPAN_RECALL_FLOOR),
            ("answer_candidate_recall", ANSWER_RECALL_FLOOR),
            ("boundary_type_accuracy", TYPE_ACCURACY_FLOOR),
        ):
            value = float(metrics.get(field, 0.0))
            if value < floor:
                generalization_failures.append(
                    f"{split} {field} {value:.3f} is below {floor:.3f}"
                )
        if float(metrics.get("offset_validity_rate", 0.0)) != 1.0:
            generalization_failures.append(f"{split} offsets are not all valid")
        if float(metrics.get("proposal_overflow_rate", 1.0)) != 0.0:
            generalization_failures.append(f"{split} proposal overflow is nonzero")

        focus = split_summary.get("focus", {})
        per_token = focus.get("per_token", {})
        if {int(token_id) for token_id in per_token} != heldout_ids:
            invalid.append(f"counterbalance {split} focus-token set changed")
        for token_id, token_metrics in per_token.items():
            begin_count = int(token_metrics.get("gold_begin_count", 0))
            inside_count = int(token_metrics.get("gold_inside_count", 0))
            begin_error = float(token_metrics.get("begin_as_inside_rate", 1.0))
            inside_error = float(token_metrics.get("inside_as_begin_rate", 1.0))
            if min(begin_count, inside_count) < MINIMUM_FOCUS_CLASS_SUPPORT:
                invalid.append(
                    f"{split} token {token_id} lacks B/I support"
                )
            if begin_error > BOUNDARY_ERROR_CEILING:
                generalization_failures.append(
                    f"{split} token {token_id} B->I {begin_error:.3f} "
                    f"exceeds {BOUNDARY_ERROR_CEILING:.3f}"
                )
            if inside_error > BOUNDARY_ERROR_CEILING:
                generalization_failures.append(
                    f"{split} token {token_id} I->B {inside_error:.3f} "
                    f"exceeds {BOUNDARY_ERROR_CEILING:.3f}"
                )
        focus_results[split] = focus.get("overall", {})

    checkpoint_eligible = (
        checkpoint.get("eligible") is True
        and candidate.get("checkpoint_eligible") is True
    )
    if invalid:
        branch = "invalid_boundary_counterbalance_comparison"
        next_action = (
            "Repair checkpoint, manifest, focus-pool, or audit provenance "
            "before interpreting Phase 35."
        )
    elif regression_failures:
        branch = "boundary_counterbalance_rejected"
        next_action = (
            "Reject the counterbalance because it regressed retained cells or "
            "did not repair the original scene-route boundary failures."
        )
    elif not checkpoint_eligible:
        branch = "boundary_counterbalance_span_gate_ineligible"
        next_action = (
            "The retained and refreshed diagnostics are interpretable, but no "
            "eligible training-panel checkpoint exists; do not extend training."
        )
    elif generalization_failures:
        branch = "boundary_counterbalance_generalization_insufficient"
        next_action = (
            "The data intervention did not compose to refreshed token "
            "identities; inspect per-token errors without changing architecture."
        )
    else:
        branch = "boundary_counterbalance_diagnostic_pass"
        next_action = (
            "Run the unchanged full Phase 34 proposer evaluation against the "
            "Phase 31/32 gate. Promote only if that inherited gate also passes."
        )

    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "regression_failures": regression_failures,
        "generalization_failures": generalization_failures,
        "regression_comparisons": regression_comparisons,
        "focus_results": focus_results,
        "candidate_checkpoint_eligible": checkpoint_eligible,
        "thresholds": {
            "boundary_error_ceiling": BOUNDARY_ERROR_CEILING,
            "end_spill_ceiling": END_SPILL_CEILING,
            "minimum_focus_class_support": MINIMUM_FOCUS_CLASS_SUPPORT,
            "span_precision_floor": SPAN_PRECISION_FLOOR,
            "span_recall_floor": SPAN_RECALL_FLOOR,
            "answer_recall_floor": ANSWER_RECALL_FLOOR,
            "type_accuracy_floor": TYPE_ACCURACY_FLOOR,
        },
        "full_phase34_evaluation_authorized": (
            branch == "boundary_counterbalance_diagnostic_pass"
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
