"""Dependency-free decision logic for the Phase 36a crossed-identity audit."""

from __future__ import annotations


CROSSED_IDENTITY_AUDIT_VERSION = 1
TARGET_SPLITS = ("lexical", "transfer")
EXPECTED_GROUP_SIZES = {"trained": 8, "heldout": 4, "legacy": 2}
FOCUS_ERROR_CEILING = 0.02
SEVERE_FOCUS_ERROR_FLOOR = 0.10
MINIMUM_CLASS_SUPPORT = 20
MAX_SPAN_METRIC_DROP = 0.02


def _identity_cells(summary: dict, model: str) -> dict:
    cells = {}
    model_summary = summary.get("models", {}).get(model, {})
    for split in TARGET_SPLITS:
        identities = model_summary.get("splits", {}).get(split, {}).get(
            "identities", {}
        )
        for token_id, value in identities.items():
            cells[(split, int(token_id))] = value
    return cells


def crossed_identity_decision(summary: dict) -> dict:
    """Interpret identity membership after holding prompt context fixed."""

    invalid = []
    if summary.get("crossed_boundary_identity_audit_version") != (
        CROSSED_IDENTITY_AUDIT_VERSION
    ):
        invalid.append("crossed-identity audit version is invalid")
    if summary.get("training_changes") != "none":
        invalid.append("Phase 36a unexpectedly changed training")
    if summary.get("decoder_changes") != "none":
        invalid.append("Phase 36a unexpectedly changed decoding")
    phase35 = summary.get("phase35_inputs", {})
    if phase35.get("decision_branch") != "boundary_counterbalance_rejected":
        invalid.append("Phase 35 input was not the official rejected branch")
    if phase35.get("checkpoint_eligible") is not False:
        invalid.append("Phase 35 diagnostic checkpoint eligibility changed")
    if phase35.get("checkpoint_promotion_authorized") is not False:
        invalid.append("Phase 35 promotion provenance is invalid")
    if phase35.get("full_phase34_evaluation_authorized") is not False:
        invalid.append("Phase 35 full-evaluation provenance is invalid")
    model_summaries = summary.get("models", {})
    phase34d_checkpoint = model_summaries.get("phase34d", {}).get(
        "checkpoint", {}
    )
    phase35_checkpoint = model_summaries.get("phase35", {}).get(
        "checkpoint", {}
    )
    if phase35_checkpoint.get("sha256") != phase35.get("checkpoint_sha256"):
        invalid.append("Phase 35 crossed checkpoint hash changed")
    if phase35_checkpoint.get("checkpoint_eligible") is not False:
        invalid.append("Phase 35 crossed checkpoint eligibility changed")
    if phase35_checkpoint.get("boundary_counterbalance_version") != 2:
        invalid.append("Phase 35 crossed checkpoint version changed")
    if phase34d_checkpoint.get("boundary_counterbalance_version") is not None:
        invalid.append("Phase 34d baseline unexpectedly has counterbalance data")
    if phase34d_checkpoint.get("tokenizer_sha256") != phase35_checkpoint.get(
        "tokenizer_sha256"
    ):
        invalid.append("Phase 34d and Phase 35 crossed tokenizers differ")
    for label, checkpoint in (
        ("Phase 34d", phase34d_checkpoint),
        ("Phase 35", phase35_checkpoint),
    ):
        if checkpoint.get("architecture") != "explicit_offset_candidate_proposer":
            invalid.append(f"{label} crossed architecture changed")
        if checkpoint.get("boundary_objective_version") != 1:
            invalid.append(f"{label} crossed boundary objective changed")
        weight = checkpoint.get("boundary_loss_weight")
        if weight is None or abs(float(weight) - 1.0) > 1.0e-9:
            invalid.append(f"{label} crossed boundary weight changed")
        if any(
            checkpoint.get(field) is not None
            for field in (
                "token_width_geometry_version",
                "token_end_geometry_version",
                "factorized_boundary_type_version",
            )
        ):
            invalid.append(f"{label} crossed proposer geometry changed")

    identities = summary.get("focus_identities", {})
    groups = {}
    all_ids = set()
    for group, expected_size in EXPECTED_GROUP_SIZES.items():
        entries = identities.get(group, ())
        ids = {int(entry.get("token_id", -1)) for entry in entries}
        if len(entries) != expected_size or len(ids) != expected_size:
            invalid.append(f"Phase 36a {group} identity set is not registered")
        if all_ids & ids:
            invalid.append(f"Phase 36a {group} identities overlap another group")
        all_ids.update(ids)
        groups[group] = ids

    panel = summary.get("panel_validation", {})
    for split in TARGET_SPLITS:
        split_panel = panel.get(split, {})
        if int(split_panel.get("invalid_contexts", -1)) != 0:
            invalid.append(f"Phase 36a {split} panel has invalid contexts")
        if int(split_panel.get("source_pairs", 0)) < 1:
            invalid.append(f"Phase 36a {split} panel has no source pairs")
        if int(split_panel.get("identities", 0)) != len(all_ids):
            invalid.append(f"Phase 36a {split} identity count changed")

    baseline_cells = _identity_cells(summary, "phase34d")
    candidate_cells = _identity_cells(summary, "phase35")
    expected_cells = {
        (split, token_id) for split in TARGET_SPLITS for token_id in all_ids
    }
    if set(baseline_cells) != expected_cells:
        invalid.append("Phase 34d crossed cells are incomplete")
    if set(candidate_cells) != expected_cells:
        invalid.append("Phase 35 crossed cells are incomplete")

    token_results = {}
    span_regressions = []
    for token_id in sorted(all_ids):
        group = next(group for group, ids in groups.items() if token_id in ids)
        split_results = {}
        for split in TARGET_SPLITS:
            before = baseline_cells.get((split, token_id), {})
            after = candidate_cells.get((split, token_id), {})
            if before.get("group") != group or after.get("group") != group:
                invalid.append(
                    f"{split} token {token_id} crossed group changed"
                )
            before_focus = before.get("focus", {}).get("overall", {})
            after_focus = after.get("focus", {}).get("overall", {})
            begin_support = int(after_focus.get("gold_begin_count", 0))
            inside_support = int(after_focus.get("gold_inside_count", 0))
            if min(begin_support, inside_support) < MINIMUM_CLASS_SUPPORT:
                invalid.append(
                    f"{split} token {token_id} lacks crossed B/I support"
                )
            before_error = float(
                before_focus.get("begin_as_inside_rate", 1.0)
            )
            after_error = float(after_focus.get("begin_as_inside_rate", 1.0))
            before_metrics = before.get("metrics", {})
            after_metrics = after.get("metrics", {})
            precision_change = float(
                after_metrics.get("exact_span_precision", 0.0)
            ) - float(before_metrics.get("exact_span_precision", 0.0))
            recall_change = float(
                after_metrics.get("exact_span_recall", 0.0)
            ) - float(before_metrics.get("exact_span_recall", 0.0))
            if precision_change < -MAX_SPAN_METRIC_DROP:
                span_regressions.append(
                    f"{split} token {token_id} span precision fell "
                    f"{-precision_change:.3f}"
                )
            if recall_change < -MAX_SPAN_METRIC_DROP:
                span_regressions.append(
                    f"{split} token {token_id} span recall fell "
                    f"{-recall_change:.3f}"
                )
            split_results[split] = {
                "phase34d_begin_as_inside_rate": before_error,
                "phase35_begin_as_inside_rate": after_error,
                "begin_as_inside_change": after_error - before_error,
                "phase35_begin_minus_inside_margin": after.get(
                    "focus_geometry", {}
                ).get("gold_B", {}).get("mean_begin_minus_inside_margin"),
                "exact_span_precision_change": precision_change,
                "exact_span_recall_change": recall_change,
            }
        maximum_error = max(
            split_results[split]["phase35_begin_as_inside_rate"]
            for split in TARGET_SPLITS
        )
        minimum_error = min(
            split_results[split]["phase35_begin_as_inside_rate"]
            for split in TARGET_SPLITS
        )
        token_results[str(token_id)] = {
            "group": group,
            "splits": split_results,
            "passes_both_splits": maximum_error <= FOCUS_ERROR_CEILING,
            "severe_in_both_splits": (
                minimum_error >= SEVERE_FOCUS_ERROR_FLOOR
            ),
        }

    group_results = {}
    for group, ids in groups.items():
        values = [token_results[str(token_id)] for token_id in ids]
        group_results[group] = {
            "tokens": len(values),
            "passing_tokens": sum(
                value["passes_both_splits"] for value in values
            ),
            "severe_tokens": sum(
                value["severe_in_both_splits"] for value in values
            ),
        }

    original_heldout = phase35.get("original_heldout_focus", {})
    originally_failing = 0
    for token_id in groups.get("heldout", set()):
        token = original_heldout.get(str(token_id), {})
        if any(
            float(token.get(split, 1.0)) > FOCUS_ERROR_CEILING
            for split in TARGET_SPLITS
        ):
            originally_failing += 1

    if invalid:
        branch = "invalid_crossed_identity_audit"
        next_action = (
            "Repair Phase 35 provenance or crossed-panel matching before "
            "interpreting identity transfer."
        )
    elif (
        group_results["trained"]["passing_tokens"] == 8
        and group_results["heldout"]["severe_tokens"] >= 3
    ):
        branch = "trained_identity_memorization_dominant"
        next_action = (
            "Register one paired identity-invariance objective while keeping "
            "the Phase 34d architecture and decoder fixed; do not merely add "
            "more counterbalance rows."
        )
    elif (
        group_results["heldout"]["passing_tokens"] >= 3
        and originally_failing >= 3
    ):
        branch = "context_distribution_interaction_indicated"
        next_action = (
            "Cross train and held-out identities through every source split "
            "before considering an architecture or objective change."
        )
    elif (
        group_results["heldout"]["passing_tokens"] >= 1
        and group_results["heldout"]["severe_tokens"] >= 1
    ):
        branch = "heterogeneous_identity_priors_confirmed"
        next_action = (
            "Use the paired margin geometry to design an identity-invariance "
            "test; a larger undifferentiated focus pool is not isolated."
        )
    elif all(
        result["passing_tokens"] == result["tokens"]
        for result in group_results.values()
    ):
        branch = "broad_cross_identity_transfer"
        next_action = (
            "Treat the Phase 35 held-out failure as a context interaction and "
            "redesign only the split-crossing curriculum."
        )
    else:
        branch = "mixed_crossed_identity_result"
        next_action = (
            "Inspect per-token margins and complete-span regressions before "
            "registering another training variable."
        )

    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "token_results": token_results,
        "group_results": group_results,
        "originally_failing_heldout_tokens": originally_failing,
        "complete_span_regressions": span_regressions,
        "thresholds": {
            "focus_error_ceiling": FOCUS_ERROR_CEILING,
            "severe_focus_error_floor": SEVERE_FOCUS_ERROR_FLOOR,
            "minimum_class_support": MINIMUM_CLASS_SUPPORT,
            "maximum_span_metric_drop": MAX_SPAN_METRIC_DROP,
        },
        "training_authorized": False,
        "decoder_change_authorized": False,
        "full_phase34_evaluation_authorized": False,
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
