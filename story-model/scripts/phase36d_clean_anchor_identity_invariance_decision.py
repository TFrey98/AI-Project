"""Dependency-free Phase 36d clean-anchor objective gate logic."""

from __future__ import annotations

from copy import deepcopy

try:
    from scripts.phase36b_identity_invariance_decision import _retained_comparison
except ModuleNotFoundError:
    from phase36b_identity_invariance_decision import _retained_comparison  # type: ignore


CLEAN_ANCHOR_IDENTITY_INVARIANCE_AUDIT_VERSION = 1
TARGET_SPLITS = ("lexical", "transfer")
EXPECTED_IDENTITY_COUNTS = {"trained": 8, "heldout": 4, "legacy": 2}
FOCUS_ERROR_CEILING = 0.02
MAX_SPAN_METRIC_DROP = 0.02
MINIMUM_CLASS_SUPPORT = 20


def _all_registered_identities(summary: dict) -> tuple[tuple[int, ...], list[str]]:
    """Read every registered identity (14 total), never guess them."""
    problems = []
    identities = summary.get("focus_identities", {})
    ids = []
    for group, expected in EXPECTED_IDENTITY_COUNTS.items():
        entries = identities.get(group, ())
        group_ids = sorted({int(entry["token_id"]) for entry in entries})
        if len(entries) != expected or len(group_ids) != expected:
            problems.append(f"expected {expected} distinct {group} identities")
        ids.extend(group_ids)
    if len(ids) != len(set(ids)):
        problems.append("registered identity groups overlap")
    return tuple(sorted(set(ids))), problems


def _identity_cells(summary: dict, model: str) -> dict:
    cells = {}
    for split in TARGET_SPLITS:
        identities = (
            summary.get("models", {}).get(model, {}).get("splits", {})
            .get(split, {}).get("identities", {})
        )
        for token_id, value in identities.items():
            cells[(split, int(token_id))] = value
    return cells


def clean_anchor_identity_invariance_decision(
    crossed_summary: dict,
    baseline_retained: dict,
    candidate_retained: dict,
) -> dict:
    """Gate Phase 36d: all 14 identities, span quality vs Phase 35, retained safety."""
    crossed = deepcopy(crossed_summary)
    baseline = deepcopy(baseline_retained)
    candidate = deepcopy(candidate_retained)

    invalid = []
    if crossed.get("clean_anchor_identity_invariance_audit_version") != (
        CLEAN_ANCHOR_IDENTITY_INVARIANCE_AUDIT_VERSION
    ):
        invalid.append("clean-anchor identity-invariance audit version is invalid")
    if crossed.get("decoder_changes") != "none":
        invalid.append("Phase 36d unexpectedly changed decoding")
    if crossed.get("architecture_changes") != "none":
        invalid.append("Phase 36d unexpectedly changed architecture")

    models = crossed.get("models", {})
    phase35 = models.get("phase35", {}).get("checkpoint", {})
    phase36d = models.get("phase36d", {}).get("checkpoint", {})
    if phase35.get("paired_identity_invariance_version") is not None:
        invalid.append("Phase 35 crossed baseline unexpectedly used identity invariance")
    if phase36d.get("paired_identity_invariance_version") != 3:
        invalid.append("Phase 36d crossed candidate is not version 3")
    if phase36d.get("identity_invariance_supervision_mode") != "clean_anchor":
        invalid.append("Phase 36d crossed candidate is not clean-anchor supervision")
    if phase35.get("tokenizer_sha256") != phase36d.get("tokenizer_sha256"):
        invalid.append("Phase 35/36d crossed tokenizers differ")
    for field in ("architecture", "boundary_objective_version", "boundary_loss_weight",
                  "boundary_counterbalance_version"):
        if phase35.get(field) is None and field != "boundary_counterbalance_version":
            invalid.append(f"Phase 35 crossed {field} is missing")

    for split in TARGET_SPLITS:
        panel = crossed.get("panel_validation", {}).get(split, {})
        if int(panel.get("invalid_contexts", -1)) != 0:
            invalid.append(f"Phase 36d {split} panel has invalid contexts")
        if int(panel.get("identities", 0)) != 14:
            invalid.append(f"Phase 36d {split} identity count changed")

    all_ids, identity_problems = _all_registered_identities(crossed)
    invalid.extend(identity_problems)

    phase35_cells = _identity_cells(crossed, "phase35")
    phase36d_cells = _identity_cells(crossed, "phase36d")
    expected_cells = {(split, token_id) for split in TARGET_SPLITS for token_id in all_ids}
    if not expected_cells <= set(phase35_cells) or not expected_cells <= set(phase36d_cells):
        invalid.append("crossed identity cells are incomplete")

    if baseline.get("checkpoint_sha256") != phase35.get("sha256"):
        invalid.append("baseline retained checkpoint is not the Phase 35 baseline")
    if candidate.get("checkpoint_sha256") != phase36d.get("sha256"):
        invalid.append("candidate retained checkpoint differs from the crossed audit")
    if candidate.get("checkpoint_paired_identity_invariance_version") != 3:
        invalid.append("candidate retained invariance version differs")

    transfer_failures = []
    span_regressions = []
    token_results = {}
    for token_id in all_ids:
        split_results = {}
        for split in TARGET_SPLITS:
            before = phase35_cells.get((split, token_id), {})
            after = phase36d_cells.get((split, token_id), {})
            focus = after.get("focus", {}).get("overall", {})
            begin_support = int(focus.get("gold_begin_count", 0))
            inside_support = int(focus.get("gold_inside_count", 0))
            if min(begin_support, inside_support) < MINIMUM_CLASS_SUPPORT:
                invalid.append(f"{split} token {token_id} lacks crossed B/I support")
            error = float(focus.get("begin_as_inside_rate", 1.0))
            if error > FOCUS_ERROR_CEILING:
                transfer_failures.append(
                    f"{split} token {token_id} B->I {error:.3f} exceeds "
                    f"{FOCUS_ERROR_CEILING:.3f}"
                )
            before_metrics, after_metrics = before.get("metrics", {}), after.get("metrics", {})
            precision_change = float(after_metrics.get("exact_span_precision", 0.0)) - float(
                before_metrics.get("exact_span_precision", 0.0)
            )
            recall_change = float(after_metrics.get("exact_span_recall", 0.0)) - float(
                before_metrics.get("exact_span_recall", 0.0)
            )
            if precision_change < -MAX_SPAN_METRIC_DROP:
                span_regressions.append(
                    f"{split} token {token_id} span precision fell "
                    f"{-precision_change:.3f} vs Phase 35"
                )
            if recall_change < -MAX_SPAN_METRIC_DROP:
                span_regressions.append(
                    f"{split} token {token_id} span recall fell "
                    f"{-recall_change:.3f} vs Phase 35"
                )
            split_results[split] = {
                "begin_as_inside_rate": error,
                "exact_span_precision_change_vs_phase35": precision_change,
                "exact_span_recall_change_vs_phase35": recall_change,
            }
        token_results[str(token_id)] = {"splits": split_results}

    retained_failures, retained_comparisons = _retained_comparison(baseline, candidate)
    if candidate.get("checkpoint_eligible") is not True:
        retained_failures.append("Phase 36d training-panel checkpoint is ineligible")

    transfer_ok = not transfer_failures
    span_ok = not span_regressions
    retained_ok = not retained_failures

    if invalid:
        branch = "invalid_clean_anchor_identity_invariance_comparison"
        next_action = "Repair provenance, support, or audit inputs before interpreting Phase 36d."
    elif not retained_ok:
        branch = "clean_anchor_identity_invariance_regressed_retained"
        next_action = (
            "Reject: a retained Phase 31/32 safety condition regressed or no "
            "eligible checkpoint exists."
        )
    elif not transfer_ok:
        branch = "clean_anchor_identity_invariance_transfer_failed"
        next_action = (
            "Clean-anchor masking did not preserve identity transfer for all "
            "14 registered identities; inspect per-identity failures."
        )
    elif not span_ok:
        branch = "clean_anchor_identity_invariance_span_still_damaged"
        next_action = (
            "Removing the swapped copy's direct supervision did not recover "
            "complete-span quality either; the damage is not explained by "
            "double supervision alone. Inspect the JSD term itself next."
        )
    else:
        branch = "clean_anchor_identity_invariance_diagnostic_pass"
        next_action = "Run the unchanged inherited full Phase 34 gate."

    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "transfer_failures": transfer_failures,
        "span_regressions_vs_phase35": span_regressions,
        "retained_regression_failures": retained_failures,
        "retained_comparisons": retained_comparisons,
        "token_results": token_results,
        "transfer_ok": transfer_ok,
        "span_ok": span_ok,
        "retained_ok": retained_ok,
        "candidate_checkpoint_eligible": candidate.get("checkpoint_eligible") is True,
        "thresholds": {
            "focus_error_ceiling": FOCUS_ERROR_CEILING,
            "maximum_span_metric_drop": MAX_SPAN_METRIC_DROP,
            "minimum_class_support": MINIMUM_CLASS_SUPPORT,
        },
        "full_phase34_evaluation_authorized": (
            branch == "clean_anchor_identity_invariance_diagnostic_pass"
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
