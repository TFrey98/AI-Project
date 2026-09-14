"""Dependency-free Phase 36c begin-only positional-ablation gate logic."""

from __future__ import annotations

from copy import deepcopy

from story_model.provenance import canonical_json_sha256

try:
    from scripts.phase36b_identity_invariance_decision import _retained_comparison
except ModuleNotFoundError:
    from phase36b_identity_invariance_decision import _retained_comparison  # type: ignore


BEGIN_ONLY_IDENTITY_INVARIANCE_AUDIT_VERSION = 1
TARGET_SPLITS = ("lexical", "transfer")
EXPECTED_TRAINED_COUNT = 8
PREVIOUSLY_SEVERE_IDENTITIES = (366, 433, 493, 357)
FOCUS_ERROR_CEILING = 0.02
MAX_SPAN_METRIC_DROP = 0.02
MINIMUM_CLASS_SUPPORT = 20
SWAP_OPPORTUNITY_SOURCES = ("phase31_train", "phase32_train", "counterbalance_train")


def _trained_identities(summary: dict) -> tuple[tuple[int, ...], list[str]]:
    """Read the registered trained focus-token IDs, never guess them."""
    entries = summary.get("focus_identities", {}).get("trained", ())
    ids = tuple(sorted({int(entry["token_id"]) for entry in entries}))
    problems = []
    if len(ids) != EXPECTED_TRAINED_COUNT or len(entries) != EXPECTED_TRAINED_COUNT:
        problems.append(
            f"expected {EXPECTED_TRAINED_COUNT} distinct trained identities, "
            f"found {len(ids)}"
        )
    return ids, problems


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


def begin_only_identity_invariance_decision(
    crossed_summary: dict,
    baseline_retained: dict,
    candidate_retained: dict,
) -> dict:
    """Gate the begin-only ablation against transfer, span, and retained safety."""
    crossed = deepcopy(crossed_summary)
    baseline = deepcopy(baseline_retained)
    candidate = deepcopy(candidate_retained)

    invalid = []
    if crossed.get("begin_only_identity_invariance_audit_version") != (
        BEGIN_ONLY_IDENTITY_INVARIANCE_AUDIT_VERSION
    ):
        invalid.append("begin-only identity-invariance audit version is invalid")
    if crossed.get("decoder_changes") != "none":
        invalid.append("Phase 36c unexpectedly changed decoding")
    if crossed.get("architecture_changes") != "none":
        invalid.append("Phase 36c unexpectedly changed architecture")

    models = crossed.get("models", {})
    phase35 = models.get("phase35", {}).get("checkpoint", {})
    phase36b = models.get("phase36b", {}).get("checkpoint", {})
    phase36c = models.get("phase36c", {}).get("checkpoint", {})
    if phase35.get("paired_identity_invariance_version") is not None:
        invalid.append("Phase 35 crossed baseline unexpectedly used identity invariance")
    if phase36b.get("paired_identity_invariance_version") != 1:
        invalid.append("Phase 36b crossed baseline is not version 1")
    if phase36c.get("paired_identity_invariance_version") != 2:
        invalid.append("Phase 36c crossed candidate is not version 2")
    if phase36c.get("identity_invariance_position_mode") != "begin_only":
        invalid.append("Phase 36c crossed candidate is not begin-only")
    tokenizer_hashes = {
        checkpoint.get("tokenizer_sha256")
        for checkpoint in (phase35, phase36b, phase36c)
    }
    if len(tokenizer_hashes) != 1:
        invalid.append("Phase 35/36b/36c crossed tokenizers differ")
    for field in (
        "architecture", "boundary_objective_version", "boundary_loss_weight",
        "boundary_counterbalance_version",
    ):
        if phase36b.get(field) != phase36c.get(field):
            invalid.append(f"Phase 36b/36c crossed {field} differs")
    if phase36b.get("identity_invariance_swap_pool_sha256") != phase36c.get(
        "identity_invariance_swap_pool_sha256"
    ):
        invalid.append("Phase 36b/36c crossed swap-pool hash differs")
    if crossed.get("identity_swap_pool_sha256") != phase36c.get(
        "identity_invariance_swap_pool_sha256"
    ):
        invalid.append("summary swap-pool hash does not match the checkpoint")

    position_counts = crossed.get("phase36c_position_counts", {})
    if int(position_counts.get("I", -1)) != 0:
        invalid.append("Phase 36c generated at least one I:route pair")
    if int(position_counts.get("B", 0)) <= 0:
        invalid.append("Phase 36c generated no B:route pairs")
    if phase36c.get("identity_invariance_total_swap_position_counts") != position_counts:
        invalid.append("checkpoint and summary position counts differ")

    opportunity = crossed.get("swap_opportunity_audit", {})
    if set(opportunity) != set(SWAP_OPPORTUNITY_SOURCES):
        invalid.append("swap-opportunity audit is missing a required source")
    for source in SWAP_OPPORTUNITY_SOURCES:
        entry = opportunity.get(source, {})
        if entry.get("eligible_rows_did_not_fall") is not True:
            invalid.append(f"{source} pairing opportunity fell under begin-only")
        if int(entry.get("begin_and_inside_eligible_rows", -1)) < 0:
            invalid.append(f"{source} eligibility counts are missing")

    for split in TARGET_SPLITS:
        panel = crossed.get("panel_validation", {}).get(split, {})
        if int(panel.get("invalid_contexts", -1)) != 0:
            invalid.append(f"Phase 36c {split} panel has invalid contexts")
        if int(panel.get("identities", 0)) != 14:
            invalid.append(f"Phase 36c {split} identity count changed")

    trained_identities, trained_problems = _trained_identities(crossed)
    invalid.extend(trained_problems)
    checked_identities = trained_identities + PREVIOUSLY_SEVERE_IDENTITIES

    phase35_cells = _identity_cells(crossed, "phase35")
    phase36c_cells = _identity_cells(crossed, "phase36c")
    expected_cells = {
        (split, token_id) for split in TARGET_SPLITS for token_id in checked_identities
    }
    if not expected_cells <= set(phase35_cells) or not expected_cells <= set(phase36c_cells):
        invalid.append("crossed identity cells are incomplete for the checked identities")

    if baseline.get("checkpoint_sha256") != phase35.get("sha256"):
        invalid.append("baseline retained checkpoint is not the Phase 35 baseline")
    if candidate.get("checkpoint_sha256") != phase36c.get("sha256"):
        invalid.append("candidate retained checkpoint differs from the crossed audit")
    if candidate.get("checkpoint_paired_identity_invariance_version") != 2:
        invalid.append("candidate retained invariance version differs")

    transfer_failures = []
    span_regressions = []
    token_results = {}
    for token_id in checked_identities:
        split_results = {}
        for split in TARGET_SPLITS:
            before = phase35_cells.get((split, token_id), {})
            after = phase36c_cells.get((split, token_id), {})
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
        retained_failures.append("Phase 36c training-panel checkpoint is ineligible")

    transfer_ok = not transfer_failures
    span_ok = not span_regressions
    retained_ok = not retained_failures

    if invalid:
        branch = "invalid_begin_only_identity_invariance_comparison"
        next_action = "Repair provenance, support, or audit inputs before interpreting Phase 36c."
    elif not retained_ok:
        branch = "begin_only_identity_invariance_regressed_retained"
        next_action = (
            "Reject the begin-only ablation: a retained Phase 31/32 safety "
            "condition regressed or no eligible checkpoint exists."
        )
    elif not transfer_ok:
        branch = "begin_only_identity_invariance_span_recovered_transfer_regressed"
        next_action = (
            "Inside-position pairing was necessary for identity transfer; the "
            "ablation disproves itself. Do not adopt begin-only pairing."
        )
    elif not span_ok:
        branch = "begin_only_identity_invariance_transfer_preserved_span_still_damaged"
        next_action = (
            "Inside-position pairing was not the sole damaging component. The "
            "next variable is masking the swapped copy's supervised loss to "
            "the selected position, not lowering the JSD weight."
        )
    else:
        branch = "begin_only_identity_invariance_diagnostic_pass"
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
            branch == "begin_only_identity_invariance_diagnostic_pass"
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
