"""Dependency-free Phase 36g consistency-weight ablation gate logic.

Phase 36g varies exactly one thing: the clean-anchor consistency weight
(lambda = 1.0 reference vs. 0.1 candidate). Both arms restart from Phase
33c and share pairing mode, supervision mode, swap pool, mixture,
architecture, decoder, optimizer, schedule, and seed, and are compared at
one predeclared step.

Unlike the Phase 36c/36d gates, checkpoint eligibility does NOT rename the
scientific outcome here. The branch reports what the metrics did; retained
metrics and eligibility are reported independently and gate authorization.
An absent eligible checkpoint remains substantive, but it is not itself a
retained-metric regression.
"""

from __future__ import annotations

from copy import deepcopy

try:
    from scripts.phase36b_identity_invariance_decision import _retained_comparison
except ModuleNotFoundError:
    from phase36b_identity_invariance_decision import _retained_comparison  # type: ignore


CONSISTENCY_WEIGHT_AUDIT_VERSION = 1
TARGET_SPLITS = ("lexical", "transfer")
EXPECTED_IDENTITY_COUNTS = {"trained": 8, "heldout": 4, "legacy": 2}
FOCUS_ERROR_CEILING = 0.02
MAX_SPAN_METRIC_DROP = 0.02
MINIMUM_CLASS_SUPPORT = 20
REFERENCE_WEIGHT = 1.0
CANDIDATE_WEIGHT = 0.1
# Everything the two arms must share. Only the consistency weight may differ.
SHARED_ARM_FIELDS = (
    "architecture",
    "boundary_objective_version",
    "boundary_loss_weight",
    "boundary_counterbalance_version",
    "paired_identity_invariance_version",
    "identity_invariance_supervision_mode",
    "identity_invariance_position_mode",
    "identity_invariance_swap_pool_sha256",
    "tokenizer_sha256",
    "parent_phase33_checkpoint",
    "seed",
)


def _registered_identities(summary: dict) -> tuple[tuple[int, ...], list[str]]:
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


def consistency_weight_decision(
    crossed_summary: dict,
    baseline_retained: dict,
    candidate_retained: dict,
) -> dict:
    """Judge a bounded consistency-weight reduction on its own terms."""
    crossed = deepcopy(crossed_summary)
    baseline = deepcopy(baseline_retained)
    candidate_retained = deepcopy(candidate_retained)

    invalid = []
    if crossed.get("consistency_weight_ablation_audit_version") != (
        CONSISTENCY_WEIGHT_AUDIT_VERSION
    ):
        invalid.append("consistency-weight audit version is invalid")
    if crossed.get("decoder_changes") != "none":
        invalid.append("Phase 36g unexpectedly changed decoding")
    if crossed.get("architecture_changes") != "none":
        invalid.append("Phase 36g unexpectedly changed architecture")

    models = crossed.get("models", {})
    phase35 = models.get("phase35", {}).get("checkpoint", {})
    reference = models.get("reference", {}).get("checkpoint", {})
    candidate = models.get("candidate", {}).get("checkpoint", {})

    if phase35.get("paired_identity_invariance_version") is not None:
        invalid.append("Phase 35 span baseline unexpectedly used identity invariance")
    for label, arm, weight in (
        ("reference", reference, REFERENCE_WEIGHT),
        ("candidate", candidate, CANDIDATE_WEIGHT),
    ):
        arm_weight = arm.get("identity_invariance_loss_weight")
        if arm_weight is None or abs(float(arm_weight) - weight) > 1e-9:
            invalid.append(f"{label} arm weight is not {weight}")
    # The whole point of the ablation: only the weight may differ.
    for field in SHARED_ARM_FIELDS:
        if reference.get(field) != candidate.get(field):
            invalid.append(f"arms differ in {field}, not only the consistency weight")
    if reference.get("step") != candidate.get("step"):
        invalid.append("arms were not compared at the same predeclared step")
    if reference.get("step") is None:
        invalid.append("arm step is missing")

    for split in TARGET_SPLITS:
        panel = crossed.get("panel_validation", {}).get(split, {})
        if int(panel.get("invalid_contexts", -1)) != 0:
            invalid.append(f"Phase 36g {split} panel has invalid contexts")
        if int(panel.get("identities", 0)) != 14:
            invalid.append(f"Phase 36g {split} identity count changed")

    all_ids, identity_problems = _registered_identities(crossed)
    invalid.extend(identity_problems)

    phase35_cells = _identity_cells(crossed, "phase35")
    reference_cells = _identity_cells(crossed, "reference")
    candidate_cells = _identity_cells(crossed, "candidate")
    expected_cells = {(split, token_id) for split in TARGET_SPLITS for token_id in all_ids}
    for label, cells in (
        ("phase35", phase35_cells),
        ("reference", reference_cells),
        ("candidate", candidate_cells),
    ):
        if not expected_cells <= set(cells):
            invalid.append(f"{label} crossed identity cells are incomplete")

    if baseline.get("checkpoint_sha256") != phase35.get("sha256"):
        invalid.append("baseline retained checkpoint is not the Phase 35 baseline")
    if candidate_retained.get("checkpoint_sha256") != candidate.get("sha256"):
        invalid.append("candidate retained checkpoint differs from the crossed audit")

    transfer_failures = []
    span_regressions = []
    token_results = {}
    for token_id in all_ids:
        split_results = {}
        for split in TARGET_SPLITS:
            before = phase35_cells.get((split, token_id), {})
            arm = reference_cells.get((split, token_id), {})
            after = candidate_cells.get((split, token_id), {})
            focus = after.get("focus", {}).get("overall", {})
            if min(
                int(focus.get("gold_begin_count", 0)),
                int(focus.get("gold_inside_count", 0)),
            ) < MINIMUM_CLASS_SUPPORT:
                invalid.append(f"{split} token {token_id} lacks crossed B/I support")
            error = float(focus.get("begin_as_inside_rate", 1.0))
            if error > FOCUS_ERROR_CEILING:
                transfer_failures.append(
                    f"{split} token {token_id} B->I {error:.3f} exceeds "
                    f"{FOCUS_ERROR_CEILING:.3f}"
                )
            before_metrics = before.get("metrics", {})
            after_metrics = after.get("metrics", {})
            arm_metrics = arm.get("metrics", {})
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
                "candidate_begin_as_inside_rate": error,
                "reference_begin_as_inside_rate": float(
                    arm.get("focus", {}).get("overall", {}).get(
                        "begin_as_inside_rate", 1.0
                    )
                ),
                "candidate_precision_change_vs_phase35": precision_change,
                "candidate_recall_change_vs_phase35": recall_change,
                "reference_precision_change_vs_phase35": (
                    float(arm_metrics.get("exact_span_precision", 0.0))
                    - float(before_metrics.get("exact_span_precision", 0.0))
                ),
                "reference_recall_change_vs_phase35": (
                    float(arm_metrics.get("exact_span_recall", 0.0))
                    - float(before_metrics.get("exact_span_recall", 0.0))
                ),
            }
        token_results[str(token_id)] = {"splits": split_results}

    retained_failures, retained_comparisons = _retained_comparison(
        baseline, candidate_retained
    )
    candidate_eligible = candidate_retained.get("checkpoint_eligible") is True

    transfer_ok = not transfer_failures
    span_ok = not span_regressions
    retained_ok = not retained_failures

    if invalid:
        branch = "invalid_consistency_weight_comparison"
        next_action = (
            "Repair arm provenance or panel support before interpreting the "
            "consistency-weight ablation."
        )
    elif transfer_ok and span_ok:
        branch = "reduced_consistency_promising"
        next_action = (
            "Lower consistency strength is a promising improvement: identity "
            "transfer held and span quality reached Phase 35 compatibility. "
            "Promotion still requires the absolute quality floors and the "
            "inherited gate."
        )
    elif span_ok and not transfer_ok:
        branch = "reduced_consistency_tradeoff"
        next_action = (
            "The tested weight exposes a tradeoff. This does not prove every "
            "intermediate weight would fail; a bounded search between the two "
            "tested values is the registered follow-up."
        )
    elif transfer_ok and not span_ok:
        branch = "reduced_consistency_no_span_recovery"
        next_action = (
            "This reduction provides no span-quality recovery. Consistency "
            "strength is not the operative variable for complete spans; the "
            "next target is a different span-training objective."
        )
    else:
        branch = "reduced_consistency_rejected"
        next_action = "Both outcomes worsened. Reject the candidate weight."

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
        "candidate_checkpoint_eligible": candidate_eligible,
        "comparison_step": reference.get("step"),
        "thresholds": {
            "focus_error_ceiling": FOCUS_ERROR_CEILING,
            "maximum_span_metric_drop": MAX_SPAN_METRIC_DROP,
            "minimum_class_support": MINIMUM_CLASS_SUPPORT,
            "reference_weight": REFERENCE_WEIGHT,
            "candidate_weight": CANDIDATE_WEIGHT,
        },
        # Authorization requires the metric outcome AND retained safety AND an
        # eligible checkpoint. Eligibility is reported separately above so an
        # absent checkpoint never gets recorded as a metric regression.
        "full_phase34_evaluation_authorized": (
            branch == "reduced_consistency_promising"
            and retained_ok
            and candidate_eligible
        ),
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
