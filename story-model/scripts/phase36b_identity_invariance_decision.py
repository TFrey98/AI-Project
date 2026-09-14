"""Dependency-free Phase 36b paired identity-invariance gate logic."""

from __future__ import annotations

from copy import deepcopy

from story_model.provenance import canonical_json_sha256

try:
    from scripts.phase35_boundary_counterbalance_decision import (
        END_SPILL_CEILING,
        MAX_BEGIN_AS_INSIDE_INCREASE,
        MAX_OUTSIDE_AS_INSIDE_INCREASE,
    )
except ModuleNotFoundError:
    END_SPILL_CEILING = 0.01
    MAX_BEGIN_AS_INSIDE_INCREASE = 0.02
    MAX_OUTSIDE_AS_INSIDE_INCREASE = 0.005


IDENTITY_INVARIANCE_AUDIT_VERSION = 1
EXPECTED_GROUP_SIZES = {"trained": 8, "heldout": 4, "legacy": 2}
PREVIOUSLY_SEVERE_IDENTITIES = {366, 433, 493, 357}
TARGET_SPLITS = ("lexical", "transfer")
FOCUS_ERROR_CEILING = 0.02
MINIMUM_CLASS_SUPPORT = 20
MAX_SPAN_METRIC_DROP = 0.02
MAX_TYPE_ACCURACY_LOSS = 0.01
MINIMUM_MEAN_ERROR_IMPROVEMENT = 0.02


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


def _metric_cells(summary: dict) -> dict:
    cells = {}
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary.get("datasets", {}).get(dataset_name, {})
        for split, split_summary in dataset.get("splits", {}).items():
            if "metrics" in split_summary:
                cells[f"{dataset_name}/{split}"] = split_summary["metrics"]
            for skill, metrics in split_summary.get("per_skill", {}).items():
                cells[f"{dataset_name}/{split}/{skill}"] = metrics
    return cells


def _retained_comparison(baseline: dict, candidate: dict) -> tuple[list, dict]:
    failures, comparisons = [], {}
    before_cells, after_cells = _metric_cells(baseline), _metric_cells(candidate)
    if set(before_cells) != set(after_cells):
        return ["baseline and candidate retained cells differ"], comparisons
    for label in sorted(before_cells):
        before, after = before_cells[label], after_cells[label]
        begin_change = float(after["gold_begin_as_inside_rate"]) - float(
            before["gold_begin_as_inside_rate"]
        )
        outside_change = float(after["gold_outside_as_inside_rate"]) - float(
            before["gold_outside_as_inside_rate"]
        )
        type_change = float(after["positive_type_accuracy"]) - float(
            before["positive_type_accuracy"]
        )
        end_spill = float(after["end_spill_rate"])
        comparisons[label] = {
            "gold_begin_as_inside_change": begin_change,
            "gold_outside_as_inside_change": outside_change,
            "positive_type_accuracy_change": type_change,
            "candidate_end_spill_rate": end_spill,
        }
        if begin_change > MAX_BEGIN_AS_INSIDE_INCREASE:
            failures.append(f"{label} B->I increased {begin_change:.3f}")
        if outside_change > MAX_OUTSIDE_AS_INSIDE_INCREASE:
            failures.append(f"{label} O->I increased {outside_change:.3f}")
        if type_change < -MAX_TYPE_ACCURACY_LOSS:
            failures.append(f"{label} type accuracy fell {-type_change:.3f}")
        if end_spill > END_SPILL_CEILING:
            failures.append(f"{label} end spill {end_spill:.3f} exceeds 0.010")
    for split in TARGET_SPLITS:
        label = f"phase31_regression/{split}/scene_route"
        metrics = after_cells.get(label)
        if metrics is None or float(metrics["gold_begin_as_inside_rate"]) > 0.02:
            failures.append(f"{label} does not meet the 0.020 B->I ceiling")
    return failures, comparisons


def identity_invariance_decision(
    crossed_summary: dict, baseline_retained: dict, candidate_retained: dict
) -> dict:
    crossed = deepcopy(crossed_summary)
    baseline = deepcopy(baseline_retained)
    candidate = deepcopy(candidate_retained)
    invalid = []
    if crossed.get("paired_identity_invariance_audit_version") != 1:
        invalid.append("paired identity-invariance audit version is invalid")
    if crossed.get("decoder_changes") != "none":
        invalid.append("Phase 36b unexpectedly changed decoding")
    phase36a = crossed.get("phase36a_inputs", {})
    if phase36a.get("decision_branch") != "trained_identity_memorization_dominant":
        invalid.append("Phase 36a did not authorize identity invariance")
    if any(
        phase36a.get(field) is not False
        for field in ("checkpoint_promotion_authorized", "full_phase34_evaluation_authorized")
    ):
        invalid.append("Phase 36a authorization provenance is invalid")

    models = crossed.get("models", {})
    phase35 = models.get("phase35", {}).get("checkpoint", {})
    phase36b = models.get("phase36b", {}).get("checkpoint", {})
    if phase35.get("sha256") != phase36a.get("phase35_checkpoint_sha256"):
        invalid.append("Phase 35 crossed baseline changed")
    if phase35.get("checkpoint_eligible") is not False:
        invalid.append("Phase 35 baseline eligibility changed")
    for label, checkpoint in (("Phase 35", phase35), ("Phase 36b", phase36b)):
        if checkpoint.get("architecture") != "explicit_offset_candidate_proposer":
            invalid.append(f"{label} architecture changed")
        if checkpoint.get("boundary_objective_version") != 1:
            invalid.append(f"{label} boundary objective changed")
        if abs(float(checkpoint.get("boundary_loss_weight", -1.0)) - 1.0) > 1e-9:
            invalid.append(f"{label} boundary loss weight changed")
        if checkpoint.get("boundary_counterbalance_version") != 2:
            invalid.append(f"{label} counterbalance version changed")
        if any(checkpoint.get(field) is not None for field in (
            "token_width_geometry_version", "token_end_geometry_version",
            "factorized_boundary_type_version",
        )):
            invalid.append(f"{label} proposer geometry changed")
    if phase35.get("paired_identity_invariance_version") is not None:
        invalid.append("Phase 35 unexpectedly used identity invariance")
    if phase36b.get("paired_identity_invariance_version") != 1:
        invalid.append("Phase 36b checkpoint lacks invariance provenance")
    if abs(float(phase36b.get("identity_invariance_loss_weight", -1.0)) - 1.0) > 1e-9:
        invalid.append("Phase 36b identity-invariance weight changed")
    for field in ("tokenizer_sha256", "counterbalance_manifest_sha256",
                  "parent_phase33_checkpoint", "parent_phase33_step"):
        if phase35.get(field) != phase36b.get(field):
            invalid.append(f"Phase 35/36b {field} differs")

    pool = crossed.get("identity_swap_pool", {})
    pool_hash = canonical_json_sha256(pool)
    if crossed.get("identity_swap_pool_sha256") != pool_hash:
        invalid.append("summary swap-pool hash changed")
    if phase36b.get("identity_invariance_swap_pool_sha256") != pool_hash:
        invalid.append("checkpoint swap-pool hash changed")

    groups, all_ids = {}, set()
    for group, expected_size in EXPECTED_GROUP_SIZES.items():
        entries = crossed.get("focus_identities", {}).get(group, ())
        ids = {int(entry.get("token_id", -1)) for entry in entries}
        if len(entries) != expected_size or len(ids) != expected_size or all_ids & ids:
            invalid.append(f"Phase 36b {group} identities are invalid")
        groups[group] = ids
        all_ids.update(ids)
    if not PREVIOUSLY_SEVERE_IDENTITIES <= all_ids:
        invalid.append("a previously-severe identity is missing")
    if set(pool.get("excluded_registered_token_ids", ())) != all_ids:
        invalid.append("swap pool does not exclude exactly the registered IDs")

    for split in TARGET_SPLITS:
        panel = crossed.get("panel_validation", {}).get(split, {})
        if int(panel.get("invalid_contexts", -1)) != 0 or int(
            panel.get("identities", 0)
        ) != 14:
            invalid.append(f"Phase 36b {split} panel is invalid")

    before_cells = _identity_cells(crossed, "phase35")
    after_cells = _identity_cells(crossed, "phase36b")
    expected_cells = {(split, token_id) for split in TARGET_SPLITS for token_id in all_ids}
    if set(before_cells) != expected_cells or set(after_cells) != expected_cells:
        invalid.append("crossed identity cells are incomplete")

    if baseline.get("checkpoint_sha256") != phase35.get("sha256"):
        invalid.append("baseline retained checkpoint differs")
    if candidate.get("checkpoint_sha256") != phase36b.get("sha256"):
        invalid.append("candidate retained checkpoint differs")
    if candidate.get("checkpoint_paired_identity_invariance_version") != 1:
        invalid.append("candidate retained invariance version differs")
    if candidate.get("checkpoint_identity_invariance_swap_pool_sha256") != pool_hash:
        invalid.append("candidate retained swap-pool hash differs")

    retained_failures, retained_comparisons = _retained_comparison(baseline, candidate)
    if candidate.get("checkpoint_eligible") is not True:
        retained_failures.append("Phase 36b training-panel checkpoint is ineligible")

    token_results, span_regressions = {}, []
    for token_id in sorted(all_ids):
        group = next(group for group, ids in groups.items() if token_id in ids)
        split_results = {}
        for split in TARGET_SPLITS:
            before, after = before_cells.get((split, token_id), {}), after_cells.get((split, token_id), {})
            focus = after.get("focus", {}).get("overall", {})
            if min(int(focus.get("gold_begin_count", 0)), int(focus.get("gold_inside_count", 0))) < 20:
                invalid.append(f"{split} token {token_id} lacks B/I support")
            before_error = float(before.get("focus", {}).get("overall", {}).get("begin_as_inside_rate", 1.0))
            after_error = float(focus.get("begin_as_inside_rate", 1.0))
            before_metrics, after_metrics = before.get("metrics", {}), after.get("metrics", {})
            precision_change = float(after_metrics.get("exact_span_precision", 0.0)) - float(before_metrics.get("exact_span_precision", 0.0))
            recall_change = float(after_metrics.get("exact_span_recall", 0.0)) - float(before_metrics.get("exact_span_recall", 0.0))
            if precision_change < -0.02 or recall_change < -0.02:
                span_regressions.append(f"{split} token {token_id} complete-span metric fell")
            split_results[split] = {
                "phase35_begin_as_inside_rate": before_error,
                "phase36b_begin_as_inside_rate": after_error,
                "begin_as_inside_change": after_error - before_error,
            }
        token_results[str(token_id)] = {
            "group": group,
            "previously_severe": token_id in PREVIOUSLY_SEVERE_IDENTITIES,
            "splits": split_results,
            "passes_both_splits": all(
                split_results[split]["phase36b_begin_as_inside_rate"] <= 0.02
                for split in TARGET_SPLITS
            ),
        }

    trained_passing = sum(token_results[str(i)]["passes_both_splits"] for i in groups.get("trained", set()))
    severe_passing = sum(token_results[str(i)]["passes_both_splits"] for i in PREVIOUSLY_SEVERE_IDENTITIES if str(i) in token_results)
    changes = [token_results[str(i)]["splits"][split]["begin_as_inside_change"] for i in PREVIOUSLY_SEVERE_IDENTITIES if str(i) in token_results for split in TARGET_SPLITS]
    mean_improvement = -sum(changes) / len(changes) if changes else 0.0
    safety_failures = retained_failures + span_regressions
    if invalid:
        branch = "invalid_identity_invariance_comparison"
        next_action = "Repair provenance before interpreting Phase 36b."
    elif trained_passing < 8 or safety_failures:
        branch = "identity_invariance_regressed_trained"
        next_action = "Reject this setting because a retained safety condition regressed."
    elif severe_passing == 4:
        branch = "identity_invariance_diagnostic_pass"
        next_action = "Run the unchanged inherited full Phase 34 gate."
    elif severe_passing > 0 or mean_improvement >= 0.02:
        branch = "identity_invariance_partial_transfer"
        next_action = "Report per-identity transfer; do not promote."
    else:
        branch = "identity_invariance_no_transfer"
        next_action = "Reject paired identity invariance at this setting."
    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "retained_regression_failures": retained_failures,
        "crossed_span_regressions": span_regressions,
        "retained_comparisons": retained_comparisons,
        "token_results": token_results,
        "trained_identities_passing": trained_passing,
        "previously_severe_identities_passing": severe_passing,
        "mean_previously_severe_error_improvement": mean_improvement,
        "candidate_checkpoint_eligible": candidate.get("checkpoint_eligible") is True,
        "thresholds": {"focus_error_ceiling": 0.02, "minimum_class_support": 20,
                       "maximum_span_metric_drop": 0.02,
                       "minimum_mean_error_improvement": 0.02,
                       "identity_invariance_loss_weight": 1.0},
        "full_phase34_evaluation_authorized": branch == "identity_invariance_diagnostic_pass",
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
