"""Dependency-free Phase 34h token-identity/prior decision."""

from __future__ import annotations

from collections import defaultdict


MINIMUM_TARGET_STARTS = 100
MINIMUM_TARGET_ERRORS = 20
MINIMUM_CONTRAST_STARTS = 20
MINIMUM_SPLIT_CONTRAST_STARTS = 10
MINIMUM_REFERENCE_POSITIVE_LABELS = 10
INSIDE_PRIOR_FLOOR = 0.80
PRIOR_ERROR_CAPTURE_FLOOR = 0.60
PRIOR_RATE_RATIO_FLOOR = 3.0
PRIOR_SPLIT_RATE_RATIO_FLOOR = 2.0
DISCOVERY_IDENTITY_STARTS = 5
DISCOVERY_IDENTITY_ERRORS = 3
DISCOVERY_IDENTITY_ERROR_RATE = 0.50
CONFIRMATION_ERROR_CAPTURE_FLOOR = 0.50
CONFIRMATION_RATE_RATIO_FLOOR = 3.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _error(row: dict) -> bool:
    return row["baseline_predicted_class"] == "I"


def _bucket(rows) -> dict:
    rows = tuple(rows)
    errors = sum(_error(row) for row in rows)
    return {
        "gold_begin_count": len(rows),
        "begin_as_inside_count": errors,
        "begin_as_inside_rate": _rate(errors, len(rows)),
    }


def _contrast(rows, selected) -> dict:
    rows = tuple(rows)
    selected = tuple(selected)
    selected_keys = {id(row) for row in selected}
    complement = tuple(row for row in rows if id(row) not in selected_keys)
    selected_metrics = _bucket(selected)
    complement_metrics = _bucket(complement)
    selected_rate = selected_metrics["begin_as_inside_rate"]
    complement_rate = complement_metrics["begin_as_inside_rate"]
    ratio = (
        selected_rate / complement_rate
        if complement_rate
        else (999.0 if selected_rate else 1.0)
    )
    return {
        "selected": selected_metrics,
        "complement": complement_metrics,
        "error_capture": _rate(
            selected_metrics["begin_as_inside_count"],
            _bucket(rows)["begin_as_inside_count"],
        ),
        "error_rate_ratio": ratio,
    }


def _reference_route_prior(statistics: dict) -> tuple[int, float]:
    training = statistics.get("reference", {}).get("train", {})
    typed = training.get("typed_labels", {})
    begin_count = int(typed.get("B:route", 0))
    inside_count = int(typed.get("I:route", 0))
    positive = begin_count + inside_count
    return positive, _rate(inside_count, positive)


def _identity_groups(rows, field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return dict(groups)


def _supported_discovery_values(rows, field: str) -> tuple:
    values = []
    for value, group in _identity_groups(rows, field).items():
        metrics = _bucket(group)
        if (
            metrics["gold_begin_count"] >= DISCOVERY_IDENTITY_STARTS
            and metrics["begin_as_inside_count"]
            >= DISCOVERY_IDENTITY_ERRORS
            and metrics["begin_as_inside_rate"]
            >= DISCOVERY_IDENTITY_ERROR_RATE
        ):
            values.append(value)
    return tuple(sorted(values, key=str))


def _confirmation(rows, field: str, values: tuple) -> dict:
    selected = tuple(row for row in rows if row[field] in values)
    result = _contrast(rows, selected)
    result["discovered_values"] = list(values)
    result["passed"] = (
        result["selected"]["gold_begin_count"]
        >= MINIMUM_CONTRAST_STARTS
        and result["complement"]["gold_begin_count"]
        >= MINIMUM_CONTRAST_STARTS
        and result["error_capture"] >= CONFIRMATION_ERROR_CAPTURE_FLOOR
        and result["error_rate_ratio"] >= CONFIRMATION_RATE_RATIO_FLOOR
    )
    return result


def _token_rows(rows, token_statistics: dict) -> list[dict]:
    grouped = _identity_groups(rows, "token_id")
    total_errors = _bucket(rows)["begin_as_inside_count"]
    summaries = []
    for token_id, group in grouped.items():
        statistics = token_statistics[str(token_id)]
        metrics = _bucket(group)
        reference_positive, reference_inside_share = (
            _reference_route_prior(statistics)
        )
        summaries.append(
            {
                "token_id": token_id,
                "token_bytes_hex": group[0]["token_bytes_hex"],
                "token_utf8": group[0].get("token_utf8"),
                **metrics,
                "error_capture": _rate(
                    metrics["begin_as_inside_count"], total_errors
                ),
                "per_split": {
                    split: _bucket(
                        row for row in group if row["split"] == split
                    )
                    for split in ("lexical", "transfer")
                },
                "reference_route_positive_labels": reference_positive,
                "reference_route_inside_share": reference_inside_share,
                "reference": statistics.get("reference", {}),
                "familiar_phase31_train_val": statistics.get(
                    "familiar_phase31_train_val", {}
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -row["begin_as_inside_count"],
            -row["gold_begin_count"],
            row["token_id"],
        ),
    )


def _invalid_decision(reasons) -> dict:
    return {
        "branch": "invalid_token_identity_audit",
        "training_authorized": False,
        "checkpoint_promotion_authorized": False,
        "invalid_reasons": list(reasons),
        "next_action": (
            "Repair the Phase 34e row/token join before changing the "
            "model or data."
        ),
    }


def token_identity_decision(rows, token_statistics: dict) -> dict:
    """Select the next single variable from the frozen Phase 34e rows."""

    target = tuple(
        row
        for row in rows
        if row.get("dataset") == "phase31_regression"
        and row.get("split") in {"lexical", "transfer"}
        and row.get("skill") == "scene_route"
    )
    invalid = []
    if not target:
        invalid.append(
            "input contains no Phase 31 lexical/transfer scene_route rows"
        )
    required = {
        "baseline_predicted_class",
        "token_id",
        "token_bytes_hex",
        "token_width",
        "token_byte_offset",
        "first_byte",
    }
    for index, row in enumerate(target):
        missing = sorted(required - set(row))
        if missing:
            invalid.append(
                f"target row {index} is missing: {', '.join(missing)}"
            )
            continue
        if row["baseline_predicted_class"] not in {"O", "B", "I"}:
            invalid.append(
                f"target row {index} has an invalid predicted class"
            )
        try:
            width = int(row["token_width"])
            offset = int(row["token_byte_offset"])
            token_id = str(int(row["token_id"]))
        except (TypeError, ValueError):
            invalid.append(f"target row {index} has invalid token metadata")
            continue
        if width < 1 or offset < 0 or offset >= width:
            invalid.append(
                f"target row {index} has invalid token geometry"
            )
        elif (
            width == 2
            and offset == 0
            and token_id not in token_statistics
        ):
            invalid.append(
                f"target row {index} token {token_id} has no statistics"
            )
    if invalid:
        return _invalid_decision(invalid)

    focus = tuple(
        row
        for row in target
        if int(row.get("token_width", 0)) == 2
        and int(row.get("token_byte_offset", -1)) == 0
    )
    focus_metrics = _bucket(focus)
    if focus_metrics["gold_begin_count"] < MINIMUM_TARGET_STARTS:
        invalid.append(
            "width-2/offset-0 target starts "
            f"{focus_metrics['gold_begin_count']} are below "
            f"{MINIMUM_TARGET_STARTS}"
        )
    if focus_metrics["begin_as_inside_count"] < MINIMUM_TARGET_ERRORS:
        invalid.append(
            "width-2/offset-0 target errors "
            f"{focus_metrics['begin_as_inside_count']} are below "
            f"{MINIMUM_TARGET_ERRORS}"
        )
    if invalid:
        return _invalid_decision(invalid)

    conflict_ids = []
    for token_id in sorted({int(row["token_id"]) for row in focus}):
        positive, inside_share = _reference_route_prior(
            token_statistics[str(token_id)]
        )
        if (
            positive >= MINIMUM_REFERENCE_POSITIVE_LABELS
            and inside_share >= INSIDE_PRIOR_FLOOR
        ):
            conflict_ids.append(token_id)
    prior_selected = tuple(
        row for row in focus if int(row["token_id"]) in conflict_ids
    )
    prior_contrast = _contrast(focus, prior_selected)
    prior_split_contrasts = {
        split: _contrast(
            (row for row in focus if row["split"] == split),
            (
                row
                for row in focus
                if row["split"] == split
                and int(row["token_id"]) in conflict_ids
            ),
        )
        for split in ("lexical", "transfer")
    }
    prior_passed = (
        prior_contrast["selected"]["gold_begin_count"]
        >= MINIMUM_CONTRAST_STARTS
        and prior_contrast["complement"]["gold_begin_count"]
        >= MINIMUM_CONTRAST_STARTS
        and prior_contrast["error_capture"]
        >= PRIOR_ERROR_CAPTURE_FLOOR
        and prior_contrast["error_rate_ratio"] >= PRIOR_RATE_RATIO_FLOOR
        and all(
            contrast["selected"]["gold_begin_count"]
            >= MINIMUM_SPLIT_CONTRAST_STARTS
            and contrast["complement"]["gold_begin_count"]
            >= MINIMUM_SPLIT_CONTRAST_STARTS
            and contrast["error_rate_ratio"]
            >= PRIOR_SPLIT_RATE_RATIO_FLOOR
            for contrast in prior_split_contrasts.values()
        )
    )

    lexical = tuple(row for row in focus if row["split"] == "lexical")
    transfer = tuple(row for row in focus if row["split"] == "transfer")
    discovered_token_ids = _supported_discovery_values(
        lexical, "token_id"
    )
    token_confirmation = _confirmation(
        transfer, "token_id", discovered_token_ids
    )
    discovered_first_bytes = _supported_discovery_values(
        lexical, "first_byte"
    )
    byte_confirmation = _confirmation(
        transfer, "first_byte", discovered_first_bytes
    )

    if prior_passed:
        branch = "train_label_prior_conflict_indicated"
        next_action = (
            "Counterbalance only the implicated route-token boundary "
            "labels, then rerun the unchanged Phase 34d objective."
        )
    elif token_confirmation["passed"]:
        branch = "token_identity_concentration_indicated"
        next_action = (
            "The same exact BPE identities fail in lexical discovery and "
            "transfer confirmation without a sufficient label-prior "
            "explanation. Test a controlled token/byte representation "
            "interaction next."
        )
    elif byte_confirmation["passed"]:
        branch = "byte_identity_concentration_indicated"
        next_action = (
            "The exact tokens do not transfer, but their first-byte "
            "identity does. Test that byte-identity interaction as the "
            "next single variable."
        )
    else:
        branch = "factorized_boundary_type_head_indicated"
        next_action = (
            "No independently supported identity or label-prior bucket "
            "explains the residual. Separate boundary detection from type "
            "classification next."
        )

    return {
        "branch": branch,
        "training_authorized": False,
        "checkpoint_promotion_authorized": False,
        "invalid_reasons": [],
        "target": _bucket(target),
        "width_2_offset_0_target": focus_metrics,
        "reference_inside_prior": {
            "token_ids": conflict_ids,
            "contrast": prior_contrast,
            "per_split": prior_split_contrasts,
            "passed": prior_passed,
        },
        "lexical_to_transfer_token_confirmation": token_confirmation,
        "lexical_to_transfer_first_byte_confirmation": byte_confirmation,
        "token_identities": _token_rows(focus, token_statistics),
        "thresholds": {
            "minimum_target_starts": MINIMUM_TARGET_STARTS,
            "minimum_target_errors": MINIMUM_TARGET_ERRORS,
            "minimum_contrast_starts": MINIMUM_CONTRAST_STARTS,
            "minimum_split_contrast_starts": (
                MINIMUM_SPLIT_CONTRAST_STARTS
            ),
            "minimum_reference_positive_labels": (
                MINIMUM_REFERENCE_POSITIVE_LABELS
            ),
            "inside_prior_floor": INSIDE_PRIOR_FLOOR,
            "prior_error_capture_floor": PRIOR_ERROR_CAPTURE_FLOOR,
            "prior_rate_ratio_floor": PRIOR_RATE_RATIO_FLOOR,
            "prior_split_rate_ratio_floor": (
                PRIOR_SPLIT_RATE_RATIO_FLOOR
            ),
            "discovery_identity_starts": DISCOVERY_IDENTITY_STARTS,
            "discovery_identity_errors": DISCOVERY_IDENTITY_ERRORS,
            "discovery_identity_error_rate": (
                DISCOVERY_IDENTITY_ERROR_RATE
            ),
            "confirmation_error_capture_floor": (
                CONFIRMATION_ERROR_CAPTURE_FLOOR
            ),
            "confirmation_rate_ratio_floor": (
                CONFIRMATION_RATE_RATIO_FLOOR
            ),
        },
        "next_action": next_action,
    }
