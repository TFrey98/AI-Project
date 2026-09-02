"""Dependency-free Phase 34g token-end premise decision."""

from __future__ import annotations


MINIMUM_TARGET_STARTS = 100
MINIMUM_TARGET_ERRORS = 20
MINIMUM_BUCKET_STARTS = 20
TOKEN_END_ERROR_CAPTURE_FLOOR = 0.80
WIDTH_TWO_TOKEN_END_ERROR_CAPTURE_FLOOR = 0.90
TOKEN_END_RATE_RATIO_FLOOR = 5.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _bucket(rows) -> dict:
    rows = tuple(rows)
    errors = sum(row["baseline_predicted_class"] == "I" for row in rows)
    return {
        "gold_begin_count": len(rows),
        "begin_as_inside_count": errors,
        "begin_as_inside_rate": _rate(errors, len(rows)),
    }


def token_end_premise_decision(rows) -> dict:
    """Test whether the Phase 34e failures concentrate at token ends."""

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
        "token_width",
        "token_byte_offset",
    }
    for index, row in enumerate(target):
        missing = sorted(required - set(row))
        if missing:
            invalid.append(
                f"target row {index} is missing: {', '.join(missing)}"
            )
            continue
        try:
            width = int(row["token_width"])
            offset = int(row["token_byte_offset"])
        except (TypeError, ValueError):
            invalid.append(
                f"target row {index} has non-integer token geometry"
            )
            continue
        if width < 1 or offset < 0 or offset >= width:
            invalid.append(
                f"target row {index} has invalid token geometry "
                f"width={width}, offset={offset}"
            )
        if row["baseline_predicted_class"] not in {"O", "B", "I"}:
            invalid.append(
                f"target row {index} has an invalid predicted class"
            )
    if invalid:
        return {
            "branch": "invalid_token_end_premise",
            "training_authorized": False,
            "invalid_reasons": invalid,
            "failures": [],
            "next_action": (
                "Regenerate the Phase 34e start-row JSONL before training."
            ),
        }

    enriched = tuple(
        {
            **row,
            "is_token_end": (
                int(row["token_byte_offset"]) + 1
                == int(row["token_width"])
            ),
        }
        for row in target
    )
    errors = tuple(
        row
        for row in enriched
        if row["baseline_predicted_class"] == "I"
    )
    token_end = tuple(row for row in enriched if row["is_token_end"])
    not_token_end = tuple(row for row in enriched if not row["is_token_end"])
    width_two = tuple(row for row in enriched if int(row["token_width"]) == 2)
    width_two_errors = tuple(
        row
        for row in width_two
        if row["baseline_predicted_class"] == "I"
    )
    token_end_errors = sum(
        row["baseline_predicted_class"] == "I" for row in token_end
    )
    not_token_end_errors = sum(
        row["baseline_predicted_class"] == "I" for row in not_token_end
    )
    width_two_end_errors = sum(
        row["is_token_end"] for row in width_two_errors
    )
    token_end_rate = _rate(token_end_errors, len(token_end))
    not_token_end_rate = _rate(not_token_end_errors, len(not_token_end))
    rate_ratio = (
        token_end_rate / not_token_end_rate
        if not_token_end_rate
        else (999.0 if token_end_rate else 1.0)
    )
    end_capture = _rate(token_end_errors, len(errors))
    width_two_end_capture = _rate(
        width_two_end_errors, len(width_two_errors)
    )

    failures = []
    checks = (
        (
            len(enriched) >= MINIMUM_TARGET_STARTS,
            f"target starts {len(enriched)} are below {MINIMUM_TARGET_STARTS}",
        ),
        (
            len(errors) >= MINIMUM_TARGET_ERRORS,
            f"target errors {len(errors)} are below {MINIMUM_TARGET_ERRORS}",
        ),
        (
            len(token_end) >= MINIMUM_BUCKET_STARTS,
            f"token-end starts {len(token_end)} are below {MINIMUM_BUCKET_STARTS}",
        ),
        (
            len(not_token_end) >= MINIMUM_BUCKET_STARTS,
            "non-token-end starts "
            f"{len(not_token_end)} are below {MINIMUM_BUCKET_STARTS}",
        ),
        (
            len(width_two) >= MINIMUM_BUCKET_STARTS,
            f"width-2 starts {len(width_two)} are below {MINIMUM_BUCKET_STARTS}",
        ),
        (
            end_capture >= TOKEN_END_ERROR_CAPTURE_FLOOR,
            f"token-end error capture {end_capture:.3f} is below "
            f"{TOKEN_END_ERROR_CAPTURE_FLOOR:.3f}",
        ),
        (
            width_two_end_capture
            >= WIDTH_TWO_TOKEN_END_ERROR_CAPTURE_FLOOR,
            f"width-2 token-end error capture {width_two_end_capture:.3f} "
            f"is below {WIDTH_TWO_TOKEN_END_ERROR_CAPTURE_FLOOR:.3f}",
        ),
        (
            rate_ratio >= TOKEN_END_RATE_RATIO_FLOOR,
            f"token-end error-rate ratio {rate_ratio:.3f} is below "
            f"{TOKEN_END_RATE_RATIO_FLOOR:.3f}",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    if failures:
        branch = "token_identity_audit_indicated"
        next_action = (
            "The existing errors do not isolate token ends strongly enough. "
            "Do not train Phase 34g; audit byte and token identity instead."
        )
    else:
        branch = "token_end_geometry_indicated"
        next_action = (
            "The frozen Phase 34e logits support the registered token-end "
            "intervention. A bounded Phase 34g run is authorized."
        )
    return {
        "branch": branch,
        "training_authorized": not failures,
        "invalid_reasons": [],
        "failures": failures,
        "target": _bucket(enriched),
        "token_end": _bucket(token_end),
        "not_token_end": _bucket(not_token_end),
        "width_two": _bucket(width_two),
        "target_error_token_end_capture": end_capture,
        "width_two_error_token_end_capture": width_two_end_capture,
        "token_end_error_rate_ratio": rate_ratio,
        "thresholds": {
            "minimum_target_starts": MINIMUM_TARGET_STARTS,
            "minimum_target_errors": MINIMUM_TARGET_ERRORS,
            "minimum_bucket_starts": MINIMUM_BUCKET_STARTS,
            "token_end_error_capture_floor": TOKEN_END_ERROR_CAPTURE_FLOOR,
            "width_two_token_end_error_capture_floor": (
                WIDTH_TWO_TOKEN_END_ERROR_CAPTURE_FLOOR
            ),
            "token_end_rate_ratio_floor": TOKEN_END_RATE_RATIO_FLOOR,
        },
        "next_action": next_action,
    }
