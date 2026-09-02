"""Dependency-free metrics and decision logic for the Phase 34b ablation."""

from __future__ import annotations


EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")
PERMISSIVE_DECODE_POLICY = "permissive"
STRICT_DECODE_POLICY = "strict"
HELPFUL_PRECISION_GAIN = 0.20
MAX_RECALL_LOSS = 0.02
MAX_ANSWER_RECALL_LOSS = 0.01


def _delta_bucket(delta: int) -> str:
    if delta <= -9:
        return "<=-9"
    if delta >= 9:
        return ">=+9"
    return f"{delta:+d}"


def false_positive_boundary_displacements(
    gold_spans, predicted_spans
) -> tuple[str, ...]:
    """Align each false positive to the nearest same-type gold span."""

    gold_exact = {
        (span.byte_start, span.byte_end, span.value_type) for span in gold_spans
    }
    displacements = []
    for predicted in predicted_spans:
        identity = (
            predicted.byte_start,
            predicted.byte_end,
            predicted.value_type,
        )
        if identity in gold_exact:
            continue
        compatible = tuple(
            gold for gold in gold_spans if gold.value_type == predicted.value_type
        )
        if not compatible:
            displacements.append("no_same_type_gold")
            continue
        nearest = min(
            compatible,
            key=lambda gold: (
                abs(predicted.byte_start - gold.byte_start)
                + abs(predicted.byte_end - gold.byte_end),
                gold.byte_start,
                gold.byte_end,
            ),
        )
        start_delta = predicted.byte_start - nearest.byte_start
        end_delta = predicted.byte_end - nearest.byte_end
        displacements.append(
            f"start={_delta_bucket(start_delta)},"
            f"end={_delta_bucket(end_delta)}"
        )
    return tuple(displacements)


def _metric_failures(label: str, metrics: dict, per_skill: bool) -> list[str]:
    failures = []
    precision_recall_floor = 0.95 if per_skill else 0.98
    answer_floor = 0.98 if per_skill else 0.995
    checks = (
        ("exact_span_precision", precision_recall_floor, True),
        ("exact_span_recall", precision_recall_floor, True),
        ("answer_candidate_recall", answer_floor, True),
        ("boundary_type_accuracy", 0.99, True),
        ("offset_validity_rate", 1.0, True),
        ("proposal_overflow_rate", 0.0, False),
    )
    for name, threshold, minimum in checks:
        actual = float(metrics[name])
        failed = actual < threshold if minimum else actual > threshold
        if failed:
            comparator = "below" if minimum else "above"
            failures.append(
                f"{label} {name} {actual:.3f} is {comparator} "
                f"{threshold:.3f}"
            )
    return failures


def strict_policy_gate_failures(summary: dict) -> tuple[str, ...]:
    failures = []
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary["datasets"][dataset_name]
        for split in EXPANDED_SPLITS:
            metrics = dataset["splits"][split]["policies"][
                STRICT_DECODE_POLICY
            ]
            label = f"{dataset_name}/{split}"
            failures.extend(_metric_failures(label, metrics, per_skill=False))
            for skill, skill_metrics in sorted(metrics["per_skill"].items()):
                failures.extend(
                    _metric_failures(
                        f"{label}/{skill}", skill_metrics, per_skill=True
                    )
                )
    return tuple(failures)


def decoder_ablation_decision(summary: dict) -> dict:
    permissive = summary["overall"][PERMISSIVE_DECODE_POLICY]
    strict = summary["overall"][STRICT_DECODE_POLICY]
    precision_gain = (
        strict["exact_span_precision"] - permissive["exact_span_precision"]
    )
    recall_change = (
        strict["exact_span_recall"] - permissive["exact_span_recall"]
    )
    answer_change = (
        strict["answer_candidate_recall"]
        - permissive["answer_candidate_recall"]
    )
    gate_failures = strict_policy_gate_failures(summary)
    split_safety = []
    for dataset_name in ("phase31_regression", "phase32"):
        for split in EXPANDED_SPLITS:
            policies = summary["datasets"][dataset_name]["splits"][split][
                "policies"
            ]
            baseline = policies[PERMISSIVE_DECODE_POLICY]
            candidate = policies[STRICT_DECODE_POLICY]
            split_safety.append(
                candidate["exact_span_recall"]
                >= baseline["exact_span_recall"] - MAX_RECALL_LOSS
                and candidate["answer_candidate_recall"]
                >= baseline["answer_candidate_recall"]
                - MAX_ANSWER_RECALL_LOSS
            )

    if not gate_failures:
        branch = "decoder_only_pass"
        next_action = (
            "Adopt strict BIO decoding and run the predicted-candidate "
            "end-to-end evaluation; do not retrain."
        )
    elif precision_gain >= HELPFUL_PRECISION_GAIN and all(split_safety):
        branch = "strict_helpful_but_insufficient"
        next_action = (
            "Retain strict decoding for the next single-variable boundary-loss "
            "experiment; do not promote this checkpoint or run the full gate."
        )
    else:
        branch = "strict_decoder_rejected"
        next_action = (
            "Retain permissive decoding and diagnose boundary supervision; "
            "do not change decoder and loss weights together."
        )
    return {
        "branch": branch,
        "precision_gain": precision_gain,
        "recall_change": recall_change,
        "answer_candidate_recall_change": answer_change,
        "helpful_precision_gain_floor": HELPFUL_PRECISION_GAIN,
        "maximum_recall_loss": MAX_RECALL_LOSS,
        "maximum_answer_candidate_recall_loss": MAX_ANSWER_RECALL_LOSS,
        "strict_gate_failures": list(gate_failures),
        "next_action": next_action,
    }
