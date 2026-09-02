"""Dependency-free accounting and decisions for the Phase 34c audit."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Union


TAG_CLASSES = ("O", "B", "I")
CURRENT_CLASS_WEIGHTS = {"O": 0.05, "B": 1.0, "I": 0.5}
ORPHAN_OUTSIDE_DOMINANCE_FLOOR = 0.80
SAFE_BEGIN_AS_INSIDE_CEILING = 0.05
BOUNDARY_BEGIN_AS_INSIDE_FLOOR = 0.10


def collapsed_tag(tag: int) -> str:
    """Collapse a type-specific Phase 34 tag to O, B, or I."""

    if tag < 0:
        raise ValueError("supervised tags cannot be negative")
    if tag == 0:
        return "O"
    return "B" if (tag - 1) % 2 == 0 else "I"


def tag_type_index(tag: int):
    if tag <= 0:
        return None
    return (tag - 1) // 2


def new_tag_audit() -> dict:
    return {
        "examples": 0,
        "supervised_bytes": 0,
        "gold_tag_counts": Counter(),
        "predicted_tag_counts": Counter(),
        "confusion_matrix": {
            gold: Counter() for gold in TAG_CLASSES
        },
        "nll_sum_by_gold_class": Counter(),
        "orphan_inside_tags": 0,
        "orphan_inside_by_gold_class": Counter(),
        "mismatched_inside_tags": 0,
        "mismatched_inside_by_gold_class": Counter(),
        "gold_begin_positions": 0,
        "gold_begin_as_inside": 0,
        "pre_start_opportunities": 0,
        "pre_start_bleed": 0,
        "end_spill_opportunities": 0,
        "end_spill": 0,
        "positive_type_comparisons": 0,
        "positive_type_correct": 0,
    }


def add_tag_sequence(
    audit: dict,
    gold_tags,
    predicted_tags,
    negative_log_likelihoods,
) -> None:
    """Add one prompt's byte tags and per-byte gold NLL to an audit."""

    gold_tags = tuple(int(tag) for tag in gold_tags)
    predicted_tags = tuple(int(tag) for tag in predicted_tags)
    nlls = tuple(float(value) for value in negative_log_likelihoods)
    if not (len(gold_tags) == len(predicted_tags) == len(nlls)):
        raise ValueError("gold tags, predictions, and NLLs must align")
    if any(tag < 0 for tag in gold_tags + predicted_tags):
        raise ValueError("tag audit accepts supervised nonnegative tags only")

    audit["examples"] += 1
    audit["supervised_bytes"] += len(gold_tags)
    gold_classes = tuple(collapsed_tag(tag) for tag in gold_tags)
    predicted_classes = tuple(collapsed_tag(tag) for tag in predicted_tags)
    grammar_active_type = None

    for gold_tag, predicted_tag, gold_class, predicted_class, nll in zip(
        gold_tags,
        predicted_tags,
        gold_classes,
        predicted_classes,
        nlls,
    ):
        audit["gold_tag_counts"][gold_class] += 1
        audit["predicted_tag_counts"][predicted_class] += 1
        audit["confusion_matrix"][gold_class][predicted_class] += 1
        audit["nll_sum_by_gold_class"][gold_class] += nll

        if gold_class == "B":
            audit["gold_begin_positions"] += 1
            if predicted_class == "I":
                audit["gold_begin_as_inside"] += 1

        gold_type = tag_type_index(gold_tag)
        predicted_type = tag_type_index(predicted_tag)
        if gold_type is not None and predicted_type is not None:
            audit["positive_type_comparisons"] += 1
            if gold_type == predicted_type:
                audit["positive_type_correct"] += 1

        if predicted_class == "O":
            grammar_active_type = None
        elif predicted_class == "B":
            grammar_active_type = predicted_type
        elif grammar_active_type is None:
            audit["orphan_inside_tags"] += 1
            audit["orphan_inside_by_gold_class"][gold_class] += 1
        elif predicted_type != grammar_active_type:
            audit["mismatched_inside_tags"] += 1
            audit["mismatched_inside_by_gold_class"][gold_class] += 1
            grammar_active_type = None

    for position, gold_class in enumerate(gold_classes):
        if (
            gold_class == "B"
            and position > 0
            and gold_classes[position - 1] == "O"
        ):
            audit["pre_start_opportunities"] += 1
            if predicted_classes[position - 1] == "I":
                audit["pre_start_bleed"] += 1
        if (
            gold_class in {"B", "I"}
            and position + 1 < len(gold_classes)
            and gold_classes[position + 1] == "O"
        ):
            audit["end_spill_opportunities"] += 1
            if predicted_classes[position + 1] == "I":
                audit["end_spill"] += 1


def merge_tag_audit(destination: dict, source: dict) -> None:
    """Merge one raw audit accumulator into another in place."""

    scalar_fields = (
        "examples",
        "supervised_bytes",
        "orphan_inside_tags",
        "mismatched_inside_tags",
        "gold_begin_positions",
        "gold_begin_as_inside",
        "pre_start_opportunities",
        "pre_start_bleed",
        "end_spill_opportunities",
        "end_spill",
        "positive_type_comparisons",
        "positive_type_correct",
    )
    counter_fields = (
        "gold_tag_counts",
        "predicted_tag_counts",
        "nll_sum_by_gold_class",
        "orphan_inside_by_gold_class",
        "mismatched_inside_by_gold_class",
    )
    for field in scalar_fields:
        destination[field] += source[field]
    for field in counter_fields:
        destination[field].update(source[field])
    for gold_class in TAG_CLASSES:
        destination["confusion_matrix"][gold_class].update(
            source["confusion_matrix"][gold_class]
        )


def _rate(
    numerator: Union[int, float], denominator: Union[int, float]
) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_tag_audit(
    audit: dict,
    class_weights: Optional[dict] = None,
) -> dict:
    """Convert raw counters into stable JSON-compatible Phase 34c metrics."""

    weights = dict(CURRENT_CLASS_WEIGHTS)
    if class_weights is not None:
        weights.update(class_weights)
    counts = {
        tag_class: int(audit["gold_tag_counts"][tag_class])
        for tag_class in TAG_CLASSES
    }
    predicted_counts = {
        tag_class: int(audit["predicted_tag_counts"][tag_class])
        for tag_class in TAG_CLASSES
    }
    confusion = {
        gold: {
            predicted: int(audit["confusion_matrix"][gold][predicted])
            for predicted in TAG_CLASSES
        }
        for gold in TAG_CLASSES
    }
    normalized = {
        gold: {
            predicted: _rate(confusion[gold][predicted], counts[gold])
            for predicted in TAG_CLASSES
        }
        for gold in TAG_CLASSES
    }
    target_weight_mass = {
        tag_class: counts[tag_class] * float(weights[tag_class])
        for tag_class in TAG_CLASSES
    }
    nll_sum = {
        tag_class: float(audit["nll_sum_by_gold_class"][tag_class])
        for tag_class in TAG_CLASSES
    }
    weighted_nll = {
        tag_class: nll_sum[tag_class] * float(weights[tag_class])
        for tag_class in TAG_CLASSES
    }
    total_weight_mass = sum(target_weight_mass.values())
    total_weighted_nll = sum(weighted_nll.values())
    positive_weight_mass = (
        counts["B"] * float(weights["B"])
        + counts["I"] * float(weights["I"])
    )
    count_balanced_outside_weight = _rate(positive_weight_mass, counts["O"])
    positive_weighted_nll_without_outside = (
        nll_sum["B"] * float(weights["B"])
        + nll_sum["I"] * float(weights["I"])
    )
    nll_balanced_outside_weight = _rate(
        positive_weighted_nll_without_outside, nll_sum["O"]
    )
    orphan_by_gold = {
        tag_class: int(audit["orphan_inside_by_gold_class"][tag_class])
        for tag_class in TAG_CLASSES
    }
    mismatch_by_gold = {
        tag_class: int(audit["mismatched_inside_by_gold_class"][tag_class])
        for tag_class in TAG_CLASSES
    }
    orphan_total = int(audit["orphan_inside_tags"])
    return {
        "examples": int(audit["examples"]),
        "supervised_bytes": int(audit["supervised_bytes"]),
        "gold_tag_counts": counts,
        "predicted_tag_counts": predicted_counts,
        "confusion_matrix": confusion,
        "row_normalized_confusion_matrix": normalized,
        "gold_outside_as_inside_rate": normalized["O"]["I"],
        "gold_begin_as_inside_count": int(audit["gold_begin_as_inside"]),
        "gold_begin_as_inside_rate": _rate(
            audit["gold_begin_as_inside"], audit["gold_begin_positions"]
        ),
        "orphan_inside_tags": orphan_total,
        "orphan_inside_by_gold_class": orphan_by_gold,
        "orphan_inside_gold_outside_fraction": _rate(
            orphan_by_gold["O"], orphan_total
        ),
        "mismatched_inside_tags": int(audit["mismatched_inside_tags"]),
        "mismatched_inside_by_gold_class": mismatch_by_gold,
        "pre_start_bleed_count": int(audit["pre_start_bleed"]),
        "pre_start_bleed_opportunities": int(
            audit["pre_start_opportunities"]
        ),
        "pre_start_bleed_rate": _rate(
            audit["pre_start_bleed"], audit["pre_start_opportunities"]
        ),
        "end_spill_count": int(audit["end_spill"]),
        "end_spill_opportunities": int(audit["end_spill_opportunities"]),
        "end_spill_rate": _rate(
            audit["end_spill"], audit["end_spill_opportunities"]
        ),
        "positive_type_accuracy": _rate(
            audit["positive_type_correct"],
            audit["positive_type_comparisons"],
        ),
        "loss_mass": {
            "class_weights": {
                tag_class: float(weights[tag_class])
                for tag_class in TAG_CLASSES
            },
            "target_weight_mass": target_weight_mass,
            "target_weight_mass_fraction": {
                tag_class: _rate(
                    target_weight_mass[tag_class], total_weight_mass
                )
                for tag_class in TAG_CLASSES
            },
            "unweighted_nll_sum": nll_sum,
            "mean_unweighted_nll": {
                tag_class: _rate(nll_sum[tag_class], counts[tag_class])
                for tag_class in TAG_CLASSES
            },
            "weighted_nll_sum": weighted_nll,
            "weighted_nll_fraction": {
                tag_class: _rate(
                    weighted_nll[tag_class], total_weighted_nll
                )
                for tag_class in TAG_CLASSES
            },
            "weighted_mean_loss": _rate(
                total_weighted_nll, total_weight_mass
            ),
            "count_balanced_outside_weight": (
                count_balanced_outside_weight
            ),
            "nll_balanced_outside_weight": nll_balanced_outside_weight,
        },
    }


def _decision_cells(summary: dict):
    for dataset_name, dataset in summary["datasets"].items():
        for split, split_summary in dataset["splits"].items():
            yield f"{dataset_name}/{split}", split_summary["metrics"]
            for skill, metrics in split_summary["per_skill"].items():
                yield f"{dataset_name}/{split}/{skill}", metrics


def tag_confusion_decision(summary: dict) -> dict:
    """Choose the next single variable without authorizing a training run."""

    overall = summary["overall"]
    cells = tuple(_decision_cells(summary))
    worst_label, worst_metrics = max(
        cells,
        key=lambda item: item[1]["gold_begin_as_inside_rate"],
    )
    maximum_begin_as_inside = worst_metrics["gold_begin_as_inside_rate"]
    outside_fraction = overall["orphan_inside_gold_outside_fraction"]
    orphan_total = overall["orphan_inside_tags"]

    if orphan_total == 0:
        branch = "no_orphan_inside_problem_detected"
        next_variable = "none"
        next_action = (
            "Do not change decoding or class weights; the reported Phase 34b "
            "failure is not reproduced by this checkpoint and dataset."
        )
    elif maximum_begin_as_inside >= BOUNDARY_BEGIN_AS_INSIDE_FLOOR:
        branch = "boundary_start_objective_indicated"
        next_variable = "boundary_start_supervision"
        next_action = (
            "Keep the permissive decoder and all O/B/I class weights fixed. "
            "The next controlled experiment may add only an explicit "
            "boundary-start objective; do not promote this checkpoint."
        )
    elif (
        outside_fraction >= ORPHAN_OUTSIDE_DOMINANCE_FLOOR
        and maximum_begin_as_inside <= SAFE_BEGIN_AS_INSIDE_CEILING
    ):
        branch = "outside_weight_ablation_indicated"
        next_variable = "outside_class_weight"
        next_action = (
            "Keep the permissive decoder and B/I weights fixed. The next "
            "controlled experiment may change only the O class weight, using "
            "the reported balance candidates as diagnostics rather than an "
            "automatic setting; do not promote this checkpoint."
        )
    else:
        branch = "mixed_or_ambiguous_boundary_errors"
        next_variable = "none"
        next_action = (
            "Do not retrain yet. Neither an O-only weight change nor a "
            "boundary-start objective is isolated by the pre-registered "
            "thresholds; inspect the per-cell confusion report first."
        )
    return {
        "branch": branch,
        "next_single_variable": next_variable,
        "orphan_inside_tags": orphan_total,
        "orphan_inside_gold_outside_fraction": outside_fraction,
        "orphan_outside_dominance_floor": (
            ORPHAN_OUTSIDE_DOMINANCE_FLOOR
        ),
        "maximum_gold_begin_as_inside_rate": maximum_begin_as_inside,
        "maximum_gold_begin_as_inside_cell": worst_label,
        "safe_gold_begin_as_inside_ceiling": SAFE_BEGIN_AS_INSIDE_CEILING,
        "boundary_objective_gold_begin_as_inside_floor": (
            BOUNDARY_BEGIN_AS_INSIDE_FLOOR
        ),
        "count_balanced_outside_weight": overall["loss_mass"][
            "count_balanced_outside_weight"
        ],
        "nll_balanced_outside_weight": overall["loss_mass"][
            "nll_balanced_outside_weight"
        ],
        "decoder_change_authorized": False,
        "training_change_applied": False,
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
