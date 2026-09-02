from copy import deepcopy
from math import isclose

from scripts.phase34c_tag_confusion_decision import (
    add_tag_sequence,
    merge_tag_audit,
    new_tag_audit,
    summarize_tag_audit,
    tag_confusion_decision,
)


def _summary(metrics):
    return {
        "overall": deepcopy(metrics),
        "datasets": {
            "phase31_regression": {
                "splits": {
                    "lexical": {
                        "metrics": deepcopy(metrics),
                        "per_skill": {"scene_route": deepcopy(metrics)},
                    }
                }
            }
        },
    }


def _metrics(orphan_total=100, orphan_outside=90, begin_as_inside=0.04):
    return {
        "orphan_inside_tags": orphan_total,
        "orphan_inside_gold_outside_fraction": (
            orphan_outside / orphan_total if orphan_total else 0.0
        ),
        "gold_begin_as_inside_rate": begin_as_inside,
        "loss_mass": {
            "count_balanced_outside_weight": 0.2,
            "nll_balanced_outside_weight": 0.3,
        },
    }


def test_tag_audit_exposes_orphan_origin_and_boundary_bleed():
    audit = new_tag_audit()
    add_tag_sequence(
        audit,
        gold_tags=(0, 1, 2, 0, 3, 4, 0),
        predicted_tags=(2, 2, 2, 2, 3, 4, 0),
        negative_log_likelihoods=(1.0,) * 7,
    )

    metrics = summarize_tag_audit(audit)

    assert metrics["confusion_matrix"] == {
        "O": {"O": 1, "B": 0, "I": 2},
        "B": {"O": 0, "B": 1, "I": 1},
        "I": {"O": 0, "B": 0, "I": 2},
    }
    assert metrics["orphan_inside_tags"] == 4
    assert metrics["orphan_inside_by_gold_class"] == {
        "O": 2,
        "B": 1,
        "I": 1,
    }
    assert metrics["gold_begin_as_inside_rate"] == 0.5
    assert metrics["pre_start_bleed_rate"] == 1.0
    assert metrics["end_spill_rate"] == 0.5
    assert metrics["positive_type_accuracy"] == 1.0
    assert isclose(metrics["loss_mass"]["weighted_mean_loss"], 1.0)
    assert metrics["loss_mass"]["count_balanced_outside_weight"] == 1.0
    assert metrics["loss_mass"]["nll_balanced_outside_weight"] == 1.0


def test_merge_tag_audit_preserves_counts_and_loss_mass():
    first = new_tag_audit()
    second = new_tag_audit()
    add_tag_sequence(first, (0, 1), (0, 1), (0.2, 0.3))
    add_tag_sequence(second, (2, 0), (2, 2), (0.4, 0.5))

    merge_tag_audit(first, second)
    metrics = summarize_tag_audit(first)

    assert metrics["examples"] == 2
    assert metrics["supervised_bytes"] == 4
    assert metrics["gold_tag_counts"] == {"O": 2, "B": 1, "I": 1}
    assert metrics["predicted_tag_counts"] == {"O": 1, "B": 1, "I": 2}
    actual_nll = metrics["loss_mass"]["unweighted_nll_sum"]
    assert all(
        isclose(actual_nll[tag_class], expected)
        for tag_class, expected in {"O": 0.7, "B": 0.3, "I": 0.4}.items()
    )


def test_decision_selects_boundary_objective_for_large_begin_as_inside_rate():
    decision = tag_confusion_decision(_summary(_metrics(begin_as_inside=0.10)))

    assert decision["branch"] == "boundary_start_objective_indicated"
    assert decision["next_single_variable"] == "boundary_start_supervision"
    assert decision["decoder_change_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_decision_selects_o_weight_only_when_orphans_are_outside_dominated():
    decision = tag_confusion_decision(
        _summary(_metrics(orphan_outside=80, begin_as_inside=0.05))
    )

    assert decision["branch"] == "outside_weight_ablation_indicated"
    assert decision["next_single_variable"] == "outside_class_weight"


def test_decision_stops_on_the_pre_registered_ambiguous_band():
    decision = tag_confusion_decision(
        _summary(_metrics(orphan_outside=90, begin_as_inside=0.07))
    )

    assert decision["branch"] == "mixed_or_ambiguous_boundary_errors"
    assert decision["next_single_variable"] == "none"


def test_decision_rejects_weight_changes_when_orphans_are_absent():
    decision = tag_confusion_decision(
        _summary(_metrics(orphan_total=0, orphan_outside=0))
    )

    assert decision["branch"] == "no_orphan_inside_problem_detected"
    assert decision["next_single_variable"] == "none"
