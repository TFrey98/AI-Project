from copy import deepcopy

from scripts.gate_unified_typed_span_resolver import gate_failures


def _metrics(skill: str, widths: tuple[str, ...]):
    base = {
        "end_to_end_resolve_accuracy": 1.0,
        "counterfactual_pair_resolve_accuracy": 1.0,
        "real_candidate_top1_accuracy": 1.0,
        "clarify_accuracy": 1.0,
        "clarify_sentinel_accuracy": 1.0,
        "generate_accuracy": 1.0,
        "mode_accuracy": 1.0,
        "value_missing_rate": 0.0,
        "wrong_alternative_rate": 0.0,
        "no_support_false_positive_rate": 0.0,
    }
    return {
        **base,
        "per_skill": {skill: dict(base)},
        "per_candidate_width": {width: dict(base) for width in widths},
        "per_skill_candidate_width": {
            skill: {width: dict(base) for width in widths}
        },
    }


def _summary():
    splits = ("train", "val", "lexical", "paraphrase", "transfer")
    return {
        "phase31_regression": {
            "splits": {
                split: _metrics("scene_route", ("2",)) for split in splits
            }
        },
        "phase32": {
            "splits": {
                split: _metrics(
                    "multi_turn_memory", ("2", "3", "4")
                )
                for split in splits
            }
        },
    }


def test_phase33_gate_accepts_complete_per_cell_pass():
    assert gate_failures(_summary()) == ()


def test_phase33_gate_rejects_hidden_real_candidate_regression():
    summary = deepcopy(_summary())
    summary["phase32"]["splits"]["lexical"][
        "per_skill_candidate_width"
    ]["multi_turn_memory"]["4"]["real_candidate_top1_accuracy"] = 0.99
    failures = gate_failures(summary)
    assert any(
        "phase32/lexical/multi_turn_memory/width-4" in failure
        and "real_candidate_top1_accuracy" in failure
        for failure in failures
    )
