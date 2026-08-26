import pytest

from story_model.neutral_diagnostics import (
    context_without_evidence,
    evenly_spaced_records,
    neutral_skill,
    repetition_metrics,
    stratified_neutral_records,
    summarize_diagnostic_rows,
)
from story_model.neutral_instruction import neutral_instruction_records


def test_stratified_sample_covers_every_skill_without_prefix_bias():
    records = neutral_instruction_records("train", examples_per_skill=5)
    selected = stratified_neutral_records(records, examples_per_skill=2)
    skills = [neutral_skill(record) for record in selected]

    assert len(selected) == 20
    assert set(skills) == {
        "cause_and_effect",
        "comparison",
        "contradiction_correction",
        "missing_information",
        "multi_turn_memory",
        "privacy_boundary",
        "promise_recall",
        "reference_tracking",
        "scene_route",
        "supplied_fact",
    }
    assert all(skills.count(skill) == 2 for skill in set(skills))
    assert {
        record.context.context_id.rsplit("_", 1)[-1]
        for record in selected
    } == {"0001", "0003"}


def test_evenly_spaced_sample_rejects_an_oversized_request():
    records = neutral_instruction_records("val", examples_per_skill=1)

    with pytest.raises(ValueError, match="from only"):
        evenly_spaced_records(records, len(records) + 1)


def test_context_ablation_keeps_only_question_and_stable_identity():
    records = neutral_instruction_records("train", examples_per_skill=1)
    record = next(
        record
        for record in records
        if neutral_skill(record) == "promise_recall"
    )
    ablated = context_without_evidence(record.context)

    assert ablated.character == record.context.character
    assert ablated.relationship == record.context.relationship
    assert ablated.target_response is None
    assert ablated.memories == ()
    assert ablated.world_facts == ()
    assert ablated.scene.location == "unknown location"
    assert ablated.scene.situation == ""
    assert ablated.recent_turns == (record.context.recent_turns[-1],)
    assert ablated.recent_turns[-1].role == "user"


def test_repetition_metrics_detect_word_and_phrase_loops():
    repeated_phrase = repetition_metrics(
        "The south stone is barred. " * 5
    )
    repeated_word = repetition_metrics(
        "covered covered covered again and again"
    )
    normal = repetition_metrics(
        "The eastern path is blocked, so use the bridge."
    )

    assert repeated_phrase["degenerate_loop"]
    assert repeated_phrase["repeated_trigram_fraction"] >= 0.35
    assert repeated_word["degenerate_loop"]
    assert repeated_word["max_identical_word_run"] == 3
    assert not normal["degenerate_loop"]


def test_summary_distinguishes_grounding_from_output_change():
    first = {
        "skill": "supplied_fact",
        "exact_response": False,
        "end_stop": True,
        "reference_similarity": 0.5,
        "degenerate_loop": False,
        "unique_word_ratio": 0.8,
        "context_changed": True,
        "context_helped": True,
        "context_advantage": 0.2,
        "prompt_truncated": False,
        "prompt_tokens": 500,
        "raw_prompt_tokens": 500,
    }
    second = dict(first)
    second.update(context_helped=False, context_advantage=0.0)
    summary = summarize_diagnostic_rows((first, second))

    assert summary["examples"] == 2
    assert summary["context_changed_rate"] == 1.0
    assert summary["context_helped_rate"] == 0.5
    assert summary["mean_context_advantage"] == pytest.approx(0.1)


def test_summary_reports_complete_counterfactual_pair_accuracy():
    rows = []

    for conversation_id, exact_values in (
        ("pair_a", (True, True)),
        ("pair_b", (True, False)),
    ):
        for exact_response in exact_values:
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "skill": "supplied_fact",
                    "exact_response": exact_response,
                    "end_stop": True,
                    "reference_similarity": 1.0,
                    "degenerate_loop": False,
                    "unique_word_ratio": 1.0,
                    "context_changed": True,
                    "context_helped": True,
                    "context_advantage": 1.0,
                    "prompt_truncated": False,
                    "prompt_tokens": 400,
                    "raw_prompt_tokens": 400,
                }
            )

    summary = summarize_diagnostic_rows(rows)

    assert summary["exact_response_rate"] == 0.75
    assert summary["counterfactual_pair_exact_rate"] == 0.5
