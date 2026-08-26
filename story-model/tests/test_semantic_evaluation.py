from story_model.semantic_evaluation import (
    score_diagnostic_rows,
    score_semantic_value,
    semantic_transfer_acceptance_failures,
    summarize_semantic_rows,
)


def test_semantic_value_accepts_paraphrase_with_correct_fact():
    result = score_semantic_value(
        "The harbor should display amber tonight.",
        expected_value="amber",
        alternative_value="violet",
        skill="supplied_fact",
    )

    assert result["semantic_correct"]
    assert result["semantic_status"] == "expected_only"


def test_semantic_route_understands_rejected_foil_before_selected_route():
    result = score_semantic_value(
        "Avoid the south stair; take the eastern tunnel.",
        expected_value="eastern tunnel",
        alternative_value="south stair",
        skill="scene_route",
    )

    assert result["semantic_correct"]
    assert result["semantic_status"] == "expected_selected_alternative_rejected"


def test_semantic_value_rejects_wrong_or_missing_values():
    wrong = score_semantic_value(
        "Use the south stair.",
        expected_value="eastern tunnel",
        alternative_value="south stair",
        skill="scene_route",
    )
    missing = score_semantic_value(
        "I cannot determine that.",
        expected_value="amber",
        alternative_value="violet",
        skill="supplied_fact",
    )

    assert not wrong["semantic_correct"]
    assert wrong["semantic_status"] == "alternative_only"
    assert not missing["semantic_correct"]
    assert missing["semantic_status"] == "value_missing"


def test_scored_diagnostics_report_row_and_complete_pair_semantics():
    keys = {
        "row_a": {
            "split": "val",
            "conversation_id": "pair_a",
            "skill": "supplied_fact",
            "expected_value": "amber",
            "alternative_value": "violet",
        },
        "row_b": {
            "split": "val",
            "conversation_id": "pair_a",
            "skill": "supplied_fact",
            "expected_value": "violet",
            "alternative_value": "amber",
        },
    }
    rows = (
        {
            "scenario_id": "row_a",
            "split": "val",
            "conversation_id": "pair_a",
            "skill": "supplied_fact",
            "full_context": {"response": "It is amber."},
        },
        {
            "scenario_id": "row_b",
            "split": "val",
            "conversation_id": "pair_a",
            "skill": "supplied_fact",
            "full_context": {"response": "I do not know."},
        },
    )
    scored = score_diagnostic_rows(rows, keys)
    summary = summarize_semantic_rows(scored)["val"]

    assert summary["semantic_correct_rate"] == 0.5
    assert summary["counterfactual_pair_semantic_rate"] == 0.0
    assert summary["expected_value_mention_rate"] == 0.5


def test_semantic_transfer_gate_uses_semantics_not_exact_wording():
    passing = {}

    for split in ("train", "val", "lexical", "paraphrase", "transfer"):
        passing[split] = {
            "semantic_correct_rate": 1.0,
            "counterfactual_pair_semantic_rate": 1.0,
            "end_stop_rate": 1.0,
            "degenerate_loop_rate": 0.0,
            "context_helped_rate": 1.0,
            "mean_context_advantage": 1.0,
            "exact_response_rate": 0.0,
        }

    assert semantic_transfer_acceptance_failures(passing) == ()
    failing = {split: dict(summary) for split, summary in passing.items()}
    failing["lexical"]["semantic_correct_rate"] = 0.0
    failures = semantic_transfer_acceptance_failures(failing)

    assert len(failures) == 1
    assert "lexical semantic_correct_rate" in failures[0]
