import json
from pathlib import Path

from story_model.backbones import (
    BackboneResponse,
    GenerationSettings,
    ScriptedBackbone,
)
from story_model.conversation_gate import (
    ConversationGateCase,
    load_conversation_gate,
    run_conversation_gate,
    score_conversation_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = PROJECT_ROOT / "examples" / "generic_conversation_gate.json"


def test_project_conversation_gate_is_valid_and_unique():
    cases = load_conversation_gate(GATE_PATH)

    assert len(cases) == 7
    assert len({case.case_id for case in cases}) == 7
    assert all(case.messages[-1].role == "user" for case in cases)


def test_concept_scoring_uses_complete_words_not_substrings():
    case = ConversationGateCase(
        case_id="word_boundaries",
        system="Answer the question.",
        messages=load_conversation_gate(GATE_PATH)[0].messages,
        required_concepts=(("no",),),
    )
    response = BackboneResponse(
        text="We should leave now through the tunnel.",
        backend="test",
        model="test",
    )

    result = score_conversation_response(case, response)

    assert not result.passed
    assert result.missing_concepts == (("no",),)


def test_gate_reports_missing_and_forbidden_concepts():
    case = load_conversation_gate(GATE_PATH)[0]
    response = BackboneResponse(
        text="The main gate is open.",
        backend="test",
        model="test",
    )

    result = score_conversation_response(case, response)

    assert not result.passed
    assert ("tunnel",) in result.missing_concepts
    assert result.found_forbidden == ("main gate is open",)


def test_gate_runs_cases_with_incrementing_seeds():
    cases = load_conversation_gate(GATE_PATH)[:2]
    backend = ScriptedBackbone(
        [
            "No. The tunnel is the usable route.",
            "The all-clear signal is amber.",
        ]
    )

    results = run_conversation_gate(
        backend,
        cases,
        GenerationSettings(seed=50),
    )

    assert all(result.passed for result in results)
    assert [call[1].seed for call in backend.calls] == [50, 51]


def test_gate_loader_rejects_duplicate_case_ids(tmp_path):
    gate_path = tmp_path / "duplicate.json"
    case = {
        "case_id": "same",
        "system": "Use the fact.",
        "messages": [{"role": "user", "content": "Question?"}],
        "required_concepts": [["answer"]],
    }
    gate_path.write_text(
        json.dumps({"schema_version": 1, "cases": [case, case]}),
        encoding="utf-8",
    )

    try:
        load_conversation_gate(gate_path)
    except ValueError as error:
        assert "case_id values must be unique" in str(error)
    else:
        raise AssertionError("duplicate case ids were accepted")
