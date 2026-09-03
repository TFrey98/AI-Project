from scripts.audit_token_identity_priors import (
    _enrich_start_rows,
    _record_offset_zero_labels,
    _summarize_panel,
    _token_layout,
)
from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
)


def _record():
    prompt = (
        "Two routes are available: quarry road and east gate. "
        "Memory: east gate is blocked."
    )
    return ExpandedResolverRecord(
        record_id="route-record",
        source_context_id="source",
        conversation_id="pair",
        split="train",
        skill="scene_route",
        case="supported",
        prompt=prompt,
        expected_action=RESOLVE_ACTION,
        expected_type="route",
        candidates=(
            ExpandedCandidate("quarry road", "route"),
            ExpandedCandidate("east gate", "route"),
        ),
        selected_candidate_index=0,
        response_template="Take the <|resolved_value|>.",
        expected_value="quarry road",
        alternative_value="east gate",
        source_phase="phase31",
    )


def test_offset_zero_labels_distinguish_begin_from_inside_for_same_width():
    record = _record()
    tokenizer = ByteBPETokenizer(
        merges=[(ord("q"), ord("u")), (ord("a"), ord("r"))]
    )
    layout = _token_layout(record.prompt, tokenizer)
    labels = list(_record_offset_zero_labels(record, layout))

    assert (256, "B", "B:route") in labels
    assert (257, "I", "I:route") in labels


def test_start_rows_join_to_exact_token_identity_and_bytes():
    record = _record()
    tokenizer = ByteBPETokenizer(
        merges=[(ord("q"), ord("u")), (ord("a"), ord("r"))]
    )
    byte_start = record.prompt.encode("utf-8").index(b"quarry road")
    row = {
        "record_id": record.record_id,
        "dataset": "phase31_regression",
        "split": "train",
        "skill": record.skill,
        "case": record.case,
        "source_phase": record.source_phase,
        "byte_start": byte_start,
        "token_width": 2,
        "token_byte_offset": 0,
        "baseline_predicted_class": "B",
        "baseline_exact_begin_tag": True,
    }

    enriched = _enrich_start_rows(
        (row,),
        {("phase31_regression", "train", record.record_id): record},
        tokenizer,
    )[0]

    assert enriched["token_id"] == 256
    assert enriched["token_bytes_hex"] == b"qu".hex()
    assert enriched["token_utf8"] == "qu"
    assert enriched["token_byte_values"] == [ord("q"), ord("u")]
    assert enriched["first_byte"] == ord("q")
    assert enriched["second_byte"] == ord("u")


def test_panel_summary_preserves_collapsed_and_typed_counts():
    panel = {
        "prompt_token_occurrences": 20,
        "labels": {"O": 10, "B": 2, "I": 8},
        "typed_labels": {"B:route": 2, "I:route": 8},
    }

    summary = _summarize_panel(panel)

    assert summary["inside_share_among_positive"] == 0.8
    assert summary["route_inside_share_among_positive"] == 0.8
