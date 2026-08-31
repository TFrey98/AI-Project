from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from story_model.data import ByteBPETokenizer
from story_model.models import build_model
from story_model.semantic_transfer import (
    build_semantic_answer_keys,
    semantic_transfer_splits,
)
from story_model.typed_span_resolver import (
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    GENERATE_ACTION,
    RESOLVE_ACTION,
    TYPED_SPAN_CONTROL_TOKENS,
    TypedSpanCandidate,
    TypedSpanResolver,
    build_typed_span_dataset,
    build_typed_span_records,
    encode_typed_span_records,
    load_typed_span_records,
    realize_resolver_decision,
    summarize_resolver_predictions,
    typed_span_batch,
    validate_typed_span_records,
)


def _source_splits():
    return semantic_transfer_splits(
        train_pairs_per_skill=3,
        validation_pairs_per_skill=2,
        lexical_pairs_per_skill=2,
        paraphrase_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
        seed=17,
    )


def _built_records(split="train"):
    splits = _source_splits()
    return build_typed_span_records(
        splits[split], build_semantic_answer_keys(splits), split
    )


def _tokenizer(records):
    text = "\n".join(record.prompt for record in records)
    return ByteBPETokenizer.train(
        text, vocab_size=280, min_frequency=2
    ).with_special_tokens(TYPED_SPAN_CONTROL_TOKENS)


def test_builder_preserves_pairs_and_balances_actions_and_positions():
    records = _built_records()
    report = validate_typed_span_records(records)
    assert len(records) == 24
    assert report["resolve_pairs"] == 6
    assert report["actions"] == {
        CLARIFY_ACTION: 6,
        GENERATE_ACTION: 6,
        RESOLVE_ACTION: 12,
    }
    assert report["selected_candidate_positions"] == {"0": 6, "1": 6}
    resolve = [record for record in records if record.expected_action == RESOLVE_ACTION]
    for offset in range(0, len(resolve), 2):
        first, second = resolve[offset : offset + 2]
        assert first.candidates == second.candidates
        assert first.selected_candidate_index != second.selected_candidate_index


def test_clarify_controls_cover_missing_evidence_and_wrong_type():
    clarify = [
        record
        for record in _built_records()
        if record.expected_action == CLARIFY_ACTION
    ]
    assert Counter(record.case for record in clarify) == {
        "missing_evidence": 3,
        "wrong_type": 3,
    }
    wrong_type = [record for record in clarify if record.case == "wrong_type"]
    assert all(record.candidates[0].value_type == "animal" for record in wrong_type)


def test_dataset_round_trip_writes_all_splits(tmp_path: Path):
    splits = _source_splits()
    manifest = build_typed_span_dataset(
        splits, build_semantic_answer_keys(splits), tmp_path
    )
    assert tuple(manifest["splits"]) == (
        "train",
        "val",
        "lexical",
        "paraphrase",
        "transfer",
    )
    for split in manifest["splits"]:
        loaded = load_typed_span_records(tmp_path / f"{split}.jsonl")
        assert len(loaded) == manifest["splits"][split]["examples"]
    assert (tmp_path / "manifest.json").is_file()


def test_encoding_marks_exact_candidate_tokens():
    records = _built_records()
    tokenizer = _tokenizer(records)
    encoded = encode_typed_span_records(records[:1], tokenizer, 1024)[0]
    record = records[0]
    for candidate, mask in zip(record.candidates, encoded.candidate_token_masks):
        selected_ids = [
            token
            for token, selected in zip(encoded.input_ids, mask)
            if selected
        ]
        assert tokenizer.decode(selected_ids) == candidate.text


def test_resolver_forward_has_separate_finite_losses():
    records = _built_records()[:4]
    tokenizer = _tokenizer(records)
    examples = encode_typed_span_records(records, tokenizer, 1024)
    backbone = build_model(
        {
            "name": "transformer",
            "position_encoding": "rope",
            "normalization": "layernorm",
            "feed_forward_activation": "swiglu",
            "embedding_dim": 32,
            "attention_heads": 4,
            "layers": 1,
            "feed_forward_dim": 64,
            "dropout": 0.0,
        },
        tokenizer.vocab_size,
        1024,
    )
    resolver = TypedSpanResolver(backbone)
    batch = typed_span_batch(examples, range(4), "cpu")
    output = resolver(*batch[:5], batch[5], batch[6])
    assert output.action_logits.shape == (4, 3)
    assert output.candidate_logits.shape == (4, 2)
    assert output.action_loss is not None and torch.isfinite(output.action_loss)
    assert output.candidate_loss is not None and torch.isfinite(output.candidate_loss)
    assert output.loss is not None and torch.isfinite(output.loss)


def test_realization_is_exact_and_wrong_type_fails_closed():
    resolve = next(
        record for record in _built_records() if record.expected_action == RESOLVE_ACTION
    )
    decision = realize_resolver_decision(
        resolve, RESOLVE_ACTION, resolve.selected_candidate_index
    )
    assert decision.selected_value == resolve.expected_value
    assert resolve.expected_value in decision.text

    clarify = next(
        record for record in _built_records() if record.case == "wrong_type"
    )
    guarded = realize_resolver_decision(clarify, RESOLVE_ACTION, 0)
    assert guarded.action == CLARIFY_ACTION
    assert guarded.text == CLARIFICATION_RESPONSE
    assert guarded.guarded


def test_prediction_summary_counts_pair_success_and_omission():
    rows = []
    for side in range(2):
        rows.append(
            {
                "conversation_id": "pair",
                "expected_action": RESOLVE_ACTION,
                "action_correct": True,
                "end_to_end_correct": side == 0,
                "exact_realization": side == 0,
                "value_missing": side == 1,
                "wrong_alternative": False,
                "guarded": False,
                "case": "supported",
                "skill": "supplied_fact",
            }
        )
    rows.extend(
        (
            {
                "conversation_id": "clarify",
                "expected_action": CLARIFY_ACTION,
                "action_correct": True,
                "end_to_end_correct": True,
                "exact_realization": False,
                "value_missing": False,
                "wrong_alternative": False,
                "guarded": False,
                "case": "missing_evidence",
                "skill": "supplied_fact",
            },
            {
                "conversation_id": "generate",
                "expected_action": GENERATE_ACTION,
                "action_correct": True,
                "end_to_end_correct": True,
                "exact_realization": False,
                "value_missing": False,
                "wrong_alternative": False,
                "guarded": False,
                "case": "ordinary_generation",
                "skill": "supplied_fact",
            },
        )
    )
    summary = summarize_resolver_predictions(rows)
    assert summary["end_to_end_resolve_accuracy"] == 0.5
    assert summary["counterfactual_pair_resolve_accuracy"] == 0.0
    assert summary["value_missing_rate"] == 0.5
    assert summary["clarify_accuracy"] == 1.0
    assert summary["generate_accuracy"] == 1.0


def test_resolve_record_rejects_a_selected_candidate_of_wrong_type():
    record = next(
        record for record in _built_records() if record.expected_action == RESOLVE_ACTION
    )
    wrong = TypedSpanCandidate(record.expected_value, "animal")
    candidates = list(record.candidates)
    candidates[record.selected_candidate_index] = wrong
    with pytest.raises(ValueError, match="wrong type"):
        replace(record, candidates=tuple(candidates))
