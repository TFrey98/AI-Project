from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    CLARIFY_ACTION,
    DIRECT_SPAN_SKILLS,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
    ExpandedTypedSpanResolver,
    build_expanded_dataset,
    encode_expanded_records,
    expanded_batch,
    load_expanded_records,
    realize_expanded_decision,
    summarize_expanded_predictions,
    validate_expanded_records,
)
from story_model.models import build_model
from story_model.neutral_instruction import TRAIN_LEXICON, VALIDATION_LEXICON


def _dataset(tmp_path: Path):
    return build_expanded_dataset(
        tmp_path,
        train_pairs_per_skill=12,
        validation_pairs_per_skill=12,
        lexical_pairs_per_skill=12,
        paraphrase_pairs_per_skill=12,
        transfer_pairs_per_skill=12,
        seed=17,
    )


def _tokenizer(records):
    return ByteBPETokenizer.train(
        "\n".join(record.prompt for record in records),
        vocab_size=300,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def test_builder_covers_four_skills_actions_and_candidate_widths(tmp_path):
    manifest = _dataset(tmp_path)
    assert tuple(manifest["skills"]) == DIRECT_SPAN_SKILLS
    for split, metadata in manifest["splits"].items():
        assert metadata["examples"] == 192
        assert metadata["resolve_pairs"] == 48
        assert metadata["actions"] == {
            CLARIFY_ACTION: 48,
            GENERATE_ACTION: 48,
            RESOLVE_ACTION: 96,
        }
        assert metadata["candidate_counts"] == {"2": 32, "3": 32, "4": 32}
        assert set(metadata["skills"]) == set(DIRECT_SPAN_SKILLS)
        assert (tmp_path / f"{split}.jsonl").is_file()


def test_counterfactual_pairs_keep_inventory_and_flip_evidence(tmp_path):
    _dataset(tmp_path)
    records = load_expanded_records(tmp_path / "train.jsonl")
    resolve = [record for record in records if record.expected_action == RESOLVE_ACTION]
    grouped = {}
    for record in resolve:
        grouped.setdefault(record.conversation_id, []).append(record)
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0].candidates == pair[1].candidates
        assert pair[0].selected_candidate_index != pair[1].selected_candidate_index
        assert pair[0].expected_value in pair[0].prompt
        assert pair[1].expected_value in pair[1].prompt


def test_lexical_and_paraphrase_axes_are_factorized(tmp_path):
    _dataset(tmp_path)
    train = load_expanded_records(tmp_path / "train.jsonl")
    lexical = load_expanded_records(tmp_path / "lexical.jsonl")
    paraphrase = load_expanded_records(tmp_path / "paraphrase.jsonl")

    def values(records):
        return {
            record.expected_value
            for record in records
            if record.expected_action == RESOLVE_ACTION
        }

    train_values = values(train)
    lexical_values = values(lexical)
    paraphrase_values = values(paraphrase)
    familiar = set(
        TRAIN_LEXICON.containers + TRAIN_LEXICON.names + TRAIN_LEXICON.actions
    )
    unfamiliar = set(
        VALIDATION_LEXICON.containers
        + VALIDATION_LEXICON.names
        + VALIDATION_LEXICON.actions
    )
    assert train_values <= familiar
    assert paraphrase_values <= familiar
    assert lexical_values <= unfamiliar
    assert not train_values & lexical_values


def test_phase31_jsonl_is_backward_compatible(tmp_path):
    payload = {
        "resolver_version": 1,
        "record_id": "phase31-row",
        "source_context_id": "source",
        "conversation_id": "pair",
        "split": "train",
        "skill": "supplied_fact",
        "case": "supported",
        "prompt": "Evidence says amber.",
        "expected_action": "resolve",
        "expected_type": "color",
        "candidates": [
            {"text": "amber", "value_type": "color"},
            {"text": "blue", "value_type": "color"},
        ],
        "selected_candidate_index": 0,
        "response_template": "The supplied evidence identifies <|resolved_value|>.",
        "expected_value": "amber",
        "alternative_value": "blue",
    }
    path = tmp_path / "phase31.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    record = load_expanded_records(path)[0]
    assert record.source_phase == "phase31"
    assert record.expected_value == "amber"
    assert len(record.candidates) == 2


def test_encoding_preserves_exact_candidates_through_width_four(tmp_path):
    _dataset(tmp_path)
    records = load_expanded_records(tmp_path / "train.jsonl")
    record = next(
        record
        for record in records
        if record.expected_action == RESOLVE_ACTION and len(record.candidates) == 4
    )
    tokenizer = _tokenizer(records)
    encoded = encode_expanded_records((record,), tokenizer, 1024)[0]
    assert len(encoded.candidate_token_masks) == 4
    for candidate, mask in zip(record.candidates, encoded.candidate_token_masks):
        selected = [
            token
            for token, include in zip(encoded.input_ids, mask)
            if include
        ]
        assert tokenizer.decode(selected) == candidate.text


def test_model_supports_variable_candidates_without_new_head_shapes(tmp_path):
    _dataset(tmp_path)
    records = load_expanded_records(tmp_path / "train.jsonl")[:4]
    tokenizer = _tokenizer(records)
    examples = encode_expanded_records(records, tokenizer, 1024)
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
    model = ExpandedTypedSpanResolver(backbone)
    batch = expanded_batch(examples, range(4), "cpu")
    output = model(*batch[:5], batch[5], batch[6])
    assert output.action_logits.shape == (4, 3)
    assert output.candidate_logits.shape == (4, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.action_head.out_features == 3
    assert model.candidate_score.out_features == 1


def test_realization_is_exact_and_incompatible_type_fails_closed():
    candidates = (
        ExpandedCandidate("desk drawer", "container"),
        ExpandedCandidate("stone locker", "container"),
    )
    record = ExpandedResolverRecord(
        record_id="row",
        source_context_id="source",
        conversation_id="pair",
        split="train",
        skill="multi_turn_memory",
        case="supported",
        prompt="I left it in the desk drawer.",
        expected_action=RESOLVE_ACTION,
        expected_type="container",
        candidates=candidates,
        selected_candidate_index=0,
        response_template="The recorded location is <|resolved_value|>.",
        expected_value="desk drawer",
        alternative_value="stone locker",
    )
    decision = realize_expanded_decision(record, RESOLVE_ACTION, 0)
    assert decision.text == "The recorded location is desk drawer."
    wrong = ExpandedResolverRecord(
        record_id="wrong",
        source_context_id="source",
        conversation_id="wrong",
        split="train",
        skill="multi_turn_memory",
        case="wrong_type",
        prompt="I left it in the desk drawer.",
        expected_action=CLARIFY_ACTION,
        expected_type="container",
        candidates=(ExpandedCandidate("puppies", "animal"),),
        selected_candidate_index=None,
        response_template="I cannot resolve that from the supplied evidence.",
        expected_value="desk drawer",
        alternative_value="stone locker",
    )
    guarded = realize_expanded_decision(wrong, RESOLVE_ACTION, 0)
    assert guarded.action == CLARIFY_ACTION
    assert guarded.guarded


def test_prediction_summary_reports_each_skill():
    rows = []
    for skill in DIRECT_SPAN_SKILLS:
        for side in range(2):
            rows.append(
                {
                    "conversation_id": skill,
                    "skill": skill,
                    "case": "supported",
                    "candidate_count": 2,
                    "expected_action": RESOLVE_ACTION,
                    "action_correct": True,
                    "end_to_end_correct": True,
                    "exact_realization": True,
                    "value_missing": False,
                    "wrong_alternative": False,
                    "guarded": False,
                }
            )
        for action, case in (
            (CLARIFY_ACTION, "missing_evidence"),
            (GENERATE_ACTION, "ordinary_generation"),
        ):
            rows.append(
                {
                    "conversation_id": f"{skill}:{action}",
                    "skill": skill,
                    "case": case,
                    "candidate_count": 0,
                    "expected_action": action,
                    "action_correct": True,
                    "end_to_end_correct": True,
                    "exact_realization": False,
                    "value_missing": False,
                    "wrong_alternative": False,
                    "guarded": False,
                }
            )
    summary = summarize_expanded_predictions(rows)
    assert set(summary["per_skill"]) == set(DIRECT_SPAN_SKILLS)
    assert set(summary["per_candidate_width"]) == {"2"}
    assert summary["end_to_end_resolve_accuracy"] == 1.0
    assert all(
        metrics["counterfactual_pair_resolve_accuracy"] == 1.0
        for metrics in summary["per_skill"].values()
    )
