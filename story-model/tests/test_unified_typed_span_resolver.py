from __future__ import annotations

from pathlib import Path

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
    ExpandedTypedSpanResolver,
    encode_expanded_record,
)
from story_model.models import build_model
from story_model.unified_typed_span_resolver import (
    GENERATE_MODE,
    MODE_TO_INDEX,
    NO_SUPPORT_OPTION_INDEX,
    STRUCTURED_MODE,
    UnifiedTypedSpanResolver,
    candidate_is_supported,
    encode_unified_records,
    realize_unified_decision,
    summarize_unified_predictions,
    unified_batch,
)


def _records():
    candidates = (
        ExpandedCandidate("silver drawer", "container"),
        ExpandedCandidate("amber locker", "container"),
        ExpandedCandidate("violet cabinet", "container"),
        ExpandedCandidate("copper trunk", "container"),
    )
    common = {
        "source_context_id": "source",
        "split": "train",
        "skill": "multi_turn_memory",
        "expected_type": "container",
        "candidates": candidates,
        "alternative_value": "amber locker",
    }
    return (
        ExpandedResolverRecord(
            record_id="resolve",
            conversation_id="pair",
            case="supported",
            prompt="The key is in the silver drawer. Where is the key?",
            expected_action=RESOLVE_ACTION,
            selected_candidate_index=0,
            response_template="The recorded location is <|resolved_value|>.",
            expected_value="silver drawer",
            **common,
        ),
        ExpandedResolverRecord(
            record_id="clarify",
            conversation_id="clarify",
            case="missing_evidence",
            prompt="The location was not recorded. Where is the key?",
            expected_action=CLARIFY_ACTION,
            selected_candidate_index=None,
            response_template=CLARIFICATION_RESPONSE,
            expected_value="silver drawer",
            **common,
        ),
        ExpandedResolverRecord(
            record_id="generate",
            source_context_id="source",
            conversation_id="generate",
            split="train",
            skill="multi_turn_memory",
            case="ordinary_generation",
            prompt="Acknowledge this request.",
            expected_action=GENERATE_ACTION,
            expected_type=None,
            candidates=(),
            selected_candidate_index=None,
            response_template=None,
            expected_value=None,
            alternative_value=None,
        ),
    )


def _tokenizer(records):
    return ByteBPETokenizer.train(
        "\n".join(record.prompt for record in records),
        vocab_size=288,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def _backbone(tokenizer, block_size=256):
    return build_model(
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
        block_size,
    )


def test_encoding_preserves_phase32_inputs_and_real_candidate_views():
    records = _records()
    tokenizer = _tokenizer(records)
    unified = encode_unified_records(records, tokenizer, 256)
    for record, encoded in zip(records, unified):
        phase32 = encode_expanded_record(record, tokenizer, 256)
        assert encoded.input_ids == phase32.input_ids
        assert encoded.option_view_input_ids[:4] == phase32.candidate_view_input_ids
    assert unified[0].option_target == 0
    assert unified[1].option_target == NO_SUPPORT_OPTION_INDEX
    assert unified[2].option_target == -100
    assert unified[0].mode_target == MODE_TO_INDEX[STRUCTURED_MODE]
    assert unified[2].mode_target == MODE_TO_INDEX[GENERATE_MODE]
    assert unified[0].option_valid_mask == (True, False, False, False, False)
    assert unified[1].option_valid_mask == (False, False, False, False, True)
    assert unified[2].option_valid_mask == (False, False, False, False, False)


def test_runtime_support_mask_rejects_absent_wrong_type_and_substrings():
    resolve, clarify, _ = _records()
    assert candidate_is_supported(resolve, 0)
    assert not candidate_is_supported(resolve, 1)
    assert not any(
        candidate_is_supported(clarify, index)
        for index in range(len(clarify.candidates))
    )

    wrong_type = ExpandedResolverRecord(
        record_id="wrong-type",
        source_context_id="wrong-type",
        conversation_id="wrong-type",
        split="train",
        skill="multi_turn_memory",
        case="wrong_type",
        prompt="The note mentions puppies, not a container.",
        expected_action=CLARIFY_ACTION,
        expected_type="container",
        candidates=(ExpandedCandidate("puppies", "animal"),),
        selected_candidate_index=None,
        response_template=CLARIFICATION_RESPONSE,
        expected_value="silver drawer",
        alternative_value="amber locker",
    )
    substring = ExpandedResolverRecord(
        record_id="substring",
        source_context_id="substring",
        conversation_id="substring",
        split="train",
        skill="multi_turn_memory",
        case="missing_evidence",
        prompt="The location was recorded elsewhere.",
        expected_action=CLARIFY_ACTION,
        expected_type="container",
        candidates=(ExpandedCandidate("record", "container"),),
        selected_candidate_index=None,
        response_template=CLARIFICATION_RESPONSE,
        expected_value="record",
        alternative_value=None,
    )
    assert not candidate_is_supported(wrong_type, 0)
    assert not candidate_is_supported(substring, 0)


def test_multiple_supported_candidates_remain_available_for_ranking():
    record = ExpandedResolverRecord(
        record_id="ambiguous-support",
        source_context_id="ambiguous-support",
        conversation_id="ambiguous-support",
        split="train",
        skill="multi_turn_memory",
        case="supported",
        prompt=(
            "One note names the silver drawer; another names the amber locker."
        ),
        expected_action=RESOLVE_ACTION,
        expected_type="container",
        candidates=(
            ExpandedCandidate("silver drawer", "container"),
            ExpandedCandidate("amber locker", "container"),
        ),
        selected_candidate_index=0,
        response_template="The recorded location is <|resolved_value|>.",
        expected_value="silver drawer",
        alternative_value="amber locker",
        source_phase="phase31",
    )
    tokenizer = _tokenizer((record,))
    encoded = encode_unified_records((record,), tokenizer, 256)[0]

    assert encoded.option_valid_mask == (True, True, False, False, False)


def test_no_support_view_exposes_marker_only_when_candidate_is_supported():
    records = _records()
    tokenizer = _tokenizer(records)
    encoded = encode_unified_records(records[:2], tokenizer, 256)
    supported = tokenizer.decode(
        encoded[0].option_view_input_ids[NO_SUPPORT_OPTION_INDEX][
            : encoded[0].option_view_sequence_tokens[NO_SUPPORT_OPTION_INDEX]
        ]
    )
    missing = tokenizer.decode(
        encoded[1].option_view_input_ids[NO_SUPPORT_OPTION_INDEX][
            : encoded[1].option_view_sequence_tokens[NO_SUPPORT_OPTION_INDEX]
        ]
    )
    assert "<|candidate|>" in supported
    assert "<|candidate|>" not in missing
    assert "candidate_type: none" in supported


def test_model_is_strictly_phase32_state_dict_compatible():
    records = _records()
    tokenizer = _tokenizer(records)
    phase32 = ExpandedTypedSpanResolver(_backbone(tokenizer))
    unified = UnifiedTypedSpanResolver(_backbone(tokenizer))
    assert phase32.state_dict().keys() == unified.state_dict().keys()
    unified.load_state_dict(phase32.state_dict(), strict=True)


def test_forward_scores_four_candidates_plus_sentinel_with_finite_losses():
    records = _records()
    tokenizer = _tokenizer(records)
    examples = encode_unified_records(records, tokenizer, 256)
    model = UnifiedTypedSpanResolver(_backbone(tokenizer))
    batch = unified_batch(examples, range(len(examples)), "cpu")
    output = model(*batch[:5], batch[5], batch[6], batch[7])
    assert output.mode_logits.shape == (3, 2)
    assert output.option_logits.shape == (3, 5)
    assert output.raw_option_logits.shape == (3, 5)
    assert torch.isfinite(output.raw_option_logits).all()
    assert output.option_logits[0, 1] == torch.finfo(
        output.option_logits.dtype
    ).min
    assert output.option_logits[1, NO_SUPPORT_OPTION_INDEX] != torch.finfo(
        output.option_logits.dtype
    ).min
    assert output.legacy_action_logits.shape == (3, 3)
    assert output.mode_loss is not None and torch.isfinite(output.mode_loss)
    assert output.option_loss is not None and torch.isfinite(output.option_loss)
    assert output.raw_candidate_loss is not None
    assert torch.isfinite(output.raw_candidate_loss)
    assert output.loss is not None and torch.isfinite(output.loss)


def test_realization_maps_real_sentinel_and_generate_options():
    resolve, clarify, generate = _records()
    selected = realize_unified_decision(
        resolve, MODE_TO_INDEX[STRUCTURED_MODE], 0
    )
    assert selected.action == RESOLVE_ACTION
    assert selected.selected_value == "silver drawer"
    absent = realize_unified_decision(
        clarify, MODE_TO_INDEX[STRUCTURED_MODE], NO_SUPPORT_OPTION_INDEX
    )
    assert absent.action == CLARIFY_ACTION
    unsupported = realize_unified_decision(
        resolve, MODE_TO_INDEX[STRUCTURED_MODE], 1
    )
    assert unsupported.action == CLARIFY_ACTION
    assert unsupported.guarded
    normal = realize_unified_decision(
        generate, MODE_TO_INDEX[GENERATE_MODE], 0
    )
    assert normal.action == GENERATE_ACTION


def test_summary_reports_routing_and_oracle_candidate_metrics():
    rows = (
        {
            "conversation_id": "pair",
            "skill": "multi_turn_memory",
            "case": "supported",
            "candidate_count": 4,
            "expected_action": RESOLVE_ACTION,
            "action_correct": True,
            "candidate_correct": True,
            "real_candidate_correct": True,
            "option_correct": True,
            "mode_correct": True,
            "end_to_end_correct": True,
            "exact_realization": True,
            "no_support_selected": False,
            "value_missing": False,
            "wrong_alternative": False,
            "guarded": False,
        },
        {
            "conversation_id": "pair",
            "skill": "multi_turn_memory",
            "case": "supported",
            "candidate_count": 4,
            "expected_action": RESOLVE_ACTION,
            "action_correct": True,
            "candidate_correct": True,
            "real_candidate_correct": True,
            "option_correct": True,
            "mode_correct": True,
            "end_to_end_correct": True,
            "exact_realization": True,
            "no_support_selected": False,
            "value_missing": False,
            "wrong_alternative": False,
            "guarded": False,
        },
        {
            "conversation_id": "clarify",
            "skill": "multi_turn_memory",
            "case": "missing_evidence",
            "candidate_count": 4,
            "expected_action": CLARIFY_ACTION,
            "action_correct": True,
            "candidate_correct": False,
            "real_candidate_correct": False,
            "option_correct": True,
            "mode_correct": True,
            "end_to_end_correct": True,
            "exact_realization": False,
            "no_support_selected": True,
            "value_missing": False,
            "wrong_alternative": False,
            "guarded": False,
        },
        {
            "conversation_id": "generate",
            "skill": "multi_turn_memory",
            "case": "ordinary_generation",
            "candidate_count": 0,
            "expected_action": GENERATE_ACTION,
            "action_correct": True,
            "candidate_correct": False,
            "real_candidate_correct": False,
            "option_correct": True,
            "mode_correct": True,
            "end_to_end_correct": True,
            "exact_realization": False,
            "no_support_selected": False,
            "value_missing": False,
            "wrong_alternative": False,
            "guarded": False,
        },
    )
    summary = summarize_unified_predictions(rows)
    assert summary["mode_accuracy"] == 1.0
    assert summary["real_candidate_top1_accuracy"] == 1.0
    assert summary["clarify_sentinel_accuracy"] == 1.0
    assert summary["no_support_false_positive_rate"] == 0.0
    assert summary["per_skill_candidate_width"]["multi_turn_memory"]["4"][
        "real_candidate_top1_accuracy"
    ] == 1.0
