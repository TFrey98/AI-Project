from __future__ import annotations

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    EXPANDED_CONTROL_TOKENS,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
)
from story_model.explicit_offset_candidate_proposer import (
    IGNORE_TAG,
    OUTSIDE_TAG,
    PERMISSIVE_DECODE_POLICY,
    PROPOSAL_TYPES,
    STRICT_DECODE_POLICY,
    EvidenceSpan,
    ExplicitOffsetCandidateProposer,
    begin_tag,
    decode_proposal_result,
    decode_proposed_spans,
    encode_proposal_record,
    gold_evidence_spans,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
    proposed_candidates,
    realize_proposed_decision,
    runtime_record_from_spans,
    inside_tag,
)
from story_model.models import build_model
from story_model.unified_typed_span_resolver import UnifiedTypedSpanResolver


def _record(case="supported", action=RESOLVE_ACTION):
    candidates = (
        ExpandedCandidate("ébon drawer", "container"),
        ExpandedCandidate("amber locker", "container"),
    )
    prompt = (
        "The ébon drawer was inspected. "
        "Verified memory: the token is in the ébon drawer."
    )
    if action == CLARIFY_ACTION:
        prompt = "The location was not recorded."
    return ExpandedResolverRecord(
        record_id=f"record-{case}-{action}",
        source_context_id="source",
        conversation_id="pair",
        split="train",
        skill="multi_turn_memory",
        case=case,
        prompt=prompt,
        expected_action=action,
        expected_type="container",
        candidates=candidates,
        selected_candidate_index=0 if action == RESOLVE_ACTION else None,
        response_template=(
            "The recorded location is <|resolved_value|>."
            if action == RESOLVE_ACTION
            else CLARIFICATION_RESPONSE
        ),
        expected_value="ébon drawer",
        alternative_value="amber locker",
        source_phase="phase31",
    )


def _tokenizer(records):
    return ByteBPETokenizer.train(
        "\n".join(record.prompt for record in records),
        vocab_size=288,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def _model(tokenizer, block_size=256):
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
        block_size,
    )
    return ExplicitOffsetCandidateProposer(
        UnifiedTypedSpanResolver(backbone), tokenizer
    )


def _perfect_logits(encoded, tokenizer):
    width = max(
        tokenizer.token_byte_length(token_id)
        for token_id in range(tokenizer.vocab_size)
    )
    targets = proposal_batch(
        (encoded,), (0,), tokenizer, width, "cpu"
    )[3][0]
    logits = torch.full(
        (256, width, 1 + 2 * len(PROPOSAL_TYPES)), -20.0
    )
    logits[..., OUTSIDE_TAG] = 0.0
    for token_position, byte_position in torch.nonzero(
        targets != IGNORE_TAG, as_tuple=False
    ).tolist():
        target = targets[token_position, byte_position]
        logits[token_position, byte_position, target] = 20.0
    return logits


def _token_byte_position(encoded, tokenizer, prompt_byte_position):
    cursor = 0
    for token_position in range(encoded.prompt_token_count):
        token_id = encoded.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        if cursor <= prompt_byte_position < cursor + width:
            return token_position, prompt_byte_position - cursor
        cursor += width
    raise ValueError("prompt byte position is outside the encoded prompt")


def test_gold_spans_preserve_every_utf8_occurrence_and_exact_offsets():
    record = _record()
    spans = gold_evidence_spans(record)

    assert len(spans) == 2
    assert all(span.text == "ébon drawer" for span in spans)
    assert all(span.value_type == "container" for span in spans)
    for span in spans:
        span.validate_prompt(record.prompt)
        assert record.prompt.encode("utf-8")[span.byte_start : span.byte_end] == (
            b"\xc3\xa9bon drawer"
        )


def test_encoding_and_batch_labels_cross_bpe_and_utf8_boundaries():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    width = max(
        tokenizer.token_byte_length(token_id)
        for token_id in range(tokenizer.vocab_size)
    )
    batch = proposal_batch((encoded,), (0,), tokenizer, width, "cpu")

    supervised = batch[3][0][batch[3][0] != IGNORE_TAG]
    assert len(supervised) == len(record.prompt.encode("utf-8"))
    assert int((supervised != OUTSIDE_TAG).sum()) == 2 * len(
        "ébon drawer".encode("utf-8")
    )


def test_perfect_byte_logits_decode_gold_spans_exactly():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    logits = _perfect_logits(encoded, tokenizer)

    decoded = decode_proposed_spans(record.prompt, encoded, logits, tokenizer)

    assert decoded == encoded.gold_spans


def test_strict_decoder_discards_orphan_inside_before_true_begin():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    logits = _perfect_logits(encoded, tokenizer)
    first_start = encoded.gold_spans[0].byte_start
    token_position, byte_position = _token_byte_position(
        encoded, tokenizer, first_start - 1
    )
    logits[token_position, byte_position, :] = -20.0
    logits[
        token_position, byte_position, inside_tag("container")
    ] = 20.0

    permissive = decode_proposal_result(
        record.prompt,
        encoded,
        logits,
        tokenizer,
        policy=PERMISSIVE_DECODE_POLICY,
    )
    strict = decode_proposal_result(
        record.prompt,
        encoded,
        logits,
        tokenizer,
        policy=STRICT_DECODE_POLICY,
    )

    assert permissive.orphan_inside_tags == strict.orphan_inside_tags == 1
    assert len(permissive.spans) == len(encoded.gold_spans) + 1
    assert strict.spans == encoded.gold_spans


def test_strict_decoder_closes_on_mismatched_inside_without_promoting_it():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    logits = _perfect_logits(encoded, tokenizer)
    first_start = encoded.gold_spans[0].byte_start
    before_token, before_byte = _token_byte_position(
        encoded, tokenizer, first_start - 1
    )
    start_token, start_byte = _token_byte_position(
        encoded, tokenizer, first_start
    )
    logits[before_token, before_byte, :] = -20.0
    logits[before_token, before_byte, begin_tag("person")] = 20.0
    logits[start_token, start_byte, :] = -20.0
    logits[start_token, start_byte, inside_tag("container")] = 20.0

    permissive = decode_proposal_result(
        record.prompt,
        encoded,
        logits,
        tokenizer,
        policy=PERMISSIVE_DECODE_POLICY,
    )
    strict = decode_proposal_result(
        record.prompt,
        encoded,
        logits,
        tokenizer,
        policy=STRICT_DECODE_POLICY,
    )

    assert permissive.mismatched_inside_tags == 1
    assert strict.mismatched_inside_tags == 1
    assert encoded.gold_spans[0] in permissive.spans
    assert encoded.gold_spans[0] not in strict.spans
    assert encoded.gold_spans[1] in strict.spans


def test_decoder_rejects_unknown_policy():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    logits = _perfect_logits(encoded, tokenizer)

    try:
        decode_proposed_spans(
            record.prompt,
            encoded,
            logits,
            tokenizer,
            policy="invented",
        )
    except ValueError as error:
        assert "unknown proposal decode policy" in str(error)
    else:
        raise AssertionError("unknown decode policy was accepted")


def test_proposed_candidates_require_type_boundaries_offsets_and_deduplicate():
    record = _record()
    gold = gold_evidence_spans(record)
    character_start = record.prompt.index("ébon") + 1
    inside_word_start = len(record.prompt[:character_start].encode("utf-8"))
    malformed = EvidenceSpan(
        inside_word_start,
        inside_word_start + len("bon"),
        "bon",
        "container",
    )
    wrong_type = EvidenceSpan(
        gold[0].byte_start,
        gold[0].byte_end,
        gold[0].text,
        "person",
    )

    candidates = proposed_candidates(
        record, (*gold, malformed, wrong_type)
    )

    assert candidates == (ExpandedCandidate("ébon drawer", "container"),)


def test_runtime_record_does_not_leak_gold_action_and_realizes_exact_text():
    source = _record()
    runtime = runtime_record_from_spans(source, gold_evidence_spans(source))

    assert runtime.expected_action == CLARIFY_ACTION
    assert runtime.selected_candidate_index is None
    decision = realize_proposed_decision(source, runtime, 0, 0)
    assert decision.action == RESOLVE_ACTION
    assert decision.selected_value == "ébon drawer"
    assert decision.text == "The recorded location is ébon drawer."


def test_wrong_type_rows_are_retained_only_for_oracle_regression():
    record = _record(case="wrong_type", action=CLARIFY_ACTION)
    assert not proposal_record_is_eligible(record)


def test_proposer_freezes_resolver_and_computes_finite_loss():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )
    output = model(*batch)

    assert output.loss is not None and torch.isfinite(output.loss)
    assert all(
        not parameter.requires_grad for parameter in model.resolver.parameters()
    )
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "proposal_query.weight",
        "proposal_key.weight",
        "proposal_offset_embedding.weight",
        "proposal_tag.weight",
        "proposal_tag.bias",
    }


def test_proposal_metrics_count_exact_and_missing_answers():
    resolve = _record()
    clarify = _record(action=CLARIFY_ACTION)
    resolve_row = proposal_metric_row(
        resolve, gold_evidence_spans(resolve), gold_evidence_spans(resolve)
    )
    clarify_row = proposal_metric_row(clarify, (), ())
    metrics = proposal_metrics((resolve_row, clarify_row))

    assert metrics["exact_span_precision"] == 1.0
    assert metrics["exact_span_recall"] == 1.0
    assert metrics["answer_candidate_recall"] == 1.0
    assert metrics["offset_validity_rate"] == 1.0
