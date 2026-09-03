from __future__ import annotations

import torch

from scripts.audit_explicit_offset_tag_confusion import (
    _begin_geometry_rows,
    _collapsed_model_weights,
    _summarize_begin_geometry,
)
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
    BOUNDARY_BEGIN,
    BOUNDARY_INSIDE,
    BOUNDARY_OBJECTIVE_VERSION,
    BOUNDARY_OUTSIDE,
    FACTORIZED_BOUNDARY_TYPE_VERSION,
    IGNORE_TAG,
    OUTSIDE_TAG,
    PERMISSIVE_DECODE_POLICY,
    PROPOSAL_TYPES,
    STRICT_DECODE_POLICY,
    TOKEN_END_GEOMETRY_VERSION,
    TOKEN_WIDTH_GEOMETRY_VERSION,
    TYPE_TO_INDEX,
    EvidenceSpan,
    ExplicitOffsetCandidateProposer,
    begin_tag,
    boundary_auxiliary_losses,
    checkpoint_uses_factorized_boundary_type,
    checkpoint_uses_token_end_geometry,
    checkpoint_uses_token_width_geometry,
    collapsed_boundary_targets,
    compose_factorized_tag_logits,
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
    typed_targets,
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


def _model(
    tokenizer,
    block_size=256,
    token_width_geometry=False,
    token_end_geometry=False,
    factorized_boundary_type=False,
):
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
        UnifiedTypedSpanResolver(backbone),
        tokenizer,
        token_width_geometry=token_width_geometry,
        token_end_geometry=token_end_geometry,
        factorized_boundary_type=factorized_boundary_type,
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


def test_token_width_geometry_preserves_initial_logits_and_rng_trajectory():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    batch = proposal_batch(
        (encoded,),
        (0,),
        tokenizer,
        max(tokenizer.token_byte_length(index) for index in range(
            tokenizer.vocab_size
        )),
        "cpu",
    )
    torch.manual_seed(2026)
    baseline = _model(tokenizer)
    state_after_baseline = torch.random.get_rng_state()
    torch.manual_seed(2026)
    geometry = _model(tokenizer, token_width_geometry=True)
    state_after_geometry = torch.random.get_rng_state()

    assert torch.equal(state_after_baseline, state_after_geometry)
    baseline_parameters = dict(baseline.named_parameters())
    geometry_parameters = dict(geometry.named_parameters())
    for name, parameter in baseline_parameters.items():
        assert torch.equal(parameter, geometry_parameters[name])
    width = geometry.proposal_token_width_embedding
    assert width is not None
    assert not bool((width.weight != 0).any())
    assert torch.equal(
        baseline(*batch).tag_logits,
        geometry(*batch).tag_logits,
    )


def test_token_width_geometry_is_trainable_and_versioned():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer, token_width_geometry=True)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    output = model(*batch, boundary_loss_weight=1.0)
    assert output.loss is not None
    output.loss.backward()

    width = model.proposal_token_width_embedding
    assert width is not None and width.weight.grad is not None
    assert bool((width.weight.grad != 0).any())
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "proposal_query.weight",
        "proposal_key.weight",
        "proposal_offset_embedding.weight",
        "proposal_tag.weight",
        "proposal_tag.bias",
        "proposal_token_width_embedding.weight",
    }
    assert TOKEN_WIDTH_GEOMETRY_VERSION == 1
    assert not checkpoint_uses_token_width_geometry({})
    assert checkpoint_uses_token_width_geometry(
        {"token_width_geometry_version": 1}
    )
    try:
        checkpoint_uses_token_width_geometry(
            {"token_width_geometry_version": 2}
        )
    except ValueError as error:
        assert "unsupported token-width geometry" in str(error)
    else:
        raise AssertionError("unsupported geometry version was accepted")


def test_begin_geometry_audit_uses_gold_start_token_width_and_offset():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    logits = _perfect_logits(encoded, tokenizer)

    rows = _begin_geometry_rows(logits, encoded, tokenizer)

    assert len(rows) == len(encoded.gold_spans)
    for span, row in zip(encoded.gold_spans, rows):
        token_position, byte_offset = _token_byte_position(
            encoded, tokenizer, span.byte_start
        )
        token_id = encoded.input_ids[token_position]
        assert row["token_id"] == token_id
        assert row["token_bytes_hex"] == tokenizer.token_bytes(token_id).hex()
        assert row["token_width"] == tokenizer.token_byte_length(token_id)
        assert row["token_byte_offset"] == byte_offset
        assert row["token_end"] == (
            byte_offset + 1 == tokenizer.token_byte_length(token_id)
        )
        assert row["begin_as_inside"] is False
    summary = _summarize_begin_geometry(rows)
    assert summary["overall"]["gold_begin_count"] == len(rows)
    assert summary["overall"]["begin_as_inside_rate"] == 0.0


def test_token_end_geometry_preserves_initial_logits_and_rng_trajectory():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    batch = proposal_batch(
        (encoded,),
        (0,),
        tokenizer,
        max(
            tokenizer.token_byte_length(index)
            for index in range(tokenizer.vocab_size)
        ),
        "cpu",
    )
    torch.manual_seed(2034)
    baseline = _model(tokenizer)
    state_after_baseline = torch.random.get_rng_state()
    torch.manual_seed(2034)
    geometry = _model(tokenizer, token_end_geometry=True)
    state_after_geometry = torch.random.get_rng_state()

    assert torch.equal(state_after_baseline, state_after_geometry)
    geometry_parameters = dict(geometry.named_parameters())
    for name, parameter in baseline.named_parameters():
        assert torch.equal(parameter, geometry_parameters[name])
    marker = geometry.proposal_token_end_embedding
    assert marker is not None
    assert marker.padding_idx == 0
    assert not bool((marker.weight != 0).any())
    assert torch.equal(
        baseline(*batch).tag_logits,
        geometry(*batch).tag_logits,
    )


def test_token_end_geometry_is_trainable_versioned_and_exclusive():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer, token_end_geometry=True)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    output = model(*batch, boundary_loss_weight=1.0)
    assert output.loss is not None
    output.loss.backward()

    marker = model.proposal_token_end_embedding
    assert marker is not None and marker.weight.grad is not None
    assert not bool((marker.weight.grad[0] != 0).any())
    assert bool((marker.weight.grad[1] != 0).any())
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "proposal_query.weight",
        "proposal_key.weight",
        "proposal_offset_embedding.weight",
        "proposal_tag.weight",
        "proposal_tag.bias",
        "proposal_token_end_embedding.weight",
    }
    assert TOKEN_END_GEOMETRY_VERSION == 1
    assert not checkpoint_uses_token_end_geometry({})
    assert checkpoint_uses_token_end_geometry(
        {"token_end_geometry_version": 1}
    )
    try:
        _model(
            tokenizer,
            token_width_geometry=True,
            token_end_geometry=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("stacked geometry features were accepted")


def test_factorized_targets_preserve_boundary_and_type_labels():
    targets = torch.tensor(
        [[[
            OUTSIDE_TAG,
            begin_tag("container"),
            inside_tag("container"),
            begin_tag("person"),
            inside_tag("person"),
            IGNORE_TAG,
        ]]]
    )

    boundary = collapsed_boundary_targets(targets)
    types = typed_targets(targets)

    assert boundary.tolist() == [[[
        BOUNDARY_OUTSIDE,
        BOUNDARY_BEGIN,
        BOUNDARY_INSIDE,
        BOUNDARY_BEGIN,
        BOUNDARY_INSIDE,
        IGNORE_TAG,
    ]]]
    assert types.tolist() == [[[
        IGNORE_TAG,
        TYPE_TO_INDEX["container"],
        TYPE_TO_INDEX["container"],
        TYPE_TO_INDEX["person"],
        TYPE_TO_INDEX["person"],
        IGNORE_TAG,
    ]]]


def test_factorized_composition_preserves_independent_argmaxes():
    boundary = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )
    types = torch.zeros(3, len(PROPOSAL_TYPES))
    types[0, TYPE_TO_INDEX["action"]] = 2.0
    types[1, TYPE_TO_INDEX["route"]] = 4.0
    types[2, TYPE_TO_INDEX["person"]] = 5.0

    typed = compose_factorized_tag_logits(boundary, types)

    assert typed.argmax(dim=-1).tolist() == [
        OUTSIDE_TAG,
        begin_tag("route"),
        inside_tag("person"),
    ]


def test_factorized_head_preserves_control_rng_and_shared_parameters():
    record = _record()
    tokenizer = _tokenizer((record,))
    torch.manual_seed(2035)
    baseline = _model(tokenizer)
    state_after_baseline = torch.random.get_rng_state()
    torch.manual_seed(2035)
    factorized = _model(tokenizer, factorized_boundary_type=True)
    state_after_factorized = torch.random.get_rng_state()

    assert torch.equal(state_after_baseline, state_after_factorized)
    factorized_parameters = dict(factorized.named_parameters())
    for name, parameter in baseline.named_parameters():
        assert torch.equal(parameter, factorized_parameters[name])
    assert not factorized.proposal_tag.weight.requires_grad
    assert not factorized.proposal_tag.bias.requires_grad
    trainable = {
        name
        for name, parameter in factorized.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "proposal_query.weight",
        "proposal_key.weight",
        "proposal_offset_embedding.weight",
        "proposal_boundary.weight",
        "proposal_boundary.bias",
        "proposal_type.weight",
        "proposal_type.bias",
    }


def test_factorized_head_trains_boundary_and_type_separately():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer, factorized_boundary_type=True)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    output = model(*batch, boundary_loss_weight=1.0)

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.boundary_logits is not None
    assert output.type_logits is not None
    assert output.boundary_class_loss is not None
    assert output.type_loss is not None
    assert output.type_positions > 0
    output.loss.backward()
    assert model.proposal_boundary is not None
    assert model.proposal_type is not None
    assert bool((model.proposal_boundary.weight.grad != 0).any())
    assert bool((model.proposal_type.weight.grad != 0).any())
    assert model.proposal_tag.weight.grad is None
    weights = _collapsed_model_weights(model)
    assert abs(weights["O"] - 0.05) < 1.0e-6
    assert weights["B"] == 1.0
    assert weights["I"] == 0.5
    assert FACTORIZED_BOUNDARY_TYPE_VERSION == 1
    assert checkpoint_uses_factorized_boundary_type(
        {"factorized_boundary_type_version": 1}
    )


def test_factorized_type_loss_is_zero_without_positive_bytes():
    record = _record(action=CLARIFY_ACTION)
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer, factorized_boundary_type=True)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    output = model(*batch, boundary_loss_weight=1.0)

    assert output.type_positions == 0
    assert output.type_loss is not None
    assert float(output.type_loss.detach()) == 0.0
    assert output.loss is not None and torch.isfinite(output.loss)


def test_factorized_head_is_versioned_and_excludes_geometry_features():
    record = _record()
    tokenizer = _tokenizer((record,))

    assert not checkpoint_uses_factorized_boundary_type({})
    try:
        checkpoint_uses_factorized_boundary_type(
            {"factorized_boundary_type_version": 2}
        )
    except ValueError as error:
        assert "unsupported factorized boundary/type" in str(error)
    else:
        raise AssertionError("unsupported factorized version was accepted")
    try:
        _model(
            tokenizer,
            token_width_geometry=True,
            factorized_boundary_type=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("factorized and geometry features were stacked")


def test_boundary_auxiliary_loss_supervises_only_gold_starts_and_ends():
    class_count = 1 + 2 * len(PROPOSAL_TYPES)
    logits = torch.zeros(1, 1, 7, class_count, requires_grad=True)
    targets = torch.tensor(
        [[[OUTSIDE_TAG, begin_tag("container"), inside_tag("container"),
           OUTSIDE_TAG, begin_tag("person"), inside_tag("person"),
           OUTSIDE_TAG]]]
    )

    boundary = boundary_auxiliary_losses(logits, targets)

    assert BOUNDARY_OBJECTIVE_VERSION == 1
    assert boundary.start_positions == 2
    assert boundary.end_positions == 2
    assert torch.isfinite(boundary.start_loss)
    assert torch.isfinite(boundary.end_loss)
    assert torch.allclose(
        boundary.combined_loss,
        (boundary.start_loss + boundary.end_loss) / 2.0,
    )
    boundary.combined_loss.backward()
    supervised_positions = {1, 3, 4, 6}
    for position in range(7):
        has_gradient = bool((logits.grad[0, 0, position] != 0).any())
        assert has_gradient == (position in supervised_positions)


def test_boundary_auxiliary_loss_is_finite_without_gold_spans():
    class_count = 1 + 2 * len(PROPOSAL_TYPES)
    logits = torch.zeros(1, 1, 4, class_count, requires_grad=True)
    targets = torch.full((1, 1, 4), OUTSIDE_TAG)

    boundary = boundary_auxiliary_losses(logits, targets)

    assert boundary.start_positions == 0
    assert boundary.end_positions == 0
    assert torch.isfinite(boundary.combined_loss)
    boundary.combined_loss.backward()
    assert not bool((logits.grad != 0).any())


def test_boundary_objective_adds_no_parameters_and_composes_with_bio_loss():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer)
    before = sum(parameter.numel() for parameter in model.parameters())
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    output = model(*batch, boundary_loss_weight=1.0)

    after = sum(parameter.numel() for parameter in model.parameters())
    assert before == after
    assert output.loss is not None
    assert output.tag_loss is not None
    assert output.boundary_loss is not None
    assert output.boundary_start_positions == len(encoded.gold_spans)
    assert output.boundary_end_positions == len(encoded.gold_spans)
    assert torch.allclose(
        output.loss, output.tag_loss + output.boundary_loss
    )


def test_boundary_objective_rejects_negative_weight():
    record = _record()
    tokenizer = _tokenizer((record,))
    encoded = encode_proposal_record(record, tokenizer, 256)
    model = _model(tokenizer)
    batch = proposal_batch(
        (encoded,), (0,), tokenizer, model.source_width, "cpu"
    )

    try:
        model(*batch, boundary_loss_weight=-1.0)
    except ValueError as error:
        assert "boundary loss weight" in str(error)
    else:
        raise AssertionError("negative boundary loss weight was accepted")


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
