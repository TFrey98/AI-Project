from __future__ import annotations

import torch

from scripts.audit_paired_token_intervention import (
    _collapse_legacy_logits,
    build_intervention_pairs,
    component_boundary_logits,
    summarize_model_results,
)
from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    EXPANDED_CONTROL_TOKENS,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
)
from story_model.explicit_offset_candidate_proposer import (
    ExplicitOffsetCandidateProposer,
)
from story_model.models import build_model
from story_model.unified_typed_span_resolver import UnifiedTypedSpanResolver


def _tokenizer():
    return ByteBPETokenizer(
        merges=[(ord("o"), ord("r")), (ord("r"), ord("o"))]
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def _record():
    candidates = (
        ExpandedCandidate("orchard road", "route"),
        ExpandedCandidate("rocky lane", "route"),
    )
    return ExpandedResolverRecord(
        record_id="paired-scene-route",
        source_context_id="paired-source",
        conversation_id="paired-conversation",
        split="lexical",
        skill="scene_route",
        case="supported",
        prompt=(
            "Routes: orchard road and rocky lane. "
            "The orchard road is blocked."
        ),
        expected_action=RESOLVE_ACTION,
        expected_type="route",
        candidates=candidates,
        selected_candidate_index=1,
        response_template="Take the <|resolved_value|>.",
        expected_value="rocky lane",
        alternative_value="orchard road",
        source_phase="phase31",
    )


def _model(tokenizer, factorized=False):
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
        256,
    )
    return ExplicitOffsetCandidateProposer(
        UnifiedTypedSpanResolver(backbone),
        tokenizer,
        factorized_boundary_type=factorized,
    ).eval()


def test_pair_builder_changes_only_one_equal_width_canonical_token():
    tokenizer = _tokenizer()
    pairs, invalid, target_starts, token_ids = build_intervention_pairs(
        (_record(),), tokenizer, 256
    )

    assert invalid == ()
    assert target_starts == len(pairs) == 3
    assert token_ids == {"or": 256, "ro": 257}
    assert {pair.source_token for pair in pairs} == {"or", "ro"}
    for pair in pairs:
        changed = [
            index
            for index, (original, swapped) in enumerate(
                zip(pair.original_input_ids, pair.swapped_input_ids)
            )
            if original != swapped
        ]
        assert changed == [pair.token_position]
        assert tokenizer.token_byte_length(pair.original_token_id) == 2
        assert tokenizer.token_byte_length(pair.swapped_token_id) == 2


def test_component_intervention_matches_complete_model_endpoints():
    tokenizer = _tokenizer()
    pairs, _, _, _ = build_intervention_pairs((_record(),), tokenizer, 256)
    pair = pairs[0]
    for factorized in (False, True):
        torch.manual_seed(2036)
        model = _model(tokenizer, factorized=factorized)
        original = torch.tensor([pair.original_input_ids])
        swapped = torch.tensor([pair.swapped_input_ids])
        lengths = torch.tensor([pair.sequence_tokens])
        prompt_counts = torch.tensor([pair.prompt_token_count])
        positions = torch.tensor([pair.token_position])
        logits = component_boundary_logits(
            model,
            original,
            swapped,
            lengths,
            positions,
            torch.tensor([pair.original_token_id]),
            torch.tensor([pair.swapped_token_id]),
        )
        original_output = model(original, lengths, prompt_counts)
        swapped_output = model(swapped, lengths, prompt_counts)
        if factorized:
            assert original_output.boundary_logits is not None
            assert swapped_output.boundary_logits is not None
            expected_original = original_output.boundary_logits[
                0, pair.token_position, 0
            ]
            expected_swapped = swapped_output.boundary_logits[
                0, pair.token_position, 0
            ]
        else:
            expected_original = _collapse_legacy_logits(
                original_output.tag_logits[0, pair.token_position, 0]
            )
            expected_swapped = _collapse_legacy_logits(
                swapped_output.tag_logits[0, pair.token_position, 0]
            )
        assert torch.allclose(logits["original"][0], expected_original)
        assert torch.allclose(logits["full_swap"][0], expected_swapped)
        assert set(logits) == {
            "original",
            "byte_swap_only",
            "context_swap_only",
            "full_swap",
        }


def test_model_summary_normalizes_both_swap_directions_to_ro_minus_or():
    def prediction(label, margin):
        return {
            "predicted_class": label,
            "begin_minus_inside_margin": margin,
        }

    results = (
        {
            "source_token": "or",
            "swapped_token": "ro",
            "conditions": {
                "original": prediction("B", 1.0),
                "byte_swap_only": prediction("I", 0.25),
                "context_swap_only": prediction("B", 0.75),
                "full_swap": prediction("I", 0.0),
            },
        },
        {
            "source_token": "ro",
            "swapped_token": "or",
            "conditions": {
                "original": prediction("I", 0.0),
                "byte_swap_only": prediction("B", 0.75),
                "context_swap_only": prediction("I", 0.25),
                "full_swap": prediction("B", 1.0),
            },
        },
    )

    summary = summarize_model_results(results)

    assert summary["rendered_identity"]["or"]["begin_as_inside_rate"] == 0.0
    assert summary["rendered_identity"]["ro"]["begin_as_inside_rate"] == 1.0
    assert summary["pathway"]["mean_full_ro_minus_or_margin"] == -1.0
    assert summary["pathway"]["mean_byte_ro_minus_or_margin"] == -0.75
    assert summary["pathway"]["mean_context_ro_minus_or_margin"] == -0.25
