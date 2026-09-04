from __future__ import annotations

from unittest.mock import patch

import torch

from scripts.audit_context_path_intervention import (
    _validate_replay,
    context_path_boundary_logits,
    summarize_model_results,
)
from scripts.audit_paired_token_intervention import (
    _collapse_legacy_logits,
    build_intervention_pairs,
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
        record_id="phase34k-scene-route",
        source_context_id="phase34k-source",
        conversation_id="phase34k-conversation",
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


def test_context_decomposition_matches_full_model_endpoints_and_one_forward():
    tokenizer = _tokenizer()
    pairs, _, _, _ = build_intervention_pairs((_record(),), tokenizer, 256)
    pair = pairs[0]
    for factorized in (False, True):
        torch.manual_seed(2041)
        model = _model(tokenizer, factorized=factorized)
        original = torch.tensor([pair.original_input_ids])
        swapped = torch.tensor([pair.swapped_input_ids])
        lengths = torch.tensor([pair.sequence_tokens])
        prompt_counts = torch.tensor([pair.prompt_token_count])
        positions = torch.tensor([pair.token_position])
        hidden_states = model.resolver.hidden_states
        with patch.object(
            model.resolver,
            "hidden_states",
            wraps=hidden_states,
        ) as observed:
            logits = context_path_boundary_logits(
                model,
                original,
                swapped,
                lengths,
                positions,
                torch.tensor([pair.original_token_id]),
                torch.tensor([pair.swapped_token_id]),
            )
        assert observed.call_count == 1

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
        assert torch.allclose(logits["full_token_swap"][0], expected_swapped)
        assert set(logits) == {
            "original",
            "local_key_swap_only",
            "query_swap_only",
            "full_context_swap",
            "full_token_swap",
        }


def test_summary_normalizes_each_component_to_ro_minus_or_by_stratum():
    def prediction(label, margin):
        return {
            "predicted_class": label,
            "begin_minus_inside_margin": margin,
        }

    results = (
        {
            "source_token": "or",
            "conditions": {
                "original": prediction("B", 1.0),
                "local_key_swap_only": prediction("B", 0.75),
                "query_swap_only": prediction("B", 0.50),
                "full_context_swap": prediction("I", 0.25),
                "full_token_swap": prediction("I", 0.0),
            },
        },
        {
            "source_token": "ro",
            "conditions": {
                "original": prediction("I", 0.0),
                "local_key_swap_only": prediction("I", 0.25),
                "query_swap_only": prediction("I", 0.50),
                "full_context_swap": prediction("B", 0.75),
                "full_token_swap": prediction("B", 1.0),
            },
        },
    )

    summary = summarize_model_results(results)

    assert summary["pathway"]["mean_local_key_ro_minus_or_margin"] == -0.25
    assert summary["pathway"]["mean_query_ro_minus_or_margin"] == -0.50
    assert (
        summary["pathway"]["mean_full_context_ro_minus_or_margin"]
        == -0.75
    )
    assert (
        summary["pathway"]["mean_full_token_ro_minus_or_margin"] == -1.0
    )
    assert summary["source_strata"]["or"]["pathway"]["observations"] == 1
    assert summary["source_strata"]["ro"]["pathway"]["observations"] == 1


def test_replay_contract_rejects_changed_phase34j_inputs():
    class Pair:
        def __init__(self, source_token):
            self.source_token = source_token

    pairs = tuple(Pair("or") for _ in range(2)) + tuple(
        Pair("ro") for _ in range(2)
    )
    previous = {
        "pair_validation": {
            "target_focus_starts": 4,
            "valid_pairs": 4,
            "invalid_pairs": 0,
            "source_counts": {"or": 2, "ro": 2},
        },
        "focus_token_ids": {"or": 270, "ro": 357},
        "data_files": {
            "lexical": {"sha256": "lexical-sha", "rows": 400},
            "transfer": {"sha256": "transfer-sha", "rows": 400},
        },
        "models": {
            "phase34d": {"checkpoint": {"sha256": "d-sha"}},
            "phase34i": {"checkpoint": {"sha256": "i-sha"}},
        },
    }
    data_files = {
        "lexical": {"sha256": "lexical-sha", "rows": 400},
        "transfer": {"sha256": "transfer-sha", "rows": 400},
    }
    metadata = {
        "phase34d": {"sha256": "d-sha"},
        "phase34i": {"sha256": "i-sha"},
    }

    _validate_replay(
        previous,
        pairs,
        (),
        4,
        {"or": 270, "ro": 357},
        data_files,
        metadata,
    )
    changed = {
        **data_files,
        "transfer": {"sha256": "changed", "rows": 400},
    }
    try:
        _validate_replay(
            previous,
            pairs,
            (),
            4,
            {"or": 270, "ro": 357},
            changed,
            metadata,
        )
    except ValueError as error:
        assert "transfer data hash changed" in str(error)
    else:
        raise AssertionError("changed Phase 34j data was accepted")
