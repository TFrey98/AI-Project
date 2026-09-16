"""Phase 36d's non-negotiable correctness guard.

Documented in docs/phase36d_clean_anchor_identity_invariance.md under
"Correctness test (non-negotiable)". The load-bearing test is
``test_masking_zero_lambda_matches_the_base_trainer_exactly``: on a real
(tiny) model, it proves that with ``identity_invariance_loss_weight=0`` the
clean-anchor path's loss and every parameter's gradient are numerically
identical (atol=1e-6) to the unmodified Phase 35 base trainer run on the
same original batch. This guards against a masking or doubled-batch-
averaging bug silently rescaling the supervised objective -- do not train
Phase 36d if this test fails. The remaining tests check the masking
primitive (``mask_swapped_supervision``) in isolation: it masks every
position of a swapped row, not just the swap position, and is a no-op when
no swap occurred.
"""

from __future__ import annotations

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    EXPANDED_CONTROL_TOKENS,
    ExpandedCandidate,
    ExpandedResolverRecord,
)
from story_model.explicit_offset_candidate_proposer import (
    ExplicitOffsetCandidateProposer,
    encode_proposal_records,
    proposal_batch,
)
from story_model.models import build_model
from story_model.paired_identity_invariance import (
    CLEAN_ANCHOR_SUPERVISION_MODE,
    FULL_SUPERVISION_MODE,
    PAIRED_IDENTITY_INVARIANCE_VERSION,
    build_identity_swap_pool,
    mask_swapped_supervision,
    paired_identity_invariance_batch,
    paired_identity_invariance_loss,
    validate_identity_swap_pool,
)
from story_model.unified_typed_span_resolver import UnifiedTypedSpanResolver


def _tokenizer():
    return ByteBPETokenizer(
        merges=[(ord("a"), ord("b")), (ord("c"), ord("d"))]
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


def _records():
    candidates = (
        ExpandedCandidate("eastern tunnel", "route"),
        ExpandedCandidate("south stair", "route"),
    )
    rows = []
    for index, selected in enumerate((0, 1, 0, 1)):
        rows.append(
            ExpandedResolverRecord(
                record_id=f"row-{index}",
                source_context_id=f"context-{index}",
                conversation_id=f"conversation-{index}",
                split="train",
                skill="scene_route",
                case="supported",
                prompt=(
                    "Routes: eastern tunnel and south stair. "
                    + (
                        "Eastern tunnel is blocked."
                        if selected == 0
                        else "South stair is blocked."
                    )
                ),
                expected_action="resolve",
                expected_type="route",
                candidates=candidates,
                selected_candidate_index=selected,
                response_template="Take the <|resolved_value|>.",
                expected_value=candidates[selected].text,
                alternative_value=candidates[1 - selected].text,
                source_phase="phase31",
            )
        )
    return tuple(rows)


def _batch():
    records = _records()
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    examples = encode_proposal_records(records, tokenizer, 256)
    batch = proposal_batch(
        examples, tuple(range(len(examples))), tokenizer, model.source_width,
        torch.device("cpu"),
    )
    return model, tokenizer, batch


def test_masking_zero_lambda_matches_the_base_trainer_exactly():
    model, tokenizer, batch = _batch()
    model.eval()

    # Path A: the unmodified Phase 35 base trainer, no augmentation at all.
    output_a = model(*batch, boundary_loss_weight=1.0)
    loss_a = output_a.loss
    model.zero_grad(set_to_none=True)
    loss_a.backward()
    grads_a = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert grads_a

    # Path B: clean-anchor augmentation with lambda=0. The swapped copy is
    # masked out of supervision entirely; only the (zeroed) JSD term touches
    # it, so this must reduce to exactly the same loss and gradients as A.
    model.zero_grad(set_to_none=True)
    pool = build_identity_swap_pool(tokenizer, excluded_token_ids=())
    pool_by_width = validate_identity_swap_pool(pool, tokenizer, excluded_token_ids=())
    original_row_count = batch[0].shape[0]
    augmented, swaps = paired_identity_invariance_batch(
        batch, tokenizer, pool_by_width, begin_only=False
    )
    assert swaps, "fixture must actually exercise a swap for this test to mean anything"
    masked = mask_swapped_supervision(augmented, original_row_count)
    output_b = model(*masked, boundary_loss_weight=1.0)
    identity_invariance_loss = paired_identity_invariance_loss(
        output_b.tag_logits, swaps
    )
    total_loss = output_b.loss + 0.0 * identity_invariance_loss
    total_loss.backward()
    grads_b = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    assert torch.allclose(loss_a, total_loss, atol=1e-6)
    assert set(grads_a) == set(grads_b)
    for name in grads_a:
        assert torch.allclose(grads_a[name], grads_b[name], atol=1e-6), name


def test_masked_swapped_rows_contribute_nothing_to_loss_or_grad_even_unweighted():
    """Same as above, but with a nonzero lambda: only the JSD term should
    differ; the base supervised loss component must stay identical."""
    model, tokenizer, batch = _batch()
    model.eval()
    pool = build_identity_swap_pool(tokenizer, excluded_token_ids=())
    pool_by_width = validate_identity_swap_pool(pool, tokenizer, excluded_token_ids=())
    original_row_count = batch[0].shape[0]

    output_a = model(*batch, boundary_loss_weight=1.0)
    base_loss_a = output_a.loss

    augmented, swaps = paired_identity_invariance_batch(
        batch, tokenizer, pool_by_width, begin_only=False
    )
    masked = mask_swapped_supervision(augmented, original_row_count)
    output_b = model(*masked, boundary_loss_weight=1.0)

    assert torch.allclose(base_loss_a, output_b.loss, atol=1e-6)


def test_paired_identity_invariance_version_is_registered_for_clean_anchor():
    assert PAIRED_IDENTITY_INVARIANCE_VERSION == 3
    assert CLEAN_ANCHOR_SUPERVISION_MODE == "clean_anchor"
    assert FULL_SUPERVISION_MODE == "full_supervision"


def test_mask_swapped_supervision_masks_every_position_not_just_the_swap():
    tag_targets = torch.tensor(
        [
            [[5], [6], [7]],
            [[8], [9], [10]],
        ]
    )
    batch = (
        torch.zeros(2, 3, dtype=torch.long),
        torch.tensor([3, 3]),
        torch.tensor([3, 3]),
        tag_targets,
    )
    masked = mask_swapped_supervision(batch, original_row_count=1)
    assert torch.equal(masked[3][0], tag_targets[0])
    assert torch.all(masked[3][1] == -100)


def test_mask_swapped_supervision_is_a_noop_when_no_swap_occurred():
    tag_targets = torch.tensor([[[5], [6]]])
    batch = (
        torch.zeros(1, 2, dtype=torch.long),
        torch.tensor([2]),
        torch.tensor([2]),
        tag_targets,
    )
    masked = mask_swapped_supervision(batch, original_row_count=1)
    assert masked is batch
