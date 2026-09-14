from copy import deepcopy

import pytest
import torch

from scripts.phase36c_begin_only_identity_invariance_decision import (
    begin_only_identity_invariance_decision,
)
from story_model.explicit_offset_candidate_proposer import begin_tag, inside_tag
from story_model.paired_identity_invariance import (
    IdentitySwap,
    eligible_swap_candidates,
    identity_swap_position_counts,
    paired_identity_invariance_batch,
    supervised_diagnostic_losses,
)
from story_model.provenance import canonical_json_sha256


def test_begin_only_batch_never_pairs_an_inside_position():
    targets = torch.full((3, 4, 1), -100, dtype=torch.long)
    for row in range(3):
        targets[row, 0, 0] = begin_tag("route")
        targets[row, 1, 0] = inside_tag("route")
    tokens = torch.tensor([[10, 11, 0, 0]] * 3, dtype=torch.long)
    batch = (tokens, torch.tensor([2] * 3), torch.tensor([2] * 3), targets)
    augmented, swaps = paired_identity_invariance_batch(
        batch, _WidthOneTokenizer(), {1: (20, 21, 22)}, begin_only=True
    )
    assert swaps
    assert all(swap.gold_tag == begin_tag("route") for swap in swaps)
    assert all(swap.token_position == 0 for swap in swaps)


class _WidthOneTokenizer:
    def token_byte_length(self, token_id: int) -> int:
        return 1


def test_eligible_candidates_detects_begin_only_shrinkage():
    # Position 0 (B) has no valid replacement (its only pool entry is itself);
    # position 1 (I) does. Begin-and-inside finds one candidate; begin-only
    # finds none: this is the shrinkage the hard audit must catch.
    tokens = torch.tensor([999, 555])
    targets = torch.tensor([[begin_tag("route")], [inside_tag("route")]])
    pool_by_width = {2: (999,)}
    tokenizer = _WidthTwoTokenizer()

    begin_and_inside = eligible_swap_candidates(
        tokens, 2, targets, tokenizer, pool_by_width, begin_only=False
    )
    begin_only = eligible_swap_candidates(
        tokens, 2, targets, tokenizer, pool_by_width, begin_only=True
    )
    assert len(begin_and_inside) == 1
    assert len(begin_only) == 0


class _WidthTwoTokenizer:
    def token_byte_length(self, token_id: int) -> int:
        return 2


def test_supervised_diagnostic_losses_splits_original_from_swapped():
    num_classes = 4
    logits = torch.randn(3, 2, 1, num_classes)
    targets = torch.randint(0, num_classes, (3, 2, 1))
    base_loss, swapped_loss = supervised_diagnostic_losses(logits, targets, 2)
    assert base_loss >= 0.0
    assert swapped_loss >= 0.0
    # No swap occurred: original_row_count equals the full batch.
    base_only, swapped_only = supervised_diagnostic_losses(logits, targets, 3)
    assert swapped_only == 0.0
    assert base_only >= 0.0


def test_identity_swap_position_counts_separates_begin_from_inside():
    begin, inside = begin_tag("route"), inside_tag("route")
    swaps = (
        IdentitySwap(0, 1, 0, 0, 10, 11, 2, begin),
        IdentitySwap(0, 2, 3, 0, 12, 13, 2, inside),
        IdentitySwap(0, 3, 4, 0, 14, 15, 2, begin),
    )
    counts = identity_swap_position_counts(swaps)
    assert counts == {"B": 2, "I": 1}


TRAINED = tuple(range(300, 308))
SEVERE = (366, 433, 493, 357)


def _identity_cell(error, precision=1.0, recall=1.0):
    return {
        "metrics": {"exact_span_precision": precision, "exact_span_recall": recall},
        "focus": {
            "overall": {
                "gold_begin_count": 25,
                "gold_inside_count": 25,
                "begin_as_inside_rate": error,
            }
        },
    }


def _metric():
    return {
        "gold_begin_as_inside_rate": 0.0,
        "gold_outside_as_inside_rate": 0.0,
        "positive_type_accuracy": 1.0,
        "end_spill_rate": 0.0,
    }


def _retained(checkpoint_sha, eligible=False, version=None):
    return {
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_eligible": eligible,
        "checkpoint_paired_identity_invariance_version": version,
        "datasets": {
            "phase31_regression": {
                "splits": {
                    split: {"metrics": _metric(), "per_skill": {"scene_route": _metric()}}
                    for split in ("lexical", "transfer")
                }
            },
            "phase32": {"splits": {"val": {"metrics": _metric(), "per_skill": {}}}},
        },
    }


def _checkpoint(sha, version, position_mode, swap_pool_hash, extra_counts=None):
    return {
        "sha256": sha,
        "architecture": "explicit_offset_candidate_proposer",
        "boundary_objective_version": 1,
        "boundary_loss_weight": 1.0,
        "boundary_counterbalance_version": 2,
        "paired_identity_invariance_version": version,
        "identity_invariance_position_mode": position_mode,
        "identity_invariance_swap_pool_sha256": swap_pool_hash,
        "identity_invariance_total_swap_position_counts": extra_counts,
        "tokenizer_sha256": "tok",
    }


def _crossed(precision_drop=0.0, transfer_error=0.0):
    pool = {"excluded_registered_token_ids": sorted(TRAINED + SEVERE + (270, 357))}
    pool_hash = canonical_json_sha256(pool)
    all_ids = TRAINED + SEVERE
    crossed = {
        "begin_only_identity_invariance_audit_version": 1,
        "decoder_changes": "none",
        "architecture_changes": "none",
        "identity_swap_pool_sha256": pool_hash,
        "focus_identities": {
            "trained": [{"token_id": token_id, "text": f"t{token_id}"} for token_id in TRAINED],
            "heldout": [{"token_id": token_id, "text": f"h{token_id}"} for token_id in SEVERE],
            "legacy": [],
        },
        "phase36c_position_counts": {"B": 40, "I": 0},
        "panel_validation": {
            split: {"invalid_contexts": 0, "identities": 14}
            for split in ("lexical", "transfer")
        },
        "swap_opportunity_audit": {
            source: {
                "begin_and_inside_eligible_rows": 100,
                "begin_only_eligible_rows": 100,
                "eligible_rows_did_not_fall": True,
            }
            for source in ("phase31_train", "phase32_train", "counterbalance_train")
        },
        "models": {
            "phase35": {
                "checkpoint": _checkpoint("p35", None, None, None),
                "splits": {},
            },
            "phase36b": {
                "checkpoint": _checkpoint("p36b", 1, "begin_and_inside", pool_hash),
                "splits": {},
            },
            "phase36c": {
                "checkpoint": _checkpoint(
                    "p36c", 2, "begin_only", pool_hash, {"B": 40, "I": 0}
                ),
                "splits": {},
            },
        },
    }
    for split in ("lexical", "transfer"):
        phase35_identities = {
            str(token_id): _identity_cell(0.0) for token_id in all_ids
        }
        phase36c_identities = {
            str(token_id): _identity_cell(
                transfer_error if token_id in SEVERE else 0.0,
                precision=1.0 - precision_drop,
            )
            for token_id in all_ids
        }
        crossed["models"]["phase35"]["splits"][split] = {"identities": phase35_identities}
        crossed["models"]["phase36c"]["splits"][split] = {"identities": phase36c_identities}
    return crossed


def test_diagnostic_pass_when_transfer_span_and_retained_all_hold():
    crossed = _crossed()
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=True, version=2)
    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)
    assert decision["branch"] == "begin_only_identity_invariance_diagnostic_pass"
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False


def test_transfer_regression_branch_when_severe_identities_fail():
    crossed = _crossed(transfer_error=0.5)
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=True, version=2)
    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)
    assert decision["branch"] == (
        "begin_only_identity_invariance_span_recovered_transfer_regressed"
    )


def test_span_damage_branch_when_transfer_holds_but_span_falls():
    crossed = _crossed(precision_drop=0.5)
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=True, version=2)
    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)
    assert decision["branch"] == (
        "begin_only_identity_invariance_transfer_preserved_span_still_damaged"
    )


def test_regressed_retained_branch_when_checkpoint_is_ineligible():
    crossed = _crossed()
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=False, version=2)
    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)
    assert decision["branch"] == "begin_only_identity_invariance_regressed_retained"


def test_invalid_branch_when_an_i_route_pair_was_generated_without_mutation():
    crossed = _crossed()
    crossed["phase36c_position_counts"]["I"] = 1
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=True, version=2)
    original = deepcopy((crossed, baseline, candidate))

    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)

    assert decision["branch"] == "invalid_begin_only_identity_invariance_comparison"
    assert (crossed, baseline, candidate) == original


def test_invalid_branch_when_swap_opportunity_fell():
    crossed = _crossed()
    crossed["swap_opportunity_audit"]["phase31_train"]["eligible_rows_did_not_fall"] = False
    baseline = _retained("p35")
    candidate = _retained("p36c", eligible=True, version=2)
    decision = begin_only_identity_invariance_decision(crossed, baseline, candidate)
    assert decision["branch"] == "invalid_begin_only_identity_invariance_comparison"
