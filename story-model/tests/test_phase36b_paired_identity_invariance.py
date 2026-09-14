from copy import deepcopy

import pytest
import torch

from scripts.phase36b_identity_invariance_decision import identity_invariance_decision
from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import EXPANDED_CONTROL_TOKENS
from story_model.explicit_offset_candidate_proposer import PROPOSAL_TYPES, begin_tag, inside_tag
from story_model.paired_identity_invariance import (
    IdentitySwap,
    apply_same_width_swap,
    build_identity_swap_pool,
    paired_identity_invariance_batch,
    paired_identity_invariance_loss,
    registered_identity_ids,
    validate_identity_swap_pool,
)
from story_model.provenance import canonical_json_sha256


def _tokenizer():
    pairs = ("ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr",
             "st", "uv", "wx", "yz", "or", "ro", "aa")
    return ByteBPETokenizer(
        merges=[(ord(a), ord(b)) for a, b in pairs]
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def _manifest():
    return {
        "focus_tokens": {
            "train": [{"token_id": i} for i in range(256, 264)],
            "heldout": [{"token_id": i} for i in range(264, 268)],
        },
        "phase34k_premise": {"focus_token_ids": {"or": 268, "ro": 269}},
    }


def test_swap_pool_excludes_special_and_all_registered_identities():
    tokenizer = _tokenizer()
    excluded = registered_identity_ids(_manifest())
    pool = validate_identity_swap_pool(
        build_identity_swap_pool(tokenizer, excluded), tokenizer, excluded
    )
    eligible = {token_id for values in pool.values() for token_id in values}
    assert not eligible & set(excluded)
    assert not eligible & set(tokenizer.special_token_ids.values())
    assert all(tokenizer.token_byte_length(i) == width
               for width, values in pool.items() for i in values)


def test_same_width_swap_rejects_width_mismatch():
    with pytest.raises(ValueError, match="width mismatch"):
        apply_same_width_swap(torch.tensor([[256]]), 0, 0, ord("a"), _tokenizer())


def test_paired_batch_changes_only_input_id_and_preserves_labels():
    tokenizer = _tokenizer()
    excluded = registered_identity_ids(_manifest())
    pool = validate_identity_swap_pool(
        build_identity_swap_pool(tokenizer, excluded), tokenizer, excluded
    )
    targets = torch.full((1, 4, 2), -100, dtype=torch.long)
    targets[0, 0] = torch.tensor([begin_tag("route"), inside_tag("route")])
    batch = (torch.tensor([[256, 42, 0, 0]]), torch.tensor([2]),
             torch.tensor([1]), targets)
    augmented, swaps = paired_identity_invariance_batch(batch, tokenizer, pool)
    assert len(swaps) == 1
    assert torch.equal(augmented[3][0], augmented[3][1])
    assert torch.equal(augmented[1][0], augmented[1][1])
    assert torch.nonzero(augmented[0][0] != augmented[0][1]).flatten().tolist() == [0]


def test_consistency_loss_zero_on_agreement_positive_on_disagreement():
    logits = torch.zeros((2, 1, 1, 1 + 2 * len(PROPOSAL_TYPES)))
    begin, inside = begin_tag("route"), inside_tag("route")
    logits[:, 0, 0, begin], logits[:, 0, 0, inside] = 2.0, -1.0
    swap = IdentitySwap(0, 1, 0, 0, 256, 270, 2, begin)
    assert float(paired_identity_invariance_loss(logits, (swap,))) == pytest.approx(0, abs=1e-7)
    logits[1, 0, 0, begin], logits[1, 0, 0, inside] = -2.0, 2.0
    assert float(paired_identity_invariance_loss(logits, (swap,))) > 0


TRAINED = tuple(range(300, 308))
HELDOUT = (366, 433, 493, 424)
LEGACY = (357, 270)
SEVERE = {366, 433, 493, 357}


def _cell(error, group):
    return {"group": group,
            "metrics": {"exact_span_precision": 1.0, "exact_span_recall": 1.0},
            "focus": {"overall": {"gold_begin_count": 25,
                                    "gold_inside_count": 25,
                                    "begin_as_inside_rate": error}}}


def _metric(type_accuracy=1.0):
    return {"gold_begin_as_inside_rate": 0.0,
            "gold_outside_as_inside_rate": 0.0,
            "positive_type_accuracy": type_accuracy, "end_spill_rate": 0.0}


def _retained(checkpoint, candidate=False):
    return {
        "checkpoint_sha256": checkpoint,
        "checkpoint_eligible": candidate,
        "checkpoint_paired_identity_invariance_version": 1 if candidate else None,
        "checkpoint_identity_invariance_swap_pool_sha256": None,
        "datasets": {"phase31_regression": {"splits": {
            split: {"metrics": _metric(), "per_skill": {"scene_route": _metric()}}
            for split in ("lexical", "transfer")}},
            "phase32": {"splits": {"val": {"metrics": _metric(),
                                               "per_skill": {}}}}},
    }


def _inputs(error=0.0):
    groups = {"trained": TRAINED, "heldout": HELDOUT, "legacy": LEGACY}
    all_ids = set(TRAINED + HELDOUT + LEGACY)
    pool = {"excluded_registered_token_ids": sorted(all_ids)}
    pool_hash = canonical_json_sha256(pool)
    common = {"architecture": "explicit_offset_candidate_proposer",
              "boundary_objective_version": 1, "boundary_loss_weight": 1.0,
              "boundary_counterbalance_version": 2,
              "token_width_geometry_version": None,
              "token_end_geometry_version": None,
              "factorized_boundary_type_version": None,
              "tokenizer_sha256": "tok", "counterbalance_manifest_sha256": "man",
              "parent_phase33_checkpoint": "parent", "parent_phase33_step": 1000}
    crossed = {
        "paired_identity_invariance_audit_version": 1, "decoder_changes": "none",
        "phase36a_inputs": {"decision_branch": "trained_identity_memorization_dominant",
                            "phase35_checkpoint_sha256": "p35",
                            "checkpoint_promotion_authorized": False,
                            "full_phase34_evaluation_authorized": False},
        "identity_swap_pool": pool, "identity_swap_pool_sha256": pool_hash,
        "focus_identities": {g: [{"token_id": i} for i in ids]
                             for g, ids in groups.items()},
        "panel_validation": {s: {"invalid_contexts": 0, "identities": 14}
                             for s in ("lexical", "transfer")},
        "models": {
            "phase35": {"checkpoint": {**common, "sha256": "p35",
                                           "checkpoint_eligible": False,
                                           "paired_identity_invariance_version": None},
                        "splits": {}},
            "phase36b": {"checkpoint": {**common, "sha256": "p36",
                                            "checkpoint_eligible": True,
                                            "paired_identity_invariance_version": 1,
                                            "identity_invariance_loss_weight": 1.0,
                                            "identity_invariance_swap_pool_sha256": pool_hash},
                         "splits": {}},
        },
    }
    for split in ("lexical", "transfer"):
        before, after = {}, {}
        for group, ids in groups.items():
            for token_id in ids:
                before[str(token_id)] = _cell(0.5 if token_id in SEVERE else 0.0, group)
                after[str(token_id)] = _cell(error if token_id in SEVERE else 0.0, group)
        crossed["models"]["phase35"]["splits"][split] = {"identities": before}
        crossed["models"]["phase36b"]["splits"][split] = {"identities": after}
    baseline, candidate = _retained("p35"), _retained("p36", True)
    candidate["checkpoint_identity_invariance_swap_pool_sha256"] = pool_hash
    return crossed, baseline, candidate


@pytest.mark.parametrize("error,branch", [
    (0.0, "identity_invariance_diagnostic_pass"),
    (0.25, "identity_invariance_partial_transfer"),
    (0.5, "identity_invariance_no_transfer"),
])
def test_decision_transfer_branches(error, branch):
    assert identity_invariance_decision(*_inputs(error))["branch"] == branch


def test_decision_regression_and_invalid_provenance_are_non_mutating():
    crossed, baseline, candidate = _inputs(0.0)
    crossed["models"]["phase36b"]["splits"]["lexical"]["identities"]["300"] = _cell(0.1, "trained")
    assert identity_invariance_decision(crossed, baseline, candidate)["branch"] == "identity_invariance_regressed_trained"
    crossed, baseline, candidate = _inputs(0.0)
    crossed["identity_swap_pool_sha256"] = "wrong"
    original = deepcopy((crossed, baseline, candidate))
    assert identity_invariance_decision(crossed, baseline, candidate)["branch"] == "invalid_identity_invariance_comparison"
    assert (crossed, baseline, candidate) == original
