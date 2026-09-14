from copy import deepcopy

from scripts.phase36d_clean_anchor_identity_invariance_decision import (
    clean_anchor_identity_invariance_decision,
)

TRAINED = (307, 361, 503, 274, 295, 475, 296, 436)
HELDOUT = (493, 366, 433, 424)
LEGACY = (270, 357)
ALL_IDS = TRAINED + HELDOUT + LEGACY


def _identity_cell(error=0.0, precision=1.0, recall=1.0):
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


def _retained(sha, eligible=False, version=None):
    return {
        "checkpoint_sha256": sha,
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


def _crossed(error_ids=(), precision_drop_ids=()):
    crossed = {
        "clean_anchor_identity_invariance_audit_version": 1,
        "decoder_changes": "none",
        "architecture_changes": "none",
        "focus_identities": {
            "trained": [{"token_id": i, "text": f"t{i}"} for i in TRAINED],
            "heldout": [{"token_id": i, "text": f"h{i}"} for i in HELDOUT],
            "legacy": [{"token_id": i, "text": f"l{i}"} for i in LEGACY],
        },
        "panel_validation": {
            split: {"invalid_contexts": 0, "identities": 14}
            for split in ("lexical", "transfer")
        },
        "models": {
            "phase35": {
                "checkpoint": {
                    "sha256": "p35",
                    "architecture": "explicit_offset_candidate_proposer",
                    "boundary_objective_version": 1,
                    "boundary_loss_weight": 1.0,
                    "paired_identity_invariance_version": None,
                    "tokenizer_sha256": "tok",
                },
                "splits": {},
            },
            "phase36d": {
                "checkpoint": {
                    "sha256": "p36d",
                    "architecture": "explicit_offset_candidate_proposer",
                    "boundary_objective_version": 1,
                    "boundary_loss_weight": 1.0,
                    "boundary_counterbalance_version": 2,
                    "paired_identity_invariance_version": 3,
                    "identity_invariance_supervision_mode": "clean_anchor",
                    "tokenizer_sha256": "tok",
                },
                "splits": {},
            },
        },
    }
    for split in ("lexical", "transfer"):
        before, after = {}, {}
        for token_id in ALL_IDS:
            before[str(token_id)] = _identity_cell()
            after[str(token_id)] = _identity_cell(
                error=0.5 if token_id in error_ids else 0.0,
                precision=0.5 if token_id in precision_drop_ids else 1.0,
            )
        crossed["models"]["phase35"]["splits"][split] = {"identities": before}
        crossed["models"]["phase36d"]["splits"][split] = {"identities": after}
    return crossed


def test_diagnostic_pass_when_everything_holds():
    decision = clean_anchor_identity_invariance_decision(
        _crossed(), _retained("p35"), _retained("p36d", eligible=True, version=3)
    )
    assert decision["branch"] == "clean_anchor_identity_invariance_diagnostic_pass"
    assert decision["full_phase34_evaluation_authorized"] is True
    assert decision["checkpoint_promotion_authorized"] is False


def test_transfer_failed_branch_when_any_of_14_identities_regress():
    decision = clean_anchor_identity_invariance_decision(
        _crossed(error_ids=(357,)), _retained("p35"),
        _retained("p36d", eligible=True, version=3),
    )
    assert decision["branch"] == "clean_anchor_identity_invariance_transfer_failed"


def test_span_still_damaged_branch_when_transfer_holds_but_span_falls():
    decision = clean_anchor_identity_invariance_decision(
        _crossed(precision_drop_ids=ALL_IDS), _retained("p35"),
        _retained("p36d", eligible=True, version=3),
    )
    assert decision["branch"] == "clean_anchor_identity_invariance_span_still_damaged"


def test_regressed_retained_branch_when_checkpoint_ineligible():
    decision = clean_anchor_identity_invariance_decision(
        _crossed(), _retained("p35"), _retained("p36d", eligible=False, version=3)
    )
    assert decision["branch"] == "clean_anchor_identity_invariance_regressed_retained"


def test_invalid_branch_does_not_mutate_inputs():
    crossed = _crossed()
    crossed["models"]["phase36d"]["checkpoint"]["paired_identity_invariance_version"] = 99
    baseline = _retained("p35")
    candidate = _retained("p36d", eligible=True, version=3)
    original = deepcopy((crossed, baseline, candidate))

    decision = clean_anchor_identity_invariance_decision(crossed, baseline, candidate)

    assert decision["branch"] == "invalid_clean_anchor_identity_invariance_comparison"
    assert (crossed, baseline, candidate) == original
