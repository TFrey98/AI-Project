"""Tests for Phase 36g: the bounded clean-anchor consistency-weight ablation.

Documented in docs/phase36g_consistency_weight_ablation.md. Two groups:

1. The `comparison_step` config contract. Phase 36g compares two arms at one
   predeclared step rather than at whatever step each arm's selection logic
   happened to keep, so the step must be validated up front: positive, not
   past `max_steps`, and landing on an evaluation step (otherwise the save
   would never fire and the comparison would silently not exist).
2. The Phase 36g decision logic, covering all four registered outcomes plus
   the invalid branch, and the non-mutation guarantee shared by every
   decision script in this chain.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.phase36g_consistency_weight_decision import (
    consistency_weight_decision,
)
from scripts.train_explicit_offset_candidate_proposer import _load_config

ROOT = Path(__file__).parents[1]
REFERENCE_CONFIG = ROOT / "configs/explicit_offset_clean_anchor_identity_invariance_pilot.yaml"
CANDIDATE_CONFIG = ROOT / "configs/explicit_offset_consistency_weight_0p1_pilot.yaml"


def _write(tmp_path, config) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_phase36g_config_varies_only_the_consistency_weight():
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))

    assert reference["model"] == candidate["model"]
    assert reference["tokenizer"] == candidate["tokenizer"]
    assert reference["data"] == candidate["data"]
    assert candidate["train"]["identity_invariance_loss_weight"] == 0.1
    assert reference["train"]["identity_invariance_loss_weight"] == 1.0
    ignored = {"identity_invariance_loss_weight", "comparison_step"}
    assert {
        key: value for key, value in reference["train"].items() if key not in ignored
    } == {
        key: value for key, value in candidate["train"].items() if key not in ignored
    }


def test_comparison_step_must_land_on_an_evaluation_step(tmp_path):
    config = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    config["train"]["comparison_step"] = 801
    with pytest.raises(ValueError, match="evaluation step"):
        _load_config(_write(tmp_path, config))


def test_comparison_step_cannot_exceed_max_steps(tmp_path):
    config = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    config["train"]["comparison_step"] = 2000
    with pytest.raises(ValueError, match="cannot exceed max_steps"):
        _load_config(_write(tmp_path, config))


def test_comparison_step_must_be_a_positive_integer(tmp_path):
    config = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    config["train"]["comparison_step"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        _load_config(_write(tmp_path, config))


def test_registered_candidate_config_loads():
    config = _load_config(CANDIDATE_CONFIG)
    assert config["train"]["comparison_step"] == 800
    assert config["train"]["identity_invariance_loss_weight"] == 0.1


TRAINED = (307, 361, 503, 274, 295, 475, 296, 436)
HELDOUT = (493, 366, 433, 424)
LEGACY = (270, 357)
ALL_IDS = TRAINED + HELDOUT + LEGACY


def _cell(error=0.0, precision=1.0, recall=1.0):
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


def _retained(sha, eligible=False):
    return {
        "checkpoint_sha256": sha,
        "checkpoint_eligible": eligible,
        "checkpoint_paired_identity_invariance_version": 3,
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


def _arm_checkpoint(sha, weight):
    return {
        "sha256": sha,
        "step": 800,
        "architecture": "explicit_offset_candidate_proposer",
        "boundary_objective_version": 1,
        "boundary_loss_weight": 1.0,
        "boundary_counterbalance_version": 2,
        "paired_identity_invariance_version": 3,
        "identity_invariance_supervision_mode": "clean_anchor",
        "identity_invariance_position_mode": "begin_and_inside",
        "identity_invariance_loss_weight": weight,
        "identity_invariance_swap_pool_sha256": "pool",
        "tokenizer_sha256": "tok",
        "parent_phase33_checkpoint": "phase33c",
        "seed": 1337,
    }


def _crossed(candidate_error=0.0, candidate_precision=1.0, reference_precision=1.0):
    crossed = {
        "consistency_weight_ablation_audit_version": 1,
        "training_changes": "consistency_weight_only",
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
                    "step": 1000,
                    "paired_identity_invariance_version": None,
                    "tokenizer_sha256": "tok",
                },
                "splits": {},
            },
            "reference": {"checkpoint": _arm_checkpoint("ref", 1.0), "splits": {}},
            "candidate": {"checkpoint": _arm_checkpoint("cand", 0.1), "splits": {}},
        },
    }
    for split in ("lexical", "transfer"):
        baseline, reference, candidate = {}, {}, {}
        for token_id in ALL_IDS:
            baseline[str(token_id)] = _cell()
            reference[str(token_id)] = _cell(precision=reference_precision)
            candidate[str(token_id)] = _cell(
                error=candidate_error, precision=candidate_precision
            )
        crossed["models"]["phase35"]["splits"][split] = {"identities": baseline}
        crossed["models"]["reference"]["splits"][split] = {"identities": reference}
        crossed["models"]["candidate"]["splits"][split] = {"identities": candidate}
    return crossed


def test_transfer_holds_and_span_recovers_is_promising():
    decision = consistency_weight_decision(
        _crossed(reference_precision=0.5), _retained("p35"), _retained("cand", True)
    )
    assert decision["branch"] == "reduced_consistency_promising"
    assert decision["checkpoint_promotion_authorized"] is False


def test_span_recovers_but_transfer_fails_is_a_tradeoff_not_a_refutation():
    decision = consistency_weight_decision(
        _crossed(candidate_error=0.5, reference_precision=0.5),
        _retained("p35"),
        _retained("cand", True),
    )
    assert decision["branch"] == "reduced_consistency_tradeoff"
    assert "intermediate" in decision["next_action"]


def test_transfer_holds_but_spans_remain_damaged():
    decision = consistency_weight_decision(
        _crossed(candidate_precision=0.5), _retained("p35"), _retained("cand", True)
    )
    assert decision["branch"] == "reduced_consistency_no_span_recovery"


def test_both_worsen_rejects_the_candidate():
    decision = consistency_weight_decision(
        _crossed(candidate_error=0.5, candidate_precision=0.5),
        _retained("p35"),
        _retained("cand", True),
    )
    assert decision["branch"] == "reduced_consistency_rejected"


def test_mismatched_arm_step_invalidates_without_mutation():
    crossed = _crossed()
    crossed["models"]["candidate"]["checkpoint"]["step"] = 750
    baseline, candidate = _retained("p35"), _retained("cand", True)
    original = deepcopy((crossed, baseline, candidate))

    decision = consistency_weight_decision(crossed, baseline, candidate)

    assert decision["branch"] == "invalid_consistency_weight_comparison"
    assert (crossed, baseline, candidate) == original


def test_arms_differing_in_more_than_weight_are_invalid():
    crossed = _crossed()
    crossed["models"]["candidate"]["checkpoint"][
        "identity_invariance_position_mode"
    ] = "begin_only"
    decision = consistency_weight_decision(
        crossed, _retained("p35"), _retained("cand", True)
    )
    assert decision["branch"] == "invalid_consistency_weight_comparison"


def test_eligibility_is_reported_independently_of_metric_outcomes():
    decision = consistency_weight_decision(
        _crossed(reference_precision=0.5), _retained("p35"), _retained("cand", False)
    )
    assert decision["candidate_checkpoint_eligible"] is False
    assert decision["transfer_ok"] is True
    assert decision["branch"] == "reduced_consistency_promising"
    assert decision["full_phase34_evaluation_authorized"] is False
