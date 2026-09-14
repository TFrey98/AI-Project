from copy import deepcopy

from scripts.audit_crossed_boundary_identity import (
    build_crossed_identity_panel,
)
from scripts.phase36a_crossed_identity_decision import (
    crossed_identity_decision,
)
from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    EXPANDED_CONTROL_TOKENS,
    ExpandedCandidate,
    ExpandedResolverRecord,
)


def _tokenizer():
    return ByteBPETokenizer(
        merges=[(ord("a"), ord("b")), (ord("c"), ord("d"))]
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)


def _pair(split):
    candidates = (
        ExpandedCandidate("eastern tunnel", "route"),
        ExpandedCandidate("south stair", "route"),
    )
    rows = []
    for side, selected in enumerate((1, 0)):
        rows.append(
            ExpandedResolverRecord(
                record_id=f"source-{split}-{side}",
                source_context_id=f"context-{split}-{side}",
                conversation_id=f"conversation-{split}",
                split=split,
                skill="scene_route",
                case="supported",
                prompt=(
                    "Routes: eastern tunnel and south stair. "
                    + (
                        "Eastern tunnel is blocked."
                        if side == 0
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


def test_crossed_panel_changes_only_focus_token_positions():
    tokenizer = _tokenizer()
    identities = {
        "trained": ((256, "ab"),),
        "heldout": ((257, "cd"),),
        "legacy": (),
    }
    sources = {split: _pair(split) for split in ("lexical", "transfer")}

    panel, validation = build_crossed_identity_panel(
        sources, identities, tokenizer, 256, minimum_class_support=1
    )

    assert set(panel["lexical"]) == {256, 257}
    assert validation["lexical"]["invalid_contexts"] == 0
    assert validation["transfer"]["matched_contexts"] == 2
    assert panel["lexical"][256]["focus_counts"]["B"] == (
        panel["lexical"][256]["focus_counts"]["I"]
    )


def _cell(error, precision=1.0, recall=1.0, group="trained"):
    return {
        "group": group,
        "metrics": {
            "exact_span_precision": precision,
            "exact_span_recall": recall,
        },
        "focus": {
            "overall": {
                "gold_begin_count": 25,
                "gold_inside_count": 25,
                "begin_as_inside_rate": error,
            }
        },
        "focus_geometry": {
            "gold_B": {"mean_begin_minus_inside_margin": 1.0}
        },
    }


def _summary(heldout_error=0.5):
    groups = {
        "trained": list(range(300, 308)),
        "heldout": list(range(400, 404)),
        "legacy": [270, 357],
    }
    summary = {
        "crossed_boundary_identity_audit_version": 1,
        "training_changes": "none",
        "decoder_changes": "none",
        "phase35_inputs": {
            "decision_branch": "boundary_counterbalance_rejected",
            "checkpoint_sha256": "phase35-sha",
            "checkpoint_eligible": False,
            "checkpoint_promotion_authorized": False,
            "full_phase34_evaluation_authorized": False,
            "original_heldout_focus": {
                str(token_id): {"lexical": 0.5, "transfer": 0.5}
                for token_id in groups["heldout"]
            },
        },
        "focus_identities": {
            group: [
                {"token_id": token_id, "text": f"x{index}"}
                for index, token_id in enumerate(ids)
            ]
            for group, ids in groups.items()
        },
        "panel_validation": {
            split: {
                "source_pairs": 100,
                "identities": 14,
                "invalid_contexts": 0,
            }
            for split in ("lexical", "transfer")
        },
        "models": {
            "phase34d": {
                "checkpoint": {
                    "sha256": "phase34d-sha",
                    "checkpoint_eligible": False,
                    "architecture": "explicit_offset_candidate_proposer",
                    "boundary_objective_version": 1,
                    "boundary_loss_weight": 1.0,
                    "boundary_counterbalance_version": None,
                    "token_width_geometry_version": None,
                    "token_end_geometry_version": None,
                    "factorized_boundary_type_version": None,
                    "tokenizer_sha256": "tokenizer-sha",
                },
                "splits": {},
            },
            "phase35": {
                "checkpoint": {
                    "sha256": "phase35-sha",
                    "checkpoint_eligible": False,
                    "architecture": "explicit_offset_candidate_proposer",
                    "boundary_objective_version": 1,
                    "boundary_loss_weight": 1.0,
                    "boundary_counterbalance_version": 2,
                    "token_width_geometry_version": None,
                    "token_end_geometry_version": None,
                    "factorized_boundary_type_version": None,
                    "tokenizer_sha256": "tokenizer-sha",
                },
                "splits": {},
            },
        },
    }
    for split in ("lexical", "transfer"):
        baseline = {}
        candidate = {}
        for group, ids in groups.items():
            for token_id in ids:
                baseline[str(token_id)] = _cell(0.5, group=group)
                error = 0.0 if group == "trained" else heldout_error
                candidate[str(token_id)] = _cell(error, group=group)
        summary["models"]["phase34d"]["splits"][split] = {
            "identities": baseline
        }
        summary["models"]["phase35"]["splits"][split] = {
            "identities": candidate
        }
    return summary


def test_crossed_decision_detects_trained_identity_memorization():
    decision = crossed_identity_decision(_summary())

    assert decision["branch"] == "trained_identity_memorization_dominant"
    assert decision["group_results"]["trained"]["passing_tokens"] == 8
    assert decision["training_authorized"] is False


def test_crossed_decision_separates_context_distribution_interaction():
    summary = _summary(heldout_error=0.0)
    for split in ("lexical", "transfer"):
        for token_id in (270, 357):
            summary["models"]["phase35"]["splits"][split]["identities"][
                str(token_id)
            ] = _cell(0.5, group="legacy")

    decision = crossed_identity_decision(summary)

    assert decision["branch"] == "context_distribution_interaction_indicated"


def test_crossed_decision_preserves_heterogeneous_identity_result():
    summary = _summary()
    for split in ("lexical", "transfer"):
        summary["models"]["phase35"]["splits"][split]["identities"][
            "400"
        ] = _cell(0.0, group="heldout")
        summary["models"]["phase35"]["splits"][split]["identities"][
            "300"
        ] = _cell(0.03, group="trained")

    decision = crossed_identity_decision(summary)

    assert decision["branch"] == "heterogeneous_identity_priors_confirmed"


def test_crossed_decision_records_complete_span_regressions():
    summary = _summary()
    summary["models"]["phase35"]["splits"]["lexical"]["identities"][
        "300"
    ] = _cell(0.0, precision=0.5, group="trained")

    decision = crossed_identity_decision(summary)

    assert decision["complete_span_regressions"]


def test_crossed_decision_rejects_provenance_mismatch_without_mutation():
    summary = _summary()
    summary["phase35_inputs"]["checkpoint_promotion_authorized"] = True
    original = deepcopy(summary)

    decision = crossed_identity_decision(summary)

    assert decision["branch"] == "invalid_crossed_identity_audit"
    assert summary == original
