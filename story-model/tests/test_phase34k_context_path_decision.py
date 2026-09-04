from copy import deepcopy

from scripts.phase34k_context_path_decision import context_path_decision


def _pathway(kind: str) -> dict:
    if kind == "local":
        local, query, full = -0.80, -0.20, -1.0
    elif kind == "query":
        local, query, full = -0.20, -0.80, -1.0
    elif kind == "distributed":
        local, query, full = -0.60, -0.55, -1.0
    elif kind == "supra_local":
        local, query, full = -1.20, -0.10, -1.0
    else:
        local, query, full = -0.05, -0.05, -0.10
    return {
        "observations": 250,
        "mean_local_key_ro_minus_or_margin": local,
        "mean_query_ro_minus_or_margin": query,
        "mean_full_context_ro_minus_or_margin": full,
        "mean_full_token_ro_minus_or_margin": full,
    }


def _model(factorized: bool, kind: str) -> dict:
    return {
        "checkpoint": {
            "sha256": "phase34i-sha" if factorized else "phase34d-sha",
            "boundary_objective_version": 1,
            "boundary_loss_weight": 1.0,
            "token_width_geometry_version": None,
            "token_end_geometry_version": None,
            "factorized_boundary_type_version": 1 if factorized else None,
            "tokenizer_sha256": "same-tokenizer",
        },
        "pathway": {**_pathway(kind), "observations": 500},
        "source_strata": {
            source: {"pathway": _pathway(kind)}
            for source in ("or", "ro")
        },
    }


def _summary(phase34d="local", phase34i="local") -> dict:
    return {
        "phase34j_premise": {
            "paired_token_intervention_version": 1,
            "branch": "checkpoint_specific_identity_interaction_indicated",
            "training_authorized": False,
            "full_phase34_evaluation_authorized": False,
            "checkpoint_promotion_authorized": False,
            "checkpoint_sha256": {
                "phase34d": "phase34d-sha",
                "phase34i": "phase34i-sha",
            },
        },
        "pair_validation": {
            "valid_pairs": 500,
            "invalid_pairs": 0,
            "source_counts": {"or": 250, "ro": 250},
        },
        "models": {
            "phase34d": _model(False, phase34d),
            "phase34i": _model(True, phase34i),
        },
    }


def test_local_key_dominance_requires_both_checkpoints_and_strata():
    decision = context_path_decision(_summary())

    assert decision["branch"] == "local_token_state_dominant"
    assert decision["training_authorized"] is False
    assert decision["full_phase34_evaluation_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_query_and_distributed_results_have_derived_distinct_actions():
    query = context_path_decision(_summary("query", "query"))
    distributed = context_path_decision(
        _summary("distributed", "distributed")
    )

    assert query["branch"] == "resolver_query_state_dominant"
    assert distributed["branch"] == "distributed_context_path_indicated"
    assert "direct raw-byte path is not implicated" in distributed[
        "next_action"
    ]


def test_checkpoint_or_source_stratum_disagreement_stops_selection():
    checkpoint_disagreement = context_path_decision(
        _summary("local", "query")
    )
    stratum_summary = _summary()
    stratum_summary["models"]["phase34i"]["source_strata"]["ro"][
        "pathway"
    ] = _pathway("query")
    stratum_disagreement = context_path_decision(stratum_summary)

    assert checkpoint_disagreement["branch"] == (
        "checkpoint_or_stratum_interaction_indicated"
    )
    assert stratum_disagreement["branch"] == (
        "checkpoint_or_stratum_interaction_indicated"
    )


def test_insufficient_full_context_effect_does_not_select_architecture():
    decision = context_path_decision(_summary("unresolved", "unresolved"))

    assert decision["branch"] == "context_path_decomposition_insufficient"


def test_post_tanh_shares_can_exceed_one_without_invalidating_result():
    decision = context_path_decision(
        _summary("supra_local", "supra_local")
    )

    assert decision["branch"] == "local_token_state_dominant"
    assessment = decision["context_path_assessments"]["phase34d"]["or"]
    assert assessment["local_key_effect_share"] == 1.2
    assert "need not sum" in decision["share_interpretation"]


def test_invalid_phase34j_premise_or_checkpoint_join_blocks_result():
    summary = _summary()
    summary["phase34j_premise"]["branch"] = (
        "contextual_token_identity_path_indicated"
    )
    summary["models"]["phase34i"]["checkpoint"]["sha256"] = "wrong"

    decision = context_path_decision(summary)

    assert decision["branch"] == "invalid_context_path_audit"
    assert decision["invalid_reasons"]


def test_invalid_pair_counts_and_checkpoint_provenance_are_rejected():
    summary = _summary()
    summary["pair_validation"]["invalid_pairs"] = 1
    summary["models"]["phase34d"]["checkpoint"][
        "factorized_boundary_type_version"
    ] = 1

    decision = context_path_decision(summary)

    assert decision["branch"] == "invalid_context_path_audit"
    assert any("factorized" in reason for reason in decision["invalid_reasons"])


def test_summary_input_is_not_mutated():
    summary = _summary()
    original = deepcopy(summary)

    context_path_decision(summary)

    assert summary == original
