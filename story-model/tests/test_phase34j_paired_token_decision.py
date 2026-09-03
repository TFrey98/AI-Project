from copy import deepcopy

from scripts.phase34j_paired_token_decision import paired_token_decision


def _identity_metrics(rate: float, observations: int = 250) -> dict:
    return {
        "observations": observations,
        "begin_as_inside_rate": rate,
    }


def _model(factorized: bool, pathway: str = "direct") -> dict:
    if pathway == "direct":
        byte_shift, context_shift = -0.8, -0.1
    elif pathway == "context":
        byte_shift, context_shift = -0.1, -0.8
    elif pathway == "distributed":
        byte_shift, context_shift = -0.6, -0.6
    else:
        byte_shift, context_shift = -0.2, -0.2
    strata = {
        source: {
            "rendered_identity": {
                "or": _identity_metrics(0.10),
                "ro": _identity_metrics(0.70),
            }
        }
        for source in ("or", "ro")
    }
    return {
        "checkpoint": {
            "boundary_objective_version": 1,
            "boundary_loss_weight": 1.0,
            "token_width_geometry_version": None,
            "token_end_geometry_version": None,
            "factorized_boundary_type_version": 1 if factorized else None,
            "tokenizer_sha256": "same-tokenizer",
        },
        "rendered_identity": {
            "or": _identity_metrics(0.10, observations=500),
            "ro": _identity_metrics(0.70, observations=500),
        },
        "source_strata": strata,
        "pathway": {
            "observations": 500,
            "mean_full_ro_minus_or_margin": -1.0,
            "mean_byte_ro_minus_or_margin": byte_shift,
            "mean_context_ro_minus_or_margin": context_shift,
        },
    }


def _summary(phase34d_path="direct", phase34i_path="direct") -> dict:
    return {
        "pair_validation": {
            "valid_pairs": 500,
            "invalid_pairs": 0,
            "source_counts": {"or": 250, "ro": 250},
        },
        "models": {
            "phase34d": _model(False, phase34d_path),
            "phase34i": _model(True, phase34i_path),
        },
    }


def test_direct_byte_path_is_selected_only_when_both_checkpoints_agree():
    decision = paired_token_decision(_summary())

    assert decision["branch"] == "direct_byte_identity_path_indicated"
    assert decision["training_authorized"] is False
    assert decision["full_phase34_evaluation_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_contextual_and_distributed_paths_have_distinct_branches():
    contextual = paired_token_decision(_summary("context", "context"))
    distributed = paired_token_decision(
        _summary("distributed", "distributed")
    )

    assert contextual["branch"] == (
        "contextual_token_identity_path_indicated"
    )
    assert distributed["branch"] == "distributed_identity_path_indicated"


def test_mixed_checkpoint_pathways_stop_without_selecting_architecture():
    decision = paired_token_decision(_summary("direct", "context"))

    assert decision["branch"] == "mixed_identity_pathways_indicated"


def test_noncausal_identity_difference_selects_prompt_context_audit():
    summary = _summary()
    for model in summary["models"].values():
        for stratum in model["source_strata"].values():
            stratum["rendered_identity"]["or"][
                "begin_as_inside_rate"
            ] = 0.30
            stratum["rendered_identity"]["ro"][
                "begin_as_inside_rate"
            ] = 0.35

    decision = paired_token_decision(summary)

    assert decision["branch"] == "prompt_context_proxy_indicated"


def test_identity_effect_in_only_one_checkpoint_is_reported_separately():
    summary = _summary()
    for stratum in summary["models"]["phase34d"][
        "source_strata"
    ].values():
        stratum["rendered_identity"]["ro"][
            "begin_as_inside_rate"
        ] = 0.15

    decision = paired_token_decision(summary)

    assert decision["branch"] == (
        "checkpoint_specific_identity_interaction_indicated"
    )


def test_invalid_pair_contract_or_checkpoint_provenance_blocks_result():
    summary = _summary()
    summary["pair_validation"]["invalid_pairs"] = 1
    summary["models"]["phase34i"]["checkpoint"][
        "factorized_boundary_type_version"
    ] = None

    decision = paired_token_decision(summary)

    assert decision["branch"] == "invalid_paired_token_audit"
    assert decision["invalid_reasons"]


def test_missing_model_observations_invalidate_the_paired_comparison():
    summary = _summary()
    summary["models"]["phase34d"]["rendered_identity"]["or"][
        "observations"
    ] = 499

    decision = paired_token_decision(summary)

    assert decision["branch"] == "invalid_paired_token_audit"
    assert any(
        "observations" in reason for reason in decision["invalid_reasons"]
    )


def test_summary_input_is_not_mutated():
    summary = _summary()
    original = deepcopy(summary)

    paired_token_decision(summary)

    assert summary == original
