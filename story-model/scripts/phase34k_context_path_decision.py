"""Dependency-free decision logic for Phase 34k context decomposition."""

from __future__ import annotations


MODEL_LABELS = ("phase34d", "phase34i")
TOKEN_LABELS = ("or", "ro")
EXPECTED_PHASE34J_VERSION = 1
EXPECTED_PHASE34J_BRANCH = (
    "checkpoint_specific_identity_interaction_indicated"
)
EXPECTED_BOUNDARY_OBJECTIVE_VERSION = 1
EXPECTED_BOUNDARY_LOSS_WEIGHT = 1.0
EXPECTED_FACTORIZED_VERSION = 1
MINIMUM_VALID_PAIRS = 400
MINIMUM_SOURCE_PAIRS = 200
FULL_CONTEXT_MARGIN_EFFECT_FLOOR = 0.25
DOMINANT_EFFECT_SHARE_FLOOR = 0.70
OTHER_EFFECT_SHARE_CEILING = 0.30
DISTRIBUTED_EFFECT_SHARE_FLOOR = 0.30


def _model_provenance_failures(label: str, model: dict) -> list[str]:
    failures = []
    checkpoint = model.get("checkpoint", {})
    if checkpoint.get("boundary_objective_version") != (
        EXPECTED_BOUNDARY_OBJECTIVE_VERSION
    ):
        failures.append(f"{label} lacks boundary objective version 1")
    weight = checkpoint.get("boundary_loss_weight")
    if weight is None or abs(
        float(weight) - EXPECTED_BOUNDARY_LOSS_WEIGHT
    ) > 1.0e-9:
        failures.append(f"{label} boundary loss weight is not 1.0")
    for field in (
        "token_width_geometry_version",
        "token_end_geometry_version",
    ):
        if checkpoint.get(field) is not None:
            failures.append(f"{label} unexpectedly declares {field}")
    factorized = checkpoint.get("factorized_boundary_type_version")
    if label == "phase34d" and factorized is not None:
        failures.append("phase34d unexpectedly uses factorized heads")
    if label == "phase34i" and factorized != EXPECTED_FACTORIZED_VERSION:
        failures.append("phase34i does not use factorized-head version 1")
    return failures


def _pathway_assessment(pathway: dict) -> dict:
    full_delta = float(
        pathway.get("mean_full_context_ro_minus_or_margin", 0.0)
    )
    local_delta = float(
        pathway.get("mean_local_key_ro_minus_or_margin", 0.0)
    )
    query_delta = float(
        pathway.get("mean_query_ro_minus_or_margin", 0.0)
    )
    full_effect = -full_delta
    local_effect = max(0.0, -local_delta)
    query_effect = max(0.0, -query_delta)
    if full_effect > 0.0:
        local_share = local_effect / full_effect
        query_share = query_effect / full_effect
    else:
        local_share = 0.0
        query_share = 0.0

    if full_effect < FULL_CONTEXT_MARGIN_EFFECT_FLOOR:
        branch = "unresolved"
    elif (
        local_share >= DOMINANT_EFFECT_SHARE_FLOOR
        and query_share <= OTHER_EFFECT_SHARE_CEILING
    ):
        branch = "local_key"
    elif (
        query_share >= DOMINANT_EFFECT_SHARE_FLOOR
        and local_share <= OTHER_EFFECT_SHARE_CEILING
    ):
        branch = "resolver_query"
    elif (
        local_share >= DISTRIBUTED_EFFECT_SHARE_FLOOR
        and query_share >= DISTRIBUTED_EFFECT_SHARE_FLOOR
    ):
        branch = "distributed"
    else:
        branch = "unresolved"
    return {
        "branch": branch,
        "full_context_margin_effect": full_effect,
        "local_key_margin_effect": local_effect,
        "resolver_query_margin_effect": query_effect,
        "local_key_effect_share": local_share,
        "resolver_query_effect_share": query_share,
        "raw_margin_deltas": {
            "full_context_ro_minus_or": full_delta,
            "local_key_ro_minus_or": local_delta,
            "resolver_query_ro_minus_or": query_delta,
        },
    }


def context_path_decision(summary: dict) -> dict:
    """Localize the contextual identity effect without authorizing training."""

    invalid = []
    premise = summary.get("phase34j_premise", {})
    if premise.get("paired_token_intervention_version") != (
        EXPECTED_PHASE34J_VERSION
    ):
        invalid.append("Phase 34j premise has the wrong audit version")
    if premise.get("branch") != EXPECTED_PHASE34J_BRANCH:
        invalid.append(
            "Phase 34j premise is not the checkpoint-specific identity branch"
        )
    for field in (
        "training_authorized",
        "full_phase34_evaluation_authorized",
        "checkpoint_promotion_authorized",
    ):
        if premise.get(field) is not False:
            invalid.append(f"Phase 34j premise has invalid {field}")

    validation = summary.get("pair_validation", {})
    valid_pairs = int(validation.get("valid_pairs", 0))
    invalid_pairs = int(validation.get("invalid_pairs", 0))
    source_counts = validation.get("source_counts", {})
    if valid_pairs < MINIMUM_VALID_PAIRS:
        invalid.append(
            f"valid pair count {valid_pairs} is below {MINIMUM_VALID_PAIRS}"
        )
    if invalid_pairs:
        invalid.append(
            f"{invalid_pairs} focus starts violate the paired-token contract"
        )
    for source in TOKEN_LABELS:
        count = int(source_counts.get(source, 0))
        if count < MINIMUM_SOURCE_PAIRS:
            invalid.append(
                f"source token {source!r} has {count} pairs, below "
                f"{MINIMUM_SOURCE_PAIRS}"
            )

    models = summary.get("models", {})
    if set(models) != set(MODEL_LABELS):
        invalid.append("audit must contain exactly Phase 34d and Phase 34i")
    tokenizer_hashes = set()
    checkpoint_hashes = premise.get("checkpoint_sha256", {})
    assessments = {}
    for label in MODEL_LABELS:
        model = models.get(label)
        if not isinstance(model, dict):
            continue
        invalid.extend(_model_provenance_failures(label, model))
        checkpoint = model.get("checkpoint", {})
        tokenizer_hash = checkpoint.get("tokenizer_sha256")
        if tokenizer_hash:
            tokenizer_hashes.add(tokenizer_hash)
        if checkpoint.get("sha256") != checkpoint_hashes.get(label):
            invalid.append(
                f"{label} checkpoint does not match the Phase 34j premise"
            )
        aggregate_observations = int(
            model.get("pathway", {}).get("observations", 0)
        )
        if aggregate_observations != valid_pairs:
            invalid.append(
                f"{label} pathway observations {aggregate_observations} do "
                f"not match {valid_pairs} pairs"
            )
        assessments[label] = {}
        for source in TOKEN_LABELS:
            expected = int(source_counts.get(source, 0))
            stratum = model.get("source_strata", {}).get(source, {})
            observations = int(
                stratum.get("pathway", {}).get("observations", 0)
            )
            if observations != expected:
                invalid.append(
                    f"{label} source {source!r} pathway observations "
                    f"{observations} do not match {expected}"
                )
            assessments[label][source] = _pathway_assessment(
                stratum.get("pathway", {})
            )
    if len(tokenizer_hashes) != 1:
        invalid.append("Phase 34d and Phase 34i tokenizers do not match")

    if invalid:
        branch = "invalid_context_path_audit"
        next_action = (
            "Correct the Phase 34j provenance, pair contract, or checkpoint "
            "join before interpreting the decomposition."
        )
    else:
        component_branches = {
            assessments[label][source]["branch"]
            for label in MODEL_LABELS
            for source in TOKEN_LABELS
        }
        if component_branches == {"local_key"}:
            branch = "local_token_state_dominant"
            next_action = (
                "Design one training-only compositional boundary "
                "counterbalance targeted at local token states, with "
                "refreshed token-disjoint held-out values."
            )
        elif component_branches == {"resolver_query"}:
            branch = "resolver_query_state_dominant"
            next_action = (
                "Audit a boundary-local or span-conditioned query that does "
                "not reuse the pooled final resolver state before training."
            )
        elif component_branches == {"distributed"}:
            branch = "distributed_context_path_indicated"
            next_action = (
                "Design a boundary-specific contextual representation that "
                "replaces both the BPE-token key and pooled resolver query; "
                "the direct raw-byte path is not implicated."
            )
        elif "unresolved" in component_branches:
            branch = "context_path_decomposition_insufficient"
            next_action = (
                "Inspect per-pair local/query margin interactions and "
                "projection activations; do not choose a new architecture."
            )
        else:
            branch = "checkpoint_or_stratum_interaction_indicated"
            next_action = (
                "The dominant contextual component changes by checkpoint or "
                "source stratum. Inspect proposal_key/proposal_query feature "
                "distributions before authorizing another model."
            )

    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "context_path_assessments": assessments,
        "thresholds": {
            "minimum_valid_pairs": MINIMUM_VALID_PAIRS,
            "minimum_source_pairs": MINIMUM_SOURCE_PAIRS,
            "full_context_margin_effect_floor": (
                FULL_CONTEXT_MARGIN_EFFECT_FLOOR
            ),
            "dominant_effect_share_floor": DOMINANT_EFFECT_SHARE_FLOOR,
            "other_effect_share_ceiling": OTHER_EFFECT_SHARE_CEILING,
            "distributed_effect_share_floor": (
                DISTRIBUTED_EFFECT_SHARE_FLOOR
            ),
        },
        "share_interpretation": (
            "Shares are post-tanh B-minus-I margin effects and need not sum "
            "to one or remain below one."
        ),
        "training_authorized": False,
        "full_phase34_evaluation_authorized": False,
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
