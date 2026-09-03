"""Dependency-free decision logic for the Phase 34j paired audit."""

from __future__ import annotations


MODEL_LABELS = ("phase34d", "phase34i")
TOKEN_LABELS = ("or", "ro")
EXPECTED_BOUNDARY_OBJECTIVE_VERSION = 1
EXPECTED_BOUNDARY_LOSS_WEIGHT = 1.0
EXPECTED_FACTORIZED_VERSION = 1
MINIMUM_VALID_PAIRS = 400
MINIMUM_SOURCE_PAIRS = 200
IDENTITY_ERROR_GAP_FLOOR = 0.20
IDENTITY_RATE_RATIO_FLOOR = 3.0
PATHWAY_FULL_MARGIN_EFFECT_FLOOR = 0.25
PATHWAY_SHARE_FLOOR = 0.60
PATHWAY_OTHER_SHARE_CEILING = 0.40


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


def _identity_assessment(model: dict) -> dict:
    failures = []
    strata = model.get("source_strata", {})
    details = {}
    for source in TOKEN_LABELS:
        rendered = strata.get(source, {}).get("rendered_identity", {})
        or_metrics = rendered.get("or", {})
        ro_metrics = rendered.get("ro", {})
        or_count = int(or_metrics.get("observations", 0))
        ro_count = int(ro_metrics.get("observations", 0))
        or_rate = float(or_metrics.get("begin_as_inside_rate", 0.0))
        ro_rate = float(ro_metrics.get("begin_as_inside_rate", 0.0))
        gap = ro_rate - or_rate
        ratio_denominator = max(or_rate, 0.01)
        ratio = ro_rate / ratio_denominator
        details[source] = {
            "or_observations": or_count,
            "ro_observations": ro_count,
            "or_begin_as_inside_rate": or_rate,
            "ro_begin_as_inside_rate": ro_rate,
            "ro_minus_or_error_gap": gap,
            "error_rate_ratio": ratio,
        }
        if min(or_count, ro_count) < MINIMUM_SOURCE_PAIRS:
            failures.append(
                f"{source} source stratum has fewer than "
                f"{MINIMUM_SOURCE_PAIRS} paired renderings"
            )
        if gap < IDENTITY_ERROR_GAP_FLOOR:
            failures.append(
                f"{source} source stratum error gap {gap:.3f} is below "
                f"{IDENTITY_ERROR_GAP_FLOOR:.3f}"
            )
        if ratio < IDENTITY_RATE_RATIO_FLOOR:
            failures.append(
                f"{source} source stratum rate ratio {ratio:.3f} is below "
                f"{IDENTITY_RATE_RATIO_FLOOR:.3f}"
            )
    return {
        "causal_identity_gate_passed": not failures,
        "failures": failures,
        "source_strata": details,
    }


def _pathway_assessment(model: dict) -> dict:
    pathway = model.get("pathway", {})
    full_effect = -float(
        pathway.get("mean_full_ro_minus_or_margin", 0.0)
    )
    byte_effect = max(
        0.0,
        -float(pathway.get("mean_byte_ro_minus_or_margin", 0.0)),
    )
    context_effect = max(
        0.0,
        -float(pathway.get("mean_context_ro_minus_or_margin", 0.0)),
    )
    if full_effect > 0.0:
        byte_share = byte_effect / full_effect
        context_share = context_effect / full_effect
    else:
        byte_share = 0.0
        context_share = 0.0

    if full_effect < PATHWAY_FULL_MARGIN_EFFECT_FLOOR:
        branch = "unresolved"
    elif (
        byte_share >= PATHWAY_SHARE_FLOOR
        and context_share < PATHWAY_OTHER_SHARE_CEILING
    ):
        branch = "direct_byte"
    elif (
        context_share >= PATHWAY_SHARE_FLOOR
        and byte_share < PATHWAY_OTHER_SHARE_CEILING
    ):
        branch = "contextual_token"
    elif (
        byte_share >= PATHWAY_OTHER_SHARE_CEILING
        and context_share >= PATHWAY_OTHER_SHARE_CEILING
    ):
        branch = "distributed"
    else:
        branch = "unresolved"
    return {
        "branch": branch,
        "full_margin_effect": full_effect,
        "direct_byte_margin_effect": byte_effect,
        "contextual_token_margin_effect": context_effect,
        "direct_byte_effect_share": byte_share,
        "contextual_token_effect_share": context_share,
    }


def paired_token_decision(summary: dict) -> dict:
    """Select the next representation test without authorizing training."""

    invalid = []
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
    for label in MODEL_LABELS:
        model = models.get(label)
        if not isinstance(model, dict):
            continue
        invalid.extend(_model_provenance_failures(label, model))
        tokenizer_hash = model.get("checkpoint", {}).get("tokenizer_sha256")
        if tokenizer_hash:
            tokenizer_hashes.add(tokenizer_hash)
        for token in TOKEN_LABELS:
            observations = int(
                model.get("rendered_identity", {})
                .get(token, {})
                .get("observations", 0)
            )
            if observations != valid_pairs:
                invalid.append(
                    f"{label} rendered {token!r} observations "
                    f"{observations} do not match {valid_pairs} pairs"
                )
        pathway_observations = int(
            model.get("pathway", {}).get("observations", 0)
        )
        if pathway_observations != valid_pairs:
            invalid.append(
                f"{label} pathway observations {pathway_observations} do "
                f"not match {valid_pairs} pairs"
            )
        for source in TOKEN_LABELS:
            expected = int(source_counts.get(source, 0))
            rendered = model.get("source_strata", {}).get(source, {}).get(
                "rendered_identity", {}
            )
            for token in TOKEN_LABELS:
                observations = int(
                    rendered.get(token, {}).get("observations", 0)
                )
                if observations != expected:
                    invalid.append(
                        f"{label} source {source!r} rendered {token!r} "
                        f"observations {observations} do not match {expected}"
                    )
    if len(tokenizer_hashes) != 1:
        invalid.append("Phase 34d and Phase 34i tokenizers do not match")

    identity = {
        label: _identity_assessment(models[label])
        for label in MODEL_LABELS
        if label in models
    }
    pathways = {
        label: _pathway_assessment(models[label])
        for label in MODEL_LABELS
        if label in models
    }
    if invalid:
        branch = "invalid_paired_token_audit"
        next_action = (
            "Correct checkpoint provenance or pair construction before "
            "interpreting the intervention."
        )
    else:
        causal = {
            label: identity[label]["causal_identity_gate_passed"]
            for label in MODEL_LABELS
        }
        if not any(causal.values()):
            branch = "prompt_context_proxy_indicated"
            next_action = (
                "The error does not follow token identity within fixed "
                "contexts. Audit the scene-route prompt structures next."
            )
        elif not all(causal.values()):
            branch = "checkpoint_specific_identity_interaction_indicated"
            next_action = (
                "Token identity is causal in only one architecture. Compare "
                "its boundary features before authorizing another model."
            )
        else:
            pathway_labels = {
                pathways[label]["branch"] for label in MODEL_LABELS
            }
            if pathway_labels == {"direct_byte"}:
                branch = "direct_byte_identity_path_indicated"
                next_action = (
                    "Test a boundary scorer that masks current-byte identity "
                    "while retaining content features outside that scorer."
                )
            elif pathway_labels == {"contextual_token"}:
                branch = "contextual_token_identity_path_indicated"
                next_action = (
                    "Test a training-only compositional boundary "
                    "counterbalance with refreshed token-disjoint held-out "
                    "values."
                )
            elif pathway_labels == {"distributed"}:
                branch = "distributed_identity_path_indicated"
                next_action = (
                    "The identity effect uses both representation paths. "
                    "Test a BPE-invariant raw-byte boundary encoder next."
                )
            else:
                branch = "mixed_identity_pathways_indicated"
                next_action = (
                    "The checkpoints localize the same identity failure to "
                    "different paths. Stop and inspect per-model paired "
                    "margins before choosing a new architecture."
                )

    return {
        "branch": branch,
        "invalid_reasons": invalid,
        "identity_assessments": identity,
        "pathway_assessments": pathways,
        "thresholds": {
            "minimum_valid_pairs": MINIMUM_VALID_PAIRS,
            "minimum_source_pairs": MINIMUM_SOURCE_PAIRS,
            "identity_error_gap_floor": IDENTITY_ERROR_GAP_FLOOR,
            "identity_rate_ratio_floor": IDENTITY_RATE_RATIO_FLOOR,
            "pathway_full_margin_effect_floor": (
                PATHWAY_FULL_MARGIN_EFFECT_FLOOR
            ),
            "pathway_share_floor": PATHWAY_SHARE_FLOOR,
            "pathway_other_share_ceiling": (
                PATHWAY_OTHER_SHARE_CEILING
            ),
        },
        "training_authorized": False,
        "full_phase34_evaluation_authorized": False,
        "checkpoint_promotion_authorized": False,
        "next_action": next_action,
    }
