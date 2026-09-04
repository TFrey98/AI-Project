"""Frozen local-key/query decomposition for Phase 34k."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_BEGIN,
    BOUNDARY_INSIDE,
    BOUNDARY_OUTSIDE,
)
from story_model.runtime import resolve_device

try:
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.audit_paired_token_intervention import (
        CLASS_NAMES,
        FOCUS_BYTES,
        TARGET_SPLITS,
        _checkpoint_metadata,
        _collapse_legacy_logits,
        _validate_checkpoint_pair,
        build_intervention_pairs,
    )
    from scripts.phase34j_paired_token_decision import paired_token_decision
    from scripts.phase34k_context_path_decision import context_path_decision
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_explicit_offset_tag_confusion import (  # type: ignore[no-redef]
        _load_model,
    )
    from audit_paired_token_intervention import (  # type: ignore[no-redef]
        CLASS_NAMES,
        FOCUS_BYTES,
        TARGET_SPLITS,
        _checkpoint_metadata,
        _collapse_legacy_logits,
        _validate_checkpoint_pair,
        build_intervention_pairs,
    )
    from phase34j_paired_token_decision import (  # type: ignore[no-redef]
        paired_token_decision,
    )
    from phase34k_context_path_decision import (  # type: ignore[no-redef]
        context_path_decision,
    )


CONTEXT_PATH_INTERVENTION_VERSION = 1
EXPECTED_PHASE34J_VERSION = 1
EXPECTED_PHASE34J_BRANCH = (
    "checkpoint_specific_identity_interaction_indicated"
)
CONDITIONS = (
    "original",
    "local_key_swap_only",
    "query_swap_only",
    "full_context_swap",
    "full_token_swap",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def context_path_boundary_logits(
    model,
    original_tokens: torch.Tensor,
    swapped_tokens: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_positions: torch.Tensor,
    original_token_ids: torch.Tensor,
    swapped_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score five key/query/byte combinations after one batched forward."""

    if model.token_width_geometry or model.token_end_geometry:
        raise ValueError("Phase 34k requires the pre-geometry proposer")
    if original_tokens.shape != swapped_tokens.shape:
        raise ValueError("paired token tensors must have the same shape")
    batch_size = len(original_tokens)
    if not all(
        len(value) == batch_size
        for value in (
            sequence_lengths,
            token_positions,
            original_token_ids,
            swapped_token_ids,
        )
    ):
        raise ValueError("context-path tensors have inconsistent batches")

    hidden = model.resolver.hidden_states(
        torch.cat((original_tokens, swapped_tokens), dim=0)
    )
    original_hidden, swapped_hidden = hidden.split(batch_size, dim=0)
    batch_indices = torch.arange(batch_size, device=original_tokens.device)
    query_positions = sequence_lengths - 1
    original_key = model.proposal_key(
        original_hidden[batch_indices, token_positions]
    )
    swapped_key = model.proposal_key(
        swapped_hidden[batch_indices, token_positions]
    )
    original_query = model.proposal_query(
        original_hidden[batch_indices, query_positions]
    )
    swapped_query = model.proposal_query(
        swapped_hidden[batch_indices, query_positions]
    )
    original_byte_ids = model.token_byte_ids[original_token_ids, 0]
    swapped_byte_ids = model.token_byte_ids[swapped_token_ids, 0]
    embeddings = model.resolver.backbone.token_embeddings
    original_bytes = embeddings(original_byte_ids)
    swapped_bytes = embeddings(swapped_byte_ids)
    offset = model.proposal_offset_embedding.weight[0]

    def score(
        key: torch.Tensor,
        query: torch.Tensor,
        byte: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.tanh(key + query + byte + offset)
        if model.factorized_boundary_type:
            if model.proposal_boundary is None:
                raise RuntimeError("factorized checkpoint has no boundary head")
            return model.proposal_boundary(states)
        return _collapse_legacy_logits(model.proposal_tag(states))

    return {
        "original": score(original_key, original_query, original_bytes),
        "local_key_swap_only": score(
            swapped_key, original_query, original_bytes
        ),
        "query_swap_only": score(
            original_key, swapped_query, original_bytes
        ),
        "full_context_swap": score(
            swapped_key, swapped_query, original_bytes
        ),
        "full_token_swap": score(
            swapped_key, swapped_query, swapped_bytes
        ),
    }


def _prediction(logits: torch.Tensor) -> dict:
    probabilities = F.softmax(logits.float(), dim=-1)
    predicted = int(logits.argmax())
    return {
        "predicted_class": CLASS_NAMES[predicted],
        "begin_minus_inside_margin": float(
            logits[BOUNDARY_BEGIN] - logits[BOUNDARY_INSIDE]
        ),
        "outside_probability": float(probabilities[BOUNDARY_OUTSIDE]),
        "begin_probability": float(probabilities[BOUNDARY_BEGIN]),
        "inside_probability": float(probabilities[BOUNDARY_INSIDE]),
    }


@torch.no_grad()
def evaluate_model(model, pairs, batch_size: int, progress_label: str):
    device = next(model.parameters()).device
    model.eval()
    results = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    started = time.monotonic()
    for batch_number, start in enumerate(range(0, len(pairs), batch_size), 1):
        selected = pairs[start : start + batch_size]
        original_tokens = torch.tensor(
            [pair.original_input_ids for pair in selected],
            dtype=torch.long,
            device=device,
        )
        swapped_tokens = torch.tensor(
            [pair.swapped_input_ids for pair in selected],
            dtype=torch.long,
            device=device,
        )
        sequence_lengths = torch.tensor(
            [pair.sequence_tokens for pair in selected],
            dtype=torch.long,
            device=device,
        )
        token_positions = torch.tensor(
            [pair.token_position for pair in selected],
            dtype=torch.long,
            device=device,
        )
        original_token_ids = torch.tensor(
            [pair.original_token_id for pair in selected],
            dtype=torch.long,
            device=device,
        )
        swapped_token_ids = torch.tensor(
            [pair.swapped_token_id for pair in selected],
            dtype=torch.long,
            device=device,
        )
        logits = context_path_boundary_logits(
            model,
            original_tokens,
            swapped_tokens,
            sequence_lengths,
            token_positions,
            original_token_ids,
            swapped_token_ids,
        )
        for offset, pair in enumerate(selected):
            results.append(
                {
                    **pair.metadata(),
                    "conditions": {
                        condition: _prediction(values[offset])
                        for condition, values in logits.items()
                    },
                }
            )
        if batch_number % 5 == 0 or batch_number == total_batches:
            completed = min(start + batch_size, len(pairs))
            elapsed = max(time.monotonic() - started, 1.0e-9)
            print(
                f"{progress_label}: {completed:,}/{len(pairs):,} pairs "
                f"({completed / elapsed:.1f} pairs/s)",
                flush=True,
            )
    return tuple(results)


def _condition_summary(predictions) -> dict:
    predictions = tuple(predictions)
    classes = Counter(row["predicted_class"] for row in predictions)
    return {
        "observations": len(predictions),
        "predicted_classes": {
            label: int(classes[label]) for label in CLASS_NAMES
        },
        "begin_as_inside_count": int(classes["I"]),
        "begin_as_inside_rate": (
            classes["I"] / len(predictions) if predictions else 0.0
        ),
        "mean_begin_minus_inside_margin": (
            sum(row["begin_minus_inside_margin"] for row in predictions)
            / len(predictions)
            if predictions
            else 0.0
        ),
    }


def _oriented_margin_delta(row: dict, condition: str) -> float:
    original = row["conditions"]["original"][
        "begin_minus_inside_margin"
    ]
    intervened = row["conditions"][condition][
        "begin_minus_inside_margin"
    ]
    if row["source_token"] == "or":
        return intervened - original
    return original - intervened


def _summarize_rows(results) -> dict:
    results = tuple(results)
    pathways = {
        condition: [_oriented_margin_delta(row, condition) for row in results]
        for condition in CONDITIONS[1:]
    }
    transitions = {
        condition: Counter(
            f"{row['conditions']['original']['predicted_class']}->"
            f"{row['conditions'][condition]['predicted_class']}"
            for row in results
        )
        for condition in CONDITIONS[1:]
    }
    count = len(results)
    return {
        "conditions": {
            condition: _condition_summary(
                row["conditions"][condition] for row in results
            )
            for condition in CONDITIONS
        },
        "pathway": {
            "observations": count,
            "margin_orientation": "rendered_ro_minus_rendered_or",
            "mean_local_key_ro_minus_or_margin": (
                sum(pathways["local_key_swap_only"]) / count
                if count
                else 0.0
            ),
            "mean_query_ro_minus_or_margin": (
                sum(pathways["query_swap_only"]) / count
                if count
                else 0.0
            ),
            "mean_full_context_ro_minus_or_margin": (
                sum(pathways["full_context_swap"]) / count
                if count
                else 0.0
            ),
            "mean_full_token_ro_minus_or_margin": (
                sum(pathways["full_token_swap"]) / count
                if count
                else 0.0
            ),
        },
        "transitions_from_original": {
            condition: dict(sorted(counts.items()))
            for condition, counts in transitions.items()
        },
    }


def summarize_model_results(results) -> dict:
    summary = _summarize_rows(results)
    summary["source_strata"] = {
        source: _summarize_rows(
            row for row in results if row["source_token"] == source
        )
        for source in FOCUS_BYTES
    }
    return summary


def _validated_phase34j_premise(summary: dict, path: Path) -> dict:
    if summary.get("paired_token_intervention_version") != (
        EXPECTED_PHASE34J_VERSION
    ):
        raise ValueError("input is not a Phase 34j paired-token audit")
    decision = paired_token_decision(summary)
    embedded = summary.get("decision")
    if embedded is not None and embedded != decision:
        raise ValueError("embedded Phase 34j decision is stale or inconsistent")
    if decision["branch"] != EXPECTED_PHASE34J_BRANCH:
        raise ValueError(
            "Phase 34j did not select checkpoint-specific identity interaction"
        )
    for field in (
        "training_authorized",
        "full_phase34_evaluation_authorized",
        "checkpoint_promotion_authorized",
    ):
        if decision.get(field) is not False:
            raise ValueError(f"Phase 34j has invalid {field}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "paired_token_intervention_version": (
            summary["paired_token_intervention_version"]
        ),
        "branch": decision["branch"],
        "training_authorized": decision["training_authorized"],
        "full_phase34_evaluation_authorized": (
            decision["full_phase34_evaluation_authorized"]
        ),
        "checkpoint_promotion_authorized": (
            decision["checkpoint_promotion_authorized"]
        ),
        "checkpoint_sha256": {
            label: summary["models"][label]["checkpoint"]["sha256"]
            for label in ("phase34d", "phase34i")
        },
    }


def _validate_replay(
    phase34j_summary: dict,
    pairs,
    invalid_pairs,
    target_starts: int,
    token_ids: dict,
    data_files: dict,
    checkpoint_metadata: dict,
) -> None:
    previous = phase34j_summary.get("pair_validation", {})
    source_counts = Counter(pair.source_token for pair in pairs)
    comparisons = {
        "target_focus_starts": target_starts,
        "valid_pairs": len(pairs),
        "invalid_pairs": len(invalid_pairs),
    }
    for field, current in comparisons.items():
        if int(previous.get(field, -1)) != current:
            raise ValueError(f"Phase 34j replay changed {field}")
    for label in FOCUS_BYTES:
        if int(previous.get("source_counts", {}).get(label, -1)) != (
            source_counts[label]
        ):
            raise ValueError(f"Phase 34j replay changed {label!r} count")
    if phase34j_summary.get("focus_token_ids") != token_ids:
        raise ValueError("Phase 34j replay changed focus token IDs")
    for split in TARGET_SPLITS:
        previous_file = phase34j_summary.get("data_files", {}).get(split, {})
        current_file = data_files[split]
        if previous_file.get("sha256") != current_file["sha256"]:
            raise ValueError(f"Phase 34j {split} data hash changed")
        if int(previous_file.get("rows", -1)) != current_file["rows"]:
            raise ValueError(f"Phase 34j {split} row count changed")
    for label in ("phase34d", "phase34i"):
        previous_hash = (
            phase34j_summary.get("models", {})
            .get(label, {})
            .get("checkpoint", {})
            .get("sha256")
        )
        if previous_hash != checkpoint_metadata[label]["sha256"]:
            raise ValueError(f"Phase 34j {label} checkpoint hash changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase34d-checkpoint", type=Path, required=True)
    parser.add_argument("--phase34i-checkpoint", type=Path, required=True)
    parser.add_argument("--phase34j-summary", type=Path, required=True)
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    phase34j_summary = json.loads(
        args.phase34j_summary.read_text(encoding="utf-8")
    )
    phase34j_premise = _validated_phase34j_premise(
        phase34j_summary, args.phase34j_summary
    )
    device = torch.device(resolve_device(args.device))
    paths = {
        "phase34d": args.phase34d_checkpoint,
        "phase34i": args.phase34i_checkpoint,
    }
    models = {}
    tokenizers = {}
    checkpoints = {}
    block_sizes = {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label] = model
        tokenizers[label] = tokenizer
        checkpoints[label] = checkpoint
        block_sizes[label] = block_size
    _validate_checkpoint_pair(checkpoints, tokenizers)
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 34d and Phase 34i block sizes differ")
    tokenizer = tokenizers["phase34d"]
    block_size = block_sizes["phase34d"]

    records = []
    data_files = {}
    for split in TARGET_SPLITS:
        path = args.phase31_data_dir / f"{split}.jsonl"
        loaded = load_expanded_records(path)
        records.extend(loaded)
        data_files[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(loaded),
        }
    pairs, invalid_pairs, target_starts, token_ids = (
        build_intervention_pairs(tuple(records), tokenizer, block_size)
    )
    if not pairs:
        raise ValueError("no Phase 34k intervention pairs were constructed")
    checkpoint_metadata = {
        label: _checkpoint_metadata(
            paths[label], checkpoints[label], tokenizers[label]
        )
        for label in ("phase34d", "phase34i")
    }
    _validate_replay(
        phase34j_summary,
        pairs,
        invalid_pairs,
        target_starts,
        token_ids,
        data_files,
        checkpoint_metadata,
    )

    model_results = {
        label: evaluate_model(model, pairs, args.batch_size, label)
        for label, model in models.items()
    }
    report_models = {
        label: {
            "checkpoint": checkpoint_metadata[label],
            **summarize_model_results(model_results[label]),
        }
        for label in ("phase34d", "phase34i")
    }
    source_counts = Counter(pair.source_token for pair in pairs)
    summary = {
        "context_path_intervention_version": (
            CONTEXT_PATH_INTERVENTION_VERSION
        ),
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint_changes": "none",
        "target": "Phase 34j's 514 paired scene_route gold starts",
        "decomposition": (
            "proposal_key(h_current) versus proposal_query(h_final)"
        ),
        "conditions": list(CONDITIONS),
        "phase34j_premise": phase34j_premise,
        "focus_token_ids": token_ids,
        "pair_validation": {
            "target_focus_starts": target_starts,
            "valid_pairs": len(pairs),
            "invalid_pairs": len(invalid_pairs),
            "source_counts": {
                label: int(source_counts[label]) for label in FOCUS_BYTES
            },
            "invalid_rows": list(invalid_pairs),
        },
        "data_files": data_files,
        "models": report_models,
    }
    summary["decision"] = context_path_decision(summary)
    rows_path = args.output.with_name(f"{args.output.stem}.rows.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows_path.write_text(
        "".join(
            json.dumps(
                {
                    **pair.metadata(),
                    "models": {
                        label: model_results[label][index]["conditions"]
                        for label in ("phase34d", "phase34i")
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for index, pair in enumerate(pairs)
        ),
        encoding="utf-8",
    )
    print(
        f"pairs: {len(pairs)}/{target_starts}, invalid {len(invalid_pairs)}"
    )
    for label in ("phase34d", "phase34i"):
        for source in FOCUS_BYTES:
            pathway = report_models[label]["source_strata"][source][
                "pathway"
            ]
            assessment = summary["decision"]["context_path_assessments"][
                label
            ][source]
            print(
                f"{label}/{source}: full/local/query margin shifts "
                f"{pathway['mean_full_context_ro_minus_or_margin']:.3f}/"
                f"{pathway['mean_local_key_ro_minus_or_margin']:.3f}/"
                f"{pathway['mean_query_ro_minus_or_margin']:.3f}; "
                f"shares {assessment['local_key_effect_share']:.3f}/"
                f"{assessment['resolver_query_effect_share']:.3f}; "
                f"{assessment['branch']}"
            )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    for reason in decision["invalid_reasons"]:
        print(f"- {reason}")
    print(f"next action: {decision['next_action']}")
    print(f"summary: {args.output}")
    print(f"rows: {rows_path}")
    if decision["branch"] == "invalid_context_path_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
