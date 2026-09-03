"""Frozen paired token intervention for Phase 34j."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_BEGIN,
    BOUNDARY_INSIDE,
    BOUNDARY_OBJECTIVE_VERSION,
    BOUNDARY_OUTSIDE,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    PROPOSAL_TYPES,
    EncodedProposalExample,
    encode_proposal_record,
    gold_evidence_spans,
    proposal_record_is_eligible,
)
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device

try:
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.phase34j_paired_token_decision import paired_token_decision
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_explicit_offset_tag_confusion import (  # type: ignore[no-redef]
        _load_model,
    )
    from phase34j_paired_token_decision import (  # type: ignore[no-redef]
        paired_token_decision,
    )


PAIRED_TOKEN_INTERVENTION_VERSION = 1
FOCUS_BYTES = {"or": b"or", "ro": b"ro"}
SWAPPED_LABEL = {"or": "ro", "ro": "or"}
TARGET_SPLITS = ("lexical", "transfer")
TARGET_SKILL = "scene_route"
CLASS_NAMES = ("O", "B", "I")


@dataclass(frozen=True)
class InterventionPair:
    pair_id: str
    record_id: str
    split: str
    case: str
    byte_start: int
    token_position: int
    source_token: str
    swapped_token: str
    original_token_id: int
    swapped_token_id: int
    sequence_tokens: int
    prompt_token_count: int
    original_input_ids: tuple[int, ...]
    swapped_input_ids: tuple[int, ...]

    def metadata(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "record_id": self.record_id,
            "split": self.split,
            "case": self.case,
            "byte_start": self.byte_start,
            "token_position": self.token_position,
            "source_token": self.source_token,
            "swapped_token": self.swapped_token,
            "original_token_id": self.original_token_id,
            "swapped_token_id": self.swapped_token_id,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for label, token_bytes in FOCUS_BYTES.items():
        token_ids = tokenizer.encode(token_bytes.decode("ascii"))
        if len(token_ids) != 1:
            raise ValueError(f"{label!r} is not one canonical BPE token")
        token_id = token_ids[0]
        if tokenizer.token_bytes(token_id) != token_bytes:
            raise ValueError(f"{label!r} token bytes do not round-trip")
        result[label] = token_id
    if len(set(result.values())) != len(result):
        raise ValueError("focus strings resolve to the same token ID")
    return result


def _token_at_byte_start(
    example: EncodedProposalExample,
    tokenizer,
    byte_start: int,
) -> tuple[int, int]:
    cursor = 0
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        if cursor <= byte_start < cursor + width:
            return token_position, byte_start - cursor
        cursor += width
    raise ValueError(f"byte position {byte_start} is outside the prompt")


def build_intervention_pairs(records, tokenizer, block_size: int):
    """Build canonical equal-length `or`/`ro` pairs at gold starts."""

    token_ids = _canonical_token_ids(tokenizer)
    label_by_id = {token_id: label for label, token_id in token_ids.items()}
    pairs = []
    invalid = []
    target_starts = 0
    for record in records:
        if (
            record.skill != TARGET_SKILL
            or record.split not in TARGET_SPLITS
            or not proposal_record_is_eligible(record)
        ):
            continue
        example = encode_proposal_record(record, tokenizer, block_size)
        prompt_ids = list(example.input_ids[: example.prompt_token_count])
        for span in gold_evidence_spans(record):
            if span.value_type != "route":
                continue
            token_position, byte_offset = _token_at_byte_start(
                example, tokenizer, span.byte_start
            )
            original_token_id = prompt_ids[token_position]
            if original_token_id not in label_by_id or byte_offset != 0:
                continue
            target_starts += 1
            source_label = label_by_id[original_token_id]
            swapped_label = SWAPPED_LABEL[source_label]
            swapped_token_id = token_ids[swapped_label]
            swapped_prompt_ids = list(prompt_ids)
            swapped_prompt_ids[token_position] = swapped_token_id
            prompt_bytes = bytearray(record.prompt.encode("utf-8"))
            prompt_bytes[span.byte_start : span.byte_start + 2] = (
                FOCUS_BYTES[swapped_label]
            )
            pair_id = f"{record.record_id}@{span.byte_start}"
            try:
                swapped_prompt = bytes(prompt_bytes).decode("utf-8")
            except UnicodeDecodeError:
                invalid.append(
                    {"pair_id": pair_id, "reason": "swap produced invalid UTF-8"}
                )
                continue
            if tokenizer.encode(swapped_prompt) != swapped_prompt_ids:
                invalid.append(
                    {
                        "pair_id": pair_id,
                        "reason": "swapped prompt is not canonically tokenized",
                    }
                )
                continue
            if tokenizer.decode(swapped_prompt_ids) != swapped_prompt:
                invalid.append(
                    {"pair_id": pair_id, "reason": "swapped prompt did not decode"}
                )
                continue
            swapped_input_ids = list(example.input_ids)
            swapped_input_ids[token_position] = swapped_token_id
            pairs.append(
                InterventionPair(
                    pair_id=pair_id,
                    record_id=record.record_id,
                    split=record.split,
                    case=record.case,
                    byte_start=span.byte_start,
                    token_position=token_position,
                    source_token=source_label,
                    swapped_token=swapped_label,
                    original_token_id=original_token_id,
                    swapped_token_id=swapped_token_id,
                    sequence_tokens=example.sequence_tokens,
                    prompt_token_count=example.prompt_token_count,
                    original_input_ids=example.input_ids,
                    swapped_input_ids=tuple(swapped_input_ids),
                )
            )
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("paired intervention IDs are not unique")
    return tuple(pairs), tuple(invalid), target_starts, token_ids


def _collapse_legacy_logits(tag_logits: torch.Tensor) -> torch.Tensor:
    if tag_logits.shape[-1] != 1 + 2 * len(PROPOSAL_TYPES):
        raise ValueError("legacy typed logits have an unexpected width")
    return torch.stack(
        (
            tag_logits[..., 0],
            tag_logits[..., 1::2].max(dim=-1).values,
            tag_logits[..., 2::2].max(dim=-1).values,
        ),
        dim=-1,
    )


@torch.no_grad()
def component_boundary_logits(
    model,
    original_tokens: torch.Tensor,
    swapped_tokens: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_positions: torch.Tensor,
    original_token_ids: torch.Tensor,
    swapped_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score original, byte-only, context-only, and full interventions."""

    if model.token_width_geometry or model.token_end_geometry:
        raise ValueError("Phase 34j requires the pre-geometry proposer")
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
        raise ValueError("paired intervention tensors have inconsistent batches")
    hidden = model.resolver.hidden_states(
        torch.cat((original_tokens, swapped_tokens), dim=0)
    )
    original_hidden, swapped_hidden = hidden.split(batch_size, dim=0)
    batch_indices = torch.arange(batch_size, device=original_tokens.device)
    query_positions = sequence_lengths - 1
    original_context = (
        model.proposal_key(original_hidden[batch_indices, token_positions])
        + model.proposal_query(
            original_hidden[batch_indices, query_positions]
        )
    )
    swapped_context = (
        model.proposal_key(swapped_hidden[batch_indices, token_positions])
        + model.proposal_query(swapped_hidden[batch_indices, query_positions])
    )
    original_byte_ids = model.token_byte_ids[original_token_ids, 0]
    swapped_byte_ids = model.token_byte_ids[swapped_token_ids, 0]
    embeddings = model.resolver.backbone.token_embeddings
    original_bytes = embeddings(original_byte_ids)
    swapped_bytes = embeddings(swapped_byte_ids)
    offset = model.proposal_offset_embedding.weight[0]

    def score(context: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
        states = torch.tanh(context + byte + offset)
        if model.factorized_boundary_type:
            if model.proposal_boundary is None:
                raise RuntimeError("factorized checkpoint has no boundary head")
            return model.proposal_boundary(states)
        return _collapse_legacy_logits(model.proposal_tag(states))

    return {
        "original": score(original_context, original_bytes),
        "byte_swap_only": score(original_context, swapped_bytes),
        "context_swap_only": score(swapped_context, original_bytes),
        "full_swap": score(swapped_context, swapped_bytes),
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
        logits = component_boundary_logits(
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
            elapsed = max(time.monotonic() - started, 1.0e-9)
            print(
                f"{progress_label}: {min(start + batch_size, len(pairs)):,}/"
                f"{len(pairs):,} pairs "
                f"({min(start + batch_size, len(pairs)) / elapsed:.1f} pairs/s)",
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


def summarize_model_results(results) -> dict:
    rendered = {label: [] for label in FOCUS_BYTES}
    strata = {
        source: {label: [] for label in FOCUS_BYTES}
        for source in FOCUS_BYTES
    }
    full_deltas = []
    byte_deltas = []
    context_deltas = []
    transitions = Counter()
    for row in results:
        source = row["source_token"]
        swapped = row["swapped_token"]
        conditions = row["conditions"]
        rendered[source].append(conditions["original"])
        rendered[swapped].append(conditions["full_swap"])
        strata[source][source].append(conditions["original"])
        strata[source][swapped].append(conditions["full_swap"])
        transitions[
            f"{source}:{conditions['original']['predicted_class']}->"
            f"{swapped}:{conditions['full_swap']['predicted_class']}"
        ] += 1
        original_margin = conditions["original"][
            "begin_minus_inside_margin"
        ]
        byte_margin = conditions["byte_swap_only"][
            "begin_minus_inside_margin"
        ]
        context_margin = conditions["context_swap_only"][
            "begin_minus_inside_margin"
        ]
        full_margin = conditions["full_swap"][
            "begin_minus_inside_margin"
        ]
        if source == "or":
            full_deltas.append(full_margin - original_margin)
            byte_deltas.append(byte_margin - original_margin)
            context_deltas.append(context_margin - original_margin)
        else:
            full_deltas.append(original_margin - full_margin)
            byte_deltas.append(original_margin - byte_margin)
            context_deltas.append(original_margin - context_margin)
    count = len(results)
    return {
        "rendered_identity": {
            label: _condition_summary(values)
            for label, values in rendered.items()
        },
        "source_strata": {
            source: {
                "rendered_identity": {
                    label: _condition_summary(values)
                    for label, values in renderings.items()
                }
            }
            for source, renderings in strata.items()
        },
        "pathway": {
            "observations": count,
            "margin_orientation": "rendered_ro_minus_rendered_or",
            "mean_full_ro_minus_or_margin": (
                sum(full_deltas) / count if count else 0.0
            ),
            "mean_byte_ro_minus_or_margin": (
                sum(byte_deltas) / count if count else 0.0
            ),
            "mean_context_ro_minus_or_margin": (
                sum(context_deltas) / count if count else 0.0
            ),
        },
        "full_swap_transitions": dict(sorted(transitions.items())),
    }


def _checkpoint_metadata(path: Path, checkpoint: dict, tokenizer) -> dict:
    extra = checkpoint.get("extra", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "step": checkpoint.get("step", 0),
        "checkpoint_eligible": extra.get("checkpoint_eligible"),
        "architecture": extra.get("architecture"),
        "explicit_offset_proposer_version": extra.get(
            "explicit_offset_proposer_version"
        ),
        "boundary_objective_version": extra.get(
            "boundary_objective_version"
        ),
        "boundary_loss_weight": extra.get("boundary_loss_weight"),
        "token_width_geometry_version": extra.get(
            "token_width_geometry_version"
        ),
        "token_end_geometry_version": extra.get(
            "token_end_geometry_version"
        ),
        "factorized_boundary_type_version": extra.get(
            "factorized_boundary_type_version"
        ),
        "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
    }


def _validate_checkpoint_pair(checkpoints: dict, tokenizers: dict) -> None:
    for label, expected_factorized in (("phase34d", None), ("phase34i", 1)):
        extra = checkpoints[label].get("extra", {})
        if extra.get("architecture") != "explicit_offset_candidate_proposer":
            raise ValueError(f"{label} is not an explicit-offset proposer")
        if extra.get("explicit_offset_proposer_version") != (
            EXPLICIT_OFFSET_PROPOSER_VERSION
        ):
            raise ValueError(f"{label} has an unsupported proposer version")
        if extra.get("boundary_objective_version") != (
            BOUNDARY_OBJECTIVE_VERSION
        ):
            raise ValueError(f"{label} lacks the Phase 34d boundary objective")
        if abs(float(extra.get("boundary_loss_weight", -1.0)) - 1.0) > 1.0e-9:
            raise ValueError(f"{label} boundary loss weight is not 1.0")
        if extra.get("token_width_geometry_version") is not None:
            raise ValueError(f"{label} unexpectedly uses token-width geometry")
        if extra.get("token_end_geometry_version") is not None:
            raise ValueError(f"{label} unexpectedly uses token-end geometry")
        if extra.get("factorized_boundary_type_version") != expected_factorized:
            raise ValueError(f"{label} factorized-head provenance is wrong")
    hashes = {
        canonical_json_sha256(tokenizer.to_dict())
        for tokenizer in tokenizers.values()
    }
    if len(hashes) != 1:
        raise ValueError("Phase 34d and Phase 34i tokenizers differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase34d-checkpoint", type=Path, required=True)
    parser.add_argument("--phase34i-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase34i-decision",
        type=Path,
        default=Path("runs/phase34i-factorized-head-decision.json"),
    )
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
    phase34i_decision = json.loads(
        args.phase34i_decision.read_text(encoding="utf-8")
    )
    if phase34i_decision.get("branch") != "factorized_head_rejected":
        raise ValueError("Phase 34i decision did not reject factorization")
    if phase34i_decision.get("checkpoint_promotion_authorized") is not False:
        raise ValueError("Phase 34i decision has invalid promotion provenance")

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
        raise ValueError("no Phase 34j intervention pairs were constructed")
    source_counts = Counter(pair.source_token for pair in pairs)
    model_results = {
        label: evaluate_model(model, pairs, args.batch_size, label)
        for label, model in models.items()
    }
    report_models = {}
    for label in ("phase34d", "phase34i"):
        report_models[label] = {
            "checkpoint": _checkpoint_metadata(
                paths[label], checkpoints[label], tokenizers[label]
            ),
            **summarize_model_results(model_results[label]),
        }
    summary = {
        "paired_token_intervention_version": (
            PAIRED_TOKEN_INTERVENTION_VERSION
        ),
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint_changes": "none",
        "target": "Phase 31 lexical+transfer scene_route gold starts",
        "intervention": (
            "canonical equal-width atomic or<->ro token replacement at one "
            "gold start"
        ),
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
        "phase34i_decision": {
            "path": str(args.phase34i_decision),
            "sha256": _sha256(args.phase34i_decision),
            "branch": phase34i_decision["branch"],
        },
        "data_files": data_files,
        "models": report_models,
    }
    summary["decision"] = paired_token_decision(summary)
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
        rendered = report_models[label]["rendered_identity"]
        pathway = report_models[label]["pathway"]
        print(
            f"{label}: or B->I {rendered['or']['begin_as_inside_rate']:.3f}, "
            f"ro B->I {rendered['ro']['begin_as_inside_rate']:.3f}, "
            f"full/byte/context margin shifts "
            f"{pathway['mean_full_ro_minus_or_margin']:.3f}/"
            f"{pathway['mean_byte_ro_minus_or_margin']:.3f}/"
            f"{pathway['mean_context_ro_minus_or_margin']:.3f}"
        )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    for reason in decision["invalid_reasons"]:
        print(f"- {reason}")
    print(f"next action: {decision['next_action']}")
    print(f"summary: {args.output}")
    print(f"rows: {rows_path}")
    if decision["branch"] == "invalid_paired_token_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
