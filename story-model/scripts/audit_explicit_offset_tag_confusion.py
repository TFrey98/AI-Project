"""Audit Phase 34 byte-tag confusion without changing decoding or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    EXCLUDED_PROPOSER_CASES,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    ExplicitOffsetCandidateProposer,
    checkpoint_uses_factorized_boundary_type,
    checkpoint_uses_token_end_geometry,
    checkpoint_uses_token_width_geometry,
    encode_proposal_records,
    proposal_batch,
    proposal_record_is_eligible,
)
from story_model.models import build_model
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device
from story_model.unified_typed_span_resolver import (
    SUPPORT_MASK_VERSION,
    UnifiedTypedSpanResolver,
)
try:
    from scripts.phase34c_tag_confusion_decision import (
        CURRENT_CLASS_WEIGHTS,
        add_tag_sequence,
        collapsed_tag,
        merge_tag_audit,
        new_tag_audit,
        summarize_tag_audit,
        tag_confusion_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34c_tag_confusion_decision import (  # type: ignore[no-redef]
        CURRENT_CLASS_WEIGHTS,
        add_tag_sequence,
        collapsed_tag,
        merge_tag_audit,
        new_tag_audit,
        summarize_tag_audit,
        tag_confusion_decision,
    )


TAG_CONFUSION_AUDIT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path, device: torch.device):
    checkpoint = read_checkpoint(path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "explicit_offset_candidate_proposer":
        raise ValueError("checkpoint is not a Phase 34 candidate proposer")
    if (
        extra.get("explicit_offset_proposer_version")
        != EXPLICIT_OFFSET_PROPOSER_VERSION
    ):
        raise ValueError("checkpoint has an unsupported proposer version")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("checkpoint does not contain the Phase 33c router")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 34c requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    resolver = UnifiedTypedSpanResolver(backbone)
    model = ExplicitOffsetCandidateProposer(
        resolver,
        tokenizer,
        token_width_geometry=checkpoint_uses_token_width_geometry(extra),
        token_end_geometry=checkpoint_uses_token_end_geometry(extra),
        factorized_boundary_type=checkpoint_uses_factorized_boundary_type(
            extra
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size, checkpoint


def _collapsed_model_weights(model) -> dict:
    if model.factorized_boundary_type:
        weights = (
            model.boundary_class_weights.detach().float().cpu().tolist()
        )
        if len(weights) != 3:
            raise ValueError("factorized proposer has invalid O/B/I weights")
        return {
            "O": float(weights[0]),
            "B": float(weights[1]),
            "I": float(weights[2]),
        }
    weights = model.tag_class_weights.detach().float().cpu().tolist()
    begin_weights = tuple(weights[1::2])
    inside_weights = tuple(weights[2::2])
    if not begin_weights or not inside_weights:
        raise ValueError("proposer has no typed BIO class weights")
    if len(set(begin_weights)) != 1 or len(set(inside_weights)) != 1:
        raise ValueError("Phase 34c requires shared weights across BIO types")
    return {
        "O": float(weights[0]),
        "B": float(begin_weights[0]),
        "I": float(inside_weights[0]),
    }


def _flatten_prompt_predictions(logits, example, tokenizer):
    prompt_logits = []
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        prompt_logits.append(logits[token_position, :width].float())
    if not prompt_logits:
        raise ValueError(f"{example.record_id} has no prompt bytes")
    flat_logits = torch.cat(prompt_logits, dim=0)
    gold = torch.tensor(example.prompt_byte_tags, dtype=torch.long)
    if len(flat_logits) != len(gold):
        raise RuntimeError("flattened logits do not align with prompt byte tags")
    predicted = flat_logits.argmax(dim=-1)
    nll = F.cross_entropy(flat_logits, gold, reduction="none")
    return gold.tolist(), predicted.tolist(), nll.tolist()


def _begin_geometry_rows(logits, example, tokenizer) -> list[dict]:
    token_ranges = []
    cursor = 0
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        token_width = tokenizer.token_byte_length(token_id)
        token_ranges.append(
            (cursor, cursor + token_width, token_position, token_width)
        )
        cursor += token_width
    rows = []
    for span in example.gold_spans:
        for token_start, token_end, token_position, token_width in token_ranges:
            if token_start <= span.byte_start < token_end:
                byte_offset = span.byte_start - token_start
                break
        else:
            raise RuntimeError("gold start is outside prompt token ranges")
        token_id = example.input_ids[token_position]
        token_bytes = tokenizer.token_bytes(token_id)
        try:
            token_utf8 = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_utf8 = None
        predicted = int(logits[token_position, byte_offset].argmax())
        rows.append(
            {
                "token_id": token_id,
                "token_bytes_hex": token_bytes.hex(),
                "token_utf8": token_utf8,
                "token_width": token_width,
                "token_byte_offset": byte_offset,
                "token_alignment": (
                    "at_token_start" if byte_offset == 0 else "inside_token"
                ),
                "token_end": byte_offset + 1 == token_width,
                "begin_as_inside": collapsed_tag(predicted) == "I",
            }
        )
    return rows


def _summarize_begin_geometry(rows) -> dict:
    rows = tuple(rows)

    def summarize(group) -> dict:
        group = tuple(group)
        errors = sum(bool(row["begin_as_inside"]) for row in group)
        return {
            "gold_begin_count": len(group),
            "begin_as_inside_count": errors,
            "begin_as_inside_rate": errors / len(group) if group else 0.0,
        }

    dimensions = {}
    for field in (
        "token_id",
        "token_bytes_hex",
        "token_width",
        "token_byte_offset",
        "token_alignment",
        "token_end",
    ):
        values = sorted({str(row[field]) for row in rows})
        dimensions[field] = {
            value: summarize(
                row for row in rows if str(row[field]) == value
            )
            for value in values
        }
    return {"overall": summarize(rows), "dimensions": dimensions}


@torch.no_grad()
def evaluate_records(
    model,
    tokenizer,
    block_size,
    records,
    device,
    batch_size,
    label,
    progress_every,
):
    records = tuple(
        record for record in records if proposal_record_is_eligible(record)
    )
    examples = encode_proposal_records(records, tokenizer, block_size)
    if len(examples) != len(records):
        raise RuntimeError("eligible records and encoded examples diverged")
    overall = new_tag_audit()
    per_skill = defaultdict(new_tag_audit)
    begin_geometry = []
    started = time.monotonic()
    batch_total = (len(examples) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(
        range(0, len(examples), batch_size), start=1
    ):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        batch = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        output = model(batch[0], batch[1], batch[2])
        logits = output.tag_logits.detach().cpu()
        for offset, index in enumerate(indices):
            gold, predicted, nll = _flatten_prompt_predictions(
                logits[offset], examples[index], tokenizer
            )
            row_audit = new_tag_audit()
            add_tag_sequence(row_audit, gold, predicted, nll)
            merge_tag_audit(overall, row_audit)
            merge_tag_audit(per_skill[records[index].skill], row_audit)
            for row in _begin_geometry_rows(
                logits[offset], examples[index], tokenizer
            ):
                begin_geometry.append(
                    {
                        **row,
                        "split": records[index].split,
                        "skill": records[index].skill,
                    }
                )
        if progress_every and (
            batch_number % progress_every == 0 or batch_number == batch_total
        ):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            completed = min(start + batch_size, len(examples))
            print(
                f"{label}: {completed:,}/{len(examples):,} rows "
                f"({completed / elapsed:.1f} rows/s)",
                flush=True,
            )
    return overall, dict(per_skill), len(records), begin_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument(
        "--phase32-data-dir",
        type=Path,
        default=Path("data/character/expanded_typed_span_resolver"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")

    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size, checkpoint = _load_model(
        args.checkpoint, device
    )
    class_weights = _collapsed_model_weights(model)
    if any(
        abs(class_weights[tag_class] - expected) > 1.0e-6
        for tag_class, expected in CURRENT_CLASS_WEIGHTS.items()
    ):
        raise ValueError(
            "Phase 34c requires the original O/B/I class weights; "
            f"found {class_weights}"
        )
    summary = {
        "tag_confusion_audit_version": TAG_CONFUSION_AUDIT_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", 0),
        "checkpoint_eligible": checkpoint.get("extra", {}).get(
            "checkpoint_eligible"
        ),
        "checkpoint_boundary_objective_version": checkpoint.get(
            "extra", {}
        ).get("boundary_objective_version"),
        "checkpoint_boundary_loss_weight": checkpoint.get("extra", {}).get(
            "boundary_loss_weight", 0.0
        ),
        "checkpoint_token_width_geometry_version": checkpoint.get(
            "extra", {}
        ).get("token_width_geometry_version"),
        "checkpoint_token_end_geometry_version": checkpoint.get(
            "extra", {}
        ).get("token_end_geometry_version"),
        "checkpoint_factorized_boundary_type_version": checkpoint.get(
            "extra", {}
        ).get("factorized_boundary_type_version"),
        "checkpoint_boundary_counterbalance_version": checkpoint.get(
            "extra", {}
        ).get("boundary_counterbalance_version"),
        "checkpoint_counterbalance_manifest_sha256": checkpoint.get(
            "extra", {}
        ).get("counterbalance_manifest_sha256"),
        "checkpoint_tokenizer_sha256": canonical_json_sha256(
            tokenizer.to_dict()
        ),
        "device": str(device),
        "decoder_policy": "unchanged_permissive",
        "training_changes": "none",
        "class_weights": class_weights,
        "excluded_cases": list(EXCLUDED_PROPOSER_CASES),
        "datasets": {
            "phase31_regression": {"splits": {}},
            "phase32": {"splits": {}},
        },
    }
    aggregate = new_tag_audit()
    target_begin_geometry = []
    for dataset_name, data_dir in (
        ("phase31_regression", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        for split in EXPANDED_SPLITS:
            label = f"{dataset_name}/{split}"
            all_records = load_expanded_records(data_dir / f"{split}.jsonl")
            (
                split_audit,
                skill_audits,
                eligible_rows,
                begin_geometry,
            ) = evaluate_records(
                model,
                tokenizer,
                block_size,
                all_records,
                device,
                args.batch_size,
                label,
                args.progress_every,
            )
            merge_tag_audit(aggregate, split_audit)
            if dataset_name == "phase31_regression" and split in {
                "lexical",
                "transfer",
            }:
                target_begin_geometry.extend(
                    row
                    for row in begin_geometry
                    if row["skill"] == "scene_route"
                )
            metrics = summarize_tag_audit(split_audit, class_weights)
            summary["datasets"][dataset_name]["splits"][split] = {
                "rows": len(all_records),
                "eligible_rows": eligible_rows,
                "excluded_rows": len(all_records) - eligible_rows,
                "metrics": metrics,
                "per_skill": {
                    skill: summarize_tag_audit(audit, class_weights)
                    for skill, audit in sorted(skill_audits.items())
                },
            }
            print(
                f"{label}: O->I {metrics['gold_outside_as_inside_rate']:.3f}, "
                f"B->I {metrics['gold_begin_as_inside_rate']:.3f}, "
                f"orphan-I {metrics['orphan_inside_tags']:,}, "
                f"pre-start {metrics['pre_start_bleed_rate']:.3f}, "
                f"end-spill {metrics['end_spill_rate']:.3f}",
                flush=True,
            )
    summary["overall"] = summarize_tag_audit(aggregate, class_weights)
    summary["target_begin_geometry"] = _summarize_begin_geometry(
        target_begin_geometry
    )
    summary["decision"] = tag_confusion_decision(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    print(f"next single variable: {decision['next_single_variable']}")
    print(f"next action: {decision['next_action']}")
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
