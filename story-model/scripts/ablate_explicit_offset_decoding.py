"""Compare permissive and strict BIO decoding on identical Phase 34 logits."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    DECODE_POLICIES,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    PERMISSIVE_DECODE_POLICY,
    STRICT_DECODE_POLICY,
    ExplicitOffsetCandidateProposer,
    decode_proposal_result,
    encode_proposal_records,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
)
from story_model.models import build_model
from story_model.runtime import resolve_device
from story_model.unified_typed_span_resolver import (
    SUPPORT_MASK_VERSION,
    UnifiedTypedSpanResolver,
)
try:
    from scripts.phase34b_decoder_decision import (
        decoder_ablation_decision,
        false_positive_boundary_displacements,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34b_decoder_decision import (  # type: ignore[no-redef]
        decoder_ablation_decision,
        false_positive_boundary_displacements,
    )


DECODER_ABLATION_VERSION = 1


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
        raise ValueError("Phase 34b requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    resolver = UnifiedTypedSpanResolver(backbone)
    model = ExplicitOffsetCandidateProposer(resolver, tokenizer)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size, checkpoint


def _flat_decoder_metrics(rows) -> dict:
    rows = tuple(rows)
    metrics = proposal_metrics(rows)
    metrics.update(
        {
            "false_positive_spans": sum(
                row["predicted_span_count"] - row["exact_span_matches"]
                for row in rows
            ),
            "false_negative_spans": sum(
                row["gold_span_count"] - row["exact_span_matches"]
                for row in rows
            ),
            "orphan_inside_tags": sum(
                row["orphan_inside_tags"] for row in rows
            ),
            "mismatched_inside_tags": sum(
                row["mismatched_inside_tags"] for row in rows
            ),
            "invalid_utf8_spans": sum(
                row["invalid_utf8_spans"] for row in rows
            ),
            "truncated_spans": sum(row["truncated_spans"] for row in rows),
            "false_positive_boundary_displacement": dict(
                sorted(
                    Counter(
                        displacement
                        for row in rows
                        for displacement in row["boundary_displacements"]
                    ).items()
                )
            ),
        }
    )
    return metrics


def summarize_decoder_rows(rows) -> dict:
    rows = tuple(rows)
    metrics = _flat_decoder_metrics(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["skill"]].append(row)
    metrics["per_skill"] = {
        skill: _flat_decoder_metrics(skill_rows)
        for skill, skill_rows in sorted(grouped.items())
    }
    return metrics


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
    rows = {policy: [] for policy in DECODE_POLICIES}
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
            record = records[index]
            example = examples[index]
            for policy in DECODE_POLICIES:
                decoded = decode_proposal_result(
                    record.prompt,
                    example,
                    logits[offset],
                    tokenizer,
                    policy=policy,
                )
                row = proposal_metric_row(
                    record, example.gold_spans, decoded.spans
                )
                row.update(
                    {
                        "policy": policy,
                        "orphan_inside_tags": decoded.orphan_inside_tags,
                        "mismatched_inside_tags": (
                            decoded.mismatched_inside_tags
                        ),
                        "invalid_utf8_spans": decoded.invalid_utf8_spans,
                        "truncated_spans": decoded.truncated_spans,
                        "boundary_displacements": (
                            false_positive_boundary_displacements(
                                example.gold_spans, decoded.spans
                            )
                        ),
                    }
                )
                rows[policy].append(row)
        if progress_every and (
            batch_number % progress_every == 0 or batch_number == batch_total
        ):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            completed = min(start + batch_size, len(examples))
            print(
                f"{label}: {completed:,}/{len(examples):,} rows "
                f"({completed / elapsed:.1f} rows/s; one forward/two decodes)",
                flush=True,
            )
    return rows, {
        policy: summarize_decoder_rows(policy_rows)
        for policy, policy_rows in rows.items()
    }


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
    summary = {
        "decoder_ablation_version": DECODER_ABLATION_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", 0),
        "checkpoint_eligible": checkpoint.get("extra", {}).get(
            "checkpoint_eligible"
        ),
        "device": str(device),
        "logit_reuse": (
            "one proposer forward per batch; both policies decode identical logits"
        ),
        "policies": {
            PERMISSIVE_DECODE_POLICY: (
                "orphan or type-mismatched I starts a new span"
            ),
            STRICT_DECODE_POLICY: (
                "only B starts; orphan or type-mismatched I is discarded"
            ),
        },
        "datasets": {
            "phase31_regression": {"splits": {}},
            "phase32": {"splits": {}},
        },
    }
    all_rows = {policy: [] for policy in DECODE_POLICIES}
    output_rows = []
    for dataset_name, data_dir in (
        ("phase31_regression", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        for split in EXPANDED_SPLITS:
            label = f"{dataset_name}/{split}"
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            rows, metrics = evaluate_records(
                model,
                tokenizer,
                block_size,
                records,
                device,
                args.batch_size,
                label,
                args.progress_every,
            )
            summary["datasets"][dataset_name]["splits"][split] = {
                "policies": metrics
            }
            for policy in DECODE_POLICIES:
                all_rows[policy].extend(rows[policy])
                output_rows.extend(rows[policy])
            permissive = metrics[PERMISSIVE_DECODE_POLICY]
            strict = metrics[STRICT_DECODE_POLICY]
            print(
                f"{label}: precision "
                f"{permissive['exact_span_precision']:.3f}->"
                f"{strict['exact_span_precision']:.3f}, recall "
                f"{permissive['exact_span_recall']:.3f}->"
                f"{strict['exact_span_recall']:.3f}, answer "
                f"{permissive['answer_candidate_recall']:.3f}->"
                f"{strict['answer_candidate_recall']:.3f}",
                flush=True,
            )
    summary["overall"] = {
        policy: summarize_decoder_rows(rows)
        for policy, rows in all_rows.items()
    }
    summary["decision"] = decoder_ablation_decision(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    print(f"next action: {decision['next_action']}")
    print(f"rows: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
