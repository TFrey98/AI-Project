"""Evaluate Phase 34 proposal and frozen-resolver behavior end to end."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    CLARIFY_ACTION,
    EXPANDED_SPLITS,
    GENERATE_ACTION,
    MAX_CANDIDATES,
    RESOLVED_VALUE_MARKER,
    RESOLVE_ACTION,
    load_expanded_records,
    summarize_expanded_predictions,
)
from story_model.explicit_offset_candidate_proposer import (
    EXCLUDED_PROPOSER_CASES,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    ExplicitOffsetCandidateProposer,
    decode_proposed_spans,
    encode_proposal_records,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
    realize_proposed_decision,
    runtime_record_from_spans,
)
from story_model.models import build_model
from story_model.runtime import resolve_device
from story_model.unified_typed_span_resolver import (
    GENERATE_MODE,
    MODE_TO_INDEX,
    NO_SUPPORT_OPTION_INDEX,
    SUPPORT_MASK_VERSION,
    UnifiedTypedSpanResolver,
    encode_unified_records,
    unified_batch,
)


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
        raise ValueError("Phase 34 evaluation requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    resolver = UnifiedTypedSpanResolver(backbone)
    model = ExplicitOffsetCandidateProposer(resolver, tokenizer)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size, checkpoint


def _rate(rows, key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _augment_end_to_end_metrics(metrics: dict, rows: tuple[dict, ...]) -> None:
    resolve = tuple(row for row in rows if row["expected_action"] == RESOLVE_ACTION)
    clarify = tuple(row for row in rows if row["expected_action"] == CLARIFY_ACTION)
    metrics.update(
        {
            "candidate_inventory_recall": _rate(
                resolve, "answer_candidate_present"
            ),
            "real_candidate_top1_accuracy": _rate(
                resolve, "real_candidate_correct"
            ),
            "mode_accuracy": _rate(rows, "mode_correct"),
            "clarify_sentinel_accuracy": _rate(
                clarify, "no_support_selected"
            ),
            "no_support_false_positive_rate": _rate(
                resolve, "no_support_selected"
            ),
        }
    )


def _summarize(rows: list[dict]) -> dict:
    frozen_rows = tuple(rows)
    metrics = summarize_expanded_predictions(frozen_rows)
    _augment_end_to_end_metrics(metrics, frozen_rows)
    metrics["proposer"] = proposal_metrics(frozen_rows)
    grouped = defaultdict(list)
    for row in frozen_rows:
        grouped[row["skill"]].append(row)
    for skill, skill_rows in grouped.items():
        group = tuple(skill_rows)
        _augment_end_to_end_metrics(metrics["per_skill"][skill], group)
        metrics["per_skill"][skill]["proposer"] = proposal_metrics(group)
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
    eligible_records = tuple(
        record for record in records if proposal_record_is_eligible(record)
    )
    excluded = len(records) - len(eligible_records)
    examples = encode_proposal_records(eligible_records, tokenizer, block_size)
    rows = []
    started = time.monotonic()
    batch_total = (len(examples) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(
        range(0, len(examples), batch_size), start=1
    ):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        proposer_input = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        proposer_output = model(
            proposer_input[0], proposer_input[1], proposer_input[2]
        )
        proposal_logits = proposer_output.tag_logits.detach().cpu()
        predicted_spans = tuple(
            decode_proposed_spans(
                eligible_records[index].prompt,
                examples[index],
                proposal_logits[offset],
                tokenizer,
            )
            for offset, index in enumerate(indices)
        )
        source_records = tuple(eligible_records[index] for index in indices)
        runtime_records = tuple(
            runtime_record_from_spans(record, spans)
            for record, spans in zip(source_records, predicted_spans)
        )
        resolver_examples = encode_unified_records(
            runtime_records, tokenizer, block_size
        )
        resolver_input = unified_batch(
            resolver_examples, range(len(resolver_examples)), device
        )
        resolver_output = model.resolver(*resolver_input[:5])
        mode_predictions = resolver_output.mode_logits.argmax(dim=-1).tolist()
        option_predictions = resolver_output.option_logits.argmax(dim=-1).tolist()
        positions = torch.arange(MAX_CANDIDATES, device=device).unsqueeze(0)
        inventory_valid = positions < resolver_input[7].unsqueeze(1)
        raw_real_logits = resolver_output.raw_option_logits[
            :, :MAX_CANDIDATES
        ].masked_fill(
            ~inventory_valid,
            torch.finfo(resolver_output.raw_option_logits.dtype).min,
        )
        real_predictions = raw_real_logits.argmax(dim=-1).tolist()

        for offset, (source, runtime, spans) in enumerate(
            zip(source_records, runtime_records, predicted_spans)
        ):
            mode_index = mode_predictions[offset]
            option_index = option_predictions[offset]
            decision = realize_proposed_decision(
                source, runtime, mode_index, option_index
            )
            expected_mode = (
                MODE_TO_INDEX[GENERATE_MODE]
                if source.expected_action == GENERATE_ACTION
                else 0
            )
            expected_runtime_index = next(
                (
                    index
                    for index, candidate in enumerate(runtime.candidates)
                    if candidate.text == source.expected_value
                    and candidate.value_type == source.expected_type
                ),
                None,
            )
            answer_present = (
                source.expected_action != RESOLVE_ACTION
                or expected_runtime_index is not None
            )
            candidate_correct = (
                source.expected_action == RESOLVE_ACTION
                and expected_runtime_index is not None
                and option_index == expected_runtime_index
            )
            real_candidate_correct = (
                source.expected_action == RESOLVE_ACTION
                and expected_runtime_index is not None
                and real_predictions[offset] == expected_runtime_index
            )
            action_correct = decision.action == source.expected_action
            end_to_end = action_correct and (
                source.expected_action != RESOLVE_ACTION or candidate_correct
            )
            expected_text = (
                source.response_template.replace(
                    RESOLVED_VALUE_MARKER, source.expected_value
                )
                if source.expected_action == RESOLVE_ACTION
                else None
            )
            proposal_row = proposal_metric_row(
                source, examples[indices[offset]].gold_spans, spans
            )
            row = {
                **proposal_row,
                "source_phase": source.source_phase,
                "source_context_id": source.source_context_id,
                "source_candidate_count": len(source.candidates),
                "candidate_count": len(runtime.candidates),
                "expected_candidate_index": expected_runtime_index,
                "predicted_option_index": option_index,
                "predicted_real_candidate_index": real_predictions[offset],
                "predicted_action": decision.action,
                "selected_value": decision.selected_value,
                "realized_text": decision.text,
                "answer_candidate_present": answer_present,
                "mode_correct": mode_index == expected_mode,
                "action_correct": action_correct,
                "candidate_correct": candidate_correct,
                "real_candidate_correct": real_candidate_correct,
                "end_to_end_correct": end_to_end,
                "exact_realization": (
                    source.expected_action == RESOLVE_ACTION
                    and decision.action == RESOLVE_ACTION
                    and decision.selected_value == source.expected_value
                    and decision.text == expected_text
                ),
                "no_support_selected": (
                    option_index == NO_SUPPORT_OPTION_INDEX
                ),
                "value_missing": (
                    source.expected_action == RESOLVE_ACTION
                    and decision.selected_value != source.expected_value
                ),
                "wrong_alternative": (
                    source.expected_action == RESOLVE_ACTION
                    and decision.selected_value == source.alternative_value
                ),
                "guarded": decision.guarded,
            }
            rows.append(row)
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
    return rows, _summarize(rows), excluded


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
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", 0),
        "checkpoint_eligible": checkpoint.get("extra", {}).get(
            "checkpoint_eligible"
        ),
        "device": str(device),
        "excluded_cases": list(EXCLUDED_PROPOSER_CASES),
        "phase31_regression": {"splits": {}},
        "phase32": {"splits": {}},
    }
    all_rows = []
    for dataset_name, data_dir in (
        ("phase31_regression", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        summary[dataset_name]["excluded_rows"] = 0
        for split in EXPANDED_SPLITS:
            label = f"{dataset_name}/{split}"
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            rows, metrics, excluded = evaluate_records(
                model,
                tokenizer,
                block_size,
                records,
                device,
                args.batch_size,
                label,
                args.progress_every,
            )
            all_rows.extend(rows)
            summary[dataset_name]["excluded_rows"] += excluded
            summary[dataset_name]["splits"][split] = metrics
            proposer = metrics["proposer"]
            print(
                f"{label}: span_f1={proposer['exact_span_f1']:.3f}, "
                f"inventory={metrics['candidate_inventory_recall']:.3f}, "
                f"resolve={metrics['end_to_end_resolve_accuracy']:.3f}, "
                f"pairs={metrics['counterfactual_pair_resolve_accuracy']:.3f}, "
                f"clarify={metrics['clarify_accuracy']:.3f}, "
                f"missing={metrics['value_missing_rate']:.3f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_rows
        ),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"predictions: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
