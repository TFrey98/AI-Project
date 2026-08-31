"""Evaluate Phase 32 and re-run the complete Phase 31 regression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    RESOLVED_VALUE_MARKER,
    RESOLVER_ACTIONS,
    RESOLVE_ACTION,
    ExpandedTypedSpanResolver,
    encode_expanded_records,
    expanded_batch,
    load_expanded_records,
    realize_expanded_decision,
    summarize_expanded_predictions,
)
from story_model.models import build_model
from story_model.runtime import resolve_device


def _load_model(path: Path, device: torch.device):
    checkpoint = read_checkpoint(path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "expanded_typed_span_resolver":
        raise ValueError("checkpoint is not an expanded typed-span resolver")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    tokenizer = tokenizer_from_dict(extra["tokenizer"])
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("expanded resolver checkpoint requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    model = ExpandedTypedSpanResolver(backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size


@torch.no_grad()
def evaluate_records(model, tokenizer, block_size, records, device, batch_size):
    examples = encode_expanded_records(records, tokenizer, block_size)
    rows = []
    for start in range(0, len(records), batch_size):
        indices = tuple(range(start, min(start + batch_size, len(records))))
        batch = expanded_batch(examples, indices, device)
        output = model(*batch[:5])
        action_predictions = output.action_logits.argmax(dim=-1).tolist()
        candidate_predictions = output.candidate_logits.argmax(dim=-1).tolist()
        for offset, index in enumerate(indices):
            record = records[index]
            action = RESOLVER_ACTIONS[action_predictions[offset]]
            candidate_index = candidate_predictions[offset]
            decision = realize_expanded_decision(record, action, candidate_index)
            action_correct = action == record.expected_action
            candidate_correct = (
                record.selected_candidate_index is not None
                and candidate_index == record.selected_candidate_index
            )
            end_to_end = action_correct and (
                record.expected_action != RESOLVE_ACTION or candidate_correct
            )
            expected_text = (
                record.response_template.replace(
                    RESOLVED_VALUE_MARKER, record.expected_value
                )
                if record.expected_action == RESOLVE_ACTION
                else None
            )
            exact = (
                record.expected_action == RESOLVE_ACTION
                and decision.action == RESOLVE_ACTION
                and decision.selected_value == record.expected_value
                and decision.text == expected_text
            )
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_context_id": record.source_context_id,
                    "conversation_id": record.conversation_id,
                    "source_phase": record.source_phase,
                    "split": record.split,
                    "skill": record.skill,
                    "case": record.case,
                    "candidate_count": len(record.candidates),
                    "expected_action": record.expected_action,
                    "predicted_action": action,
                    "expected_candidate_index": record.selected_candidate_index,
                    "predicted_candidate_index": candidate_index,
                    "expected_value": record.expected_value,
                    "alternative_value": record.alternative_value,
                    "selected_value": decision.selected_value,
                    "realized_text": decision.text,
                    "action_correct": action_correct,
                    "candidate_correct": candidate_correct,
                    "end_to_end_correct": end_to_end,
                    "exact_realization": exact,
                    "value_missing": (
                        record.expected_action == RESOLVE_ACTION
                        and decision.selected_value != record.expected_value
                    ),
                    "wrong_alternative": (
                        record.expected_action == RESOLVE_ACTION
                        and decision.selected_value == record.alternative_value
                    ),
                    "guarded": decision.guarded,
                }
            )
    return rows, summarize_expanded_predictions(rows)


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size = _load_model(args.checkpoint, device)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "phase31_regression": {"splits": {}},
        "phase32": {"splits": {}},
    }
    all_rows = []
    datasets = (
        ("phase31_regression", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    )
    for dataset_name, data_dir in datasets:
        for split in EXPANDED_SPLITS:
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            rows, metrics = evaluate_records(
                model,
                tokenizer,
                block_size,
                records,
                device,
                args.batch_size,
            )
            all_rows.extend(rows)
            summary[dataset_name]["splits"][split] = metrics
            print(
                f"{dataset_name}/{split}: "
                f"resolve={metrics['end_to_end_resolve_accuracy']:.3f}, "
                f"pairs={metrics['counterfactual_pair_resolve_accuracy']:.3f}, "
                f"clarify={metrics['clarify_accuracy']:.3f}, "
                f"generate={metrics['generate_accuracy']:.3f}, "
                f"missing={metrics['value_missing_rate']:.3f}, "
                f"wrong={metrics['wrong_alternative_rate']:.3f}"
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
