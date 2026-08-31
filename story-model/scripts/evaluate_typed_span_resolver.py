"""Evaluate Phase 31 actions, candidate selection, and exact realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.models import build_model
from story_model.runtime import resolve_device
from story_model.typed_span_resolver import (
    RESOLVER_ACTIONS,
    RESOLVE_ACTION,
    TYPED_SPAN_SPLITS,
    TypedSpanResolver,
    encode_typed_span_records,
    load_typed_span_records,
    realize_resolver_decision,
    summarize_resolver_predictions,
    typed_span_batch,
)


def _load_model(path: Path, device: torch.device):
    checkpoint = read_checkpoint(path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "typed_span_resolver":
        raise ValueError("checkpoint is not a typed-span resolver")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    tokenizer = tokenizer_from_dict(extra["tokenizer"])
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("typed-span checkpoint requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(
        config["model"], tokenizer.vocab_size, block_size
    )
    model = TypedSpanResolver(backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size


@torch.no_grad()
def evaluate_split(model, tokenizer, block_size, records, device, batch_size):
    examples = encode_typed_span_records(records, tokenizer, block_size)
    rows = []
    for start in range(0, len(records), batch_size):
        indices = tuple(range(start, min(start + batch_size, len(records))))
        batch = typed_span_batch(examples, indices, device)
        output = model(*batch[:5])
        action_predictions = output.action_logits.argmax(dim=-1).tolist()
        candidate_predictions = output.candidate_logits.argmax(dim=-1).tolist()
        for offset, index in enumerate(indices):
            record = records[index]
            action = RESOLVER_ACTIONS[action_predictions[offset]]
            candidate_index = candidate_predictions[offset]
            decision = realize_resolver_decision(
                record, action, candidate_index
            )
            action_correct = action == record.expected_action
            candidate_correct = (
                record.selected_candidate_index is not None
                and candidate_index == record.selected_candidate_index
            )
            end_to_end_correct = (
                action_correct
                and (
                    record.expected_action != RESOLVE_ACTION
                    or candidate_correct
                )
            )
            exact_realization = (
                record.expected_action == RESOLVE_ACTION
                and decision.action == RESOLVE_ACTION
                and decision.selected_value == record.expected_value
                and decision.text
                == record.response_template.replace(
                    "<|resolved_value|>", record.expected_value
                )
            )
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_context_id": record.source_context_id,
                    "conversation_id": record.conversation_id,
                    "split": record.split,
                    "skill": record.skill,
                    "case": record.case,
                    "expected_action": record.expected_action,
                    "predicted_action": action,
                    "expected_candidate_index": record.selected_candidate_index,
                    "predicted_candidate_index": candidate_index,
                    "expected_value": record.expected_value,
                    "alternative_value": record.alternative_value,
                    "realized_text": decision.text,
                    "selected_value": decision.selected_value,
                    "action_correct": action_correct,
                    "candidate_correct": candidate_correct,
                    "end_to_end_correct": end_to_end_correct,
                    "exact_realization": exact_realization,
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
    return rows, summarize_resolver_predictions(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size = _load_model(args.checkpoint, device)
    all_rows = []
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "splits": {},
    }
    for split in TYPED_SPAN_SPLITS:
        records = load_typed_span_records(args.data_dir / f"{split}.jsonl")
        rows, metrics = evaluate_split(
            model,
            tokenizer,
            block_size,
            records,
            device,
            args.batch_size,
        )
        all_rows.extend(rows)
        summary["splits"][split] = metrics
        print(
            f"{split}: resolve={metrics['end_to_end_resolve_accuracy']:.3f}, "
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
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"predictions: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
