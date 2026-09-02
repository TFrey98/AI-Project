"""Evaluate Phase 33c and re-run the complete Phase 31/32 regression gate."""

from __future__ import annotations

import argparse
import json
import time
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
    RESOLVER_ACTIONS,
    RESOLVE_ACTION,
    load_expanded_records,
)
from story_model.models import build_model
from story_model.runtime import resolve_device
from story_model.unified_typed_span_resolver import (
    GENERATE_MODE,
    MODE_TO_INDEX,
    NO_SUPPORT_OPTION_INDEX,
    SUPPORT_MASK_VERSION,
    STRUCTURED_MODE,
    UnifiedTypedSpanResolver,
    encode_unified_records,
    realize_unified_decision,
    summarize_unified_predictions,
    unified_batch,
)


def _load_model(path: Path, device: torch.device):
    checkpoint = read_checkpoint(path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "unified_typed_span_resolver":
        raise ValueError("checkpoint is not a Phase 33 unified resolver")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("checkpoint predates the Phase 33c count router")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    tokenizer = tokenizer_from_dict(extra["tokenizer"])
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("unified resolver checkpoint requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    model = UnifiedTypedSpanResolver(backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size


@torch.no_grad()
def evaluate_records(
    model,
    tokenizer,
    block_size,
    records,
    device,
    batch_size,
    label="dataset",
    progress_every=50,
):
    examples = encode_unified_records(records, tokenizer, block_size)
    rows = []
    started = time.monotonic()
    batches = (len(records) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(records), batch_size), start=1):
        indices = tuple(range(start, min(start + batch_size, len(records))))
        batch = unified_batch(examples, indices, device)
        output = model(*batch[:5])
        mode_predictions = output.mode_logits.argmax(dim=-1).tolist()
        option_predictions = output.option_logits.argmax(dim=-1).tolist()
        candidate_positions = torch.arange(
            MAX_CANDIDATES, device=device
        ).unsqueeze(0)
        inventory_valid = candidate_positions < batch[7].unsqueeze(1)
        real_candidate_predictions = (
            output.raw_option_logits[:, :MAX_CANDIDATES]
            .masked_fill(
                ~inventory_valid,
                torch.finfo(output.raw_option_logits.dtype).min,
            )
            .argmax(dim=-1)
            .tolist()
        )
        legacy_predictions = output.legacy_action_logits.argmax(dim=-1).tolist()
        for offset, index in enumerate(indices):
            record = records[index]
            mode_index = mode_predictions[offset]
            option_index = option_predictions[offset]
            real_candidate_index = real_candidate_predictions[offset]
            eligible_candidate_indices = [
                position
                for position, valid in enumerate(
                    examples[index].option_valid_mask[:MAX_CANDIDATES]
                )
                if valid
            ]
            legacy_multi_action_index = int(
                output.legacy_action_logits[offset, :2].argmax().item()
            )
            multi_candidate_action_correct = None
            if (
                len(eligible_candidate_indices) >= 2
                and record.expected_action in (RESOLVE_ACTION, CLARIFY_ACTION)
            ):
                expected_multi_action_index = RESOLVER_ACTIONS.index(
                    record.expected_action
                )
                multi_candidate_action_correct = (
                    legacy_multi_action_index == expected_multi_action_index
                )
            decision = realize_unified_decision(record, mode_index, option_index)
            # Report the guarded runtime action.  A malformed structured choice
            # must fail closed to clarification rather than appear as resolve.
            action = decision.action
            expected_mode = (
                MODE_TO_INDEX[GENERATE_MODE]
                if record.expected_action == GENERATE_ACTION
                else MODE_TO_INDEX[STRUCTURED_MODE]
            )
            mode_correct = mode_index == expected_mode
            if record.expected_action == RESOLVE_ACTION:
                option_correct = option_index == record.selected_candidate_index
            elif record.expected_action == CLARIFY_ACTION:
                option_correct = option_index == NO_SUPPORT_OPTION_INDEX
            else:
                option_correct = True
            action_correct = action == record.expected_action
            candidate_correct = (
                record.selected_candidate_index is not None
                and option_index == record.selected_candidate_index
            )
            real_candidate_correct = (
                record.selected_candidate_index is not None
                and real_candidate_index == record.selected_candidate_index
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
                    "predicted_mode": (
                        GENERATE_MODE
                        if mode_index == MODE_TO_INDEX[GENERATE_MODE]
                        else STRUCTURED_MODE
                    ),
                    "expected_candidate_index": record.selected_candidate_index,
                    "predicted_option_index": option_index,
                    "predicted_real_candidate_index": real_candidate_index,
                    "eligible_candidate_indices": eligible_candidate_indices,
                    "eligible_candidate_count": len(eligible_candidate_indices),
                    "legacy_predicted_action": RESOLVER_ACTIONS[
                        legacy_predictions[offset]
                    ],
                    "multi_candidate_predicted_action": RESOLVER_ACTIONS[
                        legacy_multi_action_index
                    ],
                    "multi_candidate_action_correct": (
                        multi_candidate_action_correct
                    ),
                    "expected_value": record.expected_value,
                    "alternative_value": record.alternative_value,
                    "selected_value": decision.selected_value,
                    "realized_text": decision.text,
                    "mode_correct": mode_correct,
                    "option_correct": option_correct,
                    "action_correct": action_correct,
                    "candidate_correct": candidate_correct,
                    "real_candidate_correct": real_candidate_correct,
                    "end_to_end_correct": end_to_end,
                    "exact_realization": exact,
                    "no_support_selected": (
                        option_index == NO_SUPPORT_OPTION_INDEX
                    ),
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
        if progress_every and (
            batch_number % progress_every == 0 or batch_number == batches
        ):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            completed = min(start + batch_size, len(records))
            print(
                f"{label}: {completed:,}/{len(records):,} rows "
                f"({completed / elapsed:.1f} rows/s)",
                flush=True,
            )
    return rows, summarize_unified_predictions(rows)


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
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
            label = f"{dataset_name}/{split}"
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            rows, metrics = evaluate_records(
                model,
                tokenizer,
                block_size,
                records,
                device,
                args.batch_size,
                label=label,
                progress_every=args.progress_every,
            )
            all_rows.extend(rows)
            summary[dataset_name]["splits"][split] = metrics
            print(
                f"{label}: resolve={metrics['end_to_end_resolve_accuracy']:.3f}, "
                f"pairs={metrics['counterfactual_pair_resolve_accuracy']:.3f}, "
                f"candidate={metrics['real_candidate_top1_accuracy']:.3f}, "
                f"multi_action={metrics['multi_candidate_action_accuracy']:.3f}, "
                f"sentinel={metrics['clarify_sentinel_accuracy']:.3f}, "
                f"generate={metrics['generate_accuracy']:.3f}, "
                f"missing={metrics['value_missing_rate']:.3f}, "
                f"wrong={metrics['wrong_alternative_rate']:.3f}",
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
