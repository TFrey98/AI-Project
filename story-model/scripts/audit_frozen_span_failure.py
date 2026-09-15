"""Phase 36e: frozen span-failure audit across Phase 35, 36b, and 36d.

No training, weights, decoder, or threshold changes. This compares three
already-trained checkpoints on the identical Phase 36a crossed-identity
panel and explains *how* complete-span quality differs, rather than just
reporting aggregate precision/recall:

- row-level transitions (still correct / regressed / newly fixed / still
  wrong) relative to the Phase 35 baseline;
- a per-gold-span error taxonomy (missing entirely, misplaced start, early
  ending, fragmentation, late ending), located relative to the row's focus
  token;
- collapsed positive-vs-outside discrimination, independent of B/I identity,
  since the JSD consistency term never directly constrains the O logit;
- breakdowns by identity group, split, and gold-span-length bucket, plus an
  explicit, honestly-caveated swap-exposure proxy at the identity-group
  level (no per-row swap log exists to do better); and
- explicit checkpoint-step reporting, since the audited checkpoints were
  selected at different steps and any comparison must stay qualified by
  that fact rather than treated as step-matched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    begin_tag,
    decode_proposed_spans,
    encode_proposal_records,
    inside_tag,
    proposal_batch,
    proposal_metric_row,
)
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device
from story_model.span_failure_analysis import (
    classify_gold_span_prediction,
    positive_vs_outside_confusion,
    span_length_bucket,
    spurious_predicted_spans,
)

try:
    from scripts.audit_crossed_boundary_identity import (
        TARGET_SPLITS,
        _read_json,
        _sha256,
        build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from scripts.audit_explicit_offset_tag_confusion import _load_model
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore


FROZEN_SPAN_FAILURE_AUDIT_VERSION = 1
FOCUS_WINDOW_TOKENS = 4


def _flat_predicted_tags(tag_logits, example, tokenizer) -> tuple[int, ...]:
    """Argmax per-byte predicted tags, aligned to ``example.prompt_byte_tags``."""
    tags = tag_logits.argmax(dim=-1)
    flat = []
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        for byte_offset in range(width):
            flat.append(int(tags[token_position, byte_offset]))
    if len(flat) != len(example.prompt_byte_tags):
        raise RuntimeError("predicted tags do not align with gold byte tags")
    return tuple(flat)


def _decode_spans(tag_logits, example, tokenizer, record):
    return decode_proposed_spans(record.prompt, example, tag_logits, tokenizer)


def _focus_byte_start(example, tokenizer, focus_token_id: int) -> int | None:
    cursor = 0
    for position in range(example.prompt_token_count):
        token_id = example.input_ids[position]
        width = tokenizer.token_byte_length(token_id)
        if token_id == focus_token_id:
            return cursor
        cursor += width
    return None


def _focus_window_bounds(
    example, tokenizer, focus_byte_start: int
) -> tuple[int, int]:
    """A small byte window around the focus token, for the local O-vs-positive check."""
    positions = []
    cursor = 0
    for position in range(example.prompt_token_count):
        token_id = example.input_ids[position]
        width = tokenizer.token_byte_length(token_id)
        positions.append((cursor, cursor + width))
        cursor += width
    focus_index = next(
        index
        for index, (start, _) in enumerate(positions)
        if start == focus_byte_start
    )
    low = max(0, focus_index - FOCUS_WINDOW_TOKENS)
    high = min(len(positions) - 1, focus_index + FOCUS_WINDOW_TOKENS)
    return positions[low][0], positions[high][1]


@torch.no_grad()
def evaluate_rows(model, tokenizer, block_size, records, focus_token_id, device, batch_size):
    """Per-row predictions, error classifications, and confusion breakdowns."""
    examples = encode_proposal_records(records, tokenizer, block_size)
    rows = {}
    for start in range(0, len(examples), batch_size):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        batch = proposal_batch(examples, indices, tokenizer, model.source_width, device)
        output = model(batch[0], batch[1], batch[2])
        logits = output.tag_logits.detach().cpu()
        for offset, index in enumerate(indices):
            example = examples[index]
            record = records[index]
            predicted_tags = _flat_predicted_tags(logits[offset], example, tokenizer)
            gold_tags = example.prompt_byte_tags
            predicted_spans = _decode_spans(logits[offset], example, tokenizer, record)
            gold_spans = example.gold_spans
            metric_row = proposal_metric_row(record, gold_spans, predicted_spans)
            row_correct = (
                metric_row["exact_span_matches"] == metric_row["gold_span_count"] == metric_row["predicted_span_count"]
            )
            focus_start = _focus_byte_start(example, tokenizer, focus_token_id)
            error_types = []
            for span in gold_spans:
                error_types.append(
                    classify_gold_span_prediction(
                        span.byte_start,
                        span.byte_end,
                        begin_tag(span.value_type),
                        inside_tag(span.value_type),
                        predicted_tags,
                    )
                )
            spurious = spurious_predicted_spans(predicted_spans, gold_spans)
            global_confusion = positive_vs_outside_confusion(gold_tags, predicted_tags)
            local_confusion = None
            if focus_start is not None:
                low, high = _focus_window_bounds(example, tokenizer, focus_start)
                local_confusion = positive_vs_outside_confusion(
                    gold_tags[low:high], predicted_tags[low:high]
                )
            rows[record.record_id] = {
                "row_correct": row_correct,
                "gold_span_count": metric_row["gold_span_count"],
                "predicted_span_count": metric_row["predicted_span_count"],
                "error_types": error_types,
                "spurious_span_count": len(spurious),
                "focus_byte_start": focus_start,
                "gold_span_byte_length": (
                    gold_spans[0].byte_end - gold_spans[0].byte_start
                    if gold_spans
                    else None
                ),
                "gold_span_distance_from_focus_bytes": (
                    gold_spans[0].byte_start - focus_start
                    if gold_spans and focus_start is not None
                    else None
                ),
                "global_positive_vs_outside": global_confusion,
                "local_positive_vs_outside": local_confusion,
            }
    return rows


def _checkpoint_step_info(checkpoint: dict) -> dict:
    extra = checkpoint.get("extra", {})
    return {
        "step": checkpoint.get("step", 0),
        "checkpoint_eligible": extra.get("checkpoint_eligible"),
        "paired_identity_invariance_version": extra.get("paired_identity_invariance_version"),
        "identity_invariance_supervision_mode": extra.get("identity_invariance_supervision_mode"),
        "identity_invariance_position_mode": extra.get("identity_invariance_position_mode"),
    }


def _transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and after_correct:
        return "still_correct"
    if before_correct and not after_correct:
        return "regressed"
    if not before_correct and after_correct:
        return "newly_fixed"
    return "still_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase35-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36b-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36d-checkpoint", type=Path, required=True)
    parser.add_argument("--phase31-data-dir", type=Path,
                        default=Path("data/character/typed_span_resolver"))
    parser.add_argument("--phase35-data-dir", type=Path,
                        default=Path("data/character/boundary_counterbalance"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    paths = {
        "phase35": args.phase35_checkpoint,
        "phase36b": args.phase36b_checkpoint,
        "phase36d": args.phase36d_checkpoint,
    }
    models, tokenizers, checkpoints, block_sizes = {}, {}, {}, {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label], tokenizers[label] = model, tokenizer
        checkpoints[label], block_sizes[label] = checkpoint, block_size
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 35/36b/36d block sizes differ")
    tokenizer, block_size = tokenizers["phase35"], block_sizes["phase35"]
    tokenizer_hashes = {
        canonical_json_sha256(t.to_dict()) for t in tokenizers.values()
    }
    if len(tokenizer_hashes) != 1:
        raise ValueError("Phase 35/36b/36d tokenizers differ")

    steps = {label: _checkpoint_step_info(checkpoints[label]) for label in paths}
    steps_matched = len({info["step"] for info in steps.values()}) == 1
    if not steps_matched:
        print(
            "WARNING: checkpoints were selected at different steps "
            + ", ".join(f"{label}={info['step']}" for label, info in steps.items())
            + " -- any comparison below must stay qualified by this, not "
            "treated as a step-matched comparison.",
            flush=True,
        )

    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    identities = canonical_focus_identities(manifest, tokenizer)
    expected_identities = {
        group: [{"token_id": token_id, "text": text} for token_id, text in values]
        for group, values in identities.items()
    }
    identity_group = {
        token_id: group
        for group, values in identities.items()
        for token_id, _ in values
    }

    source_records = {}
    for split in TARGET_SPLITS:
        path = args.phase31_data_dir / f"{split}.jsonl"
        records = load_expanded_records(path)
        expected = manifest["source_phase31_files"][split]
        if expected["sha256"] != _sha256(path) or expected["rows"] != len(records):
            raise ValueError(f"Phase 31 {split} data changed")
        source_records[split] = records
    panel, validation = build_crossed_identity_panel(
        source_records, identities, tokenizer, block_size
    )

    # rows[label][split][token_id] = {record_id: row_analysis}
    rows = {label: {} for label in paths}
    for label, model in models.items():
        for split in TARGET_SPLITS:
            rows[label][split] = {}
            for token_id in sorted(panel[split]):
                records = panel[split][token_id]["records"]
                rows[label][split][token_id] = evaluate_rows(
                    model, tokenizer, block_size, records, token_id, device,
                    args.batch_size,
                )
        print(f"{label}: evaluated all identities/splits", flush=True)

    transitions = {
        "phase36b_vs_phase35": Counter(),
        "phase36d_vs_phase35": Counter(),
    }
    error_type_counts = defaultdict(lambda: defaultdict(Counter))
    span_length_error_counts = defaultdict(lambda: defaultdict(Counter))
    group_confusion = defaultdict(lambda: defaultdict(list))
    spurious_totals = defaultdict(lambda: defaultdict(int))

    for split in TARGET_SPLITS:
        for token_id in sorted(panel[split]):
            group = identity_group[token_id]
            baseline_rows = rows["phase35"][split][token_id]
            for candidate_label in ("phase36b", "phase36d"):
                candidate_rows = rows[candidate_label][split][token_id]
                key = f"{candidate_label}_vs_phase35"
                for record_id, before in baseline_rows.items():
                    after = candidate_rows[record_id]
                    transitions[key][
                        _transition(before["row_correct"], after["row_correct"])
                    ] += 1
                    for error_type in after["error_types"]:
                        error_type_counts[candidate_label][split][error_type] += 1
                        if after["gold_span_byte_length"] is not None:
                            bucket = span_length_bucket(after["gold_span_byte_length"])
                            span_length_error_counts[candidate_label][bucket][
                                error_type
                            ] += 1
                    spurious_totals[candidate_label][split] += after[
                        "spurious_span_count"
                    ]
                    if after["local_positive_vs_outside"] is not None:
                        group_confusion[candidate_label][group].append(
                            after["local_positive_vs_outside"]
                        )

    def _mean_confusion(entries: list) -> dict:
        if not entries:
            return {}
        keys = (
            "gold_outside_as_positive_rate",
            "gold_positive_as_outside_rate",
        )
        return {
            key: sum(entry[key] for entry in entries) / len(entries)
            for key in keys
        }

    summary = {
        "frozen_span_failure_audit_version": FROZEN_SPAN_FAILURE_AUDIT_VERSION,
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint_changes": "none",
        "checkpoints": steps,
        "checkpoint_steps_matched": steps_matched,
        "focus_identities": expected_identities,
        "panel_validation": validation,
        "scaffold_note": (
            "every row in this panel uses the same fixed route-form variant "
            "(CROSSED_ROUTE_VARIANT=0); scaffold is not a variable in this "
            "data and is not reported as a breakdown axis"
        ),
        "swap_exposure_note": (
            "no per-row swap log was recorded during training; the closest "
            "available proxy is identity-group membership (trained rows were "
            "the Phase 35 counterbalance focus; held-out/legacy rows were "
            "never counterbalance-trained), not literal per-row swap "
            "incidence during Phase 36b/36d training"
        ),
        "row_transitions_vs_phase35": {
            key: dict(counter) for key, counter in transitions.items()
        },
        "error_type_counts_by_split": {
            label: {split: dict(counter) for split, counter in splits.items()}
            for label, splits in error_type_counts.items()
        },
        "error_type_counts_by_span_length_bucket": {
            label: {bucket: dict(counter) for bucket, counter in buckets.items()}
            for label, buckets in span_length_error_counts.items()
        },
        "spurious_span_totals_by_split": {
            label: dict(splits) for label, splits in spurious_totals.items()
        },
        "local_positive_vs_outside_confusion_by_identity_group": {
            label: {
                group: _mean_confusion(entries)
                for group, entries in groups.items()
            }
            for label, groups in group_confusion.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
