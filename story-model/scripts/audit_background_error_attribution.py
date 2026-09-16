"""Phase 36f: frozen background-error attribution audit.

No training, weights, decoder, or threshold changes. Extends Phase 36e's
spurious-span finding with three specific outputs:

1. The same spurious-span census run against Phase 35, to establish how much
   of the background false-positive pattern predates any identity-invariance
   training at all (Phase 35 never saw the swap mechanism).
2. Per-spurious-span attribution to the actual emitting input token: its
   token id, byte width, the byte offset within that token (a 1-byte
   predicted span need not come from a 1-byte input token -- it can be one
   byte carved out of a longer token), whether that token id is a member of
   the training swap pool (Phase 36b/36d only; not applicable to Phase 35,
   which never trained with a swap pool), the predicted type, and whether
   the span opens with a grammatically clean B tag or an orphan I (an inside
   tag with no preceding begin). Phase 36e's `census_spurious_span_fragments`
   recorded the row's crossed-panel focus identity in `token_id`, which is
   NOT necessarily the token that produced a given spurious span -- this
   audit records the actual emitting token separately and explicitly.
3. Positive-vs-outside confusion across the whole prompt, broken down by
   section and split, normalized by gold-O byte count in that section (not
   total section length). Phase 36e computed this per row but never
   exported it; this is that data, aggregated and exported.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    encode_proposal_records,
    proposal_batch,
    tag_is_begin,
)
from story_model.paired_identity_invariance import validate_identity_swap_pool
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device
from story_model.span_failure_analysis import spurious_predicted_spans

try:
    from scripts.audit_crossed_boundary_identity import (
        TARGET_SPLITS,
        _read_json,
        _sha256,
        build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.audit_frozen_span_failure import _decode_spans, _flat_predicted_tags
    from scripts.census_spurious_span_fragments import (
        SECTION_MARKERS,
        prompt_section_at,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from audit_frozen_span_failure import _decode_spans, _flat_predicted_tags  # type: ignore
    from census_spurious_span_fragments import (  # type: ignore
        SECTION_MARKERS, prompt_section_at,
    )


BACKGROUND_ERROR_ATTRIBUTION_AUDIT_VERSION = 1
ALL_SECTIONS = ("preamble",) + SECTION_MARKERS


def _token_byte_ranges(example, tokenizer) -> tuple[tuple[int, int, int], ...]:
    """(token_position, byte_start, byte_end) for every prompt token."""
    ranges = []
    cursor = 0
    for position in range(example.prompt_token_count):
        token_id = example.input_ids[position]
        width = tokenizer.token_byte_length(token_id)
        ranges.append((position, cursor, cursor + width))
        cursor += width
    return tuple(ranges)


def _token_owning_byte(token_ranges, byte_offset: int) -> tuple[int, int, int]:
    for position, start, end in token_ranges:
        if start <= byte_offset < end:
            return position, start, end
    raise ValueError(f"byte offset {byte_offset} is not covered by any token")


def _load_swap_pool(checkpoint, tokenizer):
    extra = checkpoint.get("extra", {})
    pool = extra.get("identity_invariance_swap_pool")
    if not isinstance(pool, dict):
        return None
    excluded = pool.get("excluded_registered_token_ids", ())
    return validate_identity_swap_pool(pool, tokenizer, excluded)


def _pool_membership(token_id: int, pool_by_width) -> str:
    if pool_by_width is None:
        return "not_applicable"
    for token_ids in pool_by_width.values():
        if token_id in token_ids:
            return "pool_member"
    return "not_pool_member"


def run_checkpoint(
    label, model, tokenizer, block_size, panel, device, batch_size, pool_by_width,
    max_examples=25,
):
    """Attribution stats and positive-vs-O confusion, one checkpoint."""
    attribution_counter = Counter()
    width_within_token_histogram = Counter()
    opening_kind_histogram = Counter()
    pool_membership_histogram = Counter()
    section_confusion = defaultdict(lambda: defaultdict(Counter))
    total_spurious = 0
    total_rows = 0
    samples = []

    for split in TARGET_SPLITS:
        for token_id in sorted(panel[split]):
            records = panel[split][token_id]["records"]
            examples = encode_proposal_records(records, tokenizer, block_size)
            for start in range(0, len(examples), batch_size):
                indices = tuple(range(start, min(start + batch_size, len(examples))))
                batch = proposal_batch(
                    examples, indices, tokenizer, model.source_width, device
                )
                with torch.no_grad():
                    output = model(batch[0], batch[1], batch[2])
                logits = output.tag_logits.detach().cpu()
                for offset, index in enumerate(indices):
                    total_rows += 1
                    record = records[index]
                    example = examples[index]
                    predicted_tags = _flat_predicted_tags(logits[offset], example, tokenizer)
                    gold_tags = example.prompt_byte_tags
                    predicted_spans = _decode_spans(logits[offset], example, tokenizer, record)
                    gold_spans = example.gold_spans
                    spurious = spurious_predicted_spans(predicted_spans, gold_spans)
                    prompt_bytes = record.prompt.encode("utf-8")
                    token_ranges = _token_byte_ranges(example, tokenizer)

                    # Positive-vs-O confusion, every byte, bucketed by section.
                    cursor_section = None
                    for byte_position, (gold, predicted) in enumerate(
                        zip(gold_tags, predicted_tags)
                    ):
                        section = prompt_section_at(prompt_bytes, byte_position)
                        counter = section_confusion[split][section]
                        if gold == 0:
                            counter["gold_outside_total"] += 1
                            if predicted != 0:
                                counter["gold_outside_as_positive"] += 1
                        else:
                            counter["gold_positive_total"] += 1
                            if predicted == 0:
                                counter["gold_positive_as_outside"] += 1

                    for span in spurious:
                        total_spurious += 1
                        emit_position, token_start, token_end = _token_owning_byte(
                            token_ranges, span.byte_start
                        )
                        emitting_token_id = example.input_ids[emit_position]
                        token_width = token_end - token_start
                        offset_within_token = span.byte_start - token_start
                        membership = _pool_membership(emitting_token_id, pool_by_width)
                        opening_tag = predicted_tags[span.byte_start]
                        opening_kind = "clean_begin" if tag_is_begin(opening_tag) else "orphan_inside"
                        section = prompt_section_at(prompt_bytes, span.byte_start)

                        attribution_counter[(emitting_token_id, token_width, membership)] += 1
                        width_within_token_histogram[(token_width, offset_within_token)] += 1
                        opening_kind_histogram[opening_kind] += 1
                        pool_membership_histogram[membership] += 1

                        if len(samples) < max_examples:
                            samples.append(
                                {
                                    "record_id": record.record_id,
                                    "split": split,
                                    "crossed_panel_identity_token_id": token_id,
                                    "emitting_token_id": emitting_token_id,
                                    "token_width": token_width,
                                    "byte_offset_within_token": offset_within_token,
                                    "pool_membership": membership,
                                    "predicted_type": span.value_type,
                                    "opening_kind": opening_kind,
                                    "section": section,
                                    "span_text": span.text,
                                    "span_byte_length": span.byte_end - span.byte_start,
                                }
                            )

    top_emitting_tokens = [
        {
            "token_id": token_id,
            "token_width": width,
            "pool_membership": membership,
            "spurious_span_count": count,
        }
        for (token_id, width, membership), count in attribution_counter.most_common(40)
    ]

    positive_vs_outside = {}
    for split, sections in section_confusion.items():
        positive_vs_outside[split] = {}
        for section, counter in sections.items():
            outside_total = counter["gold_outside_total"]
            positive_total = counter["gold_positive_total"]
            positive_vs_outside[split][section] = {
                "gold_outside_total": outside_total,
                "gold_outside_as_positive_count": counter["gold_outside_as_positive"],
                "gold_outside_as_positive_rate": (
                    counter["gold_outside_as_positive"] / outside_total
                    if outside_total
                    else 0.0
                ),
                "gold_positive_total": positive_total,
                "gold_positive_as_outside_count": counter["gold_positive_as_outside"],
                "gold_positive_as_outside_rate": (
                    counter["gold_positive_as_outside"] / positive_total
                    if positive_total
                    else 0.0
                ),
            }

    return {
        "label": label,
        "total_rows": total_rows,
        "total_spurious_spans": total_spurious,
        "distinct_emitting_tokens": len(attribution_counter),
        "top_emitting_tokens": top_emitting_tokens,
        "pool_membership_histogram": dict(pool_membership_histogram),
        "opening_kind_histogram": dict(opening_kind_histogram),
        "byte_offset_within_token_histogram": {
            f"width={width}_offset={offset}": count
            for (width, offset), count in sorted(width_within_token_histogram.items())
        },
        "positive_vs_outside_confusion_by_section": positive_vs_outside,
        "sample_examples": samples,
    }


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
    tokenizer_hashes = {canonical_json_sha256(t.to_dict()) for t in tokenizers.values()}
    if len(tokenizer_hashes) != 1:
        raise ValueError("Phase 35/36b/36d tokenizers differ")

    steps = {label: checkpoints[label].get("step", 0) for label in paths}
    steps_matched = len(set(steps.values())) == 1
    if not steps_matched:
        print(
            "WARNING: checkpoints were selected at different steps "
            + ", ".join(f"{label}={step}" for label, step in steps.items())
            + " -- retain this qualification, do not treat as step-matched.",
            flush=True,
        )

    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    identities = canonical_focus_identities(manifest, tokenizer)

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

    results = {}
    for label in paths:
        pool_by_width = _load_swap_pool(checkpoints[label], tokenizer)
        results[label] = run_checkpoint(
            label, models[label], tokenizer, block_size, panel, device,
            args.batch_size, pool_by_width,
        )
        print(
            f"{label}: {results[label]['total_spurious_spans']} spurious spans, "
            f"{results[label]['distinct_emitting_tokens']} distinct emitting tokens",
            flush=True,
        )

    summary = {
        "background_error_attribution_audit_version": (
            BACKGROUND_ERROR_ATTRIBUTION_AUDIT_VERSION
        ),
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint_changes": "none",
        "checkpoint_steps": steps,
        "checkpoint_steps_matched": steps_matched,
        "panel_validation": validation,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
