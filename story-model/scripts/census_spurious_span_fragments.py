"""Non-cherry-picked census of every spurious predicted span in the panel.

Extends Phase 36e: instead of a handful of hand-picked example rows, this
walks every row in the crossed-identity panel (14 identities x 2 splits x
200 rows) and records, for every spurious predicted span, its byte length
and which structured prompt section it falls in. The point is to test —
not assume — whether spurious predictions cluster in specific sections or
at specific lengths, since a three-row trace is not evidence on its own.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    encode_proposal_records,
    proposal_batch,
)
from story_model.runtime import resolve_device
from story_model.span_failure_analysis import span_length_bucket, spurious_predicted_spans

try:
    from scripts.audit_crossed_boundary_identity import (
        TARGET_SPLITS,
        _read_json,
        _sha256,
        build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.audit_frozen_span_failure import _decode_spans
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from audit_frozen_span_failure import _decode_spans  # type: ignore


SECTION_MARKERS = (
    "<|character|>",
    "<|relationship|>",
    "<|scene|>",
    "<|world_facts|>",
    "<|memories|>",
    "<|conversation|>",
)


def prompt_section_at(prompt_bytes: bytes, byte_offset: int) -> str:
    """Byte-exact lookup of which structured section a byte offset falls in."""
    best_position = -1
    best_marker = "preamble"
    for marker in SECTION_MARKERS:
        marker_bytes = marker.encode("utf-8")
        position = prompt_bytes.rfind(marker_bytes, 0, byte_offset)
        if position > best_position:
            best_position = position
            best_marker = marker
    return best_marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase31-data-dir", type=Path,
                        default=Path("data/character/typed_span_resolver"))
    parser.add_argument("--phase35-data-dir", type=Path,
                        default=Path("data/character/boundary_counterbalance"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-examples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size, checkpoint = _load_model(args.checkpoint, device)

    manifest = _read_json(args.phase35_data_dir / "manifest.json")
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

    length_histogram = Counter()
    section_histogram = Counter()
    text_histogram = Counter()
    section_by_length = {}
    total_spurious = 0
    total_rows = 0
    examples = []

    for split in TARGET_SPLITS:
        for token_id in sorted(panel[split]):
            records = panel[split][token_id]["records"]
            encoded = encode_proposal_records(records, tokenizer, block_size)
            for start in range(0, len(encoded), args.batch_size):
                indices = tuple(range(start, min(start + args.batch_size, len(encoded))))
                batch = proposal_batch(
                    encoded, indices, tokenizer, model.source_width, device
                )
                with torch.no_grad():
                    output = model(batch[0], batch[1], batch[2])
                logits = output.tag_logits.detach().cpu()
                for offset, index in enumerate(indices):
                    total_rows += 1
                    record = records[index]
                    example = encoded[index]
                    predicted_spans = _decode_spans(logits[offset], example, tokenizer, record)
                    gold_spans = example.gold_spans
                    spurious = spurious_predicted_spans(predicted_spans, gold_spans)
                    prompt_bytes = record.prompt.encode("utf-8")
                    for span in spurious:
                        total_spurious += 1
                        length = span.byte_end - span.byte_start
                        bucket = span_length_bucket(length)
                        section = prompt_section_at(prompt_bytes, span.byte_start)
                        length_histogram[length] += 1
                        section_histogram[section] += 1
                        text_histogram[span.text] += 1
                        section_by_length.setdefault(bucket, Counter())[section] += 1
                        if len(examples) < args.sample_examples:
                            examples.append(
                                {
                                    "record_id": record.record_id,
                                    "split": split,
                                    "token_id": token_id,
                                    "text": span.text,
                                    "byte_length": length,
                                    "section": section,
                                    "byte_start": span.byte_start,
                                    "byte_end": span.byte_end,
                                }
                            )

    summary = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "total_rows": total_rows,
        "total_spurious_spans": total_spurious,
        "spurious_spans_per_row": total_spurious / total_rows if total_rows else 0.0,
        "byte_length_histogram": dict(sorted(length_histogram.items())),
        "section_histogram": dict(section_histogram),
        "section_histogram_fraction": {
            section: count / total_spurious
            for section, count in section_histogram.items()
        }
        if total_spurious
        else {},
        "section_by_length_bucket": {
            bucket: dict(counter) for bucket, counter in section_by_length.items()
        },
        "most_common_spurious_text": text_histogram.most_common(30),
        "sample_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{args.label}: {total_spurious} spurious spans across {total_rows} rows "
          f"({total_spurious/total_rows:.2f}/row)")
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
