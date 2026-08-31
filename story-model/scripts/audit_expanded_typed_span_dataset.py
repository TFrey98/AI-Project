"""Audit every Phase 31/32 row with the real Phase 31 tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    encode_expanded_record,
    load_expanded_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/typed_span_resolver_pilot/best.pt"),
    )
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
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()
    checkpoint = read_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("extra", {}).get("architecture") != "typed_span_resolver":
        raise ValueError("audit checkpoint must be the Phase 31 resolver")
    tokenizer = tokenizer_from_dict(checkpoint["extra"]["tokenizer"])
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 31 checkpoint does not contain byte-BPE")

    total = 0
    maximum = (0, "")
    for dataset_name, data_dir in (
        ("phase31", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        for split in EXPANDED_SPLITS:
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            split_maximum = (0, "")
            for record in records:
                encoded = encode_expanded_record(
                    record, tokenizer, args.block_size
                )
                lengths = (
                    encoded.sequence_tokens,
                    *encoded.candidate_view_sequence_tokens,
                )
                record_maximum = max(lengths)
                if record_maximum > split_maximum[0]:
                    split_maximum = (record_maximum, record.record_id)
                if record_maximum > maximum[0]:
                    maximum = (record_maximum, record.record_id)
            total += len(records)
            print(
                f"{dataset_name}/{split}: {len(records):,}/{len(records):,} "
                f"encoded, max {split_maximum[0]:,} tokens "
                f"({split_maximum[1]})"
            )
    print(f"tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"block size: {args.block_size}")
    print(f"maximum encoded length: {maximum[0]:,} ({maximum[1]})")
    print(f"expanded resolver tokenizer audit: passed ({total:,}/{total:,})")


if __name__ == "__main__":
    main()
