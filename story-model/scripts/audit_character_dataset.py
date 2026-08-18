"""Gate production character data on coverage and encoding quality."""

from __future__ import annotations

import argparse

from story_model.character_data import CHARACTER_CONTROL_TOKENS
from story_model.character_training import (
    CHARACTER_BEHAVIOR_TAGS,
    audit_character_training_records,
    load_character_training_records,
)
from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--min-conversations", type=int, default=20)
    parser.add_argument("--min-tag-examples", type=int, default=5)
    parser.add_argument(
        "--required-tag",
        action="append",
        choices=sorted(CHARACTER_BEHAVIOR_TAGS),
        dest="required_tags",
        help=(
            "Required behavior category. Repeat to select a subset; "
            "the default requires every category."
        ),
    )
    args = parser.parse_args()

    checkpoint = read_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    tokenizer_data = (
        checkpoint.get("extra", {}).get("tokenizer")
    )

    if tokenizer_data is None:
        raise ValueError("checkpoint has no tokenizer metadata")

    source_tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(source_tokenizer, ByteBPETokenizer):
        raise ValueError("character audit requires byte-BPE")

    tokenizer = source_tokenizer.with_special_tokens(
        CHARACTER_CONTROL_TOKENS
    )
    records = load_character_training_records(args.data)
    required_tags = (
        args.required_tags
        if args.required_tags is not None
        else CHARACTER_BEHAVIOR_TAGS
    )
    report = audit_character_training_records(
        records,
        tokenizer,
        block_size=args.block_size,
        min_examples=args.min_examples,
        min_conversations=args.min_conversations,
        min_examples_per_tag=args.min_tag_examples,
        required_tags=required_tags,
    )

    print(f"data: {args.data}")
    print(f"examples: {report['examples']:,}")
    print(f"conversations: {report['conversations']:,}")
    print(f"characters: {report['characters']:,}")
    print(
        "sequence tokens: "
        f"min {report['sequence_tokens_min']:,}, "
        f"mean {report['sequence_tokens_mean']:,.1f}, "
        f"max {report['sequence_tokens_max']:,}"
    )
    print(f"complete turns dropped: {report['dropped_turns']:,}")
    print("behavior coverage:")

    for tag, count in report["tag_counts"].items():
        print(f"- {tag}: {count:,}")

    for warning in report["warnings"]:
        print(f"warning: {warning}")

    if report["errors"]:
        for error in report["errors"]:
            print(f"error: {error}")

        raise SystemExit("character dataset audit: failed")

    print("character dataset audit: passed")


if __name__ == "__main__":
    main()
