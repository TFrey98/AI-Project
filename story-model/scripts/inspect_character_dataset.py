"""Inspect token budgets and response masks before character training."""

from __future__ import annotations

import argparse

from story_model.character_data import CHARACTER_CONTROL_TOKENS
from story_model.character_training import (
    RESPONSE_IGNORE_INDEX,
    encode_character_training_records,
    load_character_training_records,
    summarize_character_examples,
)
from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--block-size", type=int, default=2048)
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
        raise ValueError("character dataset requires byte-BPE")

    tokenizer = source_tokenizer.with_special_tokens(
        CHARACTER_CONTROL_TOKENS
    )
    records = load_character_training_records(args.data)
    examples = encode_character_training_records(
        records,
        tokenizer,
        args.block_size,
    )
    summary = summarize_character_examples(records, examples)

    if any(
        target != RESPONSE_IGNORE_INDEX
        for example in examples
        for target in example.target_ids[
            example.sequence_tokens :
        ]
    ):
        raise RuntimeError("right-padding target mask failed")

    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data}")
    print(f"vocabulary: {tokenizer.vocab_size:,}")
    print(f"block size: {args.block_size:,}")
    print(f"examples: {summary['examples']:,}")
    print(f"conversations: {summary['conversations']:,}")
    print(f"characters: {summary['characters']:,}")
    print(
        "sequence tokens: "
        f"min {summary['sequence_tokens_min']:,}, "
        f"mean {summary['sequence_tokens_mean']:,.1f}, "
        f"max {summary['sequence_tokens_max']:,}"
    )
    print(
        f"maximum prompt tokens: {summary['prompt_tokens_max']:,}"
    )
    print(
        f"supervised response tokens: {summary['supervised_tokens']:,}"
    )
    print(f"complete turns dropped: {summary['dropped_turns']:,}")
    print("response-only masking: passed")


if __name__ == "__main__":
    main()
