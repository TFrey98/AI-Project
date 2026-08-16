"""Inspect control-token compression and checkpoint warm starting."""

from __future__ import annotations

import argparse

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    load_character_context,
    serialize_character_prompt,
)
from story_model.checkpoint import (
    load_model_warm_start,
    read_checkpoint,
)
from story_model.data import (
    ByteBPETokenizer,
    tokenizer_from_dict,
)
from story_model.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Phase 13 byte-BPE checkpoint.",
    )
    parser.add_argument(
        "--context",
        default="examples/character_context.json",
        help="Character context JSON to inspect.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Proposed character-model context length.",
    )
    args = parser.parse_args()

    if args.block_size < 1:
        raise ValueError("block-size must be positive")

    checkpoint = read_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    extra = checkpoint.get("extra", {})
    tokenizer_data = extra.get("tokenizer")
    config = extra.get("config")

    if tokenizer_data is None or config is None:
        raise ValueError(
            "checkpoint must contain tokenizer and config metadata"
        )

    source_tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(source_tokenizer, ByteBPETokenizer):
        raise ValueError(
            "character token inspection requires byte-BPE"
        )

    tokenizer = source_tokenizer.with_special_tokens(
        CHARACTER_CONTROL_TOKENS
    )

    for token in CHARACTER_CONTROL_TOKENS:
        token_id = tokenizer.special_token_ids[token]

        if tokenizer.encode(token) != [token_id]:
            raise RuntimeError(
                f"control token is not atomic: {token}"
            )

    context = load_character_context(args.context)
    prompt = serialize_character_prompt(context)
    source_prompt_tokens = source_tokenizer.encode(prompt)
    prompt_tokens = tokenizer.encode(prompt)

    if tokenizer.decode(prompt_tokens) != prompt:
        raise RuntimeError(
            "expanded tokenizer failed prompt roundtrip"
        )

    source_block_size = int(config["data"]["block_size"])
    position_encoding = config["model"].get(
        "position_encoding",
        "learned",
    )

    if (
        args.block_size != source_block_size
        and position_encoding != "rope"
    ):
        raise ValueError(
            "context expansion requires a RoPE checkpoint"
        )

    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=args.block_size,
    )
    expanded_parameters = load_model_warm_start(
        model=model,
        checkpoint=checkpoint,
        source_vocabulary_size=source_tokenizer.vocab_size,
        destination_vocabulary_size=tokenizer.vocab_size,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"source vocabulary: {source_tokenizer.vocab_size}")
    print(f"control tokens: {len(CHARACTER_CONTROL_TOKENS)}")
    print(f"expanded vocabulary: {tokenizer.vocab_size}")
    print(f"source prompt tokens: {len(source_prompt_tokens):,}")
    print(f"expanded prompt tokens: {len(prompt_tokens):,}")
    print(f"proposed context: {args.block_size:,}")
    print(
        "tokens remaining after prompt: "
        f"{args.block_size - len(prompt_tokens):,}"
    )
    print(f"parameters: {parameter_count:,}")
    print(
        "expanded parameters: "
        + ", ".join(expanded_parameters)
    )
    print("control-token roundtrip: passed")
    print("warm-start weights: passed")


if __name__ == "__main__":
    main()
