"""Evaluate a checkpoint on held-out, behavior-tagged responses."""

from __future__ import annotations

import argparse

from story_model.character_evaluate import evaluate_character_records
from story_model.character_training import load_character_training_records
from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.models import build_model
from story_model.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = read_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    extra = checkpoint.get("extra", {})
    config = extra.get("config")
    tokenizer_data = extra.get("tokenizer")

    if config is None or tokenizer_data is None:
        raise ValueError(
            "checkpoint must contain config and tokenizer metadata"
        )

    tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("character evaluation requires byte-BPE")

    block_size = (
        args.block_size
        if args.block_size is not None
        else int(config["data"]["block_size"])
    )
    configured_block_size = int(config["data"]["block_size"])

    if (
        block_size != configured_block_size
        and config["model"].get("position_encoding") != "rope"
    ):
        raise ValueError(
            "context-length override requires a RoPE checkpoint"
        )

    device = resolve_device(args.device)
    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=block_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    records = load_character_training_records(args.data)
    report = evaluate_character_records(
        model,
        records,
        tokenizer,
        block_size,
        device,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"completed updates: {checkpoint.get('step', 0)}")
    print(f"data: {args.data}")
    print(f"device: {device}")
    print(
        "response-only: "
        f"loss {report['loss']:.4f}, "
        f"perplexity {report['perplexity']:.4f}, "
        f"tokens {report['tokens']:,}, "
        f"examples {report['examples']:,}"
    )
    print("behavior categories:")

    for tag, category in report["categories"].items():
        print(
            f"- {tag}: loss {category['loss']:.4f}, "
            f"perplexity {category['perplexity']:.4f}, "
            f"tokens {category['tokens']:,}, "
            f"examples {category['examples']:,}"
        )


if __name__ == "__main__":
    main()
