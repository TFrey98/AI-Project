"""Overfit a small dataset and require correct free-running responses."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from story_model.character_chat import generate_character_response
from story_model.character_data import CHARACTER_CONTROL_TOKENS
from story_model.character_evaluate import evaluate_character_generations
from story_model.character_training import (
    encode_character_training_records,
    load_character_training_records,
)
from story_model.checkpoint import (
    load_model_warm_start,
    read_checkpoint,
    save_checkpoint,
)
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default="data/character/smoke/train.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "checkpoints/transformer_character_generation_overfit/final.pt"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = read_checkpoint(args.checkpoint, map_location="cpu")
    extra = checkpoint.get("extra", {})
    tokenizer_data = extra.get("tokenizer")
    source_config = extra.get("config")

    if tokenizer_data is None or source_config is None:
        raise ValueError(
            "checkpoint must contain tokenizer and config metadata"
        )

    source_tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(source_tokenizer, ByteBPETokenizer):
        raise ValueError("character overfit requires byte-BPE")

    tokenizer = source_tokenizer.with_special_tokens(
        CHARACTER_CONTROL_TOKENS
    )
    records = load_character_training_records(args.data)
    examples = encode_character_training_records(
        records,
        tokenizer,
        args.block_size,
    )
    model_config = dict(source_config["model"])
    model_config["dropout"] = 0.0
    model = build_model(
        model_config,
        vocabulary_size=tokenizer.vocab_size,
        block_size=args.block_size,
    )
    expanded = load_model_warm_start(
        model=model,
        checkpoint=checkpoint,
        source_vocabulary_size=source_tokenizer.vocab_size,
        destination_vocabulary_size=tokenizer.vocab_size,
    )
    model = model.to(device)
    inputs = torch.tensor(
        [example.input_ids for example in examples],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [example.target_ids for example in examples],
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    model.eval()

    with torch.no_grad():
        _, initial_loss_tensor = model(inputs, targets)

    assert initial_loss_tensor is not None
    initial_loss = initial_loss_tensor.item()
    print(f"step   0: loss {initial_loss:.6f}")
    model.train()

    for step in range(1, args.steps + 1):
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )

        if not torch.isfinite(gradient_norm).item():
            raise RuntimeError("non-finite gradient during overfit")

        optimizer.step()

        if step % 25 == 0 or step == args.steps:
            print(f"step {step:3d}: loss {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        _, final_loss_tensor = model(inputs, targets)

    assert final_loss_tensor is not None
    final_loss = final_loss_tensor.item()

    def generate(record, index):
        return generate_character_response(
            model=model,
            tokenizer=tokenizer,
            context=replace(record.context, target_response=None),
            block_size=args.block_size,
            device=device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
            greedy=True,
        )

    report = evaluate_character_generations(records, generate)
    diagnostic_config = {
        "tokenizer": {
            "type": "byte_bpe",
            "vocab_size": tokenizer.base_vocab_size,
            "special_tokens": list(tokenizer.special_tokens),
        },
        "model": model_config,
        "data": {
            "type": "character_jsonl",
            "train_path": args.data,
            "val_path": args.data,
            "block_size": args.block_size,
            "batch_size": len(examples),
        },
        "train": {
            "device": args.device,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "max_steps": args.steps,
        },
        "checkpoint": {"dir": "diagnostic"},
    }
    save_checkpoint(
        args.output,
        model,
        optimizer,
        step=args.steps,
        extra={
            "config": diagnostic_config,
            "tokenizer": tokenizer.to_dict(),
            "diagnostic": {
                "source_checkpoint": args.checkpoint,
                "data": args.data,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "generation_passed": report["all_passed"],
            },
        },
    )

    print(f"device: {device}")
    print(f"examples: {len(records)}")
    print(f"block size: {args.block_size:,}")
    print(
        "expanded parameters: "
        + (", ".join(expanded) if expanded else "none")
    )
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss: {final_loss:.6f}")

    for result in report["results"]:
        print(
            f"{result['context_id']}: "
            f"exact={result['exact_response']}, "
            f"end={result['end_stop']}, "
            f"prefix={result['prefix_fraction']:.1%}, "
            f"similarity={result['similarity']:.1%}, "
            f"stop={result['stop_reason']}"
        )

    print(f"checkpoint: {args.output}")

    if final_loss >= args.loss_threshold:
        raise RuntimeError(
            "teacher-forced loss did not reach the diagnostic threshold"
        )
    if not report["all_passed"]:
        raise RuntimeError(
            "free-running responses did not exactly reproduce the dataset"
        )

    print("multi-example free-running overfit: passed")


if __name__ == "__main__":
    main()
