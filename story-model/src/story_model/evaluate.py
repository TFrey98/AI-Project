"""Deterministically evaluate a saved language-model checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import read_checkpoint
from story_model.data import load_text_splits
from story_model.generate import load_generation_metadata
from story_model.models import build_model
from story_model.runtime import resolve_device


def read_evaluation_data_config(
    config_path: str,
) -> dict:
    """Read a full training config or a data-only YAML mapping."""

    loaded = yaml.safe_load(
        Path(config_path).read_text(encoding="utf-8")
    )

    if not isinstance(loaded, dict):
        raise ValueError(
            "evaluation data config must be a YAML mapping"
        )

    data_config = loaded.get("data", loaded)

    if not isinstance(data_config, dict):
        raise ValueError(
            "evaluation data section must be a YAML mapping"
        )

    return data_config


def calculate_bits_per_byte(
    mean_loss: float,
    token_count: int,
    byte_count: int,
) -> float:
    if token_count < 1:
        raise ValueError("token_count must be positive")

    if byte_count < 1:
        raise ValueError("byte_count must be positive")

    total_nats = mean_loss * token_count
    return total_nats / (math.log(2.0) * byte_count)


@torch.no_grad()
def evaluate_token_stream(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str | torch.device,
) -> tuple[float, int]:
    """Return mean next-token loss over every target in a stream."""

    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional token stream")

    if len(data) < 2:
        raise ValueError("data must contain at least two tokens")

    if block_size < 1:
        raise ValueError("block_size must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.eval()

    prediction_count = len(data) - 1
    full_token_count = (
        prediction_count // block_size
    ) * block_size

    total_negative_log_likelihood = 0.0
    evaluated_tokens = 0

    if full_token_count > 0:
        inputs = data[:full_token_count].reshape(
            -1,
            block_size,
        )
        targets = data[1 : full_token_count + 1].reshape(
            -1,
            block_size,
        )

        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[
                start : start + batch_size
            ].to(device)
            batch_targets = targets[
                start : start + batch_size
            ].to(device)

            _, loss = model(
                batch_inputs,
                batch_targets,
            )

            if loss is None:
                raise RuntimeError(
                    "Model did not return evaluation loss"
                )

            token_count = batch_inputs.numel()
            total_negative_log_likelihood += (
                loss.item() * token_count
            )
            evaluated_tokens += token_count

    if full_token_count < prediction_count:
        tail_inputs = data[
            full_token_count:-1
        ].unsqueeze(0).to(device)
        tail_targets = data[
            full_token_count + 1:
        ].unsqueeze(0).to(device)

        _, tail_loss = model(
            tail_inputs,
            tail_targets,
        )

        if tail_loss is None:
            raise RuntimeError(
                "Model did not return evaluation loss"
            )

        tail_count = tail_inputs.numel()
        total_negative_log_likelihood += (
            tail_loss.item() * tail_count
        )
        evaluated_tokens += tail_count

    if was_training:
        model.train()

    mean_loss = (
        total_negative_log_likelihood
        / evaluated_tokens
    )

    return mean_loss, evaluated_tokens


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str | None = None,
    split: str = "both",
    device_name: str | None = None,
    data_config: dict | None = None,
) -> dict:
    if split not in {"train", "val", "both"}:
        raise ValueError(
            "split must be 'train', 'val', or 'both'"
        )

    checkpoint = read_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    config, tokenizer = load_generation_metadata(
        checkpoint,
        config_path,
    )

    configured_device = (
        device_name
        if device_name is not None
        else config["train"]["device"]
    )
    device = resolve_device(configured_device)

    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=config["data"]["block_size"],
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)

    evaluation_data_config = (
        data_config
        if data_config is not None
        else config["data"]
    )
    train_text, val_text = load_text_splits(
        evaluation_data_config
    )

    train_encoded = tokenizer.encode(train_text)
    val_encoded = tokenizer.encode(val_text)

    available_splits = {
        "train": torch.tensor(
            train_encoded,
            dtype=torch.long,
        ),
        "val": torch.tensor(
            val_encoded,
            dtype=torch.long,
        ),
    }
    available_tokens = {
        "train": train_encoded,
        "val": val_encoded,
    }
    selected_splits = (
        ("train", "val")
        if split == "both"
        else (split,)
    )

    results = {
        "checkpoint": checkpoint_path,
        "step": int(checkpoint.get("step", 0)),
        "device": str(device),
        "data_override": data_config is not None,
        "splits": {},
    }

    for split_name in selected_splits:
        loss, token_count = evaluate_token_stream(
            model=model,
            data=available_splits[split_name],
            block_size=config["data"]["block_size"],
            batch_size=int(
                evaluation_data_config.get(
                    "batch_size",
                    config["data"]["batch_size"],
                )
            ),
            device=device,
        )

        target_byte_count = sum(
            tokenizer.token_byte_length(token_id)
            for token_id in available_tokens[split_name][1:]
        )

        results["splits"][split_name] = {
            "loss": loss,
            "perplexity": math.exp(loss),
            "tokens": token_count,
            "bytes": target_byte_count,
            "bits_per_byte": calculate_bits_per_byte(
                mean_loss=loss,
                token_count=token_count,
                byte_count=target_byte_count,
            ),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved checkpoint.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Fallback configuration for older checkpoints "
            "without embedded metadata."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="both",
    )

    parser.add_argument(
        "--data-config",
        default=None,
        help=(
            "Optional full or data-only YAML config whose corpus "
            "replaces the checkpoint corpus for cross-evaluation."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override: auto, cpu, mps, or cuda.",
    )

    args = parser.parse_args()

    evaluation_data_config = (
        read_evaluation_data_config(args.data_config)
        if args.data_config is not None
        else None
    )

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        device_name=args.device,
        data_config=evaluation_data_config,
    )

    print(f"checkpoint: {results['checkpoint']}")
    print(f"completed updates: {results['step']}")
    print(f"device: {results['device']}")

    if args.data_config is not None:
        print(f"evaluation data: {args.data_config}")

    for split_name, metrics in results["splits"].items():
        print(
            f"{split_name}: "
            f"loss {metrics['loss']:.4f}, "
            f"perplexity {metrics['perplexity']:.4f}, "
            f"bits/byte {metrics['bits_per_byte']:.4f}, "
            f"tokens {metrics['tokens']:,}, "
            f"bytes {metrics['bytes']:,}"
        )


if __name__ == "__main__":
    main()
