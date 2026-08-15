"""Generate deterministic text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import read_checkpoint
from story_model.data import CharTokenizer, load_text
from story_model.models import build_model
from story_model.runtime import (
    resolve_device,
    seed_everything,
)


def load_generation_metadata(
    checkpoint: dict,
    config_path: str | None,
) -> tuple[dict, CharTokenizer]:
    extra = checkpoint.get("extra", {})

    config = extra.get("config")
    tokenizer_data = extra.get("tokenizer")

    if config is None:
        if config_path is None:
            raise ValueError(
                "This older checkpoint does not contain "
                "its configuration; provide --config"
            )

        config = yaml.safe_load(
            Path(config_path).read_text(
                encoding="utf-8"
            )
        )

    if tokenizer_data is not None:
        tokenizer = CharTokenizer(
            stoi=tokenizer_data["stoi"],
            itos=tokenizer_data["itos"],
        )
    else:
        text = load_text(config["data"]["path"])
        tokenizer = CharTokenizer.from_text(text)

    return config, tokenizer


def generate(
    checkpoint_path: str,
    max_new_tokens: int,
    seed: int,
    config_path: str | None = None,
) -> str:
    checkpoint = read_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    config, tokenizer = load_generation_metadata(
        checkpoint,
        config_path,
    )

    device = resolve_device(
        config["train"]["device"]
    )

    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=config["data"]["block_size"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # Seed after model construction and checkpoint loading so only
    # generation consumes the deterministic random sequence.
    seed_everything(seed)

    starting_tokens = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=device,
    )

    output = model.generate(
        starting_tokens,
        max_new_tokens=max_new_tokens,
    )

    token_ids = output[0].cpu().tolist()
    return tokenizer.decode(token_ids)


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
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    args = parser.parse_args()

    print(
        generate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()