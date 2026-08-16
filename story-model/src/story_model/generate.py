"""Generate controlled text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import read_checkpoint
from story_model.data import (
    CharTokenizer,
    Tokenizer,
    build_tokenizer,
    load_text_splits,
    tokenizer_from_dict,
)
from story_model.models import build_model
from story_model.runtime import (
    resolve_device,
    seed_everything,
)
from story_model.sampling import generate_tokens


def load_generation_metadata(
    checkpoint: dict,
    config_path: str | None,
) -> tuple[dict, Tokenizer]:
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
        tokenizer = tokenizer_from_dict(
            tokenizer_data
        )
    else:
        training_text, _ = load_text_splits(
            config["data"]
        )
        tokenizer = build_tokenizer(
            training_text=training_text,
            config=config.get("tokenizer"),
        )

    return config, tokenizer


def encode_prompt(
    tokenizer: Tokenizer,
    prompt: str,
) -> list[int]:
    if not prompt:
        raise ValueError("prompt cannot be empty")

    if isinstance(tokenizer, CharTokenizer):
        unknown = sorted(
            set(prompt) - set(tokenizer.stoi)
        )

        if unknown:
            raise ValueError(
                "Prompt contains characters outside the "
                f"checkpoint vocabulary: {unknown!r}"
            )

    return tokenizer.encode(prompt)


def generate(
    checkpoint_path: str,
    max_new_tokens: int,
    seed: int,
    config_path: str | None = None,
    prompt: str = "\n",
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
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

    # Seed after construction and loading so only decoding consumes
    # this deterministic random sequence.
    seed_everything(seed)

    prompt_tokens = encode_prompt(
        tokenizer,
        prompt,
    )

    starting_tokens = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    output = generate_tokens(
        model=model,
        starting_tokens=starting_tokens,
        max_new_tokens=max_new_tokens,
        block_size=config["data"]["block_size"],
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
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
        "--prompt",
        default="\n",
        help="Text used to begin generation.",
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

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Positive sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sample only from the k highest-scoring tokens.",
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Always choose the highest-scoring token.",
    )

    args = parser.parse_args()

    print(
        generate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )
    )


if __name__ == "__main__":
    main()
