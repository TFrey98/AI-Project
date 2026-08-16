"""Generate controlled text from a trained checkpoint.

This is the "use the model" script, as opposed to train.py ("make the
model"). Its main job, beyond running generate_tokens(), is FAITHFULLY
reconstructing the exact model architecture and tokenizer a checkpoint was
trained with — get either one even slightly wrong (a different
embedding_dim, a different vocabulary mapping) and the checkpoint's saved
weights would either fail to load or, worse, load into the wrong shapes
and produce meaningless output without necessarily raising an error.
"""

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
    """Recover the (config, tokenizer) pair a checkpoint was trained with.

    Newer checkpoints (see train.py's checkpoint_metadata) are
    self-describing: both the full training config and the tokenizer's
    to_dict() are embedded directly in the checkpoint's "extra" field, so
    this function can reconstruct everything from the checkpoint file
    alone. Two fallback paths exist for checkpoints that predate one of
    these fields: --config lets the caller supply a config file
    explicitly (for checkpoints saved before configs were embedded at
    all), and rebuilding the tokenizer from scratch via build_tokenizer()
    handles checkpoints saved before tokenizer state was embedded — this
    only works correctly for a char-level tokenizer (deterministically
    rebuildable from the training text alone) or a BPE tokenizer whose
    training is itself deterministic (see ByteBPETokenizer.train's
    tie-breaking rule in data.py), which is why that determinism was a
    deliberate design requirement, not an accident.
    """

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
        # Only ever rebuilt from the TRAINING split, never the full
        # corpus or validation split — same leakage-avoidance reasoning
        # as build_tokenizer()'s docstring in data.py: the tokenizer this
        # checkpoint trained with must never have "seen" validation text.
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
    """Encode a user-supplied prompt, with a friendly error for char-level gaps.

    A CharTokenizer's vocabulary is exactly the set of characters that
    appeared in its training text — nothing more. If a user's prompt
    contains a character the model has literally never seen (an emoji, a
    symbol, a language the corpus didn't include), CharTokenizer.encode()
    would raise a raw KeyError deep inside a dict lookup, which is
    accurate but unhelpful. This function checks for that case up front
    and raises a clear, actionable error naming exactly which characters
    are the problem. A ByteBPETokenizer never has this issue — it can
    always fall back to encoding any Unicode text as raw UTF-8 bytes, even
    for a character its merge vocabulary has never specifically learned a
    shortcut for — so this check only applies to CharTokenizer.
    """

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
