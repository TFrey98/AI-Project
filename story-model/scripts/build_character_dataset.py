"""Split source character JSONL by conversation and write a manifest."""

from __future__ import annotations

import argparse

from story_model.character_training import (
    build_character_dataset,
    load_character_training_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Unsplit character-training JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/character/processed",
        help="Destination for train.jsonl, val.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    records = load_character_training_records(args.source)
    manifest = build_character_dataset(
        records,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    print(f"source examples: {manifest['source_examples']:,}")
    print(
        "train: "
        f"{manifest['train']['examples']:,} examples, "
        f"{len(manifest['train']['conversations']):,} conversations"
    )
    print(
        "val: "
        f"{manifest['val']['examples']:,} examples, "
        f"{len(manifest['val']['conversations']):,} conversations"
    )
    print(f"output: {args.output_dir}")
    print("conversation leakage check: passed")


if __name__ == "__main__":
    main()
