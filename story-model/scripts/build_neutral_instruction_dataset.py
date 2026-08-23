"""Build the Phase 26 generic dialogue instruction curriculum."""

from __future__ import annotations

import argparse

from story_model.neutral_instruction import (
    NEUTRAL_SKILLS,
    build_neutral_instruction_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/character/neutral_instruction",
    )
    parser.add_argument(
        "--train-examples-per-skill",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--val-examples-per-skill",
        type=int,
        default=50,
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    manifest = build_neutral_instruction_dataset(
        output_dir=args.output_dir,
        train_examples_per_skill=args.train_examples_per_skill,
        validation_examples_per_skill=args.val_examples_per_skill,
        seed=args.seed,
    )

    print(f"skills: {len(NEUTRAL_SKILLS)}")
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
    print(f"split: {manifest['split_strategy']}")
    print(f"output: {args.output_dir}")
    print("neutral instruction dataset: passed")


if __name__ == "__main__":
    main()
