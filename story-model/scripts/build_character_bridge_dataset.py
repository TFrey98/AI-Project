"""Build the Phase 21B prompt-conditioning bridge dataset."""

from __future__ import annotations

import argparse

from story_model.character_bridge import bridge_records
from story_model.character_training import build_character_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/character/bridge",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    manifest = build_character_dataset(
        bridge_records(),
        output_dir=args.output_dir,
        validation_fraction=0.2,
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
    print("character bridge dataset: passed")


if __name__ == "__main__":
    main()
