"""Build the Phase 30 pointer/copy dataset."""

from __future__ import annotations

import argparse

from story_model.pointer_copy import build_pointer_copy_dataset
from story_model.semantic_transfer import SEMANTIC_TRANSFER_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/character/pointer_copy",
    )
    parser.add_argument("--train-pairs-per-skill", type=int, default=800)
    parser.add_argument("--val-pairs-per-skill", type=int, default=100)
    parser.add_argument("--lexical-pairs-per-skill", type=int, default=100)
    parser.add_argument(
        "--paraphrase-pairs-per-skill", type=int, default=100
    )
    parser.add_argument("--transfer-pairs-per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    manifest = build_pointer_copy_dataset(
        output_dir=args.output_dir,
        train_pairs_per_skill=args.train_pairs_per_skill,
        validation_pairs_per_skill=args.val_pairs_per_skill,
        lexical_pairs_per_skill=args.lexical_pairs_per_skill,
        paraphrase_pairs_per_skill=args.paraphrase_pairs_per_skill,
        transfer_pairs_per_skill=args.transfer_pairs_per_skill,
        seed=args.seed,
    )

    for split in SEMANTIC_TRANSFER_SPLITS:
        metadata = manifest[split]
        print(
            f"{split}: {metadata['examples']:,} examples, "
            f"{metadata['pairs']:,} counterfactual pairs"
        )

    print(f"answer keys: {args.output_dir}/answer_keys.json")
    print(f"output: {args.output_dir}")
    print("pointer copy dataset: passed")


if __name__ == "__main__":
    main()
