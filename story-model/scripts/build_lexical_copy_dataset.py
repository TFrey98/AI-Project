"""Build the Phase 29 sparse open-value curriculum."""

from __future__ import annotations

import argparse

from story_model.lexical_copy import build_lexical_copy_dataset
from story_model.semantic_transfer import SEMANTIC_TRANSFER_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/character/lexical_copy",
    )
    parser.add_argument("--copy-pairs-per-skill", type=int, default=1200)
    parser.add_argument("--anchor-pairs-per-skill", type=int, default=400)
    parser.add_argument(
        "--baseline-train-pairs-per-skill",
        type=int,
        default=800,
        help="Phase 28 train count needed to reproduce its val split.",
    )
    parser.add_argument("--val-pairs-per-skill", type=int, default=100)
    parser.add_argument("--lexical-pairs-per-skill", type=int, default=100)
    parser.add_argument(
        "--paraphrase-pairs-per-skill",
        type=int,
        default=100,
    )
    parser.add_argument("--transfer-pairs-per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    manifest = build_lexical_copy_dataset(
        output_dir=args.output_dir,
        copy_pairs_per_skill=args.copy_pairs_per_skill,
        anchor_pairs_per_skill=args.anchor_pairs_per_skill,
        baseline_train_pairs_per_skill=(
            args.baseline_train_pairs_per_skill
        ),
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

    print("training value diversity:")

    for skill, statistics in manifest["training_value_diversity"].items():
        exposure = statistics["copy_exposures"]
        print(
            f"- {skill}: {statistics['copy_values_exposed']:,}/"
            f"{statistics['copy_pool_values']:,} copy values exposed; "
            "exposures min/mean/max "
            f"{exposure['min']}/{exposure['mean']:.1f}/{exposure['max']}"
        )

    print(f"split: {manifest['split_strategy']}")
    print(f"answer keys: {args.output_dir}/answer_keys.json")
    print(f"output: {args.output_dir}")
    print("lexical-copy dataset: passed")


if __name__ == "__main__":
    main()
