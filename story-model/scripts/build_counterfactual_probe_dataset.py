"""Build the Phase 27 paired counterfactual grounding probe."""

from __future__ import annotations

import argparse

from story_model.counterfactual_probe import (
    COUNTERFACTUAL_SKILLS,
    build_counterfactual_probe_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/character/counterfactual_probe",
    )
    parser.add_argument("--train-pairs-per-skill", type=int, default=400)
    parser.add_argument("--val-pairs-per-skill", type=int, default=100)
    parser.add_argument("--transfer-pairs-per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    manifest = build_counterfactual_probe_dataset(
        output_dir=args.output_dir,
        train_pairs_per_skill=args.train_pairs_per_skill,
        validation_pairs_per_skill=args.val_pairs_per_skill,
        transfer_pairs_per_skill=args.transfer_pairs_per_skill,
        seed=args.seed,
    )

    print(f"skills: {len(COUNTERFACTUAL_SKILLS)}")

    for split in ("train", "val", "transfer"):
        metadata = manifest[split]
        print(
            f"{split}: {metadata['examples']:,} examples, "
            f"{metadata['pairs']:,} counterfactual pairs"
        )

    print(f"split: {manifest['split_strategy']}")
    print(f"output: {args.output_dir}")
    print("counterfactual probe dataset: passed")


if __name__ == "__main__":
    main()
