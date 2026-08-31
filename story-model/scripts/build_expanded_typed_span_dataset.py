"""Build the Phase 32 direct single-span resolver curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    build_expanded_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/character/expanded_typed_span_resolver"),
    )
    parser.add_argument("--train-pairs-per-skill", type=int, default=400)
    parser.add_argument("--val-pairs-per-skill", type=int, default=100)
    parser.add_argument("--lexical-pairs-per-skill", type=int, default=100)
    parser.add_argument("--paraphrase-pairs-per-skill", type=int, default=100)
    parser.add_argument("--transfer-pairs-per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    manifest = build_expanded_dataset(
        args.output_dir,
        train_pairs_per_skill=args.train_pairs_per_skill,
        validation_pairs_per_skill=args.val_pairs_per_skill,
        lexical_pairs_per_skill=args.lexical_pairs_per_skill,
        paraphrase_pairs_per_skill=args.paraphrase_pairs_per_skill,
        transfer_pairs_per_skill=args.transfer_pairs_per_skill,
        seed=args.seed,
    )
    for split in EXPANDED_SPLITS:
        metadata = manifest["splits"][split]
        print(
            f"{split}: {metadata['examples']:,} examples, "
            f"{metadata['resolve_pairs']:,} resolve pairs, "
            f"actions={metadata['actions']}, "
            f"candidate_counts={metadata['candidate_counts']}"
        )
    print("candidate positions: every position covered at widths 2, 3, and 4")
    print(f"output: {args.output_dir}")
    print("expanded typed-span dataset: passed")


if __name__ == "__main__":
    main()
