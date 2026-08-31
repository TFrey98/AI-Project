"""Build the Phase 31 typed whole-span resolver curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.character_training import load_character_training_records
from story_model.semantic_transfer import (
    build_semantic_answer_keys,
    semantic_transfer_splits,
)
from story_model.typed_span_resolver import (
    TYPED_SPAN_SPLITS,
    build_typed_span_dataset,
)


def _load_source_directory(path: Path):
    required = [path / f"{split}.jsonl" for split in TYPED_SPAN_SPLITS]
    required.append(path / "answer_keys.json")
    if not all(item.is_file() for item in required):
        return None
    splits = {
        split: load_character_training_records(path / f"{split}.jsonl")
        for split in TYPED_SPAN_SPLITS
    }
    answer_keys = json.loads(
        (path / "answer_keys.json").read_text(encoding="utf-8")
    )
    return splits, answer_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/character/pointer_copy"),
        help=(
            "Use the exact Phase 30 records when present; otherwise "
            "rebuild the Phase 28-equivalent source splits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument("--train-pairs-per-skill", type=int, default=800)
    parser.add_argument("--val-pairs-per-skill", type=int, default=100)
    parser.add_argument("--lexical-pairs-per-skill", type=int, default=100)
    parser.add_argument("--paraphrase-pairs-per-skill", type=int, default=100)
    parser.add_argument("--transfer-pairs-per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    loaded = _load_source_directory(args.source_dir)
    if loaded is None:
        print(
            f"source: {args.source_dir} not found; rebuilding the "
            "Phase 28-equivalent source records"
        )
        splits = semantic_transfer_splits(
            train_pairs_per_skill=args.train_pairs_per_skill,
            validation_pairs_per_skill=args.val_pairs_per_skill,
            lexical_pairs_per_skill=args.lexical_pairs_per_skill,
            paraphrase_pairs_per_skill=args.paraphrase_pairs_per_skill,
            transfer_pairs_per_skill=args.transfer_pairs_per_skill,
            seed=args.seed,
        )
        answer_keys = build_semantic_answer_keys(splits)
    else:
        print(f"source: using exact Phase 30 rows from {args.source_dir}")
        splits, answer_keys = loaded

    manifest = build_typed_span_dataset(
        splits,
        answer_keys,
        args.output_dir,
    )
    for split in TYPED_SPAN_SPLITS:
        metadata = manifest["splits"][split]
        print(
            f"{split}: {metadata['examples']:,} examples, "
            f"{metadata['resolve_pairs']:,} resolve pairs, "
            f"actions={metadata['actions']}"
        )
    print("candidate positions: exactly balanced")
    print(f"output: {args.output_dir}")
    print("typed-span resolver dataset: passed")


if __name__ == "__main__":
    main()
