"""Phase 30 evidence-pointer curriculum built from Phase 28 records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from story_model.character_training import CHARACTER_DATASET_VERSION
from story_model.corpus import sha256_text
from story_model.counterfactual_probe import _records_text, _split_metadata
from story_model.semantic_transfer import (
    SEMANTIC_TRANSFER_SPLITS,
    SEMANTIC_TRANSFER_VERSION,
    build_semantic_answer_keys,
    semantic_transfer_splits,
)


POINTER_COPY_VERSION = 2


def pointer_copy_splits(
    train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> tuple[dict, dict]:
    """Attach exact copy values without changing Phase 28 prompt content."""

    baseline = semantic_transfer_splits(
        train_pairs_per_skill=train_pairs_per_skill,
        validation_pairs_per_skill=validation_pairs_per_skill,
        lexical_pairs_per_skill=lexical_pairs_per_skill,
        paraphrase_pairs_per_skill=paraphrase_pairs_per_skill,
        transfer_pairs_per_skill=transfer_pairs_per_skill,
        seed=seed,
    )
    answer_keys = build_semantic_answer_keys(baseline)
    entries = answer_keys["entries"]
    annotated = {
        split: tuple(
            replace(
                record,
                copy_value=entries[record.context.context_id][
                    "expected_value"
                ],
            )
            for record in baseline[split]
        )
        for split in SEMANTIC_TRANSFER_SPLITS
    }
    return annotated, answer_keys


def build_pointer_copy_dataset(
    output_dir: str | Path,
    train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict:
    """Write Phase 28 rows with byte-level pointer annotations."""

    splits, answer_keys = pointer_copy_splits(
        train_pairs_per_skill=train_pairs_per_skill,
        validation_pairs_per_skill=validation_pairs_per_skill,
        lexical_pairs_per_skill=lexical_pairs_per_skill,
        paraphrase_pairs_per_skill=paraphrase_pairs_per_skill,
        transfer_pairs_per_skill=transfer_pairs_per_skill,
        seed=seed,
    )
    texts = {
        split: _records_text(records) for split, records in splits.items()
    }
    answer_keys_text = json.dumps(
        answer_keys,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, text in texts.items():
        (output_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")

    (output_dir / "answer_keys.json").write_text(
        answer_keys_text,
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": CHARACTER_DATASET_VERSION,
        "semantic_transfer_version": SEMANTIC_TRANSFER_VERSION,
        "pointer_copy_version": POINTER_COPY_VERSION,
        "seed": seed,
        "split_strategy": "phase28_records_with_byte_pointer_supervision",
        "pair_invariant": (
            "adjacent rows share a conversation_id and differ in exactly "
            "one serialized evidence line"
        ),
        "copy_annotation": (
            "copy_value identifies the exact answer bytes in prompt and target"
        ),
        "source_examples": len(splits["train"]) + len(splits["val"]),
        "diagnostic_examples": sum(
            len(splits[split])
            for split in ("lexical", "paraphrase", "transfer")
        ),
        "answer_keys_sha256": sha256_text(answer_keys_text),
    }

    for split in SEMANTIC_TRANSFER_SPLITS:
        manifest[split] = _split_metadata(splits[split], texts[split])

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
