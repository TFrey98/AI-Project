"""Phase 29 sparse-value curriculum for copying evidence into answers."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from story_model.character_training import CHARACTER_DATASET_VERSION
from story_model.corpus import sha256_text
from story_model.counterfactual_probe import (
    COUNTERFACTUAL_SKILLS,
    _build_records,
    _records_text,
    _split_metadata,
    validate_counterfactual_pairs,
)
from story_model.semantic_transfer import (
    DEVELOPMENT_LEXICON,
    SEMANTIC_TRANSFER_SPLITS,
    SEMANTIC_TRANSFER_VERSION,
    _skill_resources,
    build_semantic_answer_keys,
    semantic_transfer_splits,
)


LEXICAL_COPY_VERSION = 1
VALUE_PREFIXES = (
    "alder",
    "ash",
    "birch",
    "cedar",
    "cinder",
    "cloud",
    "copper",
    "dusk",
    "ember",
    "fern",
    "flint",
    "frost",
    "granite",
    "harbor",
    "hazel",
    "iron",
    "juniper",
    "linden",
    "moss",
    "pearl",
    "pine",
    "reed",
    "storm",
    "willow",
)
COLOR_TERMS = (
    "bloom",
    "bronze",
    "coral",
    "cream",
    "flame",
    "haze",
    "jade",
    "lavender",
    "mist",
    "plum",
    "rust",
    "saffron",
    "smoke",
    "umber",
    "wine",
    "yellow",
)
ROUTE_TERMS = (
    "arch",
    "causeway",
    "corridor",
    "crossing",
    "door",
    "footway",
    "gallery",
    "gate",
    "lane",
    "path",
    "ramp",
    "road",
    "stairway",
    "trail",
    "tunnel",
    "walk",
)
COPY_COLOR_VALUES = tuple(
    f"{prefix} {term}"
    for prefix in VALUE_PREFIXES
    for term in COLOR_TERMS
)
COPY_ROUTE_VALUES = tuple(
    f"{prefix} {term}"
    for prefix in VALUE_PREFIXES
    for term in ROUTE_TERMS
)
LEXICAL_COPY_LEXICON = replace(
    DEVELOPMENT_LEXICON,
    colors=COPY_COLOR_VALUES,
    routes=COPY_ROUTE_VALUES,
)


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _training_records_and_keys(
    copy_pairs_per_skill: int,
    anchor_pairs_per_skill: int,
    seed: int,
) -> tuple[tuple, dict[str, dict]]:
    records = []
    entries = {}

    for skill_index, skill in enumerate(COUNTERFACTUAL_SKILLS):
        (
            spec_builder,
            questions,
            responses,
            _,
            _,
        ) = _skill_resources(skill)
        anchor_specs = spec_builder(
            anchor_pairs_per_skill,
            seed + skill_index * 101,
            DEVELOPMENT_LEXICON,
            questions,
            responses,
        )
        copy_specs = spec_builder(
            copy_pairs_per_skill,
            seed + 5_000 + skill_index * 101,
            LEXICAL_COPY_LEXICON,
            questions,
            responses,
        )
        specs = list(anchor_specs + copy_specs)
        random.Random(seed + 9_000 + skill_index).shuffle(specs)
        skill_records = _build_records(
            "train",
            skill,
            tuple(specs),
            questions,
            responses,
        )
        records.extend(skill_records)

        for pair_index, spec in enumerate(specs):
            first, second = skill_records[pair_index * 2 : pair_index * 2 + 2]

            if skill == "supplied_fact":
                expected_first, expected_second = spec[2], spec[3]
            else:
                expected_first, expected_second = spec[2], spec[1]

            for record, expected, alternative in (
                (first, expected_first, expected_second),
                (second, expected_second, expected_first),
            ):
                entries[record.context.context_id] = {
                    "split": "train",
                    "conversation_id": record.conversation_id,
                    "skill": skill,
                    "expected_value": expected,
                    "alternative_value": alternative,
                }

    return tuple(records), entries


def lexical_copy_curriculum(
    copy_pairs_per_skill: int = 1_200,
    anchor_pairs_per_skill: int = 400,
    baseline_train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> tuple[dict, dict]:
    """Build sparse-value training with unchanged Phase 28 diagnostics."""

    counts = {
        "copy_pairs_per_skill": _positive_integer(
            copy_pairs_per_skill,
            "copy_pairs_per_skill",
        ),
        "anchor_pairs_per_skill": _positive_integer(
            anchor_pairs_per_skill,
            "anchor_pairs_per_skill",
        ),
        "baseline_train_pairs_per_skill": _positive_integer(
            baseline_train_pairs_per_skill,
            "baseline_train_pairs_per_skill",
        ),
        "validation_pairs_per_skill": _positive_integer(
            validation_pairs_per_skill,
            "validation_pairs_per_skill",
        ),
        "lexical_pairs_per_skill": _positive_integer(
            lexical_pairs_per_skill,
            "lexical_pairs_per_skill",
        ),
        "paraphrase_pairs_per_skill": _positive_integer(
            paraphrase_pairs_per_skill,
            "paraphrase_pairs_per_skill",
        ),
        "transfer_pairs_per_skill": _positive_integer(
            transfer_pairs_per_skill,
            "transfer_pairs_per_skill",
        ),
    }
    baseline = semantic_transfer_splits(
        train_pairs_per_skill=counts["baseline_train_pairs_per_skill"],
        validation_pairs_per_skill=counts["validation_pairs_per_skill"],
        lexical_pairs_per_skill=counts["lexical_pairs_per_skill"],
        paraphrase_pairs_per_skill=counts["paraphrase_pairs_per_skill"],
        transfer_pairs_per_skill=counts["transfer_pairs_per_skill"],
        seed=seed,
    )
    training, training_entries = _training_records_and_keys(
        copy_pairs_per_skill=counts["copy_pairs_per_skill"],
        anchor_pairs_per_skill=counts["anchor_pairs_per_skill"],
        seed=seed,
    )
    splits = {
        "train": training,
        **{
            split: baseline[split]
            for split in SEMANTIC_TRANSFER_SPLITS
            if split != "train"
        },
    }

    for split_records in splits.values():
        validate_counterfactual_pairs(split_records)

    baseline_keys = build_semantic_answer_keys(baseline)
    diagnostic_entries = {
        context_id: entry
        for context_id, entry in baseline_keys["entries"].items()
        if entry["split"] != "train"
    }
    entries = {**diagnostic_entries, **training_entries}
    expected_entries = sum(len(split) for split in splits.values())

    if len(entries) != expected_entries:
        raise RuntimeError(
            "lexical-copy answer keys do not cover every curriculum row"
        )

    answer_keys = {
        "schema_version": 1,
        "semantic_transfer_version": SEMANTIC_TRANSFER_VERSION,
        "lexical_copy_version": LEXICAL_COPY_VERSION,
        "entries": dict(sorted(entries.items())),
    }
    return splits, answer_keys


def _value_statistics(answer_keys: dict) -> dict:
    by_skill = {
        "supplied_fact": Counter(),
        "scene_route": Counter(),
    }

    for entry in answer_keys["entries"].values():
        if entry["split"] != "train":
            continue

        by_skill[entry["skill"]][entry["expected_value"]] += 1

    copy_sets = {
        "supplied_fact": set(COPY_COLOR_VALUES),
        "scene_route": set(COPY_ROUTE_VALUES),
    }
    result = {}

    for skill, counts in by_skill.items():
        copy_counts = {
            value: count
            for value, count in counts.items()
            if value in copy_sets[skill]
        }
        exposures = tuple(copy_counts.values())
        result[skill] = {
            "all_unique_values": len(counts),
            "copy_pool_values": len(copy_sets[skill]),
            "copy_values_exposed": len(copy_counts),
            "copy_exposures": {
                "min": min(exposures),
                "mean": sum(exposures) / len(exposures),
                "max": max(exposures),
            },
        }

    return result


def build_lexical_copy_dataset(
    output_dir: str | Path,
    copy_pairs_per_skill: int = 1_200,
    anchor_pairs_per_skill: int = 400,
    baseline_train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict:
    """Write Phase 29 training and unchanged Phase 28 diagnostic axes."""

    splits, answer_keys = lexical_copy_curriculum(
        copy_pairs_per_skill=copy_pairs_per_skill,
        anchor_pairs_per_skill=anchor_pairs_per_skill,
        baseline_train_pairs_per_skill=baseline_train_pairs_per_skill,
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
        "lexical_copy_version": LEXICAL_COPY_VERSION,
        "seed": seed,
        "split_strategy": "sparse_open_values_with_fixed_phase28_diagnostics",
        "pair_invariant": (
            "adjacent rows share a conversation_id and differ in exactly "
            "one serialized evidence line"
        ),
        "source_examples": len(splits["train"]) + len(splits["val"]),
        "diagnostic_examples": sum(
            len(splits[split])
            for split in ("lexical", "paraphrase", "transfer")
        ),
        "training_pairs_per_skill": {
            "copy": copy_pairs_per_skill,
            "anchor": anchor_pairs_per_skill,
        },
        "training_value_diversity": _value_statistics(answer_keys),
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
