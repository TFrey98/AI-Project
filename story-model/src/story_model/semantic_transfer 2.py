"""Phase 28 semantic and factorized-transfer counterfactual curriculum."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from story_model.character_data import serialize_character_prompt
from story_model.character_training import (
    CHARACTER_DATASET_VERSION,
    CharacterTrainingRecord,
)
from story_model.corpus import sha256_text
from story_model.counterfactual_probe import (
    COUNTERFACTUAL_SKILLS,
    SCENE_ROUTE_QUESTIONS,
    SCENE_ROUTE_RESPONSES,
    SCENE_ROUTE_TRANSFER_QUESTIONS,
    SCENE_ROUTE_TRANSFER_RESPONSES,
    SUPPLIED_FACT_QUESTIONS,
    SUPPLIED_FACT_RESPONSES,
    SUPPLIED_FACT_TRANSFER_QUESTIONS,
    SUPPLIED_FACT_TRANSFER_RESPONSES,
    _build_records,
    _positive_integer,
    _records_text,
    _scene_route_specs,
    _split_metadata,
    _supplied_fact_specs,
    validate_counterfactual_pairs,
)
from story_model.neutral_instruction import (
    TRAIN_LEXICON,
    VALIDATION_LEXICON,
    NeutralLexicon,
)


SEMANTIC_TRANSFER_VERSION = 1
SEMANTIC_TRANSFER_SPLITS = (
    "train",
    "val",
    "lexical",
    "paraphrase",
    "transfer",
)


ADDITIONAL_TRAIN_LEXICON = NeutralLexicon(
    names=("Arden", "Bela", "Corin", "Dessa", "Elian", "Fenn"),
    items=(
        "pewter badge",
        "leather folio",
        "wax tablet",
        "reed basket",
        "porcelain cup",
        "cedar case",
    ),
    containers=(
        "iron strongbox",
        "linen cabinet",
        "stable locker",
        "hallway recess",
        "cedar wardrobe",
        "counter drawer",
    ),
    locations=(
        "south signal tower",
        "granary office",
        "forest checkpoint",
        "ferry landing",
        "museum courtyard",
        "quarry shelter",
        "hilltop station",
        "east gatehouse",
    ),
    routes=(
        "timber causeway",
        "granary steps",
        "forest track",
        "ferry ramp",
        "museum arcade",
        "quarry road",
        "hill path",
        "east sally port",
    ),
    obstacles=(
        "blocked by timber",
        "unsafe after rain",
        "closed by the keeper",
        "covered in loose rock",
        "impassable at high tide",
        "under repair",
        "blocked by a wagon",
        "hidden by smoke",
    ),
    colors=(
        "ochre",
        "cerulean",
        "maroon",
        "turquoise",
        "bronze",
        "rose",
        "lime",
        "navy",
    ),
    organizations=(
        "ferry service",
        "forest rangers",
        "museum staff",
        "quarry office",
        "granary board",
        "signal corps",
        "gatehouse watch",
        "hill survey",
    ),
    signals=(
        "ferry ready",
        "trail reopened",
        "gallery cleared",
        "blasting complete",
        "grain inspected",
        "relay confirmed",
        "gate unlocked",
        "survey finished",
    ),
    documents=(
        "ferry schedule",
        "forest notice",
        "museum catalog",
        "quarry permit",
        "granary tally",
        "signal roster",
    ),
    days=("Moonday", "Tideday", "Fieldday", "Marketday"),
    roles=("ranger", "ferryman", "curator", "foreman", "miller", "warden"),
    actions=(
        "raise the striped flag",
        "wait at the east marker",
        "deliver the tally at noon",
        "leave the gate unlatched",
        "inspect the upper gallery",
        "report after the final bell",
    ),
    distractors=(
        "a sparrow landed on the rail",
        "the noon meal was delayed",
        "two windows stood open",
        "a cart passed the fountain",
        "the clouds moved east",
        "someone swept the front step",
        "the stable lamp burned low",
        "a kettle whistled nearby",
    ),
)


def _combine_lexicons(*lexicons: NeutralLexicon) -> NeutralLexicon:
    return NeutralLexicon(
        **{
            field.name: tuple(
                value
                for lexicon in lexicons
                for value in getattr(lexicon, field.name)
            )
            for field in fields(NeutralLexicon)
        }
    )


DEVELOPMENT_LEXICON = _combine_lexicons(
    TRAIN_LEXICON,
    ADDITIONAL_TRAIN_LEXICON,
)

SUPPLIED_FACT_DEVELOPMENT_QUESTIONS = SUPPLIED_FACT_QUESTIONS + (
    "What hue does {organization} associate with {signal}?",
    "How is the message {signal} indicated visually?",
    "Give the colored indication for {signal}.",
    "Which recorded hue marks {signal}?",
    "In this code, what shade communicates {signal}?",
    "What color marking tells {organization} that {signal}?",
    "Read the record: how is {signal} visually represented?",
    "Which color should be displayed to convey {signal}?",
)

SUPPLIED_FACT_DEVELOPMENT_RESPONSES = SUPPLIED_FACT_RESPONSES + (
    "The recorded hue for {signal} is {color}.",
    "{color} is the visual indication for {signal}.",
    "Display {color} to communicate {signal}.",
    "The record associates {signal} with {color}.",
    "That message is shown in {color}.",
    "The relevant shade is {color}.",
)

SCENE_ROUTE_DEVELOPMENT_QUESTIONS = SCENE_ROUTE_QUESTIONS + (
    "Which path is passable at {location}?",
    "What way remains traversable from {location}?",
    "Which route can carry us onward?",
    "Point out the viable passage.",
    "Which path has not been cut off?",
    "How can we leave without meeting the reported obstruction?",
    "Of the two named ways, which one can we still traverse?",
    "State the passable route from the evidence.",
)

SCENE_ROUTE_DEVELOPMENT_RESPONSES = SCENE_ROUTE_RESPONSES + (
    "The passable route is the {open_route}.",
    "The {open_route} is still traversable.",
    "Use the viable {open_route}.",
    "The evidence directs us through the {open_route}.",
    "We can continue by the {open_route}.",
    "The unobstructed way is the {open_route}.",
)


def _skill_resources(skill: str) -> tuple:
    if skill == "supplied_fact":
        return (
            _supplied_fact_specs,
            SUPPLIED_FACT_DEVELOPMENT_QUESTIONS,
            SUPPLIED_FACT_DEVELOPMENT_RESPONSES,
            SUPPLIED_FACT_TRANSFER_QUESTIONS,
            SUPPLIED_FACT_TRANSFER_RESPONSES,
        )
    if skill == "scene_route":
        return (
            _scene_route_specs,
            SCENE_ROUTE_DEVELOPMENT_QUESTIONS,
            SCENE_ROUTE_DEVELOPMENT_RESPONSES,
            SCENE_ROUTE_TRANSFER_QUESTIONS,
            SCENE_ROUTE_TRANSFER_RESPONSES,
        )
    raise ValueError(f"unknown counterfactual skill: {skill}")


def semantic_transfer_splits(
    train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict[str, tuple[CharacterTrainingRecord, ...]]:
    """Build training plus isolated and combined transfer splits."""

    counts = {
        "train": _positive_integer(
            train_pairs_per_skill,
            "train_pairs_per_skill",
        ),
        "val": _positive_integer(
            validation_pairs_per_skill,
            "validation_pairs_per_skill",
        ),
        "lexical": _positive_integer(
            lexical_pairs_per_skill,
            "lexical_pairs_per_skill",
        ),
        "paraphrase": _positive_integer(
            paraphrase_pairs_per_skill,
            "paraphrase_pairs_per_skill",
        ),
        "transfer": _positive_integer(
            transfer_pairs_per_skill,
            "transfer_pairs_per_skill",
        ),
    }
    split_records = {split: [] for split in SEMANTIC_TRANSFER_SPLITS}

    for skill_index, skill in enumerate(COUNTERFACTUAL_SKILLS):
        (
            spec_builder,
            development_questions,
            development_responses,
            held_out_questions,
            held_out_responses,
        ) = _skill_resources(skill)
        development_specs = spec_builder(
            counts["train"] + counts["val"],
            seed + skill_index * 101,
            DEVELOPMENT_LEXICON,
            development_questions,
            development_responses,
        )
        axis_definitions = {
            "lexical": (
                VALIDATION_LEXICON,
                development_questions,
                development_responses,
                seed + 10_000 + skill_index * 101,
            ),
            "paraphrase": (
                DEVELOPMENT_LEXICON,
                held_out_questions,
                held_out_responses,
                seed + 20_000 + skill_index * 101,
            ),
            "transfer": (
                VALIDATION_LEXICON,
                held_out_questions,
                held_out_responses,
                seed + 30_000 + skill_index * 101,
            ),
        }
        split_records["train"].extend(
            _build_records(
                "train",
                skill,
                development_specs[: counts["train"]],
                development_questions,
                development_responses,
            )
        )
        split_records["val"].extend(
            _build_records(
                "val",
                skill,
                development_specs[counts["train"] :],
                development_questions,
                development_responses,
            )
        )

        for split, (
            lexicon,
            questions,
            responses,
            split_seed,
        ) in axis_definitions.items():
            specs = spec_builder(
                counts[split],
                split_seed,
                lexicon,
                questions,
                responses,
            )
            split_records[split].extend(
                _build_records(
                    split,
                    skill,
                    specs,
                    questions,
                    responses,
                )
            )

    result = {
        split: tuple(split_records[split])
        for split in SEMANTIC_TRANSFER_SPLITS
    }

    for records in result.values():
        validate_counterfactual_pairs(records)

    context_sets = {
        split: {record.context.context_id for record in records}
        for split, records in result.items()
    }

    for index, split in enumerate(SEMANTIC_TRANSFER_SPLITS):
        for other in SEMANTIC_TRANSFER_SPLITS[index + 1 :]:
            if context_sets[split] & context_sets[other]:
                raise RuntimeError(
                    f"counterfactual context leakage: {split}/{other}"
                )

    return result


def _values_in_text(text: str, values: tuple[str, ...]) -> tuple[str, ...]:
    folded = text.casefold()
    return tuple(value for value in values if value.casefold() in folded)


def _evidence_value(text: str, values: tuple[str, ...]) -> str:
    folded = text.casefold()
    counts = {
        value: folded.count(value.casefold())
        for value in values
        if value.casefold() in folded
    }

    if not counts:
        raise ValueError("changed evidence line contains no candidate value")

    maximum = max(counts.values())
    winners = [value for value, count in counts.items() if count == maximum]

    if len(winners) != 1:
        raise ValueError(
            "changed evidence line does not identify one candidate value"
        )

    return winners[0]


def build_semantic_answer_keys(
    splits: dict[str, tuple[CharacterTrainingRecord, ...]],
) -> dict:
    """Derive a structured answer value and pair foil for every row."""

    all_colors = DEVELOPMENT_LEXICON.colors + VALIDATION_LEXICON.colors
    all_routes = DEVELOPMENT_LEXICON.routes + VALIDATION_LEXICON.routes
    entries = {}

    for split in SEMANTIC_TRANSFER_SPLITS:
        records = tuple(splits[split])

        for offset in range(0, len(records), 2):
            first, second = records[offset : offset + 2]
            skill = (
                "scene_route"
                if "scene" in first.behavior_tags
                else "supplied_fact"
            )

            if skill == "supplied_fact":
                expected_values = []

                for record in (first, second):
                    target = record.context.target_response or ""
                    matches = _values_in_text(target, all_colors)

                    if len(matches) != 1:
                        raise ValueError(
                            "supplied-fact target must contain exactly one "
                            f"known color; found {matches!r}"
                        )

                    expected_values.append(matches[0])
            else:
                first_lines = serialize_character_prompt(
                    replace(first.context, target_response=None)
                ).splitlines()
                second_lines = serialize_character_prompt(
                    replace(second.context, target_response=None)
                ).splitlines()
                routes = _values_in_text("\n".join(first_lines), all_routes)

                if len(routes) != 2:
                    raise ValueError(
                        "scene-route prompt must name exactly two routes; "
                        f"found {routes!r}"
                    )

                changed = [
                    (left, right)
                    for left, right in zip(first_lines, second_lines)
                    if left != right
                ]

                if len(changed) != 1:
                    raise ValueError(
                        "answer keys require one changed evidence line"
                    )

                blocked_first = _evidence_value(changed[0][0], routes)
                blocked_second = _evidence_value(changed[0][1], routes)

                expected_values = [
                    next(route for route in routes if route != blocked_first),
                    next(route for route in routes if route != blocked_second),
                ]

            if expected_values[0] == expected_values[1]:
                raise ValueError("counterfactual answer values must differ")

            for record, expected, alternative in (
                (first, expected_values[0], expected_values[1]),
                (second, expected_values[1], expected_values[0]),
            ):
                entries[record.context.context_id] = {
                    "split": split,
                    "conversation_id": record.conversation_id,
                    "skill": skill,
                    "expected_value": expected,
                    "alternative_value": alternative,
                }

    return {
        "schema_version": 1,
        "semantic_transfer_version": SEMANTIC_TRANSFER_VERSION,
        "entries": dict(sorted(entries.items())),
    }


def build_semantic_transfer_dataset(
    output_dir: str | Path,
    train_pairs_per_skill: int = 800,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict:
    """Write the deterministic Phase 28 dataset and semantic answer keys."""

    splits = semantic_transfer_splits(
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
    answer_keys = build_semantic_answer_keys(splits)
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
        "seed": seed,
        "split_strategy": "paired_factorized_lexical_and_paraphrase_transfer",
        "pair_invariant": (
            "adjacent rows share a conversation_id and differ in exactly "
            "one serialized evidence line"
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
