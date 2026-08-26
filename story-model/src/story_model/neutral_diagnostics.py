"""Free-running diagnostics for the neutral instruction curriculum."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Iterable

from story_model.character_data import CharacterContext


if TYPE_CHECKING:
    from story_model.character_training import CharacterTrainingRecord


_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def neutral_skill(record: CharacterTrainingRecord) -> str:
    """Recover a generated curriculum skill from its context identifier."""

    parts = record.context.context_id.split("_")

    if (
        len(parts) < 4
        or parts[0] != "neutral"
        or parts[1]
        not in {"train", "val", "lexical", "paraphrase", "transfer"}
        or not parts[-1].isdigit()
    ):
        raise ValueError(
            "neutral context_id must match neutral_<split>_<skill>_<index>"
        )

    return "_".join(parts[2:-1])


def evenly_spaced_records(
    records: Iterable[CharacterTrainingRecord],
    count: int,
) -> tuple[CharacterTrainingRecord, ...]:
    """Select records across an ordered split instead of taking its prefix."""

    records = tuple(records)

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(records):
        raise ValueError(
            f"requested {count} records from only {len(records)}"
        )

    return tuple(
        records[((2 * index + 1) * len(records)) // (2 * count)]
        for index in range(count)
    )


def stratified_neutral_records(
    records: Iterable[CharacterTrainingRecord],
    examples_per_skill: int,
) -> tuple[CharacterTrainingRecord, ...]:
    """Select an evenly distributed diagnostic sample from every skill."""

    if (
        isinstance(examples_per_skill, bool)
        or not isinstance(examples_per_skill, int)
    ):
        raise TypeError("examples_per_skill must be an integer")
    if examples_per_skill < 1:
        raise ValueError("examples_per_skill must be positive")

    grouped: dict[str, list[CharacterTrainingRecord]] = defaultdict(list)

    for record in records:
        grouped[neutral_skill(record)].append(record)

    if not grouped:
        raise ValueError("neutral diagnostic dataset cannot be empty")

    selected = []

    for skill in sorted(grouped):
        skill_records = grouped[skill]

        if len(skill_records) < examples_per_skill:
            raise ValueError(
                f"skill {skill!r} has only {len(skill_records)} examples; "
                f"requested {examples_per_skill}"
            )

        selected.extend(
            evenly_spaced_records(skill_records, examples_per_skill)
        )

    return tuple(selected)


def stratified_neutral_pairs(
    records: Iterable[CharacterTrainingRecord],
    pairs_per_skill: int,
) -> tuple[CharacterTrainingRecord, ...]:
    """Select complete counterfactual conversations from every skill."""

    if isinstance(pairs_per_skill, bool) or not isinstance(
        pairs_per_skill,
        int,
    ):
        raise TypeError("pairs_per_skill must be an integer")
    if pairs_per_skill < 1:
        raise ValueError("pairs_per_skill must be positive")

    conversations: dict[str, list[CharacterTrainingRecord]] = defaultdict(list)

    for record in records:
        conversations[record.conversation_id].append(record)

    grouped: dict[
        str,
        list[tuple[CharacterTrainingRecord, CharacterTrainingRecord]],
    ] = defaultdict(list)

    for conversation_id, pair in conversations.items():
        if len(pair) != 2:
            raise ValueError(
                f"conversation {conversation_id!r} has {len(pair)} rows; "
                "counterfactual diagnostics require exactly two"
            )

        first_skill = neutral_skill(pair[0])
        second_skill = neutral_skill(pair[1])

        if first_skill != second_skill:
            raise ValueError(
                "counterfactual conversation rows must share one skill"
            )

        grouped[first_skill].append((pair[0], pair[1]))

    selected = []

    for skill in sorted(grouped):
        pairs = grouped[skill]

        if len(pairs) < pairs_per_skill:
            raise ValueError(
                f"skill {skill!r} has only {len(pairs)} pairs; "
                f"requested {pairs_per_skill}"
            )

        for pair in evenly_spaced_records(pairs, pairs_per_skill):
            selected.extend(pair)

    return tuple(selected)


def context_without_evidence(context: CharacterContext) -> CharacterContext:
    """Keep the question and persona while removing task evidence."""

    latest_user_turn = (
        (context.recent_turns[-1],) if context.recent_turns else ()
    )

    return replace(
        context,
        scene=replace(
            context.scene,
            location="unknown location",
            time="unspecified",
            situation="",
            character_condition=(),
            objects=(),
            active_threads=(),
        ),
        memories=(),
        world_facts=(),
        recent_turns=latest_user_turn,
        target_response=None,
    )


def normalized_similarity(left: str, right: str) -> float:
    """Case- and whitespace-insensitive character similarity."""

    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def repetition_metrics(text: str) -> dict[str, int | float | bool]:
    """Measure token runs and repeated trigrams in generated text."""

    words = tuple(word.lower() for word in _WORD_PATTERN.findall(text))

    if not words:
        return {
            "words": 0,
            "unique_word_ratio": 0.0,
            "max_identical_word_run": 0,
            "repeated_trigram_fraction": 0.0,
            "degenerate_loop": False,
        }

    maximum_run = 1
    current_run = 1

    for previous, current in zip(words, words[1:]):
        if current == previous:
            current_run += 1
            maximum_run = max(maximum_run, current_run)
        else:
            current_run = 1

    trigrams = tuple(zip(words, words[1:], words[2:]))
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(
        count - 1 for count in trigram_counts.values() if count > 1
    )
    repeated_fraction = (
        repeated_trigrams / len(trigrams) if trigrams else 0.0
    )
    degenerate_loop = len(words) >= 6 and (
        maximum_run >= 3 or repeated_fraction >= 0.35
    )

    return {
        "words": len(words),
        "unique_word_ratio": len(set(words)) / len(words),
        "max_identical_word_run": maximum_run,
        "repeated_trigram_fraction": repeated_fraction,
        "degenerate_loop": degenerate_loop,
    }


def summarize_diagnostic_rows(rows: Iterable[dict]) -> dict:
    """Aggregate one split's full-context and ablated generations."""

    rows = tuple(rows)

    if not rows:
        raise ValueError("diagnostic rows cannot be empty")

    def fraction(key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows)

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    raw_prompt_tokens = [int(row["raw_prompt_tokens"]) for row in rows]

    summary = {
        "examples": len(rows),
        "skills": dict(sorted(Counter(row["skill"] for row in rows).items())),
        "exact_response_rate": fraction("exact_response"),
        "end_stop_rate": fraction("end_stop"),
        "mean_reference_similarity": mean("reference_similarity"),
        "degenerate_loop_rate": fraction("degenerate_loop"),
        "mean_unique_word_ratio": mean("unique_word_ratio"),
        "context_changed_rate": fraction("context_changed"),
        "context_helped_rate": fraction("context_helped"),
        "mean_context_advantage": mean("context_advantage"),
        "truncated_prompt_rate": fraction("prompt_truncated"),
        "prompt_tokens": {
            "min": min(prompt_tokens),
            "mean": sum(prompt_tokens) / len(prompt_tokens),
            "max": max(prompt_tokens),
        },
        "raw_prompt_tokens": {
            "min": min(raw_prompt_tokens),
            "mean": sum(raw_prompt_tokens) / len(raw_prompt_tokens),
            "max": max(raw_prompt_tokens),
        },
    }

    if all("conversation_id" in row for row in rows):
        conversations: dict[str, list[dict]] = defaultdict(list)

        for row in rows:
            conversations[str(row["conversation_id"])].append(row)

        if conversations and all(
            len(pair) == 2 for pair in conversations.values()
        ):
            summary["counterfactual_pair_exact_rate"] = sum(
                all(bool(row["exact_response"]) for row in pair)
                for pair in conversations.values()
            ) / len(conversations)

    return summary
