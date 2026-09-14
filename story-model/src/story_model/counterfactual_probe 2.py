"""Phase 27 paired evidence-grounding curriculum.

Every adjacent pair has the same serialized prompt except for one evidence
line.  That line changes the correct response, so a model cannot minimize both
examples by learning a question template while ignoring the supplied state.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from story_model.character_data import (
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    MemoryRecord,
    RelationshipState,
    SceneState,
    WorldFact,
    serialize_character_prompt,
)
from story_model.character_training import (
    CHARACTER_DATASET_VERSION,
    CharacterTrainingRecord,
    character_training_record_to_json,
)
from story_model.corpus import sha256_text
from story_model.neutral_instruction import (
    NEUTRAL_PERSONAS,
    TRAIN_LEXICON,
    VALIDATION_LEXICON,
    NeutralLexicon,
)


COUNTERFACTUAL_PROBE_VERSION = 1
COUNTERFACTUAL_SKILLS = ("scene_route", "supplied_fact")
EVIDENCE_CHANNELS = ("world_fact", "scene", "memory", "prior_turn")
COUNTERFACTUAL_ACCEPTANCE_THRESHOLDS = {
    "train": {
        "exact_response_rate": 0.90,
        "counterfactual_pair_exact_rate": 0.80,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.80,
        "mean_context_advantage": 0.15,
    },
    "val": {
        "exact_response_rate": 0.70,
        "counterfactual_pair_exact_rate": 0.50,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.75,
        "mean_context_advantage": 0.12,
    },
    "transfer": {
        "exact_response_rate": 0.50,
        "counterfactual_pair_exact_rate": 0.30,
        "end_stop_rate": 0.95,
        "degenerate_loop_rate_max": 0.05,
        "context_helped_rate": 0.60,
        "mean_context_advantage": 0.08,
    },
}


SUPPLIED_FACT_QUESTIONS = (
    "What color signals {signal} for {organization}?",
    "Which color does {organization} use for {signal}?",
    "For {organization}, what color means {signal}?",
    "If I need to recognize {signal}, which color should I watch for?",
    "Tell me the {organization} color for {signal}.",
    "What is the color assigned to {signal} by {organization}?",
    "Which colored signal represents {signal}?",
    "How is {signal} shown in the {organization} signal code?",
    "What color should appear when {organization} means {signal}?",
    "Identify the color corresponding to {signal}.",
    "In the supplied record, which color denotes {signal}?",
    "What color would confirm {signal} for {organization}?",
    "Which color is associated with {signal} in this system?",
    "Read the evidence and give me the color for {signal}.",
    "What should I look for if the message is {signal}?",
    "According to the current information, what color means {signal}?",
)

SUPPLIED_FACT_RESPONSES = (
    "{color} signals {signal} for {organization}.",
    "The color is {color}.",
    "For {signal}, watch for {color}.",
    "{organization} uses {color} to mean {signal}.",
    "The supplied evidence identifies {color}.",
    "Look for {color}; it indicates {signal}.",
    "The {signal} signal is {color}.",
    "According to the record, {color} means {signal}.",
    "It is {color} for {signal}.",
    "The correct signal color is {color}.",
    "{color} is assigned to {signal}.",
    "Use {color} as the indicator for {signal}.",
)

SUPPLIED_FACT_TRANSFER_QUESTIONS = (
    "Name the hue that communicates {signal} to {organization}.",
    "Under this code, how would {signal} be colored?",
    "Which hue conveys the message {signal}?",
    "Give only the recorded color for {signal}.",
    "How does {organization} visually mark {signal}?",
    "Consult what you were told: what color stands for {signal}?",
    "What colored indication should accompany {signal}?",
    "State the hue linked with {signal} in the evidence.",
)

SUPPLIED_FACT_TRANSFER_RESPONSES = (
    "The recorded hue is {color}.",
    "{signal} is marked in {color}.",
    "Use the {color} indication.",
    "The evidence links {color} with {signal}.",
    "For {organization}, that message appears as {color}.",
    "It should be colored {color}.",
)

SCENE_ROUTE_QUESTIONS = (
    "Which route remains usable at {location}?",
    "Should we take the {route_a} or the {route_b}?",
    "Which of these two routes is open?",
    "What route should we use to continue from {location}?",
    "Which way avoids the stated obstruction?",
    "Read the evidence and choose our usable route.",
    "Between the {route_a} and the {route_b}, where should we go?",
    "Which route is not obstructed?",
    "What is our available way forward?",
    "Which passage should we take from {location}?",
    "Identify the route that remains clear.",
    "Given the obstruction, which route can we still use?",
    "Which route does the current information leave available?",
    "How should we proceed past {location}?",
    "What is the unblocked route?",
    "Choose the viable route named in the scene.",
)

SCENE_ROUTE_RESPONSES = (
    "Use the {open_route}; the {blocked_route} is {obstacle}.",
    "Take the {open_route}.",
    "The {open_route} remains usable.",
    "We should use the {open_route}, not the {blocked_route}.",
    "The available route is the {open_route}.",
    "Proceed through the {open_route}.",
    "The evidence leaves the {open_route} open.",
    "Choose the {open_route}; it avoids the obstruction.",
    "Our clear way forward is the {open_route}.",
    "Go by the {open_route} because the {blocked_route} is {obstacle}.",
    "The {blocked_route} is unavailable, so take the {open_route}.",
    "From {location}, continue along the {open_route}.",
)

SCENE_ROUTE_TRANSFER_QUESTIONS = (
    "Name the passable way out of {location}.",
    "Which listed path has not been cut off?",
    "Point us toward the route the evidence leaves traversable.",
    "Where can we pass without meeting the reported obstruction?",
    "Of the {route_a} and the {route_b}, which can carry us onward?",
    "State the viable path from the information provided.",
    "How do we leave {location} without using the obstructed way?",
    "Which route is presently traversable?",
)

SCENE_ROUTE_TRANSFER_RESPONSES = (
    "The traversable way is the {open_route}.",
    "Follow the {open_route}; the other route is obstructed.",
    "The evidence supports using the {open_route}.",
    "We can pass by the {open_route}.",
    "Avoid the {blocked_route} and use the {open_route}.",
    "Our viable path is the {open_route}.",
)


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _draw_distinct(
    rng: random.Random,
    values: tuple[str, ...],
) -> tuple[str, str]:
    first, second = rng.sample(values, 2)
    return first, second


def _unique_specs(
    count: int,
    seed: int,
    draw: Callable[[random.Random], tuple],
) -> tuple[tuple, ...]:
    rng = random.Random(seed)
    selected = []
    seen = set()
    maximum_attempts = max(1_000, count * 100)

    for _ in range(maximum_attempts):
        spec = draw(rng)

        if spec in seen:
            continue

        seen.add(spec)
        selected.append(spec)

        if len(selected) == count:
            return tuple(selected)

    raise ValueError(
        f"could create only {len(selected)} unique specifications "
        f"for a request of {count}"
    )


def _supplied_fact_specs(
    count: int,
    seed: int,
    lexicon: NeutralLexicon,
    questions: tuple[str, ...],
    responses: tuple[str, ...],
) -> tuple[tuple, ...]:
    def draw(rng: random.Random) -> tuple:
        color_a, color_b = _draw_distinct(rng, lexicon.colors)
        question_index = rng.randrange(len(questions))
        # Exact-match evaluation is meaningful only when response form is
        # inferable from the prompt.  Tie it to the question construction
        # instead of assigning an arbitrary hidden response template.
        return (
            rng.choice(lexicon.organizations),
            rng.choice(lexicon.signals),
            color_a,
            color_b,
            rng.choice(lexicon.locations),
            question_index,
            question_index % len(responses),
            rng.choice(EVIDENCE_CHANNELS),
            rng.choice(lexicon.distractors),
        )

    return _unique_specs(count, seed, draw)


def _scene_route_specs(
    count: int,
    seed: int,
    lexicon: NeutralLexicon,
    questions: tuple[str, ...],
    responses: tuple[str, ...],
) -> tuple[tuple, ...]:
    def draw(rng: random.Random) -> tuple:
        route_a, route_b = _draw_distinct(rng, lexicon.routes)
        question_index = rng.randrange(len(questions))
        return (
            rng.choice(lexicon.locations),
            route_a,
            route_b,
            rng.choice(lexicon.obstacles),
            question_index,
            question_index % len(responses),
            rng.choice(EVIDENCE_CHANNELS),
            rng.choice(lexicon.distractors),
        )

    return _unique_specs(count, seed, draw)


def _record(
    split: str,
    skill: str,
    pair_index: int,
    side: int,
    question: str,
    target: str,
    evidence: str,
    evidence_channel: str,
    location: str,
    generic_situation: str,
    distractor: str,
    tags: tuple[str, ...],
) -> CharacterTrainingRecord:
    character_id, name, summary = NEUTRAL_PERSONAS[
        pair_index % len(NEUTRAL_PERSONAS)
    ]
    pair_id = f"counterfactual_{split}_{skill}_{pair_index:05d}"
    context_id = (
        f"neutral_{split}_{skill}_{pair_index * 2 + side:06d}"
    )
    scene_situation = generic_situation
    memories = ()
    world_facts = (
        WorldFact(
            fact_id=f"{pair_id}_distractor",
            content=distractor,
            known_by=(character_id,),
        ),
    )
    recent_turns = (
        ConversationTurn("user", "user", question),
    )

    if evidence_channel == "world_fact":
        world_facts = (
            WorldFact(
                fact_id=f"{pair_id}_evidence",
                content=evidence,
                known_by=(character_id,),
            ),
            *world_facts,
        )
    elif evidence_channel == "scene":
        scene_situation = f"{generic_situation} Evidence: {evidence}"
    elif evidence_channel == "memory":
        memories = (
            MemoryRecord(
                memory_id=f"{pair_id}_evidence",
                owner_id=character_id,
                content=evidence,
                kind="semantic",
                source="observed",
                importance=4,
            ),
        )
    elif evidence_channel == "prior_turn":
        recent_turns = (
            ConversationTurn(
                "user",
                "user",
                "Review the supplied evidence before I ask.",
            ),
            ConversationTurn("assistant", character_id, evidence),
            ConversationTurn("user", "user", question),
        )
    else:
        raise ValueError(f"unknown evidence channel: {evidence_channel}")

    return CharacterTrainingRecord(
        conversation_id=pair_id,
        behavior_tags=tags,
        context=CharacterContext(
            context_id=context_id,
            character=CharacterProfile(
                character_id=character_id,
                name=name,
                summary=summary,
                traits=("attentive", "grounded"),
                voice=("clear", "concise", "direct"),
                values=("accuracy", "relevance"),
                goals=(
                    "Answer the latest question from supplied evidence.",
                ),
                boundaries=("Does not invent missing information.",),
            ),
            relationship=RelationshipState(
                character_id=character_id,
                participant_id="user",
                participant_name="The Traveler",
                attitude="Neutral and cooperative.",
                trust=20,
                respect=20,
            ),
            scene=SceneState(
                location=location,
                time="Current scene",
                situation=scene_situation,
                participants=(character_id, "user"),
            ),
            memories=memories,
            world_facts=world_facts,
            recent_turns=recent_turns,
            target_response=target,
        ),
    )


def _supplied_fact_pair(
    split: str,
    pair_index: int,
    spec: tuple,
    questions: tuple[str, ...],
    responses: tuple[str, ...],
) -> tuple[CharacterTrainingRecord, CharacterTrainingRecord]:
    (
        organization,
        signal,
        color_a,
        color_b,
        location,
        question_index,
        response_index,
        channel,
        distractor,
    ) = spec
    question = questions[question_index].format(
        organization=organization,
        signal=signal,
    )
    records = []

    for side, color in enumerate((color_a, color_b)):
        evidence = (
            f"{organization} uses {color} to indicate {signal}."
        )
        target = responses[response_index].format(
            organization=organization,
            signal=signal,
            color=color,
        )
        records.append(
            _record(
                split=split,
                skill="supplied_fact",
                pair_index=pair_index,
                side=side,
                question=question,
                target=target,
                evidence=evidence,
                evidence_channel=channel,
                location=location,
                generic_situation=(
                    "The user is checking a signal reference."
                ),
                distractor=distractor,
                tags=("canon", "world_fact", "uncertainty"),
            )
        )

    return records[0], records[1]


def _scene_route_pair(
    split: str,
    pair_index: int,
    spec: tuple,
    questions: tuple[str, ...],
    responses: tuple[str, ...],
) -> tuple[CharacterTrainingRecord, CharacterTrainingRecord]:
    (
        location,
        route_a,
        route_b,
        obstacle,
        question_index,
        response_index,
        channel,
        distractor,
    ) = spec
    question = questions[question_index].format(
        location=location,
        route_a=route_a,
        route_b=route_b,
    )
    records = []

    for side, blocked_route in enumerate((route_a, route_b)):
        open_route = route_b if side == 0 else route_a
        evidence = (
            f"At {location}, the {blocked_route} is {obstacle}."
        )
        target = responses[response_index].format(
            location=location,
            route_a=route_a,
            route_b=route_b,
            blocked_route=blocked_route,
            open_route=open_route,
            obstacle=obstacle,
        )
        records.append(
            _record(
                split=split,
                skill="scene_route",
                pair_index=pair_index,
                side=side,
                question=question,
                target=target,
                evidence=evidence,
                evidence_channel=channel,
                location=location,
                generic_situation=(
                    f"Two routes are available: the {route_a} and "
                    f"the {route_b}."
                ),
                distractor=distractor,
                tags=("scene", "world_fact", "relationship"),
            )
        )

    return records[0], records[1]


def validate_counterfactual_pairs(
    records: tuple[CharacterTrainingRecord, ...],
) -> dict:
    """Prove adjacency and one-line prompt differences for every pair."""

    records = tuple(records)

    if not records or len(records) % 2:
        raise ValueError(
            "counterfactual records require a non-empty even row count"
        )

    skill_counts = Counter()

    for offset in range(0, len(records), 2):
        first, second = records[offset : offset + 2]

        if first.conversation_id != second.conversation_id:
            raise ValueError(
                "counterfactual pair rows must be adjacent and share "
                "conversation_id"
            )
        if first.behavior_tags != second.behavior_tags:
            raise ValueError("counterfactual pair tags must match")
        if (
            first.context.recent_turns[-1].text
            != second.context.recent_turns[-1].text
        ):
            raise ValueError("counterfactual pair questions must match")
        if first.context.target_response == second.context.target_response:
            raise ValueError("counterfactual pair targets must differ")

        first_prompt = serialize_character_prompt(
            replace(first.context, target_response=None)
        ).splitlines()
        second_prompt = serialize_character_prompt(
            replace(second.context, target_response=None)
        ).splitlines()

        if len(first_prompt) != len(second_prompt):
            raise ValueError(
                "counterfactual pair prompts must have equal line counts"
            )

        changed_lines = sum(
            left != right
            for left, right in zip(first_prompt, second_prompt)
        )

        if changed_lines != 1:
            raise ValueError(
                "counterfactual pair prompts must differ in exactly one "
                f"evidence line; observed {changed_lines}"
            )

        skill = "scene_route" if "scene" in first.behavior_tags else (
            "supplied_fact"
        )
        skill_counts[skill] += 1

    return {
        "examples": len(records),
        "pairs": len(records) // 2,
        "skills": dict(sorted(skill_counts.items())),
    }


def _build_records(
    split: str,
    skill: str,
    specs: tuple[tuple, ...],
    questions: tuple[str, ...],
    responses: tuple[str, ...],
) -> tuple[CharacterTrainingRecord, ...]:
    pair_builder = (
        _scene_route_pair if skill == "scene_route" else _supplied_fact_pair
    )
    records = []

    for pair_index, spec in enumerate(specs):
        records.extend(
            pair_builder(
                split,
                pair_index,
                spec,
                questions,
                responses,
            )
        )

    return tuple(records)


def counterfactual_probe_splits(
    train_pairs_per_skill: int = 400,
    validation_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict[str, tuple[CharacterTrainingRecord, ...]]:
    """Build train, compositional validation, and transfer splits."""

    train_count = _positive_integer(
        train_pairs_per_skill,
        "train_pairs_per_skill",
    )
    validation_count = _positive_integer(
        validation_pairs_per_skill,
        "validation_pairs_per_skill",
    )
    transfer_count = _positive_integer(
        transfer_pairs_per_skill,
        "transfer_pairs_per_skill",
    )
    split_records = {"train": [], "val": [], "transfer": []}

    for skill_index, skill in enumerate(COUNTERFACTUAL_SKILLS):
        if skill == "supplied_fact":
            spec_builder = _supplied_fact_specs
            core_questions = SUPPLIED_FACT_QUESTIONS
            core_responses = SUPPLIED_FACT_RESPONSES
            transfer_questions = SUPPLIED_FACT_TRANSFER_QUESTIONS
            transfer_responses = SUPPLIED_FACT_TRANSFER_RESPONSES
        else:
            spec_builder = _scene_route_specs
            core_questions = SCENE_ROUTE_QUESTIONS
            core_responses = SCENE_ROUTE_RESPONSES
            transfer_questions = SCENE_ROUTE_TRANSFER_QUESTIONS
            transfer_responses = SCENE_ROUTE_TRANSFER_RESPONSES

        core_specs = spec_builder(
            train_count + validation_count,
            seed + skill_index * 101,
            TRAIN_LEXICON,
            core_questions,
            core_responses,
        )
        transfer_specs = spec_builder(
            transfer_count,
            seed + 10_000 + skill_index * 101,
            VALIDATION_LEXICON,
            transfer_questions,
            transfer_responses,
        )
        split_records["train"].extend(
            _build_records(
                "train",
                skill,
                core_specs[:train_count],
                core_questions,
                core_responses,
            )
        )
        split_records["val"].extend(
            _build_records(
                "val",
                skill,
                core_specs[train_count:],
                core_questions,
                core_responses,
            )
        )
        split_records["transfer"].extend(
            _build_records(
                "transfer",
                skill,
                transfer_specs,
                transfer_questions,
                transfer_responses,
            )
        )

    result = {
        split: tuple(records) for split, records in split_records.items()
    }

    for records in result.values():
        validate_counterfactual_pairs(records)

    context_sets = {
        split: {record.context.context_id for record in records}
        for split, records in result.items()
    }

    if (
        context_sets["train"] & context_sets["val"]
        or context_sets["train"] & context_sets["transfer"]
        or context_sets["val"] & context_sets["transfer"]
    ):
        raise RuntimeError("counterfactual context leakage between splits")

    return result


def _records_text(records: tuple[CharacterTrainingRecord, ...]) -> str:
    return "".join(
        character_training_record_to_json(record) for record in records
    )


def _split_metadata(
    records: tuple[CharacterTrainingRecord, ...],
    text: str,
) -> dict:
    tags = Counter(tag for record in records for tag in record.behavior_tags)
    skills = Counter(
        "scene_route" if "scene" in record.behavior_tags else "supplied_fact"
        for record in records
    )
    return {
        "examples": len(records),
        "pairs": len(records) // 2,
        "conversations": sorted(
            {record.conversation_id for record in records}
        ),
        "characters": sorted(
            {record.context.character.character_id for record in records}
        ),
        "behavior_tags": {
            tag: tags[tag] for tag in sorted(tags)
        },
        "skills": {
            skill: skills[skill] for skill in sorted(skills)
        },
        "sha256": sha256_text(text),
    }


def build_counterfactual_probe_dataset(
    output_dir: str | Path,
    train_pairs_per_skill: int = 400,
    validation_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict:
    """Write the deterministic three-split counterfactual probe."""

    splits = counterfactual_probe_splits(
        train_pairs_per_skill=train_pairs_per_skill,
        validation_pairs_per_skill=validation_pairs_per_skill,
        transfer_pairs_per_skill=transfer_pairs_per_skill,
        seed=seed,
    )
    texts = {
        split: _records_text(records) for split, records in splits.items()
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, text in texts.items():
        (output_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")

    manifest = {
        "dataset_version": CHARACTER_DATASET_VERSION,
        "counterfactual_probe_version": COUNTERFACTUAL_PROBE_VERSION,
        "seed": seed,
        "split_strategy": (
            "paired_counterfactual_composition_and_transfer"
        ),
        "pair_invariant": (
            "adjacent rows share a conversation_id and differ in exactly "
            "one serialized evidence line"
        ),
        # load_character_dataset_splits trains on train/val and therefore
        # intentionally treats the untouched transfer split separately.
        "source_examples": len(splits["train"]) + len(splits["val"]),
        "transfer_examples": len(splits["transfer"]),
        "train": _split_metadata(splits["train"], texts["train"]),
        "val": _split_metadata(splits["val"], texts["val"]),
        "transfer": _split_metadata(
            splits["transfer"],
            texts["transfer"],
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def counterfactual_acceptance_failures(summaries: dict) -> tuple[str, ...]:
    """Return human-readable failures for a diagnostic summary."""

    failures = []

    for split, thresholds in COUNTERFACTUAL_ACCEPTANCE_THRESHOLDS.items():
        summary = summaries.get(split)

        if not isinstance(summary, dict):
            failures.append(f"missing diagnostic split: {split}")
            continue

        for metric, threshold in thresholds.items():
            if metric.endswith("_max"):
                source_metric = metric[: -len("_max")]
                value = float(summary.get(source_metric, float("inf")))

                if value > threshold:
                    failures.append(
                        f"{split} {source_metric} {value:.3f} exceeds "
                        f"{threshold:.3f}"
                    )
            else:
                value = float(summary.get(metric, float("-inf")))

                if value < threshold:
                    failures.append(
                        f"{split} {metric} {value:.3f} is below "
                        f"{threshold:.3f}"
                    )

    return tuple(failures)
