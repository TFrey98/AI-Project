"""Deterministic, character-neutral instruction-tuning curriculum.

The foundation corpus teaches next-token continuation.  These records teach
the same Transformer to treat the structured prompt as information, engage
with the latest user question, and produce only a grounded response.  Training
and validation use disjoint vocabularies and disjoint sentence templates so a
low validation loss cannot come from seeing a paraphrase in both splits.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from story_model.character_data import (
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    MemoryRecord,
    RelationshipState,
    SceneState,
    WorldFact,
)
from story_model.character_training import (
    CHARACTER_DATASET_VERSION,
    CharacterTrainingRecord,
    character_training_record_to_json,
)
from story_model.corpus import sha256_text


NEUTRAL_INSTRUCTION_VERSION = 1
NEUTRAL_SKILLS = (
    "scene_route",
    "supplied_fact",
    "missing_information",
    "multi_turn_memory",
    "contradiction_correction",
    "reference_tracking",
    "cause_and_effect",
    "promise_recall",
    "comparison",
    "privacy_boundary",
)


@dataclass(frozen=True)
class NeutralLexicon:
    names: tuple[str, ...]
    items: tuple[str, ...]
    containers: tuple[str, ...]
    locations: tuple[str, ...]
    routes: tuple[str, ...]
    obstacles: tuple[str, ...]
    colors: tuple[str, ...]
    organizations: tuple[str, ...]
    signals: tuple[str, ...]
    documents: tuple[str, ...]
    days: tuple[str, ...]
    roles: tuple[str, ...]
    actions: tuple[str, ...]
    distractors: tuple[str, ...]


TRAIN_LEXICON = NeutralLexicon(
    names=(
        "Alden",
        "Brina",
        "Cato",
        "Delia",
        "Eamon",
        "Fara",
        "Galen",
        "Hesta",
        "Ivo",
        "Jora",
        "Keris",
        "Lucan",
    ),
    items=(
        "silver compass",
        "brass token",
        "sealed letter",
        "blue ledger",
        "ivory key",
        "canvas satchel",
        "glass vial",
        "copper map case",
        "wool cloak",
        "iron lantern",
        "oak box",
        "green journal",
    ),
    containers=(
        "desk drawer",
        "red cabinet",
        "stone locker",
        "travel chest",
        "upper shelf",
        "canvas pouch",
        "wooden crate",
        "bedside table",
        "locked cupboard",
        "archive case",
        "tool basket",
        "window alcove",
    ),
    locations=(
        "north watchtower",
        "old customs house",
        "lower archive",
        "river warehouse",
        "market cellar",
        "east courtyard",
        "hill observatory",
        "harbor office",
        "stone chapel",
        "border station",
        "mill storehouse",
        "garden pavilion",
    ),
    routes=(
        "eastern tunnel",
        "south stair",
        "covered footbridge",
        "river path",
        "service corridor",
        "north passage",
        "lower gallery",
        "garden gate",
        "old aqueduct",
        "western trail",
        "courtyard door",
        "cellar ramp",
    ),
    obstacles=(
        "barred",
        "flooded",
        "collapsed",
        "blocked by fallen stone",
        "closed for repairs",
        "guarded",
        "buried by snow",
        "damaged by fire",
    ),
    colors=(
        "amber",
        "violet",
        "scarlet",
        "white",
        "green",
        "blue",
        "orange",
        "silver",
    ),
    organizations=(
        "harbor watch",
        "northern patrol",
        "courier guild",
        "river wardens",
        "city archive",
        "market guard",
        "bridge crew",
        "border scouts",
    ),
    signals=(
        "all-clear",
        "evacuation",
        "shift change",
        "safe crossing",
        "medical assistance",
        "closed road",
        "incoming courier",
        "inspection complete",
    ),
    documents=(
        "shipping ledger",
        "watch report",
        "delivery receipt",
        "archive register",
        "inspection form",
        "travel manifest",
        "maintenance log",
        "warehouse inventory",
    ),
    days=(
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ),
    roles=(
        "driver",
        "courier",
        "witness",
        "inspector",
        "porter",
        "guide",
        "clerk",
        "messenger",
    ),
    actions=(
        "cover the signal lamp",
        "wait beside the fountain",
        "lock the lower door",
        "carry no royal insignia",
        "return before sunset",
        "keep the ledger dry",
        "watch the northern road",
        "leave the parcel unopened",
        "meet after the second bell",
        "bring two clean bandages",
        "avoid the market square",
        "send word to the watchtower",
    ),
    distractors=(
        "the weather changed before noon",
        "three carts crossed the square",
        "the kitchen served soup",
        "a bell rang in the distance",
        "the lamps were lit early",
        "a dog barked near the wall",
        "the market closed quietly",
        "rain began after supper",
    ),
)


VALIDATION_LEXICON = NeutralLexicon(
    names=("Merek", "Nadia", "Oren", "Pella", "Quill", "Sabine"),
    items=(
        "bronze sextant",
        "purple ribbon",
        "ceramic flask",
        "linen bundle",
        "black notebook",
        "tin whistle",
    ),
    containers=(
        "map cabinet",
        "granite niche",
        "rope hamper",
        "second bookcase",
        "painted coffer",
        "bench compartment",
    ),
    locations=(
        "cliff beacon",
        "canal tollhouse",
        "orchard shed",
        "west infirmary",
        "clockmaker's loft",
        "lakeside depot",
    ),
    routes=(
        "cliffside ladder",
        "drainage gallery",
        "orchard lane",
        "roof walkway",
        "canal towpath",
        "kitchen passage",
    ),
    obstacles=(
        "sealed by ice",
        "covered by a landslide",
        "closed under quarantine",
        "split by the earthquake",
        "occupied by soldiers",
        "washed away",
    ),
    colors=("gold", "indigo", "crimson", "gray", "teal", "black"),
    organizations=(
        "canal authority",
        "cliff wardens",
        "medical corps",
        "royal post",
        "lake patrol",
        "orchard council",
    ),
    signals=(
        "quarantine lifted",
        "storm warning",
        "doctor requested",
        "mail collected",
        "dock secured",
        "water released",
    ),
    documents=(
        "canal docket",
        "medical chart",
        "post register",
        "dock certificate",
        "orchard account",
        "beacon journal",
    ),
    days=("Firstday", "Secondday", "Thirdday", "Fourthday"),
    roles=("pilot", "surveyor", "physician", "collector", "steward", "scribe"),
    actions=(
        "lower the blue pennant",
        "remain near the canal lock",
        "deliver the medicine unopened",
        "close the orchard shutters",
        "check the beacon at dusk",
        "meet beneath the clock",
    ),
    distractors=(
        "fog settled over the water",
        "the baker opened late",
        "two gulls landed on the roof",
        "the clock stopped briefly",
        "a cart lost one wheel",
        "the wind turned north",
    ),
)


NEUTRAL_PERSONAS = (
    ("neutral_guide", "The Guide", "A clear and attentive guide."),
    ("neutral_clerk", "The Clerk", "A precise and neutral clerk."),
    ("neutral_scout", "The Scout", "A practical and observant scout."),
    ("neutral_keeper", "The Keeper", "A concise and careful keeper."),
)


def _select_combinations(
    pools: tuple[tuple[object, ...], ...],
    count: int,
    seed: int,
) -> tuple[tuple[object, ...], ...]:
    combinations = list(itertools.product(*pools))

    if count > len(combinations):
        raise ValueError(
            f"requested {count} examples from only "
            f"{len(combinations)} combinations"
        )

    random.Random(seed).shuffle(combinations)
    return tuple(combinations[:count])


def _turn(role: str, speaker: str, text: str) -> ConversationTurn:
    return ConversationTurn(role=role, speaker_id=speaker, text=text)


def _record(
    split: str,
    skill: str,
    index: int,
    recent_turns: tuple[ConversationTurn, ...],
    target: str,
    tags: tuple[str, ...],
    location: str,
    situation: str,
    world_facts: tuple[WorldFact, ...] = (),
    memories: tuple[MemoryRecord, ...] = (),
) -> CharacterTrainingRecord:
    character_id, name, summary = NEUTRAL_PERSONAS[
        index % len(NEUTRAL_PERSONAS)
    ]
    context_id = f"neutral_{split}_{skill}_{index:04d}"

    return CharacterTrainingRecord(
        conversation_id=context_id,
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
                goals=("Answer the latest question from supplied information.",),
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
                situation=situation,
                participants=(character_id, "user"),
            ),
            memories=memories,
            world_facts=world_facts,
            recent_turns=recent_turns,
            target_response=target,
        ),
    )


def _known_fact(
    context_id: str,
    suffix: str,
    character_id: str,
    content: str,
    status: str = "confirmed",
) -> WorldFact:
    return WorldFact(
        fact_id=f"{context_id}_{suffix}",
        content=content,
        status=status,
        known_by=(character_id,),
    )


def _character_id(index: int) -> str:
    return NEUTRAL_PERSONAS[index % len(NEUTRAL_PERSONAS)][0]


def _phrase(
    split: str,
    index: int,
    training: tuple[str, ...],
    validation: tuple[str, ...],
) -> str:
    choices = training if split == "train" else validation
    return choices[index % len(choices)]


def _scene_route_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.routes, lexicon.obstacles, lexicon.routes),
        count * 2,
        seed,
    )
    combinations = tuple(
        combo for combo in combinations if combo[0] != combo[2]
    )[:count]

    if len(combinations) != count:
        raise RuntimeError("not enough distinct route combinations")

    records = []

    for index, (closed_route, obstacle, open_route) in enumerate(combinations):
        question = _phrase(
            split,
            index,
            (
                "Can we use the {closed}? Which way remains open?",
                "Is the {closed} usable, and what is the alternative?",
                "Which route can we take instead of the {closed}?",
            ),
            (
                "Which path should we take, given the condition of the {closed}?",
                "Name the available way around the unusable {closed}.",
            ),
        ).format(closed=closed_route)
        target = _phrase(
            split,
            index,
            (
                "No. The {closed} is {obstacle}; use the {open}.",
                "The {closed} is {obstacle}. The {open} remains usable.",
                "Use the {open}, because the {closed} is {obstacle}.",
            ),
            (
                "Take the {open}. The {closed} is {obstacle}.",
                "The usable path is the {open}; the {closed} is {obstacle}.",
            ),
        ).format(closed=closed_route, obstacle=obstacle, open=open_route)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_scene_route_{index:04d}"
        records.append(
            _record(
                split,
                "scene_route",
                index,
                (_turn("user", "user", question),),
                target,
                ("scene", "world_fact"),
                lexicon.locations[index % len(lexicon.locations)],
                (
                    f"The {closed_route} is {obstacle}. The only usable "
                    f"route is the {open_route}."
                ),
                world_facts=(
                    _known_fact(
                        context_id,
                        "route",
                        character_id,
                        f"The {open_route} remains usable.",
                    ),
                ),
            )
        )

    return tuple(records)


def _supplied_fact_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.organizations, lexicon.colors, lexicon.signals),
        count,
        seed,
    )
    records = []

    for index, (organization, color, signal) in enumerate(combinations):
        question = _phrase(
            split,
            index,
            (
                "What color does the {organization} use for {signal}?",
                "Which color means {signal} to the {organization}?",
                "Tell me the {organization}'s {signal} color.",
            ),
            (
                "Identify the color that means {signal} for the {organization}.",
                "How is {signal} shown by the {organization}?",
            ),
        ).format(organization=organization, signal=signal)
        target = _phrase(
            split,
            index,
            (
                "The {organization} uses {color} for {signal}.",
                "{color_cap} means {signal}.",
                "Their {signal} signal is {color}.",
            ),
            (
                "{color_cap} is the {signal} color.",
                "The signal is {color}.",
            ),
        ).format(
            organization=organization,
            color=color,
            color_cap=color.capitalize(),
            signal=signal,
        )
        character_id = _character_id(index)
        context_id = f"neutral_{split}_supplied_fact_{index:04d}"
        fact = f"The {organization} uses {color} as its {signal} signal."
        records.append(
            _record(
                split,
                "supplied_fact",
                index,
                (_turn("user", "user", question),),
                target,
                ("voice", "canon"),
                lexicon.locations[index % len(lexicon.locations)],
                "A signal must be interpreted from the current record.",
                world_facts=(
                    _known_fact(context_id, "signal", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _missing_information_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.documents, lexicon.items, lexicon.days, lexicon.roles),
        count,
        seed,
    )
    records = []

    for index, (document, item, day, role) in enumerate(combinations):
        question = _phrase(
            split,
            index,
            (
                "Who was the {role} for the {item}?",
                "What is the name of the {item}'s {role}?",
                "Does the record identify the {role} connected to the {item}?",
            ),
            (
                "Does the {document} name the {role} responsible for the {item}?",
                "Can you determine the {role}'s identity from this record?",
            ),
        ).format(role=role, item=item, document=document)
        target = _phrase(
            split,
            index,
            (
                "The {document} does not identify the {role}.",
                "The {role}'s name is not recorded.",
                "That information is unavailable; the record does not name the {role}.",
            ),
            (
                "No. The {role}'s identity is not recorded.",
                "The available information does not specify the {role}.",
            ),
        ).format(document=document, role=role)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_missing_information_{index:04d}"
        fact = (
            f"The {document} records the {item} on {day}, but it does "
            f"not name the {role}."
        )
        records.append(
            _record(
                split,
                "missing_information",
                index,
                (_turn("user", "user", question),),
                target,
                ("uncertainty", "world_fact"),
                lexicon.locations[index % len(lexicon.locations)],
                "A record is available, but one requested detail is absent.",
                world_facts=(
                    _known_fact(context_id, "record", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _multi_turn_memory_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.items, lexicon.containers, lexicon.distractors),
        count,
        seed,
    )
    records = []

    for index, (item, container, distractor) in enumerate(combinations):
        first = f"I put the {item} in the {container} before supper."
        question = _phrase(
            split,
            index,
            (
                "Where did I put the {item}?",
                "Where is the {item} I mentioned earlier?",
                "Remind me where I left the {item}.",
            ),
            (
                "Remind me of the location of the {item}.",
                "What earlier location did I give for the {item}?",
            ),
        ).format(item=item)
        target = _phrase(
            split,
            index,
            (
                "You put the {item} in the {container}.",
                "It is in the {container}.",
                "You said the {item} was in the {container}.",
            ),
            (
                "The location you gave was the {container}.",
                "You left it in the {container}.",
            ),
        ).format(item=item, container=container)
        turns = (
            _turn("user", "user", first),
            _turn("assistant", _character_id(index), "Understood."),
            _turn("user", "user", distractor.capitalize() + "."),
            _turn("assistant", _character_id(index), "Noted."),
            _turn("user", "user", question),
        )
        records.append(
            _record(
                split,
                "multi_turn_memory",
                index,
                turns,
                target,
                ("memory", "long_context"),
                lexicon.locations[index % len(lexicon.locations)],
                "The conversation has moved away from an earlier concrete detail.",
            )
        )

    return tuple(records)


def _contradiction_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    name_pairs = tuple(
        (first, second)
        for first in lexicon.names
        for second in lexicon.names
        if first != second
    )
    combinations = _select_combinations(
        (name_pairs, lexicon.items),
        count,
        seed,
    )
    records = []

    for index, (pair, item) in enumerate(combinations):
        former, holder = pair
        question = _phrase(
            split,
            index,
            (
                "{former} still has the {item}, correct?",
                "The {item} is with {former}, isn't it?",
                "Confirm that {former} currently holds the {item}.",
            ),
            (
                "Am I right that the {item} remains with {former}?",
                "Would it be accurate to say {former} has the {item}?",
            ),
        ).format(former=former, item=item)
        target = _phrase(
            split,
            index,
            (
                "No. {holder} currently has the {item}.",
                "That is not correct; {holder} holds the {item}.",
                "The premise is wrong. The {item} is with {holder}.",
            ),
            (
                "That is incorrect; the {item} is with {holder}.",
                "No; the current holder is {holder}.",
            ),
        ).format(holder=holder, item=item)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_contradiction_correction_{index:04d}"
        fact = f"{former} returned the {item} to {holder} before dawn."
        records.append(
            _record(
                split,
                "contradiction_correction",
                index,
                (_turn("user", "user", question),),
                target,
                ("conflict", "canon"),
                lexicon.locations[index % len(lexicon.locations)],
                "The latest question contains a false premise.",
                world_facts=(
                    _known_fact(context_id, "holder", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _reference_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    name_pairs = tuple(
        (first, second)
        for first in lexicon.names
        for second in lexicon.names
        if first != second
    )
    combinations = _select_combinations(
        (name_pairs, lexicon.items, lexicon.containers),
        count,
        seed,
    )
    records = []

    for index, (pair, item, container) in enumerate(combinations):
        giver, receiver = pair
        question = _phrase(
            split,
            index,
            (
                "Where is the {item} now?",
                "Where did {receiver} place the {item}?",
                "What is the {item}'s current location?",
            ),
            (
                "After that exchange, what location contains the {item}?",
                "Track the {item} through the exchange. Where did it end up?",
            ),
        ).format(item=item, receiver=receiver)
        target = _phrase(
            split,
            index,
            (
                "The {item} is in the {container}.",
                "{receiver} placed it in the {container}.",
                "It ended up in the {container}.",
            ),
            (
                "Following the exchange, the {item} is in the {container}.",
                "Its current location is the {container}.",
            ),
        ).format(item=item, receiver=receiver, container=container)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_reference_tracking_{index:04d}"
        fact = (
            f"{giver} gave the {item} to {receiver}. "
            f"{receiver} placed it in the {container}."
        )
        records.append(
            _record(
                split,
                "reference_tracking",
                index,
                (_turn("user", "user", question),),
                target,
                ("long_context", "world_fact"),
                lexicon.locations[index % len(lexicon.locations)],
                "An object's location follows a two-person exchange.",
                world_facts=(
                    _known_fact(context_id, "exchange", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _cause_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.locations, lexicon.routes, lexicon.obstacles),
        count,
        seed,
    )
    records = []

    for index, (destination, route, obstacle) in enumerate(combinations):
        question = _phrase(
            split,
            index,
            (
                "Why can't we reach the {destination} right now?",
                "What has made the {destination} inaccessible?",
                "Explain why travel to the {destination} has stopped.",
            ),
            (
                "What is preventing travel to the {destination}?",
                "State the cause of our inability to reach the {destination}.",
            ),
        ).format(destination=destination)
        target = _phrase(
            split,
            index,
            (
                "We cannot reach it because the only route, the {route}, "
                "is {obstacle}.",
                "The {route} is {obstacle}, leaving no available way there.",
                "Travel is blocked because the sole route is {obstacle}.",
            ),
            (
                "The {route} is {obstacle}, and no alternate route is available.",
                "The only approach is the {route}, which is {obstacle}.",
            ),
        ).format(route=route, obstacle=obstacle)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_cause_and_effect_{index:04d}"
        fact = (
            f"The {route} is the only route to the {destination}; it is "
            f"{obstacle}, and no alternate transport is available."
        )
        records.append(
            _record(
                split,
                "cause_and_effect",
                index,
                (_turn("user", "user", question),),
                target,
                ("scene", "world_fact"),
                destination,
                "Travel is currently interrupted.",
                world_facts=(
                    _known_fact(context_id, "obstacle", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _promise_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.actions, lexicon.locations, lexicon.distractors),
        count,
        seed,
    )
    records = []

    for index, (action, location, distractor) in enumerate(combinations):
        promise = f"I promise to {action}."
        question = _phrase(
            split,
            index,
            (
                "What did I promise to do?",
                "Remind me of my promise.",
                "Which action did I commit to?",
            ),
            (
                "Which commitment did I make earlier?",
                "What obligation did I accept in our earlier exchange?",
            ),
        )
        target = _phrase(
            split,
            index,
            (
                "You promised to {action}.",
                "Your promise was to {action}.",
                "You committed to {action}.",
            ),
            (
                "Your commitment was to {action}.",
                "The obligation you accepted was to {action}.",
            ),
        ).format(action=action)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_promise_recall_{index:04d}"
        turns = (
            _turn("user", "user", promise),
            _turn("assistant", character_id, "I will remember that."),
            _turn("user", "user", distractor.capitalize() + "."),
            _turn("assistant", character_id, "Understood."),
            _turn("user", "user", question),
        )
        memory = MemoryRecord(
            memory_id=f"{context_id}_promise",
            owner_id=character_id,
            content=f"The traveler promised to {action}.",
            kind="promise",
            source="observed",
            importance=4,
            entities=("traveler",),
        )
        records.append(
            _record(
                split,
                "promise_recall",
                index,
                turns,
                target,
                ("relationship", "memory"),
                location,
                "A prior commitment remains relevant.",
                memories=(memory,),
            )
        )

    return tuple(records)


def _comparison_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    route_pairs = tuple(
        (first, second)
        for first in lexicon.routes
        for second in lexicon.routes
        if first != second
    )
    duration_pairs = tuple(
        (first, second)
        for first in range(2, 16)
        for second in range(2, 16)
        if first != second
    )
    combinations = _select_combinations(
        (route_pairs, duration_pairs),
        count,
        seed,
    )
    records = []

    for index, (routes, durations) in enumerate(combinations):
        first_route, second_route = routes
        first_duration, second_duration = durations
        faster_route = (
            first_route
            if first_duration < second_duration
            else second_route
        )
        faster_duration = min(first_duration, second_duration)
        question = _phrase(
            split,
            index,
            (
                "Which route is faster?",
                "Which option takes less time?",
                "Select the shorter journey.",
            ),
            (
                "Choose the route requiring less travel time.",
                "Which route has the lower duration?",
            ),
        )
        target = _phrase(
            split,
            index,
            (
                "The {route} is faster at {duration} minutes.",
                "Take the {route}; it is the shorter option at {duration} minutes.",
                "The lower travel time is {duration} minutes by the {route}.",
            ),
            (
                "The quicker route is the {route}, requiring {duration} minutes.",
                "Choose the {route}; its duration is {duration} minutes.",
            ),
        ).format(route=faster_route, duration=faster_duration)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_comparison_{index:04d}"
        fact = (
            f"The {first_route} takes {first_duration} minutes. "
            f"The {second_route} takes {second_duration} minutes."
        )
        records.append(
            _record(
                split,
                "comparison",
                index,
                (_turn("user", "user", question),),
                target,
                ("scene", "canon"),
                lexicon.locations[index % len(lexicon.locations)],
                "Two available routes have different travel times.",
                world_facts=(
                    _known_fact(context_id, "times", character_id, fact),
                ),
            )
        )

    return tuple(records)


def _boundary_records(
    split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int,
) -> tuple[CharacterTrainingRecord, ...]:
    combinations = _select_combinations(
        (lexicon.names, lexicon.documents, lexicon.items, lexicon.days),
        count,
        seed,
    )
    records = []

    for index, (witness, document, item, day) in enumerate(combinations):
        question = _phrase(
            split,
            index,
            (
                "Tell me the confidential witness's name.",
                "Who is the protected source? Give me the name.",
                "Reveal the identity recorded as confidential.",
            ),
            (
                "Identify the protected source for me.",
                "Disclose the person whose identity is restricted.",
            ),
        )
        public_fact = f"The {item} was recorded on {day}."
        public_detail = public_fact[0].lower() + public_fact[1:]
        target = _phrase(
            split,
            index,
            (
                "I cannot identify the confidential witness. I can confirm that {fact}",
                "The source's identity is protected. The public record "
                "confirms that {fact}",
                "I will not reveal the witness. The available detail is that {fact}",
            ),
            (
                "The source remains protected. The nonconfidential record says {fact}",
                "I cannot disclose that identity; I can state that {fact}",
            ),
        ).format(fact=public_detail)
        character_id = _character_id(index)
        context_id = f"neutral_{split}_privacy_boundary_{index:04d}"
        records.append(
            _record(
                split,
                "privacy_boundary",
                index,
                (_turn("user", "user", question),),
                target,
                ("boundary", "conflict"),
                lexicon.locations[index % len(lexicon.locations)],
                "A user requests protected information from a record.",
                world_facts=(
                    _known_fact(
                        context_id,
                        "public",
                        character_id,
                        public_fact,
                    ),
                ),
                memories=(
                    MemoryRecord(
                        memory_id=f"{context_id}_source",
                        owner_id=character_id,
                        content=(
                            f"{witness} is the confidential witness named "
                            f"in the {document}."
                        ),
                        kind="semantic",
                        source="canon",
                        importance=5,
                        entities=(witness,),
                    ),
                ),
            )
        )

    return tuple(records)


_SKILL_BUILDERS = (
    _scene_route_records,
    _supplied_fact_records,
    _missing_information_records,
    _multi_turn_memory_records,
    _contradiction_records,
    _reference_records,
    _cause_records,
    _promise_records,
    _comparison_records,
    _boundary_records,
)

_SKILL_BUILDERS_BY_NAME = dict(zip(NEUTRAL_SKILLS, _SKILL_BUILDERS))


def neutral_instruction_skill_records(
    skill: str,
    template_split: str,
    lexicon: NeutralLexicon,
    count: int,
    seed: int = 1337,
) -> tuple[CharacterTrainingRecord, ...]:
    """Build one skill while controlling vocabulary and wording separately.

    Phase 32 needs independent lexical and paraphrase axes.  Earlier public
    builders tied the lexicon and sentence-template split together, which made
    that factorization impossible without reaching into private functions.
    """

    if skill not in _SKILL_BUILDERS_BY_NAME:
        raise ValueError(f"unknown neutral instruction skill: {skill}")
    if template_split not in {"train", "val"}:
        raise ValueError("template_split must be 'train' or 'val'")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")

    builder = _SKILL_BUILDERS_BY_NAME[skill]
    return builder(template_split, lexicon, count, seed)


def neutral_instruction_records(
    split: str,
    examples_per_skill: int,
    seed: int = 1337,
) -> tuple[CharacterTrainingRecord, ...]:
    """Generate one explicit split with held-out vocabulary/templates."""

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if (
        isinstance(examples_per_skill, bool)
        or not isinstance(examples_per_skill, int)
        or examples_per_skill < 1
    ):
        raise ValueError("examples_per_skill must be a positive integer")

    lexicon = TRAIN_LEXICON if split == "train" else VALIDATION_LEXICON
    records = []

    for skill_index, builder in enumerate(_SKILL_BUILDERS):
        records.extend(
            builder(
                split,
                lexicon,
                examples_per_skill,
                seed + skill_index * 10_007,
            )
        )

    return tuple(records)


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
        "_".join(record.context.context_id.split("_")[2:-1])
        for record in records
    )
    return {
        "examples": len(records),
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


def build_neutral_instruction_dataset(
    output_dir: str | Path,
    train_examples_per_skill: int = 300,
    validation_examples_per_skill: int = 50,
    seed: int = 1337,
) -> dict:
    """Write deterministic train/validation JSONL plus a strict manifest."""

    training = neutral_instruction_records(
        "train",
        train_examples_per_skill,
        seed,
    )
    validation = neutral_instruction_records(
        "val",
        validation_examples_per_skill,
        seed,
    )
    train_ids = {record.context.context_id for record in training}
    val_ids = {record.context.context_id for record in validation}

    if train_ids & val_ids:
        raise RuntimeError("neutral instruction context leakage")

    training_text = _records_text(training)
    validation_text = _records_text(validation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text(
        training_text,
        encoding="utf-8",
    )
    (output_dir / "val.jsonl").write_text(
        validation_text,
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": CHARACTER_DATASET_VERSION,
        "neutral_instruction_version": NEUTRAL_INSTRUCTION_VERSION,
        "seed": seed,
        "split_strategy": "held_out_lexicon_and_templates",
        "source_examples": len(training) + len(validation),
        "train": _split_metadata(training, training_text),
        "val": _split_metadata(validation, validation_text),
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
