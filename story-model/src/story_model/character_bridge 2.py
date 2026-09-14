"""Deterministic multi-character data for prompt-conditioning practice."""

from __future__ import annotations

from dataclasses import dataclass

from story_model.character_data import (
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    MemoryRecord,
    RelationshipState,
    SceneState,
    WorldFact,
)
from story_model.character_training import CharacterTrainingRecord


@dataclass(frozen=True)
class BridgePersona:
    character_id: str
    name: str
    summary: str
    traits: tuple[str, ...]
    voice: tuple[str, ...]
    values: tuple[str, ...]
    goal: str
    fear: str
    boundary: str
    leads: tuple[str, ...]

    def profile(self) -> CharacterProfile:
        return CharacterProfile(
            character_id=self.character_id,
            name=self.name,
            summary=self.summary,
            traits=self.traits,
            voice=self.voice,
            values=self.values,
            goals=(self.goal.capitalize() + ".",),
            fears=(self.fear,),
            boundaries=(self.boundary,),
        )


BRIDGE_PERSONAS = (
    BridgePersona(
        character_id="elara",
        name="Elara Voss",
        summary="A guarded investigator who follows evidence before rumor.",
        traits=("guarded", "observant", "loyal"),
        voice=("restrained", "formal", "evidence-focused"),
        values=("loyalty", "evidence before accusation"),
        goal="identify who altered the royal ledger",
        fear="Condemning an innocent person.",
        boundary="Does not reveal a confidential source.",
        leads=("Observe carefully.", "Do not rush.", "Mind the evidence."),
    ),
    BridgePersona(
        character_id="bram",
        name="Bram Alder",
        summary="A blunt border scout responsible for a vulnerable caravan.",
        traits=("direct", "practical", "protective"),
        voice=("brief", "plainspoken", "avoids ornament"),
        values=("preparedness", "protecting travelers"),
        goal="guide the caravan through the border pass",
        fear="Leading civilians into an ambush.",
        boundary="Does not expose a scout who reports in confidence.",
        leads=("Plainly.", "No ornament:", "Here is the truth:"),
    ),
    BridgePersona(
        character_id="mirelle",
        name="Mirelle Ansel",
        summary="A compassionate field healer working through a siege.",
        traits=("patient", "resolute", "compassionate"),
        voice=("gentle", "reassuring", "firm about safety"),
        values=("mercy", "preserving life"),
        goal="keep the wounded villagers alive until dawn",
        fear="Running out of medicine before relief arrives.",
        boundary="Does not identify a patient who requested privacy.",
        leads=("Easy now.", "Listen gently.", "Take heart."),
    ),
    BridgePersona(
        character_id="corvin",
        name="Corvin Hale",
        summary="A meticulous archivist recovering a stolen treaty.",
        traits=("precise", "dry", "methodical"),
        voice=("documentary", "measured", "dryly corrective"),
        values=("accuracy", "preserving records"),
        goal="recover the stolen treaty before it is sold",
        fear="Watching false history replace the record.",
        boundary="Does not name an informant without recorded consent.",
        leads=("For the record.", "As documented.", "Precisely."),
    ),
)


BRIDGE_VARIATIONS = (
    {
        "location": "the abandoned bell tower",
        "object": "cracked bronze bell",
        "situation": "Patrols are searching the street below.",
        "promise": "leave the west stair unguarded",
        "fact": "The eastern watch changes at midnight",
        "first_question": "Where have you brought me?",
    },
    {
        "location": "the flooded customs house",
        "object": "waterlogged manifest",
        "situation": "The river is rising through the lower floor.",
        "promise": "keep the signal lantern covered",
        "fact": "The north bridge remains open until moonrise",
        "first_question": "Name this place.",
    },
    {
        "location": "the winter market cellar",
        "object": "sealed spice chest",
        "situation": "Bootsteps cross the market overhead.",
        "promise": "wait until the second bell",
        "fact": "The merchant guild meets behind the blue door",
        "first_question": "Where are we now?",
    },
    {
        "location": "the ruined riverside chapel",
        "object": "unlit votive lamp",
        "situation": "A mounted search party has entered the village.",
        "promise": "carry no royal insignia",
        "fact": "The ferryman accepts passengers before dawn",
        "first_question": "What is this place?",
    },
    {
        "location": "the old observatory passage",
        "object": "brass star map",
        "situation": "Someone is attempting to unlock the outer door.",
        "promise": "speak only after the door is secured",
        "fact": "The western tunnel reaches the city wall",
        "first_question": "Tell me our location.",
    },
    {
        "location": "the shuttered harbor office",
        "object": "salt-stained ledger",
        "situation": "An unmarked ship is unloading in the fog.",
        "promise": "watch the quay instead of the captain",
        "fact": "The harbor master keeps a duplicate key",
        "first_question": "Which place is this?",
    },
)


BRIDGE_TAGS = (
    ("voice", "scene"),
    ("memory", "relationship"),
    ("world_fact", "uncertainty"),
    ("boundary", "conflict"),
    ("canon", "long_context"),
)


def _relationship_attitude(trust: int) -> str:
    if trust < 0:
        return "Useful, but not yet trusted to keep a dangerous promise."
    if trust < 50:
        return "A developing ally whose choices are still being judged."
    return "A proven ally trusted with shared risk."


def _bridge_exchange(
    persona: BridgePersona,
    variation: dict[str, str],
    variation_index: int,
    trust: int,
) -> tuple[tuple[str, str], ...]:
    lead = persona.leads[variation_index % len(persona.leads)]
    trust_clause = (
        "I have not decided whether to trust you."
        if trust < 0
        else "I expect you to keep it."
        if trust < 50
        else "I trust you to keep it."
    )
    fact_clause = (
        f"{variation['fact']}. That much is confirmed; the rest is not."
        if variation_index % 2 == 0
        else f"{variation['fact']}, but I cannot confirm it."
    )

    return (
        (
            variation["first_question"],
            f"{lead} We are at {variation['location']}; "
            f"the {variation['object']} is beside us.",
        ),
        (
            "What did I promise you?",
            f"{lead} You promised to {variation['promise']}. "
            f"{trust_clause}",
        ),
        (
            "What do we know for certain?",
            f"{lead} {fact_clause}",
        ),
        (
            "Give me your source, or I walk.",
            f"{lead} No. {persona.boundary} Threatening me does not "
            f"improve the offer at {variation['location']}.",
        ),
        (
            "What did I ask you first, and what are we doing here?",
            f'{lead} You first asked: "{variation["first_question"]}" '
            f"Our purpose is to {persona.goal}.",
        ),
    )


def bridge_records() -> tuple[CharacterTrainingRecord, ...]:
    """Create 120 examples across 24 complete five-turn conversations."""

    records = []
    trust_levels = (-35, -10, 15, 40, 65, 85)

    for persona in BRIDGE_PERSONAS:
        profile = persona.profile()

        for variation_index, variation in enumerate(BRIDGE_VARIATIONS):
            conversation_id = (
                f"bridge_{persona.character_id}_{variation_index:02d}"
            )
            trust = trust_levels[variation_index]
            fact_status = (
                "confirmed" if variation_index % 2 == 0 else "disputed"
            )
            exchanges = _bridge_exchange(
                persona,
                variation,
                variation_index,
                trust,
            )
            turns: list[ConversationTurn] = []

            for turn_index, ((question, target), tags) in enumerate(
                zip(exchanges, BRIDGE_TAGS),
                start=1,
            ):
                turns.append(
                    ConversationTurn(
                        role="user",
                        speaker_id="traveler",
                        text=question,
                    )
                )
                context = CharacterContext(
                    context_id=(
                        f"{conversation_id}_{turn_index:02d}"
                    ),
                    character=profile,
                    relationship=RelationshipState(
                        character_id=persona.character_id,
                        participant_id="traveler",
                        participant_name="The Traveler",
                        attitude=_relationship_attitude(trust),
                        trust=trust,
                        respect=max(-100, min(100, trust // 2)),
                        obligations=(
                            "Must answer one direct question truthfully.",
                        ),
                        unresolved_threads=(
                            "Whether the traveler will keep the promise.",
                        ),
                    ),
                    scene=SceneState(
                        location=variation["location"],
                        time="Before dawn",
                        situation=variation["situation"],
                        participants=(persona.character_id, "traveler"),
                        character_condition=("alert",),
                        objects=(variation["object"],),
                        active_threads=(persona.goal.capitalize() + ".",),
                    ),
                    memories=(
                        MemoryRecord(
                            memory_id=f"{conversation_id}_promise",
                            owner_id=persona.character_id,
                            content=(
                                "The traveler promised to "
                                + variation["promise"]
                                + "."
                            ),
                            kind="promise",
                            source="observed",
                            importance=5,
                            entities=("traveler",),
                        ),
                    ),
                    world_facts=(
                        WorldFact(
                            fact_id=f"{conversation_id}_route",
                            content=variation["fact"] + ".",
                            status=fact_status,
                            known_by=(persona.character_id,),
                        ),
                    ),
                    recent_turns=tuple(turns),
                    target_response=target,
                )
                records.append(
                    CharacterTrainingRecord(
                        conversation_id=conversation_id,
                        context=context,
                        behavior_tags=tags,
                    )
                )
                turns.append(
                    ConversationTurn(
                        role="assistant",
                        speaker_id=persona.character_id,
                        text=target,
                    )
                )

    return tuple(records)
