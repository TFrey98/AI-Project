"""Structured character context and deterministic prompt serialization."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union


CHARACTER_DATA_VERSION = 1

CHARACTER_MARKER = "<|character|>"
RELATIONSHIP_MARKER = "<|relationship|>"
SCENE_MARKER = "<|scene|>"
WORLD_FACTS_MARKER = "<|world_facts|>"
MEMORIES_MARKER = "<|memories|>"
CONVERSATION_MARKER = "<|conversation|>"
USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
END_MARKER = "<|end|>"

CHARACTER_CONTROL_TOKENS = (
    CHARACTER_MARKER,
    RELATIONSHIP_MARKER,
    SCENE_MARKER,
    WORLD_FACTS_MARKER,
    MEMORIES_MARKER,
    CONVERSATION_MARKER,
    USER_MARKER,
    ASSISTANT_MARKER,
    END_MARKER,
)

MEMORY_KINDS = frozenset(
    {"episodic", "semantic", "promise", "relationship"}
)
MEMORY_SOURCES = frozenset(
    {"observed", "told", "inferred", "canon", "summary"}
)
BELIEF_STATES = frozenset(
    {"believed", "uncertain", "disbelieved"}
)
FACT_STATES = frozenset(
    {"confirmed", "disputed", "unknown"}
)
TURN_ROLES = frozenset({"user", "assistant"})

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _text(value: Any, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")

    cleaned = value.strip()

    if not cleaned and not allow_empty:
        raise ValueError(f"{label} cannot be empty")

    return cleaned


def _identifier(value: Any, label: str) -> str:
    cleaned = _text(value, label)

    if _IDENTIFIER_PATTERN.fullmatch(cleaned) is None:
        raise ValueError(
            f"{label} must contain only letters, numbers, periods, "
            "colons, underscores, or hyphens"
        )

    return cleaned


def _choice(value: Any, label: str, choices: frozenset[str]) -> str:
    cleaned = _text(value, label)

    if cleaned not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{label} must be one of: {expected}")

    return cleaned


def _strings(
    values: Iterable[Any],
    label: str,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings")

    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(
            f"{label} must be a sequence of strings"
        ) from error

    cleaner = _identifier if identifiers else _text
    cleaned = tuple(
        cleaner(value, f"{label} entry") for value in values
    )

    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{label} cannot contain duplicates")

    return cleaned


def _score(
    value: Any,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{label} must be between {minimum} and {maximum}"
        )

    return value


def _records(
    values: Iterable[Any],
    expected_type: type,
    label: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")

    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error

    if not all(isinstance(value, expected_type) for value in values):
        raise TypeError(
            f"every {label} entry must be a {expected_type.__name__}"
        )

    return values


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")

    return value


def _require_unique(values: Iterable[str], label: str) -> None:
    values = tuple(values)

    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} values")


@dataclass(frozen=True)
class CharacterProfile:
    """Stable identity, behavior, voice, and private character canon."""

    character_id: str
    name: str
    summary: str
    traits: tuple[str, ...] = field(default_factory=tuple)
    voice: tuple[str, ...] = field(default_factory=tuple)
    values: tuple[str, ...] = field(default_factory=tuple)
    goals: tuple[str, ...] = field(default_factory=tuple)
    fears: tuple[str, ...] = field(default_factory=tuple)
    boundaries: tuple[str, ...] = field(default_factory=tuple)
    secrets: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "character_id",
            _identifier(self.character_id, "character_id"),
        )
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, "character summary"),
        )

        for attribute in (
            "traits",
            "voice",
            "values",
            "goals",
            "fears",
            "boundaries",
            "secrets",
        ):
            object.__setattr__(
                self,
                attribute,
                _strings(getattr(self, attribute), attribute),
            )


@dataclass(frozen=True)
class RelationshipState:
    """Evolving relationship between the active character and user."""

    character_id: str
    participant_id: str
    participant_name: str
    attitude: str
    trust: int = 0
    affection: int = 0
    respect: int = 0
    fear: int = 0
    obligations: tuple[str, ...] = field(default_factory=tuple)
    unresolved_threads: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "character_id",
            _identifier(self.character_id, "relationship character_id"),
        )
        object.__setattr__(
            self,
            "participant_id",
            _identifier(self.participant_id, "participant_id"),
        )
        object.__setattr__(
            self,
            "participant_name",
            _text(self.participant_name, "participant_name"),
        )
        object.__setattr__(
            self,
            "attitude",
            _text(self.attitude, "relationship attitude"),
        )

        for attribute in ("trust", "affection", "respect"):
            object.__setattr__(
                self,
                attribute,
                _score(getattr(self, attribute), attribute, -100, 100),
            )

        object.__setattr__(
            self,
            "fear",
            _score(self.fear, "fear", 0, 100),
        )
        object.__setattr__(
            self,
            "obligations",
            _strings(self.obligations, "obligations"),
        )
        object.__setattr__(
            self,
            "unresolved_threads",
            _strings(self.unresolved_threads, "unresolved_threads"),
        )


@dataclass(frozen=True)
class SceneState:
    """Information immediately observable in the current scene."""

    location: str
    time: str = "unspecified"
    situation: str = ""
    participants: tuple[str, ...] = field(default_factory=tuple)
    character_condition: tuple[str, ...] = field(default_factory=tuple)
    objects: tuple[str, ...] = field(default_factory=tuple)
    active_threads: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "location",
            _text(self.location, "scene location"),
        )
        object.__setattr__(self, "time", _text(self.time, "scene time"))
        object.__setattr__(
            self,
            "situation",
            _text(self.situation, "scene situation", allow_empty=True),
        )
        object.__setattr__(
            self,
            "participants",
            _strings(
                self.participants,
                "scene participants",
                identifiers=True,
            ),
        )

        for attribute in (
            "character_condition",
            "objects",
            "active_threads",
        ):
            object.__setattr__(
                self,
                attribute,
                _strings(getattr(self, attribute), attribute),
            )


@dataclass(frozen=True)
class MemoryRecord:
    """One character-owned memory or subjective belief."""

    memory_id: str
    owner_id: str
    content: str
    kind: str = "episodic"
    source: str = "observed"
    belief: str = "believed"
    importance: int = 3
    entities: tuple[str, ...] = field(default_factory=tuple)
    shared_with: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _identifier(self.memory_id, "memory_id"),
        )
        object.__setattr__(
            self,
            "owner_id",
            _identifier(self.owner_id, "memory owner_id"),
        )
        object.__setattr__(
            self,
            "content",
            _text(self.content, "memory content"),
        )
        object.__setattr__(
            self,
            "kind",
            _choice(self.kind, "memory kind", MEMORY_KINDS),
        )
        object.__setattr__(
            self,
            "source",
            _choice(self.source, "memory source", MEMORY_SOURCES),
        )
        object.__setattr__(
            self,
            "belief",
            _choice(self.belief, "memory belief", BELIEF_STATES),
        )
        object.__setattr__(
            self,
            "importance",
            _score(self.importance, "memory importance", 1, 5),
        )
        object.__setattr__(
            self,
            "entities",
            _strings(self.entities, "memory entities"),
        )
        object.__setattr__(
            self,
            "shared_with",
            _strings(
                self.shared_with,
                "memory shared_with",
                identifiers=True,
            ),
        )

        if self.owner_id in self.shared_with:
            raise ValueError(
                "memory owner_id cannot also appear in shared_with"
            )

    def is_visible_to(self, character_id: str) -> bool:
        return (
            character_id == self.owner_id
            or character_id in self.shared_with
        )


@dataclass(frozen=True)
class WorldFact:
    """Objective or disputed world state with scoped knowledge."""

    fact_id: str
    content: str
    status: str = "confirmed"
    public: bool = False
    known_by: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fact_id",
            _identifier(self.fact_id, "fact_id"),
        )
        object.__setattr__(
            self,
            "content",
            _text(self.content, "world fact content"),
        )
        object.__setattr__(
            self,
            "status",
            _choice(self.status, "world fact status", FACT_STATES),
        )

        if not isinstance(self.public, bool):
            raise TypeError("world fact public must be a boolean")

        object.__setattr__(
            self,
            "known_by",
            _strings(
                self.known_by,
                "world fact known_by",
                identifiers=True,
            ),
        )

    def is_visible_to(self, character_id: str) -> bool:
        return self.public or character_id in self.known_by


@dataclass(frozen=True)
class ConversationTurn:
    """One complete turn retained in recent conversational context."""

    role: str
    speaker_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role",
            _choice(self.role, "turn role", TURN_ROLES),
        )
        object.__setattr__(
            self,
            "speaker_id",
            _identifier(self.speaker_id, "turn speaker_id"),
        )
        object.__setattr__(self, "text", _text(self.text, "turn text"))


@dataclass(frozen=True)
class CharacterContext:
    """Complete context used to produce one character response."""

    context_id: str
    character: CharacterProfile
    relationship: RelationshipState
    scene: SceneState
    memories: tuple[MemoryRecord, ...] = field(default_factory=tuple)
    world_facts: tuple[WorldFact, ...] = field(default_factory=tuple)
    recent_turns: tuple[ConversationTurn, ...] = field(
        default_factory=tuple
    )
    target_response: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_id",
            _identifier(self.context_id, "context_id"),
        )

        if not isinstance(self.character, CharacterProfile):
            raise TypeError("character must be a CharacterProfile")
        if not isinstance(self.relationship, RelationshipState):
            raise TypeError("relationship must be a RelationshipState")
        if not isinstance(self.scene, SceneState):
            raise TypeError("scene must be a SceneState")

        object.__setattr__(
            self,
            "memories",
            _records(self.memories, MemoryRecord, "memories"),
        )
        object.__setattr__(
            self,
            "world_facts",
            _records(self.world_facts, WorldFact, "world_facts"),
        )
        object.__setattr__(
            self,
            "recent_turns",
            _records(
                self.recent_turns,
                ConversationTurn,
                "recent_turns",
            ),
        )

        if self.relationship.character_id != self.character.character_id:
            raise ValueError(
                "relationship character_id must match the active character"
            )

        _require_unique(
            (memory.memory_id for memory in self.memories),
            "memory_id",
        )
        _require_unique(
            (fact.fact_id for fact in self.world_facts),
            "fact_id",
        )
        self._validate_turns()

        if self.target_response is not None:
            object.__setattr__(
                self,
                "target_response",
                _text(self.target_response, "target_response"),
            )

    def _validate_turns(self) -> None:
        previous_role = None

        for turn in self.recent_turns:
            expected_speaker = (
                self.relationship.participant_id
                if turn.role == "user"
                else self.character.character_id
            )

            if turn.speaker_id != expected_speaker:
                raise ValueError(
                    f"{turn.role} turn speaker_id must be "
                    f"{expected_speaker!r}"
                )
            if turn.role == previous_role:
                raise ValueError("recent conversation roles must alternate")

            previous_role = turn.role

        if self.recent_turns and self.recent_turns[-1].role != "user":
            raise ValueError(
                "the most recent conversation turn must be a user turn"
            )

    @property
    def visible_memories(self) -> tuple[MemoryRecord, ...]:
        character_id = self.character.character_id
        return tuple(
            memory
            for memory in self.memories
            if memory.is_visible_to(character_id)
        )

    @property
    def visible_world_facts(self) -> tuple[WorldFact, ...]:
        character_id = self.character.character_id
        return tuple(
            fact
            for fact in self.world_facts
            if fact.is_visible_to(character_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHARACTER_DATA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterContext":
        data = _mapping(data, "character context")
        version = data.get("schema_version")

        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != CHARACTER_DATA_VERSION
        ):
            raise ValueError(
                "unsupported character data schema version: "
                f"{version!r}"
            )

        return cls(
            context_id=data["context_id"],
            character=CharacterProfile(
                **_mapping(data["character"], "character")
            ),
            relationship=RelationshipState(
                **_mapping(data["relationship"], "relationship")
            ),
            scene=SceneState(**_mapping(data["scene"], "scene")),
            memories=tuple(
                MemoryRecord(**_mapping(item, "memory"))
                for item in data.get("memories", ())
            ),
            world_facts=tuple(
                WorldFact(**_mapping(item, "world fact"))
                for item in data.get("world_facts", ())
            ),
            recent_turns=tuple(
                ConversationTurn(
                    **_mapping(item, "conversation turn")
                )
                for item in data.get("recent_turns", ())
            ),
            target_response=data.get("target_response"),
        )


def character_context_to_json(context: CharacterContext) -> str:
    if not isinstance(context, CharacterContext):
        raise TypeError("context must be a CharacterContext")

    return json.dumps(
        context.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def character_context_from_json(text: str) -> CharacterContext:
    if not isinstance(text, str):
        raise TypeError("character context JSON must be a string")

    return CharacterContext.from_dict(json.loads(text))


def save_character_context(
    context: CharacterContext,
    path: Union[str, Path],
) -> None:
    Path(path).write_text(
        character_context_to_json(context),
        encoding="utf-8",
    )


def load_character_context(
    path: Union[str, Path],
) -> CharacterContext:
    return character_context_from_json(
        Path(path).read_text(encoding="utf-8")
    )


def _escape_control_markers(text: str) -> str:
    # Prevent user text from becoming a control marker while keeping it
    # visually unchanged for a person reading the serialized prompt.
    return text.replace("<|", "<​|")


def _structured_text(text: str) -> str:
    return " ".join(
        _escape_control_markers(text).split()
    )


def _append_values(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:
    if values:
        lines.append(
            f"{label}: "
            + "; ".join(
                _structured_text(value)
                for value in values
            )
        )


def _render_character(context: CharacterContext) -> list[str]:
    character = context.character
    lines = [
        f"id: {character.character_id}",
        f"name: {_structured_text(character.name)}",
        f"summary: {_structured_text(character.summary)}",
    ]

    for label in (
        "traits",
        "voice",
        "values",
        "goals",
        "fears",
        "boundaries",
        "secrets",
    ):
        _append_values(lines, label, getattr(character, label))

    return lines


def _render_relationship(context: CharacterContext) -> list[str]:
    relationship = context.relationship
    lines = [
        "user: "
        f"{_structured_text(relationship.participant_name)} "
        f"({relationship.participant_id})",
        f"attitude: {_structured_text(relationship.attitude)}",
        "scores: "
        f"trust={relationship.trust}; "
        f"affection={relationship.affection}; "
        f"respect={relationship.respect}; "
        f"fear={relationship.fear}",
    ]
    _append_values(
        lines,
        "obligations",
        relationship.obligations,
    )
    _append_values(
        lines,
        "unresolved",
        relationship.unresolved_threads,
    )
    return lines


def _render_scene(context: CharacterContext) -> list[str]:
    scene = context.scene
    lines = [
        f"location: {_structured_text(scene.location)}",
        f"time: {_structured_text(scene.time)}",
    ]

    if scene.situation:
        lines.append(
            f"situation: {_structured_text(scene.situation)}"
        )

    _append_values(lines, "present", scene.participants)
    _append_values(
        lines,
        "condition",
        scene.character_condition,
    )
    _append_values(lines, "objects", scene.objects)
    _append_values(lines, "unresolved", scene.active_threads)
    return lines


def _render_world_facts(context: CharacterContext) -> list[str]:
    if not context.visible_world_facts:
        return ["none"]

    return [
        f"{fact.status}: {_structured_text(fact.content)}"
        for fact in context.visible_world_facts
    ]


def _render_memories(context: CharacterContext) -> list[str]:
    if not context.visible_memories:
        return ["none"]

    lines = []

    for memory in context.visible_memories:
        metadata = [
            memory.kind,
            memory.source,
            memory.belief,
            f"importance={memory.importance}",
        ]

        if memory.entities:
            metadata.append(
                "entities=" + ",".join(memory.entities)
            )

        lines.append(
            "["
            + "; ".join(metadata)
            + "] "
            + _structured_text(memory.content)
        )

    return lines


def _render_prompt_prefix(context: CharacterContext) -> str:
    lines = [CHARACTER_MARKER]
    lines.extend(_render_character(context))
    lines.append(RELATIONSHIP_MARKER)
    lines.extend(_render_relationship(context))
    lines.append(SCENE_MARKER)
    lines.extend(_render_scene(context))
    lines.append(WORLD_FACTS_MARKER)
    lines.extend(_render_world_facts(context))
    lines.append(MEMORIES_MARKER)
    lines.extend(_render_memories(context))
    lines.append(CONVERSATION_MARKER)
    return "\n".join(lines) + "\n"


def _render_turn(turn: ConversationTurn) -> str:
    marker = USER_MARKER if turn.role == "user" else ASSISTANT_MARKER
    return f"{marker}\n{_escape_control_markers(turn.text)}\n"


def _count_tokens(
    text: str,
    token_counter: Callable[[str], int],
) -> int:
    count = token_counter(text)

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("token_counter must return an integer")
    if count < 0:
        raise ValueError("token_counter cannot return a negative count")

    return count


def serialize_character_prompt(
    context: CharacterContext,
    max_prompt_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> str:
    """Render a prompt, dropping only complete oldest turns if needed."""

    if not isinstance(context, CharacterContext):
        raise TypeError("context must be a CharacterContext")
    if (max_prompt_tokens is None) != (token_counter is None):
        raise ValueError(
            "max_prompt_tokens and token_counter must be provided together"
        )

    prefix = _render_prompt_prefix(context)
    response_cue = f"{ASSISTANT_MARKER}\n"
    rendered_turns = [_render_turn(turn) for turn in context.recent_turns]

    if max_prompt_tokens is None:
        return prefix + "".join(rendered_turns) + response_cue

    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
    ):
        raise TypeError("max_prompt_tokens must be an integer")
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")

    assert token_counter is not None
    fixed_prompt = prefix + response_cue

    if _count_tokens(fixed_prompt, token_counter) > max_prompt_tokens:
        raise ValueError(
            "character, state, and response cue exceed the prompt "
            "token budget"
        )

    selected_turns = rendered_turns[:]

    while len(selected_turns) > 1:
        candidate = prefix + "".join(selected_turns) + response_cue

        if _count_tokens(candidate, token_counter) <= max_prompt_tokens:
            break

        selected_turns.pop(0)

    prompt = prefix + "".join(selected_turns) + response_cue

    if _count_tokens(prompt, token_counter) > max_prompt_tokens:
        raise ValueError(
            "the most recent conversation turn exceeds the prompt "
            "token budget"
        )

    return prompt


def serialize_character_training_text(
    context: CharacterContext,
    max_prompt_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> str:
    """Render prompt, target character response, and end marker."""

    if context.target_response is None:
        raise ValueError(
            "character training context requires target_response"
        )

    prompt = serialize_character_prompt(
        context,
        max_prompt_tokens=max_prompt_tokens,
        token_counter=token_counter,
    )
    response = _escape_control_markers(context.target_response)
    return f"{prompt}{response}\n{END_MARKER}\n"
