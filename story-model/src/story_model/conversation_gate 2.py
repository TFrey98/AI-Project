"""Neutral conversational-competence gate for interchangeable backbones."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from story_model.backbones import (
    BackboneResponse,
    ChatMessage,
    GenerationSettings,
    LanguageBackbone,
)


CONVERSATION_GATE_VERSION = 1


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold()))


@dataclass(frozen=True)
class ConversationGateCase:
    """One simple question with deterministic concept-level checks."""

    case_id: str
    system: str
    messages: tuple[ChatMessage, ...]
    required_concepts: tuple[tuple[str, ...], ...]
    forbidden_phrases: tuple[str, ...] = ()
    manual_criteria: str = "Response is logical and relevant."

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.system.strip():
            raise ValueError("gate system message cannot be empty")
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("gate conversation must end with a user message")
        if not self.required_concepts:
            raise ValueError("gate case needs at least one required concept")
        if any(not group for group in self.required_concepts):
            raise ValueError("required concept groups cannot be empty")

    @property
    def chat_messages(self) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage(role="system", content=self.system),
            *self.messages,
        )


@dataclass(frozen=True)
class ConversationGateResult:
    case_id: str
    response: BackboneResponse
    passed: bool
    missing_concepts: tuple[tuple[str, ...], ...]
    found_forbidden: tuple[str, ...]
    manual_criteria: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "response": self.response.text,
            "backend": self.response.backend,
            "model": self.response.model,
            "finish_reason": self.response.finish_reason,
            "seed": self.response.seed,
            "automatic_pass": self.passed,
            "missing_concepts": [
                list(group) for group in self.missing_concepts
            ],
            "found_forbidden": list(self.found_forbidden),
            "manual_criteria": self.manual_criteria,
        }


def load_conversation_gate(
    path: str | Path,
) -> tuple[ConversationGateCase, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("conversation gate must be a JSON object")

    if data.get("schema_version") != CONVERSATION_GATE_VERSION:
        raise ValueError("unsupported conversation gate schema version")

    raw_cases = data.get("cases")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("conversation gate must contain cases")

    cases = []

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every conversation gate case must be an object")

        raw_messages = raw_case.get("messages")

        if not isinstance(raw_messages, list):
            raise ValueError("gate case messages must be a list")

        raw_concepts = raw_case.get("required_concepts")

        if not isinstance(raw_concepts, list):
            raise ValueError("required_concepts must be a list")

        cases.append(
            ConversationGateCase(
                case_id=str(raw_case.get("case_id", "")),
                system=str(raw_case.get("system", "")),
                messages=tuple(
                    ChatMessage(
                        role=message["role"],
                        content=message["content"],
                    )
                    for message in raw_messages
                ),
                required_concepts=tuple(
                    tuple(str(phrase) for phrase in group)
                    for group in raw_concepts
                ),
                forbidden_phrases=tuple(
                    str(phrase)
                    for phrase in raw_case.get("forbidden_phrases", ())
                ),
                manual_criteria=str(
                    raw_case.get(
                        "manual_criteria",
                        "Response is logical and relevant.",
                    )
                ),
            )
        )

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("conversation gate case_id values must be unique")

    return tuple(cases)


def score_conversation_response(
    case: ConversationGateCase,
    response: BackboneResponse,
) -> ConversationGateResult:
    normalized_response = f" {_normalized(response.text)} "
    missing = tuple(
        group
        for group in case.required_concepts
        if not any(
            f" {_normalized(phrase)} " in normalized_response
            for phrase in group
        )
    )
    forbidden = tuple(
        phrase
        for phrase in case.forbidden_phrases
        if f" {_normalized(phrase)} " in normalized_response
    )
    return ConversationGateResult(
        case_id=case.case_id,
        response=response,
        passed=not missing and not forbidden,
        missing_concepts=missing,
        found_forbidden=forbidden,
        manual_criteria=case.manual_criteria,
    )


def run_conversation_gate(
    backbone: LanguageBackbone,
    cases: Iterable[ConversationGateCase],
    settings: GenerationSettings,
) -> tuple[ConversationGateResult, ...]:
    results = []

    for index, case in enumerate(cases):
        active_settings = settings

        if settings.seed is not None:
            active_settings = GenerationSettings(
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                seed=settings.seed + index,
            )

        response = backbone.generate(
            case.chat_messages,
            active_settings,
        )
        results.append(score_conversation_response(case, response))

    return tuple(results)
