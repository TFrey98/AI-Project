"""Interchangeable language backbones for character conversations.

The character system should not depend on one model implementation.  This
module defines the small request/response contract shared by deterministic
tests and coherent local instruction models.  The HTTP implementation is
deliberately restricted to loopback addresses so character context and memory
cannot be sent to a remote service by accidentally changing a URL.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


CHAT_ROLES = frozenset({"system", "user", "assistant"})


def _clean_text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{label} cannot be empty")

    return cleaned


@dataclass(frozen=True)
class ChatMessage:
    """One role-tagged message accepted by an instruction model."""

    role: str
    content: str

    def __post_init__(self) -> None:
        role = _clean_text(self.role, "message role")

        if role not in CHAT_ROLES:
            expected = ", ".join(sorted(CHAT_ROLES))
            raise ValueError(f"message role must be one of: {expected}")

        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "content",
            _clean_text(self.content, "message content"),
        )


@dataclass(frozen=True)
class GenerationSettings:
    """Provider-neutral decoding settings used by conversation backbones."""

    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = 1337

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


@dataclass(frozen=True)
class BackboneResponse:
    """A normalized response returned by any language backbone."""

    text: str
    backend: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _clean_text(self.text, "response"))
        object.__setattr__(
            self,
            "backend",
            _clean_text(self.backend, "backend"),
        )
        object.__setattr__(self, "model", _clean_text(self.model, "model"))


class LanguageBackbone(Protocol):
    """The complete interface required by the character runtime."""

    def generate(
        self,
        messages: Iterable[ChatMessage],
        settings: GenerationSettings,
    ) -> BackboneResponse:
        """Generate one assistant response for a complete chat history."""


class ScriptedBackbone:
    """Deterministic response queue for exact system-level tests."""

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[tuple[ChatMessage, ...], GenerationSettings]] = []

    def generate(
        self,
        messages: Iterable[ChatMessage],
        settings: GenerationSettings,
    ) -> BackboneResponse:
        normalized = tuple(messages)

        if not normalized:
            raise ValueError("at least one chat message is required")

        if not self._responses:
            raise RuntimeError("scripted backbone has no responses remaining")

        self.calls.append((normalized, settings))
        return BackboneResponse(
            text=self._responses.pop(0),
            backend="scripted",
            model="deterministic-test-double",
            seed=settings.seed,
        )


def _is_loopback_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)

    if parsed.scheme not in {"http", "https"}:
        return False

    if parsed.username is not None or parsed.password is not None:
        return False

    hostname = parsed.hostname

    if hostname is None:
        return False

    if hostname.casefold() == "localhost":
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class LocalOpenAIBackbone:
    """Call a local OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = _clean_text(model, "model")
        self.endpoint = _clean_text(endpoint, "endpoint")

        if not _is_loopback_endpoint(self.endpoint):
            raise ValueError(
                "local backbone endpoint must use localhost or a "
                "loopback IP address"
            )

        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")

        self.timeout_seconds = float(timeout_seconds)

    def generate(
        self,
        messages: Iterable[ChatMessage],
        settings: GenerationSettings,
    ) -> BackboneResponse:
        normalized = tuple(messages)

        if not normalized:
            raise ValueError("at least one chat message is required")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in normalized
            ],
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "stream": False,
        }

        if settings.seed is not None:
            payload["seed"] = settings.seed

        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"local model server returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "could not reach the local model server at "
                f"{self.endpoint}: {error.reason}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "local model server returned invalid JSON"
            ) from error

        try:
            choice = response_data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "local model server response has no assistant message"
            ) from error

        usage = response_data.get("usage", {})
        return BackboneResponse(
            text=text,
            backend="local_openai_compatible",
            model=self.model,
            finish_reason=str(choice.get("finish_reason", "unknown")),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(
                usage.get("completion_tokens")
            ),
            seed=settings.seed,
        )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value
