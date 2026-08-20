"""Character-conditioned generation and multi-turn chat state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch import nn

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    END_MARKER,
    CharacterContext,
    ConversationTurn,
    serialize_character_prompt,
)
from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything
from story_model.sampling import sample_next_token


@dataclass(frozen=True)
class CharacterGeneration:
    """One generated response plus reproducibility metadata."""

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    stop_reason: str
    seed: int


@dataclass(frozen=True)
class CharacterRuntime:
    """Loaded model, tokenizer, and limits required for character use."""

    model: nn.Module
    tokenizer: ByteBPETokenizer
    block_size: int
    device: str
    checkpoint_step: int


def load_character_runtime(
    checkpoint_path: Union[str, Path],
    device: str = "auto",
) -> CharacterRuntime:
    """Reconstruct a character model from a self-describing checkpoint."""

    checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    config = extra.get("config")
    tokenizer_data = extra.get("tokenizer")

    if config is None or tokenizer_data is None:
        raise ValueError(
            "character checkpoint must contain config and tokenizer metadata"
        )

    tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("character generation requires byte-BPE")

    missing_tokens = tuple(
        token
        for token in CHARACTER_CONTROL_TOKENS
        if token not in tokenizer.special_token_ids
    )

    if missing_tokens:
        raise ValueError(
            "checkpoint tokenizer is missing character control tokens: "
            + ", ".join(missing_tokens)
        )

    block_size = int(config["data"]["block_size"])
    resolved_device = resolve_device(device)
    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=block_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()

    return CharacterRuntime(
        model=model,
        tokenizer=tokenizer,
        block_size=block_size,
        device=resolved_device,
        checkpoint_step=int(checkpoint.get("step", 0)),
    )


@torch.no_grad()
def generate_character_response(
    model: nn.Module,
    tokenizer: ByteBPETokenizer,
    context: CharacterContext,
    block_size: int,
    device: str,
    max_new_tokens: int = 160,
    seed: int = 1337,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    greedy: bool = False,
) -> CharacterGeneration:
    """Generate only the assistant response and stop at a control token."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_new_tokens >= block_size:
        raise ValueError("max_new_tokens must be smaller than block_size")
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("character generation requires byte-BPE")

    missing_tokens = tuple(
        token
        for token in CHARACTER_CONTROL_TOKENS
        if token not in tokenizer.special_token_ids
    )

    if missing_tokens:
        raise ValueError("tokenizer is missing character control tokens")

    # Reserve room for the entire response. This prevents the character
    # profile at the beginning of the prompt from silently falling out of
    # the model's visible window while the reply is being generated.
    prompt_budget = block_size - max_new_tokens
    prompt = serialize_character_prompt(
        replace(context, target_response=None),
        max_prompt_tokens=prompt_budget,
        token_counter=lambda text: len(tokenizer.encode(text)),
    )
    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )
    control_ids = set(tokenizer.special_token_ids.values())
    end_id = tokenizer.special_token_ids[END_MARKER]
    generated_ids: list[int] = []
    stop_reason = "max_tokens"

    seed_everything(seed)

    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -block_size:])
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        token_id = int(next_token.item())

        if token_id in control_ids:
            stop_reason = (
                "end" if token_id == end_id else "control_token"
            )
            break

        generated_ids.append(token_id)
        tokens = torch.cat((tokens, next_token), dim=1)

    return CharacterGeneration(
        text=tokenizer.decode(generated_ids).strip(),
        token_ids=tuple(generated_ids),
        prompt_tokens=len(prompt_ids),
        stop_reason=stop_reason,
        seed=seed,
    )


class CharacterChatSession:
    """Maintain recent turns while producing one response at a time."""

    def __init__(
        self,
        context: CharacterContext,
        response_generator: Callable[[CharacterContext, int], CharacterGeneration],
    ) -> None:
        if not isinstance(context, CharacterContext):
            raise TypeError("context must be a CharacterContext")

        self._base_context = replace(context, target_response=None)
        self._turns = list(context.recent_turns)
        self._response_generator = response_generator
        self._response_count = 0

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    @property
    def has_pending_user_turn(self) -> bool:
        return bool(self._turns and self._turns[-1].role == "user")

    def respond(self, user_text: Optional[str] = None) -> CharacterGeneration:
        """Add a user turn when supplied, then generate its response."""

        if user_text is not None:
            if self.has_pending_user_turn:
                raise ValueError(
                    "cannot add a user turn before answering the pending turn"
                )

            self._turns.append(
                ConversationTurn(
                    role="user",
                    speaker_id=(
                        self._base_context.relationship.participant_id
                    ),
                    text=user_text,
                )
            )

        if not self.has_pending_user_turn:
            raise ValueError("a user turn is required before responding")

        generation_context = replace(
            self._base_context,
            recent_turns=tuple(self._turns),
        )
        generation = self._response_generator(
            generation_context,
            self._response_count,
        )

        if not generation.text:
            raise RuntimeError(
                "model produced an empty character response; try another seed"
            )

        self._turns.append(
            ConversationTurn(
                role="assistant",
                speaker_id=self._base_context.character.character_id,
                text=generation.text,
            )
        )
        self._response_count += 1
        return generation
