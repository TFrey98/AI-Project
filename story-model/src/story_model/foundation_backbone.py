"""Adapt this project's own from-scratch checkpoint to LanguageBackbone.

docs/phase25_neutral_backbone.md describes three interchangeable backbones
for CharacterChatSession: a scripted test double, a local OpenAI-compatible
instruction server, and "the existing checkpoint generator" for testing
from-scratch model code. The Phase 25 patch shipped the first two; this
module supplies the third, so the same seven-case neutral gate that scores
an instruction model can also score a from-scratch checkpoint like
foundation-v3 directly, without standing up a separate model server.

A TransformerLanguageModel checkpoint has no chat template and no
instruction tuning — it only knows how to continue text. This adapter
renders the message list as a plain transcript ("System: ...\\nUser:
...\\nAssistant:") and asks the model to continue it, then truncates the
continuation at the first sign the model has started hallucinating the
next turn itself (a "User:" or "System:" line), since nothing in a raw
completion model stops it from writing both sides of the conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from story_model.backbones import (
    BackboneResponse,
    ChatMessage,
    GenerationSettings,
)
from story_model.checkpoint import read_checkpoint
from story_model.generate import load_generation_metadata
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything
from story_model.sampling import generate_tokens

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}

_TURN_STOPS = ("\nUser:", "\nSystem:", "\nAssistant:")


def render_transcript(messages: Iterable[ChatMessage]) -> str:
    """Render chat messages as a plain-text transcript for a base LM."""

    lines = [
        f"{_ROLE_LABELS[message.role]}: {message.content}"
        for message in messages
    ]
    lines.append("Assistant:")
    return "\n".join(lines)


def truncate_at_next_turn(continuation: str) -> str:
    """Keep only the assistant's own turn from a raw completion.

    A completion model has no notion of "stop when it's the other
    speaker's turn" — left alone, it will keep writing, including
    fabricating the next "User:" line and answering that too. Cutting at
    the first such marker keeps only what was actually generated in
    response to the real prompt.
    """

    cut = len(continuation)

    for stop in _TURN_STOPS:
        index = continuation.find(stop)

        if index != -1:
            cut = min(cut, index)

    return continuation[:cut].strip()


class FoundationCheckpointBackbone:
    """Generate chat responses from this project's own checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)

        checkpoint = read_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        config, tokenizer = load_generation_metadata(
            checkpoint,
            config_path=None,
        )

        self.tokenizer = tokenizer
        self.block_size = int(config["data"]["block_size"])
        self.device = resolve_device(device)

        model = build_model(
            config["model"],
            vocabulary_size=tokenizer.vocab_size,
            block_size=self.block_size,
            tokenizer=tokenizer,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        messages: Iterable[ChatMessage],
        settings: GenerationSettings,
    ) -> BackboneResponse:
        normalized = tuple(messages)

        if not normalized:
            raise ValueError("at least one chat message is required")

        prompt = render_transcript(normalized)
        prompt_ids = self.tokenizer.encode(prompt)

        # Reserve room for the reply the same way character_chat.py does,
        # rather than letting a long prompt silently consume the entire
        # context window and leave no room to generate anything.
        max_new_tokens = min(
            settings.max_new_tokens,
            max(self.block_size - len(prompt_ids) - 1, 1),
        )

        if len(prompt_ids) >= self.block_size:
            prompt_ids = prompt_ids[-(self.block_size - max_new_tokens):]

        starting_tokens = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        if settings.seed is not None:
            seed_everything(settings.seed)

        # This project's sampler only implements greedy/temperature/top-k
        # (see sampling.py); GenerationSettings' top_p is a nucleus-
        # sampling knob the OpenAI-compatible backend understands but this
        # one does not, so it's accepted for interface compatibility and
        # silently unused here rather than pretending to honor it.
        greedy = settings.temperature <= 0.0
        output = generate_tokens(
            model=self.model,
            starting_tokens=starting_tokens,
            max_new_tokens=max_new_tokens,
            block_size=self.block_size,
            temperature=(
                1.0 if greedy else settings.temperature
            ),
            top_k=40,
            greedy=greedy,
        )

        generated_ids = output[0, len(prompt_ids):].cpu().tolist()
        continuation = self.tokenizer.decode(generated_ids)
        text = truncate_at_next_turn(continuation)

        if not text:
            text = "(the checkpoint generated no usable continuation)"

        return BackboneResponse(
            text=text,
            backend="foundation_checkpoint",
            model=self.checkpoint_path,
            finish_reason="max_tokens",
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated_ids),
            seed=settings.seed,
        )
