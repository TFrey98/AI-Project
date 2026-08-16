"""Token selection and autoregressive decoding policies."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Choose one token from a batch of next-token logits."""

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (batch, vocabulary)"
        )

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    vocabulary_size = logits.shape[-1]

    if top_k is not None and not (
        1 <= top_k <= vocabulary_size
    ):
        raise ValueError(
            "top_k must be between 1 and the vocabulary size"
        )

    if greedy:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    scaled_logits = logits / temperature

    if top_k is not None:
        top_values, top_indices = torch.topk(
            scaled_logits,
            k=top_k,
            dim=-1,
        )

        filtered_logits = torch.full_like(
            scaled_logits,
            float("-inf"),
        )

        scaled_logits = filtered_logits.scatter(
            dim=-1,
            index=top_indices,
            src=top_values,
        )

    probabilities = F.softmax(
        scaled_logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    starting_tokens: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Autoregressively extend a batch of starting tokens."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if block_size < 1:
        raise ValueError("block_size must be positive")

    if (
        starting_tokens.ndim != 2
        or starting_tokens.shape[1] < 1
    ):
        raise ValueError(
            "starting_tokens must contain at least one token"
        )

    tokens = starting_tokens

    for _ in range(max_new_tokens):
        visible_tokens = tokens[:, -block_size:]

        logits, _ = model(visible_tokens)
        final_logits = logits[:, -1, :]

        next_token = sample_next_token(
            final_logits,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )

        tokens = torch.cat(
            (tokens, next_token),
            dim=1,
        )

    return tokens