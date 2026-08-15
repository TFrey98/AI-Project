"""A minimal language model with one causal self-attention head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Allow each position to examine itself and earlier positions."""

    def __init__(
        self,
        embedding_dim: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(
                block_size,
                block_size,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length, _ = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores * (self.embedding_dim**-0.5)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        context = weights @ values
        context = self.projection(context)

        return self.residual_dropout(context)


class AttentionLanguageModel(nn.Module):
    """Character model containing one causal attention head."""

    def __init__(
        self,
        vocabulary_size: int,
        block_size: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block_size = block_size

        self.token_embeddings = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )
        self.position_embeddings = nn.Embedding(
            block_size,
            embedding_dim,
        )

        self.pre_attention_norm = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            block_size=block_size,
            dropout=dropout,
        )

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocabulary_size)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = tokens.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block size {self.block_size}"
            )

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        token_vectors = self.token_embeddings(tokens)
        position_vectors = self.position_embeddings(positions)

        x = token_vectors + position_vectors

        normalized = self.pre_attention_norm(x)
        x = x + self.attention(normalized)

        logits = self.output(self.final_norm(x))

        loss = None

        if targets is not None:
            batch, sequence, vocabulary = logits.shape

            loss = F.cross_entropy(
                logits.reshape(batch * sequence, vocabulary),
                targets.reshape(batch * sequence),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            visible_tokens = tokens[:, -self.block_size :]

            logits, _ = self(visible_tokens)
            final_logits = logits[:, -1, :]

            probabilities = F.softmax(final_logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1,
            )

        return tokens