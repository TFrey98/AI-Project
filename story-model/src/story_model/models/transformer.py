"""A small decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head attention with a causal future-token mask."""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if embedding_dim % attention_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by attention_heads"
            )

        self.attention_heads = attention_heads
        self.head_dim = embedding_dim // attention_heads

        # One efficient projection produces queries, keys, and values.
        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )

        self.output_projection = nn.Linear(
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

        # Shape allows broadcasting over batches and heads.
        mask = mask.view(1, 1, block_size, block_size)

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, embedding_dim = x.shape

        qkv = self.qkv_projection(x)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # (B, T, C) -> (B, H, T, D)
        queries = queries.view(
            batch,
            sequence,
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        keys = keys.view(
            batch,
            sequence,
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        values = values.view(
            batch,
            sequence,
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Compare every query with every key.
        scores = queries @ keys.transpose(-2, -1)
        scores = scores * (self.head_dim**-0.5)

        mask = self.causal_mask[
            :,
            :,
            :sequence,
            :sequence,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # Each head gathers information from its visible values.
        context = weights @ values

        # (B, H, T, D) -> (B, T, C)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, sequence, embedding_dim)

        output = self.output_projection(context)
        return self.residual_dropout(output)


class FeedForward(nn.Module):
    """Position-wise nonlinear transformation."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """One pre-normalized attention and feed-forward block."""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        feed_forward_dim: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=feed_forward_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x)
        )

        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x


class TransformerLanguageModel(nn.Module):
    """Stacked decoder-only Transformer."""

    def __init__(
        self,
        vocabulary_size: int,
        block_size: int,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        layers: int = 4,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if layers < 1:
            raise ValueError("layers must be at least 1")

        self.block_size = block_size

        self.token_embeddings = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )

        self.position_embeddings = nn.Embedding(
            block_size,
            embedding_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    attention_heads=attention_heads,
                    feed_forward_dim=feed_forward_dim,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.output_projection = nn.Linear(
            embedding_dim,
            vocabulary_size,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, sequence = tokens.shape

        if sequence > self.block_size:
            raise ValueError(
                f"Sequence length {sequence} exceeds "
                f"block size {self.block_size}"
            )

        positions = torch.arange(
            sequence,
            device=tokens.device,
        )

        x = (
            self.token_embeddings(tokens)
            + self.position_embeddings(positions)
        )

        for block in self.blocks:
            x = block(x)

        logits = self.output_projection(
            self.final_norm(x)
        )

        loss = None

        if targets is not None:
            _, _, vocabulary = logits.shape

            loss = F.cross_entropy(
                logits.reshape(
                    batch * sequence,
                    vocabulary,
                ),
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
            visible_tokens = tokens[
                :,
                -self.block_size :,
            ]

            logits, _ = self(visible_tokens)
            final_logits = logits[:, -1, :]

            probabilities = F.softmax(
                final_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1,
            )

        return tokens