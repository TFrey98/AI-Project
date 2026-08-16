"""A small decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RotaryPositionEmbedding(nn.Module):
    """Rotate query and key pairs according to token position."""

    def __init__(
        self,
        head_dim: int,
        block_size: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even attention head dimension"
            )

        if block_size < 1:
            raise ValueError("block_size must be positive")

        inverse_frequencies = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            block_size,
            dtype=torch.float32,
        )

        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        self.register_buffer(
            "cosines",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sines",
            angles.sin(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.shape[-2]

        if sequence > self.cosines.shape[0]:
            raise ValueError(
                f"Sequence length {sequence} exceeds "
                f"RoPE cache size {self.cosines.shape[0]}"
            )

        cosines = self.cosines[:sequence].to(
            device=x.device,
            dtype=x.dtype,
        )
        sines = self.sines[:sequence].to(
            device=x.device,
            dtype=x.dtype,
        )

        # Broadcast across batch and attention-head dimensions.
        cosines = cosines.view(1, 1, sequence, -1)
        sines = sines.view(1, 1, sequence, -1)

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(mean_square + self.eps)

        return x * self.weight


def _build_norm(
    normalization: str,
    embedding_dim: int,
) -> nn.Module:
    if normalization == "layernorm":
        return nn.LayerNorm(embedding_dim)

    if normalization == "rmsnorm":
        return RMSNorm(embedding_dim)

    raise ValueError(
        "normalization must be 'layernorm' or 'rmsnorm'"
    )


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head attention with a causal future-token mask."""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        block_size: int,
        dropout: float,
        position_encoding: str = "learned",
    ) -> None:
        super().__init__()

        if embedding_dim % attention_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by attention_heads"
            )

        self.attention_heads = attention_heads
        self.head_dim = embedding_dim // attention_heads

        if position_encoding not in {"learned", "rope"}:
            raise ValueError(
                "position_encoding must be 'learned' or 'rope'"
            )

        self.rotary = (
            RotaryPositionEmbedding(
                head_dim=self.head_dim,
                block_size=block_size,
            )
            if position_encoding == "rope"
            else None
        )

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

        if self.rotary is not None:
            queries = self.rotary(queries)
            keys = self.rotary(keys)

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


def parameter_matched_swiglu_dim(
    embedding_dim: int,
    gelu_hidden_dim: int,
) -> int:
    """Choose a SwiGLU width with nearly the same parameters as GELU."""

    if embedding_dim < 1 or gelu_hidden_dim < 1:
        raise ValueError(
            "feed-forward dimensions must be positive"
        )

    # GELU parameters: 2*d*h + h + d
    # SwiGLU parameters: 3*d*m + 2*m + d
    numerator = gelu_hidden_dim * (
        2 * embedding_dim + 1
    )
    denominator = 3 * embedding_dim + 2

    return max(
        1,
        round(numerator / denominator),
    )


class SwiGLUFeedForward(nn.Module):
    """Gated feed-forward network using the SiLU activation."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.gate_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
        )
        self.value_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
        )
        self.output_projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(
            self.gate_projection(x)
        )
        values = self.value_projection(x)

        return self.dropout(
            self.output_projection(
                gated * values
            )
        )


def _build_feed_forward(
    activation: str,
    embedding_dim: int,
    feed_forward_dim: int,
    dropout: float,
) -> nn.Module:
    if activation == "gelu":
        return FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=feed_forward_dim,
            dropout=dropout,
        )

    if activation == "swiglu":
        hidden_dim = parameter_matched_swiglu_dim(
            embedding_dim,
            feed_forward_dim,
        )

        return SwiGLUFeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError(
        "feed_forward_activation must be 'gelu' or 'swiglu'"
    )


class TransformerBlock(nn.Module):
    """One pre-normalized attention and feed-forward block."""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        feed_forward_dim: int,
        block_size: int,
        dropout: float,
        position_encoding: str,
        normalization: str = "layernorm",
        feed_forward_activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.attention_norm = _build_norm(
            normalization,
            embedding_dim,
        )
        self.feed_forward_norm = _build_norm(
            normalization,
            embedding_dim,
        )

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            block_size=block_size,
            dropout=dropout,
            position_encoding=position_encoding,
        )

        self.feed_forward = _build_feed_forward(
            activation=feed_forward_activation,
            embedding_dim=embedding_dim,
            feed_forward_dim=feed_forward_dim,
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
        position_encoding: str = "learned",
        normalization: str = "layernorm",
        feed_forward_activation: str = "gelu",
    ) -> None:
        super().__init__()

        if layers < 1:
            raise ValueError("layers must be at least 1")

        if position_encoding not in {"learned", "rope"}:
            raise ValueError(
                "position_encoding must be 'learned' or 'rope'"
            )

        if normalization not in {"layernorm", "rmsnorm"}:
            raise ValueError(
                "normalization must be 'layernorm' or 'rmsnorm'"
            )

        if feed_forward_activation not in {"gelu", "swiglu"}:
            raise ValueError(
                "feed_forward_activation must be 'gelu' or 'swiglu'"
            )

        self.block_size = block_size
        self.position_encoding = position_encoding
        self.feed_forward_activation = feed_forward_activation

        self.token_embeddings = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )

        # Construct this table for both variants so the shared weights
        # consume the same random-number sequence during initialization.
        # The RoPE variant removes it immediately after initialization.
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
                    position_encoding=position_encoding,
                    normalization=normalization,
                    feed_forward_activation=feed_forward_activation,
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = _build_norm(
            normalization,
            embedding_dim,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            vocabulary_size,
        )

        self.apply(self._initialize_weights)

        if position_encoding == "rope":
            self.position_embeddings = None

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

        x = self.token_embeddings(tokens)

        if self.position_embeddings is not None:
            positions = torch.arange(
                sequence,
                device=tokens.device,
            )

            x = x + self.position_embeddings(positions)

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