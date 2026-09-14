"""A minimal language model with one causal self-attention head.

This model sits between BigramLanguageModel (no context at all — see
bigram.py) and TransformerLanguageModel (many layers, multi-head
attention, RoPE/RMSNorm/SwiGLU options — see transformer.py) in this
project's step-by-step build-up. It's the smallest possible model that
demonstrates the actual mechanism that makes Transformers work — a single
self-attention head letting each position gather information from earlier
positions — without any of the scaling machinery (multiple heads, stacked
layers, alternate position encodings) that a production-sized model adds
on top. Read this file first if attention itself is the part that's new;
read transformer.py once this one makes sense.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Allow each position to examine itself and earlier positions.

    See transformer.py's MultiHeadCausalSelfAttention for the fuller
    explanation of the query/key/value mechanism and the causal mask —
    this class is the single-head special case of exactly the same idea:
    one query projection, one key projection, one value projection (here
    kept as three separate nn.Linear layers, rather than one combined
    projection split apart, purely because there's no efficiency need to
    combine them at this small scale), and the same masked, scaled
    dot-product attention computation.
    """

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

        # The causal mask: position i may attend to position j only when
        # j <= i (torch.tril keeps the lower triangle, including the
        # diagonal). Precomputed once here rather than recomputed every
        # forward pass, and stored as a non-trainable buffer since it's a
        # fixed structural constraint, not a learned weight.
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

        # scores[i, j] = how strongly position i's query matches position
        # j's key — one number per (query position, key position) pair,
        # for every position in the sequence at once.
        scores = queries @ keys.transpose(-2, -1)

        # Scaling by 1/sqrt(embedding_dim) prevents the dot products from
        # growing large as embedding_dim grows, which would otherwise push
        # softmax below into an overconfident, gradient-starved regime —
        # the same reasoning as transformer.py's per-head scaling, just
        # applied to the full embedding_dim here since there's only one
        # head.
        scores = scores * (self.embedding_dim**-0.5)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        # -inf before softmax guarantees exactly zero probability for
        # future positions after softmax — no future information can leak
        # through, even slightly.
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # The actual "attend" step: each position's new representation is
        # a weighted average of every visible position's value vector,
        # weighted by how strongly that position's query matched.
        context = weights @ values
        context = self.projection(context)

        return self.residual_dropout(context)


class AttentionLanguageModel(nn.Module):
    """Character model containing one causal attention head.

    Structurally this is: embed tokens, add a LEARNED position embedding
    (see transformer.py's RotaryPositionEmbedding docstring for why later
    experiments moved to rotary position encoding instead), normalize, run
    one attention block with a residual connection, normalize again, then
    project to vocabulary logits. It's a one-block, one-head version of
    TransformerLanguageModel's general pre-norm + residual pattern.
    """

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

        # Additive position encoding: without this, "cat sat" and "sat
        # cat" would embed identically, since attention alone has no
        # inherent notion of order — it just weighs positions by content
        # similarity. Adding a position-dependent vector to each token's
        # embedding is what gives the model something to key that
        # ordering information off of.
        x = token_vectors + position_vectors

        # Pre-norm residual pattern: x = x + Attention(Norm(x)), the same
        # "add a correction to an untouched running total" shape used
        # throughout transformer.py — see TransformerBlock's docstring for
        # why this specific ordering (normalize-then-sublayer, then add)
        # matters for training deep stacks reliably.
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
            # Crop to the model's fixed context window — position
            # embeddings and the causal mask only exist up to block_size,
            # so older tokens beyond that window are simply dropped from
            # what the model can see.
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
