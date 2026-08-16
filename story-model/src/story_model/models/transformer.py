"""A small decoder-only Transformer language model.

This is the same architecture family as GPT: token embeddings, a stack of
attention + feed-forward blocks with residual connections, and a final
linear layer that scores every vocabulary token as the "next token."

The file also implements several interchangeable *building blocks*
(position encodings, normalization layers, feed-forward variants) that can
be swapped via config rather than by editing code. This mirrors how real
research/production codebases evolve: you rarely settle on one architecture
forever, so the model is built to let each design choice vary independently
and be compared experimentally (see configs/transformer_rope*.yaml,
transformer_bpe*.yaml, etc. for the ablations this project ran).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — encodes token position by rotating
    query/key vectors instead of adding a learned position vector.

    Why this exists: the original Transformer (and this project's "learned"
    mode) gives every position its own trainable embedding vector. That
    works, but it has two weaknesses relevant to a hobby-scale model like
    this one:

    1. It costs a whole embedding table (block_size x embedding_dim
       parameters) and that table has no way to generalize past the
       longest sequence length seen during training.
    2. It encodes ABSOLUTE position ("this is token #37"), but what
       attention actually needs to reason about is usually RELATIVE
       position ("this token is 3 words after that one").

    RoPE instead rotates each (query, key) vector by an angle that depends
    on its position in the sequence. The key mathematical trick: when you
    take the dot product of two rotated vectors (which is exactly what
    attention scoring does), the result only depends on the ANGLE BETWEEN
    them — i.e. the relative distance between the two positions — not on
    either position's absolute value. So relative position falls out of
    the attention math for free, with zero extra parameters, and the same
    rotation formula can be applied to sequence lengths never seen during
    training (this is why RoPE-based models can often be "stretched" to
    longer contexts more gracefully than learned position embeddings).
    """

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

        # RoPE treats each head's dimensions as head_dim/2 independent 2D
        # planes and rotates each plane by a different frequency. Low-index
        # pairs rotate fast (they distinguish nearby positions), high-index
        # pairs rotate slowly (they distinguish far-apart positions) — the
        # same "different clock speeds" idea as the sinusoidal position
        # encoding in the original "Attention Is All You Need" paper.
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

        # angles[position, pair] = position * inverse_frequencies[pair]
        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        # register_buffer (not nn.Parameter) because these are precomputed,
        # non-trainable lookup tables, not weights the optimizer should
        # touch. persistent=False keeps them out of the checkpoint file
        # entirely, since they're cheap to regenerate from block_size/head_dim
        # and don't need to round-trip through training resumption.
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

        # Apply a standard 2D rotation matrix to each (even, odd) pair of
        # dimensions: [cos -sin; sin cos] @ [even; odd]. This is the same
        # rotation you'd use to spin a point around the origin in geometry
        # class — here it's spinning vectors in each 2D "plane" of the head
        # by an angle proportional to token position.
        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    LayerNorm (the classic choice) recenters activations to zero mean AND
    rescales them to unit variance, using both a learned scale and a
    learned bias. RMSNorm skips the recentering step entirely: it only
    rescales by the root-mean-square of the activations. The intuition
    (from the original RMSNorm paper) is that the *rescaling* is what
    actually stabilizes training — the *recentering* contributes little,
    so cutting it saves computation (no mean subtraction) and parameters
    (no bias term, no per-feature mean to track). This is one of the
    changes that made LLaMA-family models faster to train than earlier
    GPT-style models without hurting quality.

    Compare to LayerNorm's formula: (x - mean(x)) / std(x) * weight + bias
    RMSNorm's formula is simpler:    x / rms(x) * weight
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # eps prevents division by zero if an input happens to be all
        # zeros (or very close to it) — a small numerical-stability guard
        # you'll see in almost every normalization layer.
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        # rsqrt(v) = 1/sqrt(v), computed directly (rather than sqrt then
        # divide) because that's the numerically standard, slightly faster
        # way to normalize by a magnitude.
        x = x * torch.rsqrt(mean_square + self.eps)

        return x * self.weight


def _build_norm(
    normalization: str,
    embedding_dim: int,
) -> nn.Module:
    """Factory for the two normalization layers this model supports.

    Keeping this as a small dictionary-like dispatch (rather than an
    if/elif sprinkled through the model) means adding a third
    normalization scheme later only touches this one function plus the
    config validation list — the rest of the model doesn't need to know
    which concrete class it's using, only that it behaves like an
    nn.Module that maps (batch, seq, dim) -> (batch, seq, dim).
    """

    if normalization == "layernorm":
        return nn.LayerNorm(embedding_dim)

    if normalization == "rmsnorm":
        return RMSNorm(embedding_dim)

    raise ValueError(
        "normalization must be 'layernorm' or 'rmsnorm'"
    )


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head attention with a causal future-token mask.

    Attention, in one sentence: for every position, produce an output
    vector that's a weighted average of every OTHER position's "value"
    vector, where the weights come from how well that position's "query"
    matches each other position's "key." This is how a Transformer lets
    token N "look at" earlier tokens to decide what it means in context,
    without recurrence (no RNN-style step-by-step state) and without
    convolution (no fixed local window) — every position can attend to
    every earlier position in a single matrix multiply.

    "Multi-head" means we don't do this once with the full embedding_dim;
    we split it into `attention_heads` smaller, independent attention
    computations (each of size head_dim = embedding_dim / attention_heads)
    run in parallel, then concatenate their outputs. Each head can learn to
    specialize — e.g. one head might track subject/verb agreement, another
    might track quotation-mark pairing — because they have separate query/
    key/value projections and never see each other's intermediate scores.

    "Causal" means position N is only allowed to attend to positions
    <= N, never anything later. This is what makes the model usable for
    left-to-right generation: at inference time we genuinely don't have
    the future tokens yet, so the model must be trained under the same
    constraint or it will have learned to "cheat" by peeking ahead.
    """

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

        # RoPE is applied per-head, right here, to queries and keys only
        # (never to values) — it's a way of tagging *where* a token is,
        # not *what* information it carries.
        self.rotary = (
            RotaryPositionEmbedding(
                head_dim=self.head_dim,
                block_size=block_size,
            )
            if position_encoding == "rope"
            else None
        )

        # One efficient projection produces queries, keys, and values.
        # Splitting this into three separate nn.Linear layers would be
        # mathematically identical, but one matmul over 3x the output
        # width is faster on GPU/MPS than three smaller matmuls, since it
        # better saturates the hardware's parallelism.
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

        # torch.tril = lower-triangular: mask[i, j] is True when j <= i,
        # i.e. "position i is allowed to see position j." This is the
        # matrix form of the causal (no-peeking-ahead) constraint.
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

        # (B, T, C) -> (B, H, T, D): split the embedding dimension into
        # `attention_heads` independent heads of size head_dim, and move
        # the head dimension next to batch so every head's attention math
        # below runs as one big batched matrix multiply instead of a
        # Python loop over heads.
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

        # Compare every query with every key: scores[.., i, j] is how much
        # position i's query matches position j's key, for every (i, j)
        # pair at once. Scaling by 1/sqrt(head_dim) keeps the dot products
        # from growing with head_dim (larger dimensions naturally produce
        # larger dot products), which would otherwise push softmax into a
        # saturated, poorly-differentiated regime and starve gradients —
        # this is the "Scaled" in "Scaled Dot-Product Attention."
        scores = queries @ keys.transpose(-2, -1)
        scores = scores * (self.head_dim**-0.5)

        mask = self.causal_mask[
            :,
            :,
            :sequence,
            :sequence,
        ]

        # Setting disallowed positions to -inf *before* softmax guarantees
        # they become exactly 0 probability after softmax, rather than
        # some small-but-nonzero probability that would leak future
        # information (however slightly) into the model's predictions.
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # Each head gathers information from its visible values: this is
        # the actual "attend" step — position i's new representation is a
        # weighted sum of every visible position's value vector.
        context = weights @ values

        # (B, H, T, D) -> (B, T, C): undo the earlier split, so downstream
        # layers see one embedding_dim-sized vector per position again,
        # now enriched with context gathered from every head.
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, sequence, embedding_dim)

        output = self.output_projection(context)
        return self.residual_dropout(output)


class FeedForward(nn.Module):
    """Position-wise nonlinear transformation (the classic GELU MLP).

    Attention lets positions exchange information; the feed-forward layer
    is where the model actually does per-position "thinking" with that
    information — attention has no nonlinearity of its own (it's just
    weighted averaging), so without this block the whole network would
    collapse into one big linear function no matter how many layers deep.
    It's called "position-wise" because the same weights are applied
    independently to every position — there's no mixing across positions
    here (that's attention's job).

    The hidden_dim is typically 4x embedding_dim: expand into a wider
    space, apply a nonlinearity, then project back down. This "expand,
    nonlinear, contract" shape is standard across nearly all Transformer
    variants — it gives the block enough capacity to represent complex
    per-position functions.
    """

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
    """Choose a SwiGLU width with nearly the same parameters as GELU.

    SwiGLU (below) needs THREE weight matrices per block instead of GELU's
    two (gate, value, and output projections vs. up and down projections).
    If you gave SwiGLU the same hidden_dim as GELU, it would end up with
    ~50% more parameters — and then any quality difference you measured
    between the two wouldn't tell you whether SwiGLU is a genuinely better
    activation, or just "more parameters is better." This function solves
    for the SwiGLU hidden_dim that makes total parameter counts roughly
    equal, so configs/transformer_bpe_swiglu.yaml and
    configs/transformer_bpe_gelu_matched.yaml are a fair, apples-to-apples
    ablation of the activation function alone. This kind of "control for
    parameter count" discipline is standard practice whenever ML
    literature claims one architectural choice beats another.
    """

    if embedding_dim < 1 or gelu_hidden_dim < 1:
        raise ValueError(
            "feed-forward dimensions must be positive"
        )

    # GELU parameters: 2*d*h + h + d   (two Linear layers, d<->h, plus biases)
    # SwiGLU parameters: 3*d*m + 2*m + d  (three Linear layers of width m)
    # Setting these equal and solving for m gives the formula below.
    numerator = gelu_hidden_dim * (
        2 * embedding_dim + 1
    )
    denominator = 3 * embedding_dim + 2

    return max(
        1,
        round(numerator / denominator),
    )


class SwiGLUFeedForward(nn.Module):
    """Gated feed-forward network using the SiLU activation (as in LLaMA/PaLM).

    Instead of one path through a single nonlinearity (GELU's
    Linear -> GELU -> Linear), SwiGLU computes TWO parallel projections of
    the input: a "gate" (passed through SiLU) and a "value" (left linear),
    then multiplies them elementwise before projecting back down. The gate
    acts like a learned, per-feature, input-dependent on/off switch that
    modulates how much of the value signal passes through — this extra
    multiplicative interaction tends to let the network represent richer
    functions per parameter than a single nonlinearity can, which is why
    swapping GELU for SwiGLU is one of the most consistent "free" quality
    wins found in later Transformer research (see the ablation in
    configs/transformer_rope_swiglu.yaml vs.
    configs/transformer_rope_gelu_matched.yaml).
    """

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
        # SiLU(x) = x * sigmoid(x), a smooth curve that's ~0 for very
        # negative inputs and ~identity for positive inputs — "SwiGLU" is
        # a portmanteau of "Swish" (SiLU's other name) + "GLU" (Gated
        # Linear Unit, the gate-times-value pattern used here).
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
        # feed_forward_dim is the config's intended GELU-equivalent width;
        # translate it to a parameter-matched SwiGLU width so switching
        # `feed_forward_activation` in a config doesn't silently change
        # the model's parameter budget too.
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
    """One pre-normalized attention and feed-forward block.

    "Pre-normalized" (pre-norm) means normalization is applied BEFORE each
    sub-layer, not after: x = x + Attention(Norm(x)), rather than the
    original Transformer paper's post-norm: x = Norm(x + Attention(x)).
    Pre-norm keeps a clean, unmodified residual stream (the running `x`)
    flowing all the way from the first block to the last, with each block
    only ever *adding* a correction to it. That makes gradients flow back
    through many stacked blocks much more reliably during training —
    post-norm Transformers are notoriously harder to train deep without
    careful learning-rate warmup, while pre-norm is far more forgiving.
    This is why virtually every modern large language model uses pre-norm.
    """

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

        # Each sub-layer gets its OWN normalization layer with its own
        # learned scale — sharing one norm between attention and
        # feed-forward would force both sub-layers to live with the same
        # rescaling, even though they tend to want different activation
        # magnitudes.
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
        # The "+" here is the residual (skip) connection: each sub-layer
        # learns a correction to ADD to x, rather than replacing x
        # outright. This means even an untrained (near-zero-output)
        # sub-layer leaves x mostly intact, which is a big part of why
        # very deep Transformer stacks are trainable at all — information
        # and gradients have a direct, unimpeded path through every block.
        x = x + self.attention(
            self.attention_norm(x)
        )

        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x


class TransformerLanguageModel(nn.Module):
    """Stacked decoder-only Transformer.

    "Decoder-only" (as opposed to the original Transformer's
    encoder-decoder design used for translation) means there's a single
    stack of causal self-attention blocks that both reads the prompt and
    generates the continuation — the same architecture family as GPT.
    There's no separate "encoder" half; every position simply attends to
    everything at-or-before it, which is all you need for
    predict-the-next-token language modeling.
    """

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

        # token_embeddings turns each token id into a learned vector — the
        # model's "vocabulary" of meaning, before any context is applied.
        self.token_embeddings = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )

        # Construct this table for both variants so the shared weights
        # consume the same random-number sequence during initialization.
        # The RoPE variant removes it immediately after initialization.
        #
        # Why bother: this project runs controlled comparisons between
        # "learned" and "rope" position encodings (see
        # test_position_variants_share_initial_weights_and_rng and
        # configs/transformer_rope*.yaml). If constructing a RoPE model
        # consumed a *different* number of random values than a learned
        # model, every other weight in the model would end up initialized
        # to different random values too, and any quality difference we
        # measured between the two runs could just be initialization luck
        # rather than a real effect of the position encoding choice. Always
        # building this table keeps both variants' random-number streams
        # in lockstep, so the ablation isolates the one variable it's
        # supposed to test.
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

        # One final normalization before the output projection — standard
        # in pre-norm Transformers, since the residual stream is otherwise
        # never normalized on its way out, only inside each block.
        self.final_norm = _build_norm(
            normalization,
            embedding_dim,
        )

        # Projects each position's final embedding_dim-sized vector into
        # one raw score ("logit") per vocabulary token — the model's
        # prediction for "what comes next," before softmax turns those
        # scores into probabilities.
        self.output_projection = nn.Linear(
            embedding_dim,
            vocabulary_size,
        )

        self.apply(self._initialize_weights)

        if position_encoding == "rope":
            self.position_embeddings = None

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        # Small, fixed-std (0.02) Gaussian initialization is the standard
        # GPT-2-style init. Weights need to start small: too large and
        # early activations/gradients can blow up before training has a
        # chance to correct them; exactly zero would make every neuron in
        # a layer compute the identical function (a symmetry that gradient
        # descent alone can never break). Biases start at exactly zero,
        # since there's no equivalent symmetry problem for them.
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

            # Additive position encoding: each position's embedding is
            # nudged by a learned vector unique to its index, so
            # "the cat sat" and "sat the cat" produce different
            # representations even though they contain the same tokens.
            # (When position_encoding == "rope", this step is skipped
            # entirely — position information is injected later, inside
            # attention, by rotating queries/keys instead.)
            x = x + self.position_embeddings(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.output_projection(
            self.final_norm(x)
        )

        loss = None

        if targets is not None:
            _, _, vocabulary = logits.shape

            # Cross-entropy compares the model's predicted probability
            # distribution over the vocabulary against the single correct
            # next token at every position simultaneously — this one
            # number is what train.py's optimizer actually minimizes.
            # reshape() flattens (batch, sequence) into one long list of
            # independent predictions, since cross_entropy scores each
            # prediction independently regardless of which sequence or
            # position it came from.
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
        """Autoregressive generation: predict one token, append it, repeat.

        This is the simple/naive generation loop (uniform random sampling
        from the full softmax distribution every step, no temperature or
        top-k control). story_model.sampling.generate_tokens is the
        production version used by the `generate` CLI, with configurable
        temperature/top-k/greedy decoding — this method mainly exists for
        quick manual testing and for the unit tests in test_transformer.py.
        """

        for _ in range(max_new_tokens):
            # A Transformer has no memory between calls — every step
            # re-runs the full forward pass over all currently visible
            # tokens. Cropping to the most recent block_size tokens is
            # required because position encodings and the causal mask are
            # only defined up to block_size; older tokens beyond that
            # window are simply forgotten, the same "context window" limit
            # every Transformer-based model has.
            visible_tokens = tokens[
                :,
                -self.block_size :,
            ]

            logits, _ = self(visible_tokens)

            # Only the prediction at the LAST position is a genuine
            # "what comes next" forecast — every earlier position's
            # prediction was already consumed when the model was trained
            # on this same sequence, so we ignore them here.
            final_logits = logits[:, -1, :]

            probabilities = F.softmax(
                final_logits,
                dim=-1,
            )

            # Sampling (rather than always taking the highest-probability
            # token) is what makes generated text varied instead of
            # deterministic and repetitive; see sampling.py for the
            # temperature/top-k/greedy tradeoffs around this choice.
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1,
            )

        return tokens
