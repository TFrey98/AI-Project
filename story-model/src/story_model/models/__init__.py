"""Construct language models from experiment configurations.

This factory function is the one place that translates a YAML config's
"model:" section into an actual, constructed nn.Module. Keeping this
translation centralized here (rather than having train.py, generate.py,
and evaluate.py each independently parse a model config) is what
guarantees all three scripts always build IDENTICAL architectures from the
same config — critical, since a checkpoint's weights are only meaningful
if the model reconstructed to load them into has exactly the shapes those
weights were trained under.
"""

from torch import nn

from story_model.models.attention import AttentionLanguageModel
from story_model.models.bigram import BigramLanguageModel
from story_model.models.transformer import TransformerLanguageModel


def build_model(
    model_config: dict,
    vocabulary_size: int,
    block_size: int,
) -> nn.Module:
    """Dispatch on model_config["name"] to build one of this project's models.

    vocabulary_size and block_size are passed in SEPARATELY from
    model_config, rather than also being read out of the config dict,
    because they aren't really "model" choices — they're determined by
    the tokenizer (vocabulary_size) and the data config (block_size),
    which are established elsewhere in each calling script. Every other
    architectural knob (embedding_dim, attention_heads, layers, ...) uses
    `.get(key, default)` rather than a required key, so a config file only
    needs to specify the values it wants to override from this project's
    baseline defaults — see configs/*.yaml, where most files only set a
    handful of keys and rely on these defaults for the rest.
    """

    name = model_config["name"]

    if name == "bigram":
        return BigramLanguageModel(vocabulary_size)

    if name == "attention":
        return AttentionLanguageModel(
            vocabulary_size=vocabulary_size,
            block_size=block_size,
            embedding_dim=model_config.get(
                "embedding_dim",
                64,
            ),
            dropout=model_config.get(
                "dropout",
                0.0,
            ),
        )

    if name == "transformer":
        return TransformerLanguageModel(
            vocabulary_size=vocabulary_size,
            block_size=block_size,
            embedding_dim=model_config.get(
                "embedding_dim",
                128,
            ),
            attention_heads=model_config.get(
                "attention_heads",
                4,
            ),
            layers=model_config.get(
                "layers",
                4,
            ),
            feed_forward_dim=model_config.get(
                "feed_forward_dim",
                512,
            ),
            dropout=model_config.get(
                "dropout",
                0.1,
            ),
            # These three knobs are what let configs/*.yaml run controlled
            # architecture ablations (RoPE vs. learned position encoding,
            # RMSNorm vs. LayerNorm, SwiGLU vs. GELU) by changing a single
            # YAML value, with every other hyperparameter — and the random
            # seed's consumption — held identical between runs. See
            # transformer.py's module docstring and each option's class
            # docstring for what each choice actually changes.
            position_encoding=model_config.get(
                "position_encoding",
                "learned",
            ),
            normalization=model_config.get(
                "normalization",
                "layernorm",
            ),
            feed_forward_activation=model_config.get(
                "feed_forward_activation",
                "gelu",
            ),
        )

    raise ValueError(f"Unknown model name: {name!r}")


__all__ = [
    "AttentionLanguageModel",
    "BigramLanguageModel",
    "TransformerLanguageModel",
    "build_model",
]
