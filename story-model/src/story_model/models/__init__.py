"""Construct language models from experiment configurations."""

from torch import nn

from story_model.models.attention import AttentionLanguageModel
from story_model.models.bigram import BigramLanguageModel
from story_model.models.transformer import TransformerLanguageModel


def build_model(
    model_config: dict,
    vocabulary_size: int,
    block_size: int,
) -> nn.Module:
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
        )

    raise ValueError(f"Unknown model name: {name!r}")


__all__ = [
    "AttentionLanguageModel",
    "BigramLanguageModel",
    "TransformerLanguageModel",
    "build_model",
]