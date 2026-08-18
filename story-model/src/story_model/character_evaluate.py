"""Exact response-only evaluation for held-out character scenarios."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch

from story_model.character_training import (
    CharacterTrainingRecord,
    encode_character_training_records,
)
from story_model.data import ByteBPETokenizer


@torch.no_grad()
def evaluate_character_records(
    model: torch.nn.Module,
    records: Iterable[CharacterTrainingRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
    device: str,
) -> dict:
    """Measure token-weighted response loss globally and per behavior."""

    records = tuple(records)
    examples = encode_character_training_records(
        records,
        tokenizer,
        block_size,
    )
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    category_totals = defaultdict(
        lambda: {
            "negative_log_likelihood": 0.0,
            "tokens": 0,
            "examples": 0,
        }
    )

    model.eval()

    for record, example in zip(records, examples):
        inputs = torch.tensor(
            [example.input_ids[: example.sequence_tokens]],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [example.target_ids[: example.sequence_tokens]],
            dtype=torch.long,
            device=device,
        )
        _, loss = model(inputs, targets)

        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError(
                f"non-finite character loss for {example.context_id}"
            )

        negative_log_likelihood = (
            loss.item() * example.supervised_tokens
        )
        total_negative_log_likelihood += negative_log_likelihood
        total_tokens += example.supervised_tokens
        tags = record.behavior_tags or ("untagged",)

        for tag in tags:
            category_totals[tag][
                "negative_log_likelihood"
            ] += negative_log_likelihood
            category_totals[tag]["tokens"] += (
                example.supervised_tokens
            )
            category_totals[tag]["examples"] += 1

    loss = total_negative_log_likelihood / total_tokens
    categories = {}

    for tag, totals in sorted(category_totals.items()):
        category_loss = (
            totals["negative_log_likelihood"]
            / totals["tokens"]
        )
        categories[tag] = {
            "loss": category_loss,
            "perplexity": math.exp(min(category_loss, 50.0)),
            "tokens": totals["tokens"],
            "examples": totals["examples"],
        }

    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "tokens": total_tokens,
        "examples": len(records),
        "categories": categories,
    }
