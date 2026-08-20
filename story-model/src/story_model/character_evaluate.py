"""Exact response-only evaluation for held-out character scenarios."""

from __future__ import annotations

import math
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Callable, Iterable

import torch

from story_model.character_training import (
    CharacterTrainingRecord,
    encode_character_training_records,
)
from story_model.character_chat import CharacterGeneration
from story_model.data import ByteBPETokenizer


def _matching_prefix_characters(first: str, second: str) -> int:
    count = 0

    for first_character, second_character in zip(first, second):
        if first_character != second_character:
            break

        count += 1

    return count


def evaluate_character_generations(
    records: Iterable[CharacterTrainingRecord],
    generator: Callable[
        [CharacterTrainingRecord, int], CharacterGeneration
    ],
) -> dict:
    """Compare free-running responses with fixed authored responses.

    This intentionally evaluates generation rather than teacher-forced
    next-token loss. A response passes only when its visible text exactly
    matches the authored target and the model independently emits the end
    marker. That is a deliberately strict diagnostic for a training-set
    overfit run; held-out behavioral review remains human-scored.
    """

    records = tuple(records)

    if not records:
        raise ValueError("character generation records cannot be empty")

    results = []
    exact_responses = 0
    end_stops = 0
    passed = 0
    prefix_fraction_total = 0.0
    similarity_total = 0.0

    for index, record in enumerate(records):
        generation = generator(record, index)

        if not isinstance(generation, CharacterGeneration):
            raise TypeError(
                "character generator must return CharacterGeneration"
            )

        reference = record.context.target_response

        if reference is None:
            raise ValueError(
                "generation evaluation records require target_response"
            )

        exact_response = generation.text == reference
        end_stop = generation.stop_reason == "end"
        scenario_passed = exact_response and end_stop
        matching_prefix = _matching_prefix_characters(
            generation.text,
            reference,
        )
        prefix_fraction = matching_prefix / max(1, len(reference))
        similarity = SequenceMatcher(
            None,
            generation.text,
            reference,
            autojunk=False,
        ).ratio()
        exact_responses += int(exact_response)
        end_stops += int(end_stop)
        passed += int(scenario_passed)
        prefix_fraction_total += prefix_fraction
        similarity_total += similarity
        results.append(
            {
                "context_id": record.context.context_id,
                "conversation_id": record.conversation_id,
                "behavior_tags": record.behavior_tags,
                "reference_response": reference,
                "generated_response": generation.text,
                "generated_tokens": len(generation.token_ids),
                "prompt_tokens": generation.prompt_tokens,
                "stop_reason": generation.stop_reason,
                "matching_prefix_characters": matching_prefix,
                "prefix_fraction": prefix_fraction,
                "similarity": similarity,
                "exact_response": exact_response,
                "end_stop": end_stop,
                "passed": scenario_passed,
            }
        )

    return {
        "examples": len(records),
        "exact_responses": exact_responses,
        "end_stops": end_stops,
        "passed": passed,
        "mean_prefix_fraction": (
            prefix_fraction_total / len(records)
        ),
        "mean_similarity": similarity_total / len(records),
        "all_passed": passed == len(records),
        "results": tuple(results),
    }


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
