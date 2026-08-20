"""Supervised character records, leakage-safe splits, and masked batches."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import torch

from story_model.character_data import (
    ASSISTANT_MARKER,
    CHARACTER_CONTROL_TOKENS,
    CharacterContext,
    serialize_character_prompt,
    serialize_character_training_text,
)
from story_model.corpus import sha256_text
from story_model.data import ByteBPETokenizer


CHARACTER_DATASET_VERSION = 1
RESPONSE_IGNORE_INDEX = -100
CHARACTER_BEHAVIOR_TAGS = frozenset(
    {
        "boundary",
        "canon",
        "conflict",
        "long_context",
        "memory",
        "relationship",
        "scene",
        "uncertainty",
        "voice",
        "world_fact",
    }
)
_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]*"
)


@dataclass(frozen=True)
class CharacterTrainingRecord:
    """One supervised response and the conversation that owns it."""

    conversation_id: str
    context: CharacterContext
    behavior_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str):
            raise TypeError("conversation_id must be a string")

        conversation_id = self.conversation_id.strip()

        if _IDENTIFIER_PATTERN.fullmatch(conversation_id) is None:
            raise ValueError(
                "conversation_id must contain only letters, numbers, "
                "periods, colons, underscores, or hyphens"
            )

        if not isinstance(self.context, CharacterContext):
            raise TypeError("context must be a CharacterContext")

        if self.context.target_response is None:
            raise ValueError(
                "character training records require target_response"
            )

        if isinstance(self.behavior_tags, (str, bytes)):
            raise TypeError(
                "behavior_tags must be a sequence of strings"
            )

        behavior_tags = tuple(self.behavior_tags)

        if not all(isinstance(tag, str) for tag in behavior_tags):
            raise TypeError("every behavior tag must be a string")
        if len(set(behavior_tags)) != len(behavior_tags):
            raise ValueError("behavior_tags cannot contain duplicates")

        unknown_tags = sorted(
            set(behavior_tags) - CHARACTER_BEHAVIOR_TAGS
        )

        if unknown_tags:
            raise ValueError(
                "unknown character behavior tags: "
                + ", ".join(unknown_tags)
            )

        object.__setattr__(
            self,
            "conversation_id",
            conversation_id,
        )
        object.__setattr__(
            self,
            "behavior_tags",
            behavior_tags,
        )


@dataclass(frozen=True)
class EncodedCharacterExample:
    """One fixed-width example with prompt labels masked from loss."""

    conversation_id: str
    context_id: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    sequence_tokens: int
    prompt_tokens: int
    supervised_tokens: int
    dropped_turns: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.target_ids):
            raise ValueError(
                "input_ids and target_ids must have equal length"
            )
        if not 1 <= self.sequence_tokens <= len(self.input_ids):
            raise ValueError("invalid sequence_tokens count")
        if not 1 <= self.supervised_tokens <= self.sequence_tokens:
            raise ValueError("invalid supervised_tokens count")


def character_training_record_to_json(
    record: CharacterTrainingRecord,
) -> str:
    """Serialize one record as one deterministic JSONL line."""

    if not isinstance(record, CharacterTrainingRecord):
        raise TypeError("record must be a CharacterTrainingRecord")

    return json.dumps(
        {
            "dataset_version": CHARACTER_DATASET_VERSION,
            "conversation_id": record.conversation_id,
            "behavior_tags": list(record.behavior_tags),
            "context": record.context.to_dict(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def character_training_record_from_json(
    text: str,
) -> CharacterTrainingRecord:
    """Parse one JSONL record and validate both wrapper and context."""

    if not isinstance(text, str):
        raise TypeError("character training JSON must be a string")

    data = json.loads(text)

    if not isinstance(data, dict):
        raise TypeError("character training record must be an object")

    version = data.get("dataset_version")

    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != CHARACTER_DATASET_VERSION
    ):
        raise ValueError(
            "unsupported character dataset version: "
            f"{version!r}"
        )

    context_data = data.get("context")

    if not isinstance(context_data, dict):
        raise TypeError("character training context must be an object")

    return CharacterTrainingRecord(
        conversation_id=data["conversation_id"],
        context=CharacterContext.from_dict(context_data),
        behavior_tags=tuple(data.get("behavior_tags", ())),
    )


def _validate_records(
    records: Iterable[CharacterTrainingRecord],
) -> tuple[CharacterTrainingRecord, ...]:
    records = tuple(records)

    if not records:
        raise ValueError("character dataset cannot be empty")
    if not all(
        isinstance(record, CharacterTrainingRecord)
        for record in records
    ):
        raise TypeError(
            "every dataset entry must be a CharacterTrainingRecord"
        )

    context_ids = [
        record.context.context_id for record in records
    ]

    if len(set(context_ids)) != len(context_ids):
        raise ValueError(
            "character dataset contains duplicate context_id values"
        )

    return records


def _records_to_jsonl(
    records: Iterable[CharacterTrainingRecord],
) -> str:
    records = _validate_records(records)
    return "".join(
        character_training_record_to_json(record)
        for record in records
    )


def save_character_training_records(
    records: Iterable[CharacterTrainingRecord],
    path: Union[str, Path],
) -> None:
    """Write validated records as UTF-8 JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _records_to_jsonl(records),
        encoding="utf-8",
    )


def load_character_training_records(
    path: Union[str, Path],
) -> tuple[CharacterTrainingRecord, ...]:
    """Load strict JSONL with line-numbered validation errors."""

    path = Path(path)
    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                f"{path}:{line_number}: blank JSONL line"
            )

        try:
            record = character_training_record_from_json(line)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{path}:{line_number}: {error}"
            ) from error

        records.append(record)

    return _validate_records(records)


def split_character_training_records(
    records: Iterable[CharacterTrainingRecord],
    validation_fraction: float = 0.2,
    seed: int = 1337,
) -> tuple[
    tuple[CharacterTrainingRecord, ...],
    tuple[CharacterTrainingRecord, ...],
]:
    """Split complete conversations so turns never leak across splits."""

    records = _validate_records(records)

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    conversation_ids = sorted(
        {record.conversation_id for record in records}
    )

    if len(conversation_ids) < 2:
        raise ValueError(
            "at least two conversations are required for a split"
        )

    random.Random(seed).shuffle(conversation_ids)
    validation_count = round(
        len(conversation_ids) * validation_fraction
    )
    validation_count = max(
        1,
        min(len(conversation_ids) - 1, validation_count),
    )
    validation_ids = set(
        conversation_ids[:validation_count]
    )

    training = tuple(
        record
        for record in records
        if record.conversation_id not in validation_ids
    )
    validation = tuple(
        record
        for record in records
        if record.conversation_id in validation_ids
    )

    return training, validation


def build_character_dataset(
    records: Iterable[CharacterTrainingRecord],
    output_dir: Union[str, Path],
    validation_fraction: float = 0.2,
    seed: int = 1337,
) -> dict:
    """Create train/validation JSONL files and a reproducibility manifest."""

    records = _validate_records(records)
    training, validation = split_character_training_records(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_text = _records_to_jsonl(training)
    validation_text = _records_to_jsonl(validation)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"

    train_path.write_text(training_text, encoding="utf-8")
    val_path.write_text(validation_text, encoding="utf-8")

    def split_metadata(
        split_records: tuple[CharacterTrainingRecord, ...],
        text: str,
    ) -> dict:
        behavior_tags = Counter(
            tag
            for record in split_records
            for tag in record.behavior_tags
        )
        return {
            "examples": len(split_records),
            "conversations": sorted(
                {
                    record.conversation_id
                    for record in split_records
                }
            ),
            "characters": sorted(
                {
                    record.context.character.character_id
                    for record in split_records
                }
            ),
            "behavior_tags": {
                tag: behavior_tags[tag]
                for tag in sorted(behavior_tags)
            },
            "sha256": sha256_text(text),
        }

    manifest = {
        "dataset_version": CHARACTER_DATASET_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_examples": len(records),
        "train": split_metadata(training, training_text),
        "val": split_metadata(validation, validation_text),
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_character_dataset_splits(
    data_config: dict,
) -> tuple[
    tuple[CharacterTrainingRecord, ...],
    tuple[CharacterTrainingRecord, ...],
    dict | None,
]:
    """Load train/validation JSONL and verify split integrity.

    A manifest is optional for hand-built fixtures, but production configs
    should provide one. Whenever it is present, hashes, example counts,
    conversation IDs, and character IDs must still match the files on
    disk. Conversation and context leakage are rejected even without a
    manifest.
    """

    if data_config.get("type") != "character_jsonl":
        raise ValueError(
            "character dataset config type must be character_jsonl"
        )

    missing_paths = [
        key
        for key in ("train_path", "val_path")
        if key not in data_config
    ]

    if missing_paths:
        raise ValueError(
            "character data config is missing: "
            + ", ".join(missing_paths)
        )

    train_path = Path(data_config["train_path"])
    val_path = Path(data_config["val_path"])
    training_text = train_path.read_text(encoding="utf-8")
    validation_text = val_path.read_text(encoding="utf-8")
    training = load_character_training_records(train_path)
    validation = load_character_training_records(val_path)

    training_conversations = {
        record.conversation_id for record in training
    }
    validation_conversations = {
        record.conversation_id for record in validation
    }
    leaked_conversations = sorted(
        training_conversations & validation_conversations
    )

    if leaked_conversations:
        raise ValueError(
            "conversation leakage between train and validation: "
            + ", ".join(leaked_conversations)
        )

    training_contexts = {
        record.context.context_id for record in training
    }
    validation_contexts = {
        record.context.context_id for record in validation
    }

    if training_contexts & validation_contexts:
        raise ValueError(
            "context_id leakage between train and validation"
        )

    manifest_path = data_config.get("manifest_path")

    if manifest_path is None:
        return training, validation, None

    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )

    if not isinstance(manifest, dict):
        raise ValueError(
            "character dataset manifest must be an object"
        )
    if manifest.get("dataset_version") != CHARACTER_DATASET_VERSION:
        raise ValueError(
            "character dataset manifest has unsupported version"
        )
    if manifest.get("source_examples") != (
        len(training) + len(validation)
    ):
        raise ValueError(
            "character dataset total does not match manifest"
        )

    def verify_split(
        split_name: str,
        records: tuple[CharacterTrainingRecord, ...],
        text: str,
    ) -> None:
        expected = manifest.get(split_name)

        if not isinstance(expected, dict):
            raise ValueError(
                f"character manifest is missing {split_name} metadata"
            )

        if expected.get("sha256") != sha256_text(text):
            raise ValueError(
                f"{split_name} character data does not match its manifest"
            )
        if expected.get("examples") != len(records):
            raise ValueError(
                f"{split_name} character example count does not match "
                "its manifest"
            )

        conversations = sorted(
            {record.conversation_id for record in records}
        )
        characters = sorted(
            {
                record.context.character.character_id
                for record in records
            }
        )
        behavior_tags = Counter(
            tag
            for record in records
            for tag in record.behavior_tags
        )

        if expected.get("conversations") != conversations:
            raise ValueError(
                f"{split_name} conversations do not match manifest"
            )
        if expected.get("characters") != characters:
            raise ValueError(
                f"{split_name} characters do not match manifest"
            )
        expected_behavior_tags = expected.get("behavior_tags")

        if (
            expected_behavior_tags is not None
            and expected_behavior_tags
            != {
                tag: behavior_tags[tag]
                for tag in sorted(behavior_tags)
            }
        ):
            raise ValueError(
                f"{split_name} behavior tags do not match manifest"
            )

    verify_split("train", training, training_text)
    verify_split("val", validation, validation_text)
    return training, validation, manifest


def _validate_character_tokenizer(
    tokenizer: ByteBPETokenizer,
) -> None:
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError(
            "character training requires a ByteBPETokenizer"
        )

    missing = [
        token
        for token in CHARACTER_CONTROL_TOKENS
        if token not in tokenizer.special_token_ids
    ]

    if missing:
        raise ValueError(
            "character tokenizer is missing control tokens: "
            + ", ".join(missing)
        )


def encode_character_training_record(
    record: CharacterTrainingRecord,
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> EncodedCharacterExample:
    """Encode one record and mask every target before the response cue.

    Inputs and targets are right-padded to ``block_size``. Padding targets
    use -100, PyTorch cross-entropy's ignore index, so they contribute no
    loss. If a record is too long, only complete oldest conversation turns
    may be removed; persona, state, the newest user turn, and the target
    response are never silently truncated.
    """

    if not isinstance(record, CharacterTrainingRecord):
        raise TypeError("record must be a CharacterTrainingRecord")
    _validate_character_tokenizer(tokenizer)

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    token_counter = lambda text: len(tokenizer.encode(text))
    prompt_budget = block_size + 1

    while True:
        try:
            prompt = serialize_character_prompt(
                record.context,
                max_prompt_tokens=prompt_budget,
                token_counter=token_counter,
            )
            training_text = serialize_character_training_text(
                record.context,
                max_prompt_tokens=prompt_budget,
                token_counter=token_counter,
            )
        except ValueError as error:
            raise ValueError(
                f"context {record.context.context_id!r} cannot fit "
                f"within block_size {block_size} without truncating "
                "required character data, the newest turn, or response"
            ) from error

        token_ids = tokenizer.encode(training_text)
        excess = len(token_ids) - (block_size + 1)

        if excess <= 0:
            break

        prompt_budget -= max(1, excess)

        if prompt_budget < 1:
            raise ValueError(
                f"context {record.context.context_id!r} cannot fit "
                f"within block_size {block_size}"
            )

    assistant_id = tokenizer.special_token_ids[ASSISTANT_MARKER]
    response_cue = max(
        index
        for index, token_id in enumerate(token_ids[:-1])
        if token_id == assistant_id
    )
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    for index in range(response_cue):
        target_ids[index] = RESPONSE_IGNORE_INDEX

    sequence_tokens = len(input_ids)
    supervised_tokens = sum(
        token_id != RESPONSE_IGNORE_INDEX
        for token_id in target_ids
    )
    included_turns = (
        prompt.count("<|user|>")
        + prompt.count("<|assistant|>")
        - 1
    )
    dropped_turns = max(
        0,
        len(record.context.recent_turns) - included_turns,
    )
    padding = block_size - sequence_tokens

    input_ids.extend([0] * padding)
    target_ids.extend([RESPONSE_IGNORE_INDEX] * padding)

    return EncodedCharacterExample(
        conversation_id=record.conversation_id,
        context_id=record.context.context_id,
        input_ids=tuple(input_ids),
        target_ids=tuple(target_ids),
        sequence_tokens=sequence_tokens,
        prompt_tokens=len(tokenizer.encode(prompt)),
        supervised_tokens=supervised_tokens,
        dropped_turns=dropped_turns,
    )


def encode_character_training_records(
    records: Iterable[CharacterTrainingRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[EncodedCharacterExample, ...]:
    records = _validate_records(records)
    return tuple(
        encode_character_training_record(
            record,
            tokenizer,
            block_size,
        )
        for record in records
    )


def get_character_batch(
    examples: Iterable[EncodedCharacterExample],
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed-width supervised examples with replacement."""

    examples = tuple(examples)

    if not examples:
        raise ValueError("character examples cannot be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    indices = torch.randint(len(examples), (batch_size,))
    inputs = torch.tensor(
        [examples[int(index)].input_ids for index in indices],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [examples[int(index)].target_ids for index in indices],
        dtype=torch.long,
    )
    return inputs.to(device), targets.to(device)


def summarize_character_examples(
    records: Iterable[CharacterTrainingRecord],
    examples: Iterable[EncodedCharacterExample],
) -> dict:
    """Return compact diagnostics used before launching fine-tuning."""

    records = _validate_records(records)
    examples = tuple(examples)

    if len(records) != len(examples):
        raise ValueError(
            "record and encoded example counts must match"
        )

    sequence_counts = [
        example.sequence_tokens for example in examples
    ]
    prompt_counts = [example.prompt_tokens for example in examples]
    supervised_counts = [
        example.supervised_tokens for example in examples
    ]

    return {
        "examples": len(records),
        "conversations": len(
            {record.conversation_id for record in records}
        ),
        "characters": len(
            {
                record.context.character.character_id
                for record in records
            }
        ),
        "sequence_tokens_min": min(sequence_counts),
        "sequence_tokens_max": max(sequence_counts),
        "sequence_tokens_mean": sum(sequence_counts) / len(examples),
        "prompt_tokens_max": max(prompt_counts),
        "supervised_tokens": sum(supervised_counts),
        "dropped_turns": sum(
            example.dropped_turns for example in examples
        ),
    }


def audit_character_training_records(
    records: Iterable[CharacterTrainingRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
    min_examples: int = 100,
    min_conversations: int = 20,
    min_examples_per_tag: int = 5,
    required_tags: Iterable[str] = CHARACTER_BEHAVIOR_TAGS,
    max_dropped_turns: int | None = None,
) -> dict:
    """Audit coverage and encoding quality before production training."""

    records = _validate_records(records)
    required_tags = tuple(sorted(set(required_tags)))
    unknown_required = sorted(
        set(required_tags) - CHARACTER_BEHAVIOR_TAGS
    )

    if unknown_required:
        raise ValueError(
            "unknown required behavior tags: "
            + ", ".join(unknown_required)
        )

    for value, label in (
        (min_examples, "min_examples"),
        (min_conversations, "min_conversations"),
        (min_examples_per_tag, "min_examples_per_tag"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < 1:
            raise ValueError(f"{label} must be positive")

    if max_dropped_turns is not None:
        if (
            isinstance(max_dropped_turns, bool)
            or not isinstance(max_dropped_turns, int)
        ):
            raise TypeError("max_dropped_turns must be an integer")
        if max_dropped_turns < 0:
            raise ValueError("max_dropped_turns cannot be negative")

    examples = encode_character_training_records(
        records,
        tokenizer,
        block_size,
    )
    summary = summarize_character_examples(records, examples)
    errors = []
    warnings = []
    tag_counts = Counter(
        tag
        for record in records
        for tag in record.behavior_tags
    )

    if len(records) < min_examples:
        errors.append(
            f"dataset has {len(records)} examples; requires {min_examples}"
        )
    if summary["conversations"] < min_conversations:
        errors.append(
            "dataset has "
            f"{summary['conversations']} conversations; "
            f"requires {min_conversations}"
        )
    if (
        max_dropped_turns is not None
        and summary["dropped_turns"] > max_dropped_turns
    ):
        errors.append(
            f"encoding drops {summary['dropped_turns']} old turns; "
            f"maximum allowed is {max_dropped_turns}"
        )

    for tag in required_tags:
        count = tag_counts[tag]

        if count < min_examples_per_tag:
            errors.append(
                f"behavior tag {tag!r} has {count} examples; "
                f"requires {min_examples_per_tag}"
            )

    untagged = sum(not record.behavior_tags for record in records)

    if untagged:
        errors.append(f"{untagged} examples have no behavior tags")

    raw_control_markers = [
        record.context.context_id
        for record in records
        if any(
            marker in record.context.target_response
            for marker in CHARACTER_CONTROL_TOKENS
        )
    ]

    if raw_control_markers:
        errors.append(
            "target responses contain raw control markers: "
            + ", ".join(raw_control_markers)
        )

    short_responses = [
        record.context.context_id
        for record, example in zip(records, examples)
        if example.supervised_tokens < 4
    ]

    if short_responses:
        warnings.append(
            "responses with fewer than four supervised tokens: "
            + ", ".join(short_responses)
        )

    long_responses = [
        record.context.context_id
        for record, example in zip(records, examples)
        if example.supervised_tokens > 256
    ]

    if long_responses:
        warnings.append(
            "responses with more than 256 supervised tokens: "
            + ", ".join(long_responses)
        )

    normalized_responses = Counter(
        " ".join(
            record.context.target_response.lower().split()
        )
        for record in records
    )
    duplicate_responses = sum(
        count - 1
        for count in normalized_responses.values()
        if count > 1
    )

    if duplicate_responses:
        warnings.append(
            f"{duplicate_responses} duplicate normalized responses"
        )
    if summary["dropped_turns"]:
        warnings.append(
            f"encoding drops {summary['dropped_turns']} old turns"
        )

    return {
        **summary,
        "passed": not errors,
        "errors": tuple(errors),
        "warnings": tuple(warnings),
        "tag_counts": {
            tag: tag_counts[tag]
            for tag in sorted(CHARACTER_BEHAVIOR_TAGS)
        },
        "untagged_examples": untagged,
        "duplicate_responses": duplicate_responses,
    }
