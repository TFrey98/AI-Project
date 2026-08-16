"""Text tokenizers, dataset loading, and random batching."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch

from story_model.corpus import sha256_text


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[t] for t in tokens)

    def token_byte_length(self, token_id: int) -> int:
        return len(
            self.itos[token_id].encode("utf-8")
        )

    def to_dict(self) -> dict:
        return {
            "type": "char",
            "version": 1,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        tokenizer_type = data.get("type", "char")

        if tokenizer_type != "char":
            raise ValueError("tokenizer data is not character-level")

        return cls(
            stoi={
                str(character): int(token_id)
                for character, token_id in data["stoi"].items()
            },
            itos={
                int(token_id): str(character)
                for token_id, character in data["itos"].items()
            },
        )


def merge_token_pair(
    tokens: list[int],
    pair: tuple[int, int],
    new_token: int,
) -> list[int]:
    """Merge non-overlapping pair occurrences from left to right."""

    merged: list[int] = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(new_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1

    return merged


@dataclass
class ByteBPETokenizer:
    """UTF-8 byte tokenizer with learned byte-pair merges."""

    merges: list[tuple[int, int]]
    _token_bytes: dict[int, bytes] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        normalized_merges: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for rank, raw_pair in enumerate(self.merges):
            if len(raw_pair) != 2:
                raise ValueError(
                    "each BPE merge must contain two token ids"
                )

            pair = (
                int(raw_pair[0]),
                int(raw_pair[1]),
            )
            new_token = 256 + rank

            if not all(
                0 <= token_id < new_token
                for token_id in pair
            ):
                raise ValueError(
                    "BPE merge references an unavailable token"
                )

            if pair in seen_pairs:
                raise ValueError("duplicate BPE merge pair")

            normalized_merges.append(pair)
            seen_pairs.add(pair)

        self.merges = normalized_merges

        token_bytes = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

        for rank, pair in enumerate(self.merges):
            token_bytes[256 + rank] = (
                token_bytes[pair[0]]
                + token_bytes[pair[1]]
            )

        self._token_bytes = token_bytes

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if not text:
            raise ValueError("BPE training text cannot be empty")

        if vocab_size < 256:
            raise ValueError(
                "BPE vocab_size must be at least 256"
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be positive"
            )

        tokens = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []

        while 256 + len(merges) < vocab_size:
            pair_counts = Counter(
                zip(tokens, tokens[1:])
            )

            if not pair_counts:
                break

            # Prefer the most frequent pair, then the numerically
            # smallest pair so training is deterministic on ties.
            best_pair, frequency = min(
                pair_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )

            if frequency < min_frequency:
                break

            new_token = 256 + len(merges)
            tokens = merge_token_pair(
                tokens,
                best_pair,
                new_token,
            )
            merges.append(best_pair)

        return cls(merges=merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))

        for rank, pair in enumerate(self.merges):
            tokens = merge_token_pair(
                tokens,
                pair,
                256 + rank,
            )

        return tokens

    def decode(self, tokens: list[int]) -> str:
        pieces: list[bytes] = []

        for token_id in tokens:
            if token_id not in self._token_bytes:
                raise ValueError(
                    f"unknown BPE token id: {token_id}"
                )

            pieces.append(self._token_bytes[token_id])

        return b"".join(pieces).decode(
            "utf-8",
            errors="replace",
        )

    def token_byte_length(self, token_id: int) -> int:
        if token_id not in self._token_bytes:
            raise ValueError(
                f"unknown BPE token id: {token_id}"
            )

        return len(self._token_bytes[token_id])

    def to_dict(self) -> dict:
        return {
            "type": "byte_bpe",
            "version": 1,
            "merges": [
                [left, right]
                for left, right in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPETokenizer":
        if data.get("type") != "byte_bpe":
            raise ValueError("tokenizer data is not byte-level BPE")

        return cls(
            merges=[
                (int(pair[0]), int(pair[1]))
                for pair in data["merges"]
            ]
        )


Tokenizer = Union[CharTokenizer, ByteBPETokenizer]


def split_text(
    text: str, train_split: float
) -> tuple[str, str]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    n = int(train_split * len(text))
    training_text = text[:n]
    validation_text = text[n:]

    if not training_text or not validation_text:
        raise ValueError(
            "text split must produce non-empty train and validation data"
        )

    return training_text, validation_text


def build_tokenizer(
    training_text: str,
    config: dict | None = None,
) -> Tokenizer:
    config = config or {}
    tokenizer_type = config.get("type", "char")

    if tokenizer_type == "char":
        return CharTokenizer.from_text(training_text)

    if tokenizer_type == "byte_bpe":
        return ByteBPETokenizer.train(
            text=training_text,
            vocab_size=config.get("vocab_size", 512),
            min_frequency=config.get(
                "min_frequency", 2
            ),
        )

    raise ValueError(
        f"Unknown tokenizer type: {tokenizer_type!r}"
    )


def tokenizer_from_dict(data: dict) -> Tokenizer:
    tokenizer_type = data.get("type", "char")

    if tokenizer_type == "char":
        return CharTokenizer.from_dict(data)

    if tokenizer_type == "byte_bpe":
        return ByteBPETokenizer.from_dict(data)

    raise ValueError(
        f"Unknown tokenizer type: {tokenizer_type!r}"
    )


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_corpus_manifest(
    data_config: dict,
) -> dict | None:
    manifest_path = data_config.get("manifest_path")

    if manifest_path is None:
        return None

    return json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )


def validate_corpus_manifest(
    data_config: dict,
    training_text: str,
    validation_text: str,
) -> dict | None:
    manifest = load_corpus_manifest(data_config)

    if manifest is None:
        return None

    for split_name, text in (
        ("train", training_text),
        ("val", validation_text),
    ):
        expected_hash = (
            manifest.get(split_name, {}).get("sha256")
        )

        if expected_hash is None:
            raise ValueError(
                f"corpus manifest is missing {split_name} SHA-256"
            )

        if sha256_text(text) != expected_hash:
            raise ValueError(
                f"{split_name} corpus does not match its manifest"
            )

    return manifest


def load_text_splits(
    data_config: dict,
) -> tuple[str, str]:
    """Load explicit document splits or a legacy split from one file."""

    has_path = "path" in data_config
    has_train_path = "train_path" in data_config
    has_val_path = "val_path" in data_config

    if has_path and (has_train_path or has_val_path):
        raise ValueError(
            "data config cannot combine path with train_path/val_path"
        )

    if has_train_path != has_val_path:
        raise ValueError(
            "data config must provide both train_path and val_path"
        )

    if has_train_path:
        training_text = load_text(data_config["train_path"])
        validation_text = load_text(data_config["val_path"])

        if not training_text or not validation_text:
            raise ValueError(
                "train and validation files must both be non-empty"
            )

        validate_corpus_manifest(
            data_config,
            training_text,
            validation_text,
        )

        return training_text, validation_text

    if has_path:
        if "manifest_path" in data_config:
            raise ValueError(
                "manifest_path requires explicit train_path/val_path"
            )

        if "train_split" not in data_config:
            raise ValueError(
                "single-file data config requires train_split"
            )

        return split_text(
            load_text(data_config["path"]),
            float(data_config["train_split"]),
        )

    raise ValueError(
        "data config requires path or train_path/val_path"
    )


def train_test_split(data: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_split * len(data))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
