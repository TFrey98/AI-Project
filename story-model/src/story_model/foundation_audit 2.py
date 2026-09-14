"""Exact tokenizer-scale audit for foundation corpus v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from story_model.checkpoint import read_checkpoint
from story_model.corpus import sha256_text
from story_model.corpus_v2 import FOUNDATION_CORPUS_VERSION
from story_model.data import ByteBPETokenizer, tokenizer_from_dict


def audit_foundation_corpus(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    minimum_training_tokens: int = 50_000_000,
    minimum_tokens_per_parameter: float = 15.0,
) -> dict[str, Any]:
    """Measure corpus tokens with the exact foundation tokenizer.

    The checkpoint must be the pre-character foundation checkpoint.  A
    tokenizer containing character control tokens is rejected so a bridge or
    character checkpoint cannot accidentally become the new language base.
    """

    if minimum_training_tokens < 1:
        raise ValueError("minimum_training_tokens must be positive")

    if minimum_tokens_per_parameter <= 0.0:
        raise ValueError("minimum_tokens_per_parameter must be positive")

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if manifest.get("version") != FOUNDATION_CORPUS_VERSION:
        raise ValueError("foundation corpus manifest has unsupported version")

    if manifest.get("split_unit") != "author":
        raise ValueError("foundation corpus manifest is not author-disjoint")

    train_text = _load_verified_split(manifest_file.parent, manifest, "train")
    val_text = _load_verified_split(manifest_file.parent, manifest, "val")
    checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
    tokenizer_data = checkpoint.get("extra", {}).get("tokenizer")

    if tokenizer_data is None:
        raise ValueError("foundation checkpoint has no tokenizer metadata")

    tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("foundation checkpoint must use byte-BPE")

    if tokenizer.special_tokens:
        raise ValueError(
            "foundation checkpoint tokenizer contains special tokens; "
            "use the pre-character foundation checkpoint"
        )

    state = checkpoint.get("model_state_dict")

    if not isinstance(state, dict) or not state:
        raise ValueError("foundation checkpoint has no model state")

    parameter_count = sum(tensor.numel() for tensor in state.values())
    training_tokens = len(tokenizer.encode(train_text))
    validation_tokens = len(tokenizer.encode(val_text))
    tokens_per_parameter = training_tokens / parameter_count
    failures: list[str] = []

    if training_tokens < minimum_training_tokens:
        failures.append(
            f"training tokens {training_tokens} < {minimum_training_tokens}"
        )

    if tokens_per_parameter < minimum_tokens_per_parameter:
        failures.append(
            "training tokens/parameter "
            f"{tokens_per_parameter:.2f} < {minimum_tokens_per_parameter:.2f}"
        )

    if failures:
        raise ValueError(
            "foundation corpus token gates failed: " + "; ".join(failures)
        )

    training_bytes = len(train_text.encode("utf-8"))
    validation_bytes = len(val_text.encode("utf-8"))

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "parameters": parameter_count,
        "vocabulary": tokenizer.vocab_size,
        "training_tokens": training_tokens,
        "validation_tokens": validation_tokens,
        "training_utf8_bytes": training_bytes,
        "validation_utf8_bytes": validation_bytes,
        "training_bytes_per_token": training_bytes / training_tokens,
        "validation_bytes_per_token": validation_bytes / validation_tokens,
        "training_tokens_per_parameter": tokens_per_parameter,
        "passed": True,
    }


def _load_verified_split(
    directory: Path,
    manifest: dict[str, Any],
    split: str,
) -> str:
    split_data = manifest.get(split)

    if not isinstance(split_data, dict):
        raise ValueError(f"foundation manifest is missing {split} metadata")

    relative_path = split_data.get("path")
    expected_hash = split_data.get("sha256")

    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"foundation manifest has incomplete {split} metadata")

    text = (directory / relative_path).read_text(encoding="utf-8")

    if sha256_text(text) != expected_hash:
        raise ValueError(f"{split} corpus does not match its manifest")

    return text
