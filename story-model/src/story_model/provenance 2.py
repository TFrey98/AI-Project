"""Stable fingerprints for tokenizers, manifests, and run identity.

The project often compares checkpoints produced on different computers.
Paths and byte counts are not sufficient to prove that two runs used the
same inputs: two files can have the same length while containing different
text.  These helpers hash a canonical JSON representation so logs and
checkpoints can identify the exact tokenizer and corpus manifest involved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_fingerprints(
    tokenizer_data: dict,
    corpus_manifest: dict | None = None,
) -> dict[str, str]:
    """Build the compact identity embedded in a training checkpoint."""

    fingerprints = {
        "tokenizer_sha256": canonical_json_sha256(tokenizer_data),
    }

    if corpus_manifest is not None:
        fingerprints["corpus_manifest_sha256"] = (
            canonical_json_sha256(corpus_manifest)
        )

        for split_name in ("train", "val"):
            split_hash = (
                corpus_manifest.get(split_name, {}).get("sha256")
            )

            if isinstance(split_hash, str):
                fingerprints[f"{split_name}_sha256"] = split_hash

    return fingerprints
