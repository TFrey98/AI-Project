"""Build deterministic train and validation corpora from text documents."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CORPUS_VERSION = 1
DOCUMENT_SEPARATOR = "\n\n\n"

GUTENBERG_START_MARKERS = (
    "START OF THE PROJECT GUTENBERG EBOOK",
    "START OF THIS PROJECT GUTENBERG EBOOK",
)
GUTENBERG_END_MARKERS = (
    "END OF THE PROJECT GUTENBERG EBOOK",
    "END OF THIS PROJECT GUTENBERG EBOOK",
)


@dataclass(frozen=True)
class CorpusDocument:
    """A normalized source document and its reproducibility metadata."""

    relative_path: str
    text: str
    utf8_bytes: int
    sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove standard Project Gutenberg wrappers when both exist."""

    lines = text.splitlines()
    start_index = None
    end_index = None

    for index, line in enumerate(lines):
        upper_line = line.upper()

        if any(
            marker in upper_line
            for marker in GUTENBERG_START_MARKERS
        ):
            start_index = index + 1
            break

    if start_index is None:
        return text

    for index in range(start_index, len(lines)):
        upper_line = lines[index].upper()

        if any(
            marker in upper_line
            for marker in GUTENBERG_END_MARKERS
        ):
            end_index = index
            break

    if end_index is None:
        return text

    return "\n".join(lines[start_index:end_index])


def normalize_document(text: str) -> str:
    """Normalize Unicode/newlines while preserving paragraph structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = unicodedata.normalize("NFC", text)

    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()

    if not normalized:
        raise ValueError("corpus document is empty after normalization")

    return normalized + "\n"


def discover_documents(
    input_dir: str | Path,
) -> list[CorpusDocument]:
    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise ValueError(
            f"corpus input directory does not exist: {input_path}"
        )

    source_paths = sorted(
        input_path.rglob("*.txt"),
        key=lambda path: path.relative_to(input_path).as_posix(),
    )

    if len(source_paths) < 2:
        raise ValueError(
            "corpus building requires at least two .txt documents"
        )

    documents = []
    source_by_hash: dict[str, str] = {}

    for source_path in source_paths:
        relative_path = source_path.relative_to(
            input_path
        ).as_posix()
        text = normalize_document(
            source_path.read_text(encoding="utf-8")
        )
        document_hash = sha256_text(text)

        if document_hash in source_by_hash:
            raise ValueError(
                "duplicate normalized corpus documents: "
                f"{source_by_hash[document_hash]!r} and "
                f"{relative_path!r}"
            )

        source_by_hash[document_hash] = relative_path

        documents.append(
            CorpusDocument(
                relative_path=relative_path,
                text=text,
                utf8_bytes=len(text.encode("utf-8")),
                sha256=document_hash,
            )
        )

    return documents


def split_documents(
    documents: list[CorpusDocument],
    validation_fraction: float,
    seed: int,
    minimum_validation_documents: int = 1,
) -> tuple[list[CorpusDocument], list[CorpusDocument]]:
    """Split complete documents, targeting a validation byte fraction."""

    if len(documents) < 2:
        raise ValueError(
            "document splitting requires at least two documents"
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if minimum_validation_documents < 1:
        raise ValueError(
            "minimum_validation_documents must be positive"
        )

    ranked = sorted(
        documents,
        key=lambda document: document.relative_path,
    )
    random.Random(seed).shuffle(ranked)

    target_validation_bytes = max(
        1,
        round(
            sum(document.utf8_bytes for document in documents)
            * validation_fraction
        ),
    )

    validation_paths: set[str] = set()
    validation_bytes = 0
    required_validation_documents = min(
        minimum_validation_documents,
        len(documents) - 1,
    )

    # Always leave at least one complete document for training.
    for document in ranked[:-1]:
        if (
            validation_bytes >= target_validation_bytes
            and len(validation_paths)
            >= required_validation_documents
        ):
            break

        validation_paths.add(document.relative_path)
        validation_bytes += document.utf8_bytes

    if not validation_paths:
        validation_paths.add(ranked[0].relative_path)

    training_documents = sorted(
        (
            document
            for document in documents
            if document.relative_path not in validation_paths
        ),
        key=lambda document: document.relative_path,
    )
    validation_documents = sorted(
        (
            document
            for document in documents
            if document.relative_path in validation_paths
        ),
        key=lambda document: document.relative_path,
    )

    return training_documents, validation_documents


def join_documents(
    documents: Iterable[CorpusDocument],
) -> str:
    texts = [document.text.rstrip() for document in documents]

    if not texts:
        raise ValueError("cannot build an empty corpus split")

    return DOCUMENT_SEPARATOR.join(texts) + "\n"


def document_manifest_entry(
    document: CorpusDocument,
    split: str,
) -> dict[str, Any]:
    return {
        "path": document.relative_path,
        "split": split,
        "utf8_bytes": document.utf8_bytes,
        "sha256": document.sha256,
    }


def build_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 1337,
    minimum_validation_documents: int = 3,
) -> dict[str, Any]:
    """Build train.txt, val.txt, and a reproducibility manifest."""

    documents = discover_documents(input_dir)
    training_documents, validation_documents = split_documents(
        documents=documents,
        validation_fraction=validation_fraction,
        seed=seed,
        minimum_validation_documents=(
            minimum_validation_documents
        ),
    )

    training_text = join_documents(training_documents)
    validation_text = join_documents(validation_documents)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.txt"
    val_path = output_path / "val.txt"
    manifest_path = output_path / "manifest.json"

    train_path.write_text(training_text, encoding="utf-8")
    val_path.write_text(validation_text, encoding="utf-8")

    entries = [
        *(
            document_manifest_entry(document, "train")
            for document in training_documents
        ),
        *(
            document_manifest_entry(document, "val")
            for document in validation_documents
        ),
    ]
    entries.sort(key=lambda entry: entry["path"])

    manifest: dict[str, Any] = {
        "version": CORPUS_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "minimum_validation_documents": (
            minimum_validation_documents
        ),
        "document_separator": DOCUMENT_SEPARATOR,
        "documents": entries,
        "train": {
            "path": "train.txt",
            "documents": len(training_documents),
            "utf8_bytes": len(training_text.encode("utf-8")),
            "sha256": sha256_text(training_text),
        },
        "val": {
            "path": "val.txt",
            "documents": len(validation_documents),
            "utf8_bytes": len(validation_text.encode("utf-8")),
            "sha256": sha256_text(validation_text),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest
