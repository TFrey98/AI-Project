"""Provenance-aware foundation corpus construction.

The original corpus builder prevents document leakage.  A larger corpus
needs one stronger boundary: every work by an author must stay in the same
split.  Otherwise validation can contain an unseen title written in a style
the model already saw from that author, making generalization look better
than it is.

This module also makes source metadata mandatory.  Public-domain status,
source identity, author identity, and broad content categories are data, not
informal notes; they belong in the reproducibility manifest beside hashes.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from story_model.corpus import (
    DOCUMENT_SEPARATOR,
    normalize_document,
    sha256_text,
)


FOUNDATION_CORPUS_VERSION = 2
SOURCE_CATALOG_VERSION = 1
AUTHOR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class FoundationDocument:
    """One normalized work plus its cataloged provenance."""

    relative_path: str
    title: str
    author: str
    author_id: str
    source: str
    source_id: str
    language: str
    rights: str
    categories: tuple[str, ...]
    text: str
    utf8_bytes: int
    sha256: str
    year: int | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class CorpusQualityGates:
    """Minimum viable scale and diversity for foundation corpus v2."""

    minimum_documents: int = 50
    minimum_authors: int = 25
    minimum_training_bytes: int = 100_000_000
    minimum_categories: int = 5
    maximum_author_fraction: float = 0.10

    def validate(self) -> None:
        for name in (
            "minimum_documents",
            "minimum_authors",
            "minimum_training_bytes",
            "minimum_categories",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

        if not 0.0 < self.maximum_author_fraction <= 1.0:
            raise ValueError(
                "maximum_author_fraction must be between 0 and 1"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "minimum_documents": self.minimum_documents,
            "minimum_authors": self.minimum_authors,
            "minimum_training_bytes": self.minimum_training_bytes,
            "minimum_categories": self.minimum_categories,
            "maximum_author_fraction": self.maximum_author_fraction,
        }


def _required_string(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"catalog document field {field!r} must be a non-empty string"
        )

    return value.strip()


def _validate_relative_text_path(value: str) -> str:
    path = PurePosixPath(value)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"catalog document path must be relative: {value!r}"
        )

    if path.suffix.lower() != ".txt":
        raise ValueError(
            f"catalog document path must end in .txt: {value!r}"
        )

    return path.as_posix()


def load_source_catalog(
    catalog_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load and strictly validate a source catalog keyed by relative path."""

    path = Path(catalog_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("catalog_version") != SOURCE_CATALOG_VERSION:
        raise ValueError("source catalog has unsupported version")

    raw_documents = data.get("documents")

    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("source catalog documents must be a non-empty list")

    catalog: dict[str, dict[str, Any]] = {}
    source_keys: set[tuple[str, str]] = set()

    for raw_entry in raw_documents:
        if not isinstance(raw_entry, dict):
            raise ValueError("each source catalog document must be an object")

        entry = dict(raw_entry)
        relative_path = _validate_relative_text_path(
            _required_string(entry, "path")
        )

        if relative_path in catalog:
            raise ValueError(
                f"duplicate source catalog path: {relative_path!r}"
            )

        for field in (
            "title",
            "author",
            "author_id",
            "source",
            "source_id",
            "language",
            "rights",
        ):
            entry[field] = _required_string(entry, field)

        if AUTHOR_ID_PATTERN.fullmatch(entry["author_id"]) is None:
            raise ValueError(
                "catalog author_id must be a lowercase stable slug: "
                f"{entry['author_id']!r}"
            )

        source_key = (
            entry["source"].casefold(),
            entry["source_id"].casefold(),
        )

        if source_key in source_keys:
            raise ValueError(
                "duplicate source/source_id pair in source catalog: "
                f"{entry['source']!r}/{entry['source_id']!r}"
            )

        source_keys.add(source_key)

        categories = entry.get("categories")

        if not isinstance(categories, list) or not categories:
            raise ValueError(
                f"catalog document {relative_path!r} needs categories"
            )

        normalized_categories: list[str] = []

        for category in categories:
            if not isinstance(category, str) or not category.strip():
                raise ValueError(
                    "catalog categories must be non-empty strings"
                )

            normalized = category.strip().casefold()

            if normalized not in normalized_categories:
                normalized_categories.append(normalized)

        entry["categories"] = tuple(normalized_categories)
        entry["path"] = relative_path

        if "year" in entry and entry["year"] is not None:
            if isinstance(entry["year"], bool):
                raise ValueError("catalog year must be an integer")

            entry["year"] = int(entry["year"])

        if "source_url" in entry and entry["source_url"] is not None:
            entry["source_url"] = _required_string(entry, "source_url")

        catalog[relative_path] = entry

    return catalog


def discover_foundation_documents(
    input_dir: str | Path,
    catalog_path: str | Path,
    required_language: str = "en",
) -> list[FoundationDocument]:
    """Normalize every cataloged file and reject inventory drift."""

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise ValueError(
            f"corpus input directory does not exist: {input_path}"
        )

    catalog = load_source_catalog(catalog_path)
    source_paths = sorted(
        input_path.rglob("*.txt"),
        key=lambda item: item.relative_to(input_path).as_posix(),
    )
    discovered_paths = {
        item.relative_to(input_path).as_posix()
        for item in source_paths
    }
    catalog_paths = set(catalog)

    uncataloged = sorted(discovered_paths - catalog_paths)
    missing = sorted(catalog_paths - discovered_paths)

    if uncataloged or missing:
        raise ValueError(
            "source catalog does not match input files: "
            f"uncataloged={uncataloged}, missing={missing}"
        )

    if len(source_paths) < 2:
        raise ValueError(
            "foundation corpus building requires at least two documents"
        )

    documents: list[FoundationDocument] = []
    path_by_hash: dict[str, str] = {}

    for source_path in source_paths:
        relative_path = source_path.relative_to(input_path).as_posix()
        metadata = catalog[relative_path]

        if metadata["language"].casefold() != required_language.casefold():
            raise ValueError(
                f"catalog document {relative_path!r} has language "
                f"{metadata['language']!r}; expected {required_language!r}"
            )

        text = normalize_document(source_path.read_text(encoding="utf-8"))
        document_hash = sha256_text(text)

        if document_hash in path_by_hash:
            raise ValueError(
                "duplicate normalized foundation documents: "
                f"{path_by_hash[document_hash]!r} and {relative_path!r}"
            )

        path_by_hash[document_hash] = relative_path
        documents.append(
            FoundationDocument(
                relative_path=relative_path,
                title=metadata["title"],
                author=metadata["author"],
                author_id=metadata["author_id"],
                source=metadata["source"],
                source_id=metadata["source_id"],
                language=metadata["language"],
                rights=metadata["rights"],
                categories=metadata["categories"],
                year=metadata.get("year"),
                source_url=metadata.get("source_url"),
                text=text,
                utf8_bytes=len(text.encode("utf-8")),
                sha256=document_hash,
            )
        )

    return documents


def split_documents_by_author(
    documents: list[FoundationDocument],
    validation_fraction: float,
    seed: int,
    minimum_validation_authors: int = 5,
) -> tuple[list[FoundationDocument], list[FoundationDocument]]:
    """Assign complete author groups to train or validation."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    if minimum_validation_authors < 1:
        raise ValueError("minimum_validation_authors must be positive")

    documents_by_author: dict[str, list[FoundationDocument]] = defaultdict(
        list
    )

    for document in documents:
        documents_by_author[document.author_id].append(document)

    if len(documents_by_author) <= minimum_validation_authors:
        raise ValueError(
            "author-level splitting requires more authors than the "
            "minimum validation-author count"
        )

    author_ids = sorted(documents_by_author)
    random.Random(seed).shuffle(author_ids)
    target_bytes = max(
        1,
        round(
            sum(document.utf8_bytes for document in documents)
            * validation_fraction
        ),
    )
    validation_author_ids: set[str] = set()
    validation_bytes = 0

    for author_id in author_ids[:-1]:
        if (
            validation_bytes >= target_bytes
            and len(validation_author_ids) >= minimum_validation_authors
        ):
            break

        validation_author_ids.add(author_id)
        validation_bytes += sum(
            document.utf8_bytes
            for document in documents_by_author[author_id]
        )

    training = sorted(
        (
            document
            for document in documents
            if document.author_id not in validation_author_ids
        ),
        key=lambda document: document.relative_path,
    )
    validation = sorted(
        (
            document
            for document in documents
            if document.author_id in validation_author_ids
        ),
        key=lambda document: document.relative_path,
    )

    return training, validation


def _join_documents(documents: Iterable[FoundationDocument]) -> str:
    texts = [document.text.rstrip() for document in documents]

    if not texts:
        raise ValueError("cannot build an empty corpus split")

    return DOCUMENT_SEPARATOR.join(texts) + "\n"


def _category_counts(
    documents: Iterable[FoundationDocument],
) -> dict[str, int]:
    counts: Counter[str] = Counter()

    for document in documents:
        counts.update(document.categories)

    return dict(sorted(counts.items()))


def _author_byte_counts(
    documents: Iterable[FoundationDocument],
) -> dict[str, int]:
    counts: Counter[str] = Counter()

    for document in documents:
        counts[document.author_id] += document.utf8_bytes

    return dict(sorted(counts.items()))


def enforce_quality_gates(
    documents: list[FoundationDocument],
    training_documents: list[FoundationDocument],
    validation_documents: list[FoundationDocument],
    gates: CorpusQualityGates,
) -> dict[str, Any]:
    """Fail loudly when the expanded corpus is still too small or leaky."""

    gates.validate()
    authors = {document.author_id for document in documents}
    categories = {
        category
        for document in documents
        for category in document.categories
    }
    training_authors = {
        document.author_id for document in training_documents
    }
    validation_authors = {
        document.author_id for document in validation_documents
    }

    if training_authors & validation_authors:
        raise ValueError("author leakage between train and validation")

    failures: list[str] = []
    training_bytes = sum(
        document.utf8_bytes for document in training_documents
    )

    if len(documents) < gates.minimum_documents:
        failures.append(
            f"documents {len(documents)} < {gates.minimum_documents}"
        )

    if len(authors) < gates.minimum_authors:
        failures.append(f"authors {len(authors)} < {gates.minimum_authors}")

    if training_bytes < gates.minimum_training_bytes:
        failures.append(
            "training bytes "
            f"{training_bytes} < {gates.minimum_training_bytes}"
        )

    if len(categories) < gates.minimum_categories:
        failures.append(
            f"categories {len(categories)} < {gates.minimum_categories}"
        )

    author_bytes = _author_byte_counts(training_documents)
    largest_author_id, largest_author_bytes = max(
        author_bytes.items(),
        key=lambda item: item[1],
    )
    largest_author_fraction = largest_author_bytes / training_bytes

    if largest_author_fraction > gates.maximum_author_fraction:
        failures.append(
            "largest training author share "
            f"{largest_author_fraction:.1%} "
            f"({largest_author_id}) > "
            f"{gates.maximum_author_fraction:.1%}"
        )

    if failures:
        raise ValueError(
            "foundation corpus quality gates failed: "
            + "; ".join(failures)
        )

    return {
        "passed": True,
        "configured": gates.to_dict(),
        "observed": {
            "documents": len(documents),
            "authors": len(authors),
            "training_bytes": training_bytes,
            "categories": len(categories),
            "largest_training_author": largest_author_id,
            "largest_training_author_fraction": largest_author_fraction,
        },
    }


def _manifest_entry(
    document: FoundationDocument,
    split: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": document.relative_path,
        "title": document.title,
        "author": document.author,
        "author_id": document.author_id,
        "source": document.source,
        "source_id": document.source_id,
        "language": document.language,
        "rights": document.rights,
        "categories": list(document.categories),
        "split": split,
        "utf8_bytes": document.utf8_bytes,
        "sha256": document.sha256,
    }

    if document.year is not None:
        entry["year"] = document.year

    if document.source_url is not None:
        entry["source_url"] = document.source_url

    return entry


def build_foundation_corpus(
    input_dir: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 1337,
    minimum_validation_authors: int = 5,
    required_language: str = "en",
    gates: CorpusQualityGates | None = None,
) -> dict[str, Any]:
    """Build author-disjoint text splits and a version-2 manifest."""

    configured_gates = gates or CorpusQualityGates()
    documents = discover_foundation_documents(
        input_dir=input_dir,
        catalog_path=catalog_path,
        required_language=required_language,
    )
    training_documents, validation_documents = split_documents_by_author(
        documents=documents,
        validation_fraction=validation_fraction,
        seed=seed,
        minimum_validation_authors=minimum_validation_authors,
    )
    quality = enforce_quality_gates(
        documents,
        training_documents,
        validation_documents,
        configured_gates,
    )

    training_text = _join_documents(training_documents)
    validation_text = _join_documents(validation_documents)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.txt"
    val_path = output_path / "val.txt"
    manifest_path = output_path / "manifest.json"
    train_path.write_text(training_text, encoding="utf-8")
    val_path.write_text(validation_text, encoding="utf-8")

    entries = [
        *(
            _manifest_entry(document, "train")
            for document in training_documents
        ),
        *(
            _manifest_entry(document, "val")
            for document in validation_documents
        ),
    ]
    entries.sort(key=lambda entry: entry["path"])

    manifest: dict[str, Any] = {
        "version": FOUNDATION_CORPUS_VERSION,
        "source_catalog_version": SOURCE_CATALOG_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "minimum_validation_authors": minimum_validation_authors,
        "required_language": required_language,
        "split_unit": "author",
        "document_separator": DOCUMENT_SEPARATOR,
        "quality_gates": quality,
        "documents": entries,
        "summary": {
            "documents": len(documents),
            "authors": len({item.author_id for item in documents}),
            "categories": _category_counts(documents),
        },
        "train": {
            "path": "train.txt",
            "documents": len(training_documents),
            "authors": len(
                {item.author_id for item in training_documents}
            ),
            "categories": _category_counts(training_documents),
            "utf8_bytes": len(training_text.encode("utf-8")),
            "sha256": sha256_text(training_text),
        },
        "val": {
            "path": "val.txt",
            "documents": len(validation_documents),
            "authors": len(
                {item.author_id for item in validation_documents}
            ),
            "categories": _category_counts(validation_documents),
            "utf8_bytes": len(validation_text.encode("utf-8")),
            "sha256": sha256_text(validation_text),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest
