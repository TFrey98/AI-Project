"""Build deterministic train and validation corpora from text documents.

The earliest configs in this project trained on one file (tiny_shakespeare)
split by raw character position: the first 90% of the text file was
"training," the last 10% was "validation." That's fine for a single short
work, but it breaks down once you train on many different books: a
position-based split has no idea where document boundaries are, so it can
easily put half of one novel in training and the other half in validation.
That's a form of *data leakage* — the model can partially "memorize" a
validation passage's neighboring sentences during training, making
validation loss look better than it should, i.e. a less honest measurement
of how well the model generalizes to text it has truly never seen.

This module fixes that by splitting at the DOCUMENT level: every source
file is assigned, whole, to either training or validation, never split
partway through. It also normalizes messy real-world text (mixed line
endings, Unicode variants, Project Gutenberg's boilerplate header/footer)
and writes out a manifest recording exactly which document went where and
its content hash — so a corpus can be verified as unchanged (or
regenerated identically) later. See data.py's load_text_splits() and
validate_corpus_manifest() for how that manifest gets checked before
every training/evaluation run.
"""

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
    # SHA-256 gives a short, fixed-length fingerprint of arbitrary-length
    # text: if even one character changes, the hash changes completely.
    # It's used throughout this project as a cheap way to answer "is this
    # exactly the text I think it is?" without storing or diffing the full
    # text — the manifest can record a corpus's identity in 64 hex
    # characters instead of megabytes of content.
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove standard Project Gutenberg wrappers when both exist.

    Free public-domain ebooks from Project Gutenberg (the source of most
    documents in data/raw/, downloaded via curl in this project) are
    wrapped in a standardized legal header and footer — licensing text,
    "how to help produce more ebooks" boilerplate, etc. — that has nothing
    to do with the actual work. Left in, that boilerplate would become
    part of the training corpus and the model would happily learn to
    generate snippets of Project Gutenberg's license text alongside prose,
    which is both undesirable output and wasted training signal. This
    function looks for the standard START/END marker lines and keeps only
    what's between them; if either marker is missing (e.g. a document
    from a different source), it conservatively leaves the text alone
    rather than guessing.
    """

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
    """Normalize Unicode/newlines while preserving paragraph structure.

    Text scraped or downloaded from varied sources rarely arrives in one
    consistent form. Two normalizations happen here:

    1. Line endings: Windows-style "\\r\\n" and old Mac-style "\\r" are
       both converted to plain "\\n", so downstream code (and the model
       itself) sees one consistent newline convention instead of the
       corpus containing an inconsistent, source-dependent mix.
    2. Unicode NFC normalization: the same visible character (e.g. "é")
       can be encoded as either one precomposed codepoint or as two
       codepoints (a base letter plus a separate combining accent) that
       render identically but compare as different strings/bytes. Without
       normalizing, the tokenizer would silently learn two different
       token sequences for what a human reads as the same character,
       depending on which encoding happened to be used in a given source
       file. NFC ("Normalization Form C") canonicalizes to the
       precomposed form, so identical-looking text is always identical
       at the byte level too — see tests/test_corpus.py's Gutenberg test
       for a concrete before/after example.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = unicodedata.normalize("NFC", text)

    # Trailing whitespace on each line and leading/trailing blank lines
    # around the whole document are stripped, but internal blank lines
    # (paragraph breaks) are deliberately preserved — normalization should
    # remove accidental noise, not the document's actual structure.
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()

    if not normalized:
        raise ValueError("corpus document is empty after normalization")

    return normalized + "\n"


def discover_documents(
    input_dir: str | Path,
) -> list[CorpusDocument]:
    """Find, normalize, and fingerprint every .txt file under input_dir.

    Requiring at least two documents (rather than accepting a single huge
    file) is what forces the document-level train/val split below to be
    meaningful — with only one document there's nothing to hold out
    wholesale, and the caller should use the single-file split_text() path
    in data.py instead. Rejecting exact duplicate documents (by content
    hash, not filename) guards against a specific, easy corpus-authoring
    mistake: accidentally including both a plain-text and a "annotated"
    copy of the same public-domain book, which would silently double that
    book's weight in training (and, worse, could land one copy in training
    and the other in validation — genuine leakage, since the model would
    then be "validated" on text nearly identical to something it trained
    on directly).
    """

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise ValueError(
            f"corpus input directory does not exist: {input_path}"
        )

    # Sorting by path (not directory-traversal order, which can vary by
    # filesystem/OS) is what makes discovery deterministic — the same
    # input_dir always produces documents in the same order, which in turn
    # is what makes split_documents()'s seeded shuffle below reproducible.
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
    """Split complete documents, targeting a validation byte fraction.

    Two goals are balanced here, and they can pull in different
    directions for a small number of documents:

    1. Hit roughly `validation_fraction` of total bytes in validation
       (e.g. 10%), so the held-out set is a reasonably representative
       SIZE of the corpus.
    2. Hold out at least `minimum_validation_documents` distinct
       documents (default 3), so validation isn't determined by a single
       author, genre, or writing style. With only one validation
       document, "validation loss" mostly measures "how well does the
       model predict THIS ONE BOOK," which can be a noisy, unrepresentative
       signal if that book happens to be stylistically unusual relative to
       the rest of the corpus. This is the same reasoning behind
       k-fold cross-validation in classical ML: a single held-out sample
       is a much noisier estimate of generalization than several.

    The seeded, deterministic shuffle (not a purely size-based or
    alphabetical pick) is what makes which documents land in validation
    reproducible across machines/runs while still being effectively
    "random" and not biased by, say, alphabetical author ordering.
    """

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

    # Sort first (undoes filesystem-dependent ordering), THEN shuffle with
    # a seeded RNG — this two-step process is what guarantees the same
    # `documents` + `seed` always produces the identical split, regardless
    # of what order discover_documents() happened to return them in.
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
    # Never require holding out more documents than would leave at least
    # one for training — minimum_validation_documents is a *target*, not
    # a guarantee, when the corpus itself is small.
    required_validation_documents = min(
        minimum_validation_documents,
        len(documents) - 1,
    )

    # Walk the shuffled list, greedily adding whole documents to
    # validation until BOTH the byte-fraction target and the
    # minimum-document-count target are satisfied. ranked[:-1] always
    # leaves at least the very last document for training, no matter how
    # small validation_fraction is — a model trained on zero documents
    # would be meaningless.
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
    # A distinctive triple-newline separator (rather than, say, a single
    # newline) marks document boundaries clearly enough in the raw text
    # that the model has some chance of learning "this is where one
    # work ends and an unrelated one begins" as a pattern, rather than
    # reading the corpus as one continuous, tonally-inconsistent document.
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
    """Build train.txt, val.txt, and a reproducibility manifest.

    This is the top-level function scripts/build_corpus.py calls. The
    manifest it writes is the corpus's "receipt": which documents went
    into which split, each document's byte count and hash, and hashes of
    the combined train.txt/val.txt themselves. data.py's
    validate_corpus_manifest() later re-hashes train.txt/val.txt before
    every training or evaluation run and compares against this manifest —
    so if data/raw/ or data/processed/ ever get edited without rerunning
    this function, that mismatch is caught immediately instead of
    silently corrupting a training run's results.
    """

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
