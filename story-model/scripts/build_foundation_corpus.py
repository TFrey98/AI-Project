"""Build the author-disjoint, provenance-aware foundation corpus v2."""

from __future__ import annotations

import argparse

from story_model.corpus_v2 import (
    CorpusQualityGates,
    build_foundation_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw_v2")
    parser.add_argument(
        "--catalog",
        default="corpus_catalogs/foundation_v2.json",
    )
    parser.add_argument("--output-dir", default="data/processed_v2")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-validation-authors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="en")
    parser.add_argument("--minimum-documents", type=int, default=50)
    parser.add_argument("--minimum-authors", type=int, default=25)
    parser.add_argument(
        "--minimum-training-bytes",
        type=int,
        default=100_000_000,
    )
    parser.add_argument("--minimum-categories", type=int, default=5)
    parser.add_argument(
        "--maximum-author-fraction",
        type=float,
        default=0.10,
    )
    args = parser.parse_args()
    manifest = build_foundation_corpus(
        input_dir=args.input_dir,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        minimum_validation_authors=args.minimum_validation_authors,
        required_language=args.language,
        gates=CorpusQualityGates(
            minimum_documents=args.minimum_documents,
            minimum_authors=args.minimum_authors,
            minimum_training_bytes=args.minimum_training_bytes,
            minimum_categories=args.minimum_categories,
            maximum_author_fraction=args.maximum_author_fraction,
        ),
    )
    print(f"documents: {manifest['summary']['documents']:,}")
    print(f"authors: {manifest['summary']['authors']:,}")
    print(f"categories: {len(manifest['summary']['categories']):,}")

    for split in ("train", "val"):
        summary = manifest[split]
        print(
            f"{split}: {summary['documents']:,} documents, "
            f"{summary['authors']:,} authors, "
            f"{summary['utf8_bytes']:,} UTF-8 bytes"
        )

    print(f"manifest: {args.output_dir.rstrip('/')}/manifest.json")
    print("foundation corpus byte and diversity gates: passed")


if __name__ == "__main__":
    main()
