"""Build deterministic document-level training and validation corpora."""

from __future__ import annotations

import argparse

from story_model.corpus import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory searched recursively for UTF-8 .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory that receives train.txt, val.txt, and manifest.json.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Approximate fraction of UTF-8 bytes assigned to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic document-assignment seed.",
    )
    parser.add_argument(
        "--minimum-validation-documents",
        type=int,
        default=3,
        help=(
            "Minimum number of complete validation documents; "
            "at least one document is always left for training."
        ),
    )

    args = parser.parse_args()

    manifest = build_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        minimum_validation_documents=(
            args.minimum_validation_documents
        ),
    )

    print(f"documents: {len(manifest['documents']):,}")
    print(
        "train: "
        f"{manifest['train']['documents']:,} documents, "
        f"{manifest['train']['utf8_bytes']:,} UTF-8 bytes"
    )
    print(
        "val: "
        f"{manifest['val']['documents']:,} documents, "
        f"{manifest['val']['utf8_bytes']:,} UTF-8 bytes"
    )
    print(
        "manifest: "
        f"{args.output_dir.rstrip('/')}/manifest.json"
    )


if __name__ == "__main__":
    main()
