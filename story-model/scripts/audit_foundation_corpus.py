"""Measure foundation corpus v2 with the original foundation tokenizer."""

from __future__ import annotations

import argparse

from story_model.foundation_audit import audit_foundation_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/processed_v2/manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/transformer_bpe_medium/best.pt",
    )
    parser.add_argument(
        "--minimum-training-tokens",
        type=int,
        default=50_000_000,
    )
    parser.add_argument(
        "--minimum-tokens-per-parameter",
        type=float,
        default=15.0,
    )
    args = parser.parse_args()
    audit = audit_foundation_corpus(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        minimum_training_tokens=args.minimum_training_tokens,
        minimum_tokens_per_parameter=args.minimum_tokens_per_parameter,
    )
    print(f"checkpoint: {audit['checkpoint']}")
    print(f"completed updates: {audit['checkpoint_step']:,}")
    print(f"parameters: {audit['parameters']:,}")
    print(f"vocabulary: {audit['vocabulary']:,}")
    print(
        "train: "
        f"{audit['training_tokens']:,} tokens, "
        f"{audit['training_utf8_bytes']:,} UTF-8 bytes, "
        f"{audit['training_bytes_per_token']:.3f} bytes/token"
    )
    print(
        "val: "
        f"{audit['validation_tokens']:,} tokens, "
        f"{audit['validation_utf8_bytes']:,} UTF-8 bytes, "
        f"{audit['validation_bytes_per_token']:.3f} bytes/token"
    )
    print(
        "training tokens/parameter: "
        f"{audit['training_tokens_per_parameter']:.2f}"
    )
    print("foundation corpus token gates: passed")


if __name__ == "__main__":
    main()
