"""Train and inspect a byte-level BPE tokenizer on a text corpus."""

from __future__ import annotations

import argparse
import time

from story_model.data import ByteBPETokenizer, load_text


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default="data/input.txt",
        help="UTF-8 training corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    text = load_text(args.path)
    byte_count = len(text.encode("utf-8"))

    started = time.perf_counter()
    tokenizer = ByteBPETokenizer.train(
        text=text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    encoded = tokenizer.encode(text)
    encoding_seconds = time.perf_counter() - started

    decoded = tokenizer.decode(encoded)

    print(f"UTF-8 bytes: {byte_count:,}")
    print(f"BPE tokens: {len(encoded):,}")
    print(f"vocabulary: {tokenizer.vocab_size:,}")
    print(f"learned merges: {len(tokenizer.merges):,}")
    print(
        f"bytes/token: {byte_count / len(encoded):.3f}"
    )
    print(
        f"token reduction: "
        f"{100 * (1 - len(encoded) / byte_count):.1f}%"
    )
    print(f"training time: {training_seconds:.2f}s")
    print(f"encoding time: {encoding_seconds:.2f}s")
    print(
        "roundtrip: "
        f"{'passed' if decoded == text else 'failed'}"
    )


if __name__ == "__main__":
    main()