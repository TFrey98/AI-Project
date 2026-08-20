"""Deterministically evaluate a saved language-model checkpoint.

train.py's `estimate_loss` measures progress cheaply, by sampling a handful
of random batches every so often — good for watching a training curve, not
precise enough for a final, reportable number. This module computes the
exact loss over an ENTIRE split (train and/or val), deterministically (no
randomness at all — every token is scored exactly once, in order), and
converts that loss into several human-interpretable metrics: perplexity
and bits-per-byte. It also supports "cross-evaluation" — scoring one
checkpoint's model+tokenizer against a DIFFERENT corpus than it was
trained on, which is how this project measured how well a
tiny_shakespeare-only model generalizes to a much larger, more varied
corpus (and vice versa).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import torch
import yaml

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, load_text_splits
from story_model.generate import load_generation_metadata
from story_model.models import build_model
from story_model.progress import ProgressReporter, format_duration
from story_model.runtime import resolve_device, synchronize_device


def read_evaluation_data_config(
    config_path: str,
) -> dict:
    """Read a full training config or a data-only YAML mapping.

    Supports two shapes for --data-config's argument so the same flag can
    point at either a complete config file (configs/transformer_bpe.yaml,
    which has a top-level "data:" section among others) or a minimal,
    hand-written data-only YAML mapping (just path/train_split or
    train_path/val_path directly, no "data:" wrapper needed) — this
    flexibility is what lets cross-evaluation reuse an existing training
    config file directly, without requiring a separate, redundant
    data-only file to be maintained for every corpus.
    """

    loaded = yaml.safe_load(
        Path(config_path).read_text(encoding="utf-8")
    )

    if not isinstance(loaded, dict):
        raise ValueError(
            "evaluation data config must be a YAML mapping"
        )

    data_config = loaded.get("data", loaded)

    if not isinstance(data_config, dict):
        raise ValueError(
            "evaluation data section must be a YAML mapping"
        )

    return data_config


def calculate_bits_per_byte(
    mean_loss: float,
    token_count: int,
    byte_count: int,
) -> float:
    """Convert a mean per-token loss (in nats) into bits per UTF-8 byte.

    This metric exists to solve a real problem this project ran into:
    cross-entropy loss and perplexity are both computed PER TOKEN, but
    "how big is a token" is entirely tokenizer-dependent — one BPE token
    might cover 4 characters while one char-level token covers exactly 1.
    That makes loss/perplexity numbers from a char-level model and a BPE
    model literally incomparable at face value: a BPE model's higher raw
    perplexity per token doesn't mean it's a worse model, since each of
    its tokens is also carrying more information.

    Bits-per-byte fixes this by normalizing to a tokenizer-independent
    unit — the UTF-8 BYTE, which every tokenizer in this project ultimately
    encodes text into one way or another. The conversion: total loss
    (mean_loss * token_count) is in "nats" (PyTorch's cross_entropy uses
    the natural log internally); dividing by log(2) converts nats to bits
    (log base conversion — this is the same math as converting any
    logarithm's base); dividing by byte_count then expresses that total
    bit-cost per byte of actual text covered. Lower bits/byte is better
    (fewer bits needed to encode/predict each byte), and — critically —
    it's the number this project actually compares across the char vs.
    BPE ablations (see the transformer_bpe* vs transformer_small
    comparisons in the training log), where raw perplexity would mislead.
    """

    if token_count < 1:
        raise ValueError("token_count must be positive")

    if byte_count < 1:
        raise ValueError("byte_count must be positive")

    total_nats = mean_loss * token_count
    return total_nats / (math.log(2.0) * byte_count)


@torch.no_grad()
def evaluate_token_stream(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str | torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[float, int]:
    """Return mean next-token loss over every target in a stream.

    Unlike train.py's estimate_loss (which samples random windows), this
    walks the ENTIRE token stream deterministically and exactly once:
    it chops the data into consecutive, non-overlapping block_size chunks
    (batched together for speed), scores every single one, and — since a
    stream's length is rarely an exact multiple of block_size — separately
    scores whatever leftover "tail" tokens remain after the last full
    chunk, so literally no token in the split goes unscored. This is what
    makes the resulting loss a precise, reproducible figure suitable for
    reporting and comparison, rather than train.py's cheaper, noisier
    running estimate.
    """

    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional token stream")

    if len(data) < 2:
        raise ValueError("data must contain at least two tokens")

    if block_size < 1:
        raise ValueError("block_size must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.eval()

    # Every token except the very first has a "predict me from what came
    # before" target, so there are exactly len(data) - 1 predictions to
    # score in total.
    prediction_count = len(data) - 1
    full_token_count = (
        prediction_count // block_size
    ) * block_size

    total_negative_log_likelihood = 0.0
    evaluated_tokens = 0

    if full_token_count > 0:
        # Reshape the flat token stream into (num_chunks, block_size)
        # windows — inputs[i] is one block_size chunk, targets[i] is that
        # same chunk shifted one token forward, mirroring the
        # (x, y) convention get_batch() uses for training.
        inputs = data[:full_token_count].reshape(
            -1,
            block_size,
        )
        targets = data[1 : full_token_count + 1].reshape(
            -1,
            block_size,
        )

        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[
                start : start + batch_size
            ].to(device)
            batch_targets = targets[
                start : start + batch_size
            ].to(device)

            _, loss = model(
                batch_inputs,
                batch_targets,
            )

            if loss is None:
                raise RuntimeError(
                    "Model did not return evaluation loss"
                )

            # model(...) returns the MEAN loss over its batch, so
            # multiplying back by token_count before accumulating (and
            # dividing by the grand total at the very end) is what
            # correctly weights every individual token equally in the
            # final average — batches of different sizes shouldn't be
            # averaged together as if they were equally sized.
            token_count = batch_inputs.numel()
            total_negative_log_likelihood += (
                loss.item() * token_count
            )
            evaluated_tokens += token_count

            if progress_callback is not None:
                progress_callback(
                    evaluated_tokens,
                    prediction_count,
                )

    if full_token_count < prediction_count:
        # The leftover tokens that didn't fill a complete block_size chunk
        # get scored separately, as one final, shorter batch of size 1 —
        # ensuring the reported loss genuinely covers every token in the
        # split, not just the ones that happened to divide evenly.
        tail_inputs = data[
            full_token_count:-1
        ].unsqueeze(0).to(device)
        tail_targets = data[
            full_token_count + 1:
        ].unsqueeze(0).to(device)

        _, tail_loss = model(
            tail_inputs,
            tail_targets,
        )

        if tail_loss is None:
            raise RuntimeError(
                "Model did not return evaluation loss"
            )

        tail_count = tail_inputs.numel()
        total_negative_log_likelihood += (
            tail_loss.item() * tail_count
        )
        evaluated_tokens += tail_count

        if progress_callback is not None:
            progress_callback(
                evaluated_tokens,
                prediction_count,
            )

    if was_training:
        model.train()

    mean_loss = (
        total_negative_log_likelihood
        / evaluated_tokens
    )

    return mean_loss, evaluated_tokens


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str | None = None,
    split: str = "both",
    device_name: str | None = None,
    data_config: dict | None = None,
    progress: bool = False,
    progress_interval: float = 10.0,
    emit: Callable[[str], None] = print,
) -> dict:
    if split not in {"train", "val", "both"}:
        raise ValueError(
            "split must be 'train', 'val', or 'both'"
        )

    if progress_interval < 0.0:
        raise ValueError("progress_interval cannot be negative")

    total_started_at = time.monotonic()

    if progress:
        emit(f"evaluation: loading checkpoint {checkpoint_path}")

    checkpoint_started_at = time.monotonic()
    checkpoint = read_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    checkpoint_seconds = time.monotonic() - checkpoint_started_at

    # Rebuilds the exact model architecture and tokenizer the checkpoint
    # was trained with, straight from its embedded metadata — see
    # generate.py's load_generation_metadata for why this matters (a
    # checkpoint's weights are only meaningful under the architecture and
    # vocabulary they were trained with).
    config, tokenizer = load_generation_metadata(
        checkpoint,
        config_path,
    )

    configured_device = (
        device_name
        if device_name is not None
        else config["train"]["device"]
    )
    device = resolve_device(configured_device)

    if progress:
        emit(
            f"evaluation: device {device}; split {split}; "
            f"checkpoint loaded in {format_duration(checkpoint_seconds)}"
        )

    model_started_at = time.monotonic()
    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=config["data"]["block_size"],
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    synchronize_device(device)
    model_seconds = time.monotonic() - model_started_at

    if progress:
        emit(
            "evaluation: model ready in "
            f"{format_duration(model_seconds)}"
        )

    # `data_config` lets a caller supply a DIFFERENT corpus than the one
    # the checkpoint was trained on ("cross-evaluation" — see --data-config
    # in main() below), while still using the checkpoint's own model
    # weights and tokenizer. This is deliberately NOT the same as
    # retraining: the tokenizer's vocabulary stays exactly as it was
    # trained (even if that vocabulary is a poor fit for the new corpus's
    # vocabulary/style), which is precisely what makes the resulting
    # bits/byte number a meaningful measure of how well the ORIGINAL
    # model generalizes to unfamiliar text.
    evaluation_data_config = (
        data_config
        if data_config is not None
        else config["data"]
    )
    if progress:
        emit("evaluation: loading and verifying corpus files")

    data_started_at = time.monotonic()
    train_text, val_text = load_text_splits(
        evaluation_data_config
    )
    data_seconds = time.monotonic() - data_started_at
    selected_splits = (
        ("train", "val")
        if split == "both"
        else (split,)
    )
    loaded_text = {
        "train": train_text,
        "val": val_text,
    }
    selected_text = {
        split_name: loaded_text[split_name]
        for split_name in selected_splits
    }
    del loaded_text, train_text, val_text

    if progress:
        emit(
            "evaluation: corpus verified in "
            f"{format_duration(data_seconds)}; encoding only "
            + ", ".join(selected_splits)
        )

    results = {
        "checkpoint": checkpoint_path,
        "step": int(checkpoint.get("step", 0)),
        "device": str(device),
        "data_override": data_config is not None,
        "splits": {},
        "timings_seconds": {
            "checkpoint_load": checkpoint_seconds,
            "model_setup": model_seconds,
            "data_load_and_verify": data_seconds,
            "encoding": {},
            "scoring": {},
        },
    }

    for split_name in selected_splits:
        text = selected_text.pop(split_name)
        utf8_bytes = len(text.encode("utf-8"))
        encoding_started_at = time.monotonic()

        if progress:
            emit(
                f"evaluation: encoding {split_name} "
                f"({utf8_bytes:,} UTF-8 bytes)"
            )

        bpe_reporter: ProgressReporter | None = None

        def report_bpe(
            completed: int,
            total: int,
            current_tokens: int,
        ) -> None:
            nonlocal bpe_reporter

            if not progress or total < 1:
                return

            if bpe_reporter is None:
                bpe_reporter = ProgressReporter(
                    label=f"encode {split_name}",
                    total=total,
                    unit="merge passes",
                    interval_seconds=progress_interval,
                    emit=emit,
                )

            bpe_reporter.update(
                completed,
                detail=f"current tokens {current_tokens:,}",
            )

        if isinstance(tokenizer, ByteBPETokenizer):
            encoded = tokenizer.encode(
                text,
                progress_callback=report_bpe,
            )
        else:
            encoded = tokenizer.encode(text)

        encoding_seconds = time.monotonic() - encoding_started_at
        results["timings_seconds"]["encoding"][split_name] = (
            encoding_seconds
        )

        if progress:
            emit(
                f"evaluation: encoded {split_name}: "
                f"{len(encoded):,} tokens in "
                f"{format_duration(encoding_seconds)}"
            )

        data = torch.tensor(encoded, dtype=torch.long)
        scoring_started_at = time.monotonic()
        scoring_reporter = (
            ProgressReporter(
                label=f"score {split_name}",
                total=len(data) - 1,
                unit="tokens",
                interval_seconds=progress_interval,
                emit=emit,
            )
            if progress
            else None
        )

        loss, token_count = evaluate_token_stream(
            model=model,
            data=data,
            block_size=config["data"]["block_size"],
            batch_size=int(
                evaluation_data_config.get(
                    "batch_size",
                    config["data"]["batch_size"],
                )
            ),
            device=device,
            progress_callback=(
                (
                    lambda completed, _total: scoring_reporter.update(
                        completed
                    )
                )
                if scoring_reporter is not None
                else None
            ),
        )
        synchronize_device(device)
        scoring_seconds = time.monotonic() - scoring_started_at
        results["timings_seconds"]["scoring"][split_name] = (
            scoring_seconds
        )

        # Every scored token except the first (which has no preceding
        # context to predict it FROM) contributes its own byte length —
        # this is what byte_count means in calculate_bits_per_byte, and is
        # why it's computed per-tokenizer (token_byte_length) rather than
        # assumed to be uniform: a BPE token can represent anywhere from 1
        # to several bytes.
        target_byte_count = sum(
            tokenizer.token_byte_length(token_id)
            for token_id in encoded[1:]
        )

        results["splits"][split_name] = {
            "loss": loss,
            # perplexity = e^loss: "the model is, on average, as
            # uncertain as if it were guessing uniformly among this many
            # equally-likely options" — a more intuitive (if still
            # tokenizer-dependent) way to read a cross-entropy loss number
            # than the raw nats value itself. Lower is better; perplexity
            # 1.0 would mean perfect, fully-confident prediction.
            "perplexity": math.exp(loss),
            "tokens": token_count,
            "bytes": target_byte_count,
            "bits_per_byte": calculate_bits_per_byte(
                mean_loss=loss,
                token_count=token_count,
                byte_count=target_byte_count,
            ),
        }

        if progress:
            emit(
                f"evaluation: scored {split_name} in "
                f"{format_duration(scoring_seconds)}"
            )

        del data, encoded, text

    results["timings_seconds"]["total"] = (
        time.monotonic() - total_started_at
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved checkpoint.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Fallback configuration for older checkpoints "
            "without embedded metadata."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="both",
    )

    parser.add_argument(
        "--data-config",
        default=None,
        help=(
            "Optional full or data-only YAML config whose corpus "
            "replaces the checkpoint corpus for cross-evaluation."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override: auto, cpu, mps, or cuda.",
    )

    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress/ETA lines (default: 10).",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stage, progress, and ETA output.",
    )

    args = parser.parse_args()

    evaluation_data_config = (
        read_evaluation_data_config(args.data_config)
        if args.data_config is not None
        else None
    )

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        device_name=args.device,
        data_config=evaluation_data_config,
        progress=not args.no_progress,
        progress_interval=args.progress_interval,
    )

    print(f"checkpoint: {results['checkpoint']}")
    print(f"completed updates: {results['step']}")
    print(f"device: {results['device']}")

    if args.data_config is not None:
        print(f"evaluation data: {args.data_config}")

    for split_name, metrics in results["splits"].items():
        print(
            f"{split_name}: "
            f"loss {metrics['loss']:.4f}, "
            f"perplexity {metrics['perplexity']:.4f}, "
            f"bits/byte {metrics['bits_per_byte']:.4f}, "
            f"tokens {metrics['tokens']:,}, "
            f"bytes {metrics['bytes']:,}"
        )

    print(
        "evaluation time: "
        f"{format_duration(results['timings_seconds']['total'])}"
    )


if __name__ == "__main__":
    main()
