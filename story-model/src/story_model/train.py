"""Train configured language models and save checkpoints."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import (
    load_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from story_model.data import (
    build_tokenizer,
    get_batch,
    load_corpus_manifest,
    load_text_splits,
    tokenizer_from_dict,
)
from story_model.models import build_model
from story_model.runtime import (
    current_memory_bytes,
    resolve_device,
    seed_everything,
    synchronize_device,
)


def learning_rate_for_step(
    step: int,
    max_steps: int,
    maximum_rate: float,
    minimum_rate: float,
    warmup_steps: int,
) -> float:
    """Linear warmup followed by cosine decay."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    if warmup_steps >= max_steps:
        raise ValueError(
            "warmup_steps must be smaller than max_steps"
        )

    if warmup_steps > 0 and step < warmup_steps:
        return maximum_rate * (
            (step + 1) / warmup_steps
        )

    decay_steps = max_steps - warmup_steps

    if decay_steps <= 1:
        return minimum_rate

    decay_position = (
        step - warmup_steps
    ) / (decay_steps - 1)

    decay_position = min(
        max(decay_position, 0.0),
        1.0,
    )

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * decay_position)
    )

    return minimum_rate + cosine * (
        maximum_rate - minimum_rate
    )


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def validation_loss_improved(
    validation_loss: float,
    best_validation_loss: float | None,
) -> bool:
    """Return whether a finite validation loss is a strict improvement."""

    return (
        math.isfinite(validation_loss)
        and (
            best_validation_loss is None
            or validation_loss < best_validation_loss
        )
    )


def save_best_validation_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_metadata: dict,
    validation_loss: float,
) -> None:
    """Save a resumable checkpoint taken immediately after evaluation."""

    best_metadata = dict(checkpoint_metadata)
    best_metadata.update(
        {
            "best_validation_loss": validation_loss,
            "best_step": step,
            "evaluation_completed_step": step,
        }
    )

    save_checkpoint(
        path,
        model,
        optimizer,
        step,
        extra=best_metadata,
    )


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    output = {}

    for split, data in (
        ("train", train_data),
        ("val", val_data),
    ):
        losses = torch.zeros(eval_iters)

        for index in range(eval_iters):
            inputs, targets = get_batch(
                data,
                block_size,
                batch_size,
                device,
            )

            _, loss = model(inputs, targets)

            if loss is None:
                raise RuntimeError(
                    "Model did not return evaluation loss"
                )

            losses[index] = loss.item()

        output[split] = losses.mean().item()

    model.train()
    return output


def train(
    config_path: str,
    resume_path: str | None = None,
) -> None:
    config = yaml.safe_load(
        Path(config_path).read_text(
            encoding="utf-8"
        )
    )

    train_config = config["train"]
    data_config = config["data"]
    checkpoint_config = config["checkpoint"]

    max_steps = int(train_config["max_steps"])
    maximum_rate = float(
        train_config["learning_rate"]
    )
    minimum_rate = float(
        train_config.get(
            "min_learning_rate",
            maximum_rate,
        )
    )
    warmup_steps = int(
        train_config.get("warmup_steps", 0)
    )
    gradient_clip = float(
        train_config.get("gradient_clip", 0.0)
    )
    log_interval = int(
        train_config.get(
            "log_interval",
            train_config["eval_interval"],
        )
    )
    weight_decay = float(
        train_config.get("weight_decay", 0.01)
    )

    if log_interval < 1:
        raise ValueError(
            "log_interval must be positive"
        )

    seed_everything(train_config["seed"])
    device = resolve_device(train_config["device"])

    training_text, validation_text = load_text_splits(
        data_config
    )

    tokenizer_data = None
    resume_checkpoint = None

    if resume_path is not None:
        resume_checkpoint = read_checkpoint(
            resume_path,
            map_location="cpu",
        )
        tokenizer_data = (
            resume_checkpoint
            .get("extra", {})
            .get("tokenizer")
        )

    if tokenizer_data is not None:
        tokenizer = tokenizer_from_dict(tokenizer_data)
    else:
        tokenizer = build_tokenizer(
            training_text=training_text,
            config=config.get("tokenizer"),
        )

    checkpoint_metadata = {
        "config": config,
        "tokenizer": tokenizer.to_dict(),
    }
    corpus_manifest = load_corpus_manifest(data_config)

    if corpus_manifest is not None:
        checkpoint_metadata["corpus_manifest"] = (
            corpus_manifest
        )

    train_data = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long,
    )
    val_data = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long,
    )

    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=data_config["block_size"],
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=maximum_rate,
        weight_decay=weight_decay,
    )

    start_step = 0
    best_validation_loss = None
    best_step = None
    evaluation_completed_step = None

    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer,
            map_location="cpu",
            restore_rng=True,
        )

        start_step = int(
            checkpoint.get("step", 0)
        )

        checkpoint_extra = checkpoint.get("extra", {})
        saved_best_validation_loss = checkpoint_extra.get(
            "best_validation_loss"
        )

        if saved_best_validation_loss is not None:
            best_validation_loss = float(
                saved_best_validation_loss
            )

        saved_best_step = checkpoint_extra.get("best_step")

        if saved_best_step is not None:
            best_step = int(saved_best_step)

        saved_evaluation_step = checkpoint_extra.get(
            "evaluation_completed_step"
        )

        if saved_evaluation_step is not None:
            evaluation_completed_step = int(
                saved_evaluation_step
            )

        saved_config = (
            checkpoint
            .get("extra", {})
            .get("config")
        )

        if (
            saved_config is not None
            and saved_config["model"] != config["model"]
        ):
            raise ValueError(
                "Checkpoint model configuration "
                "does not match the current configuration"
            )

        if start_step >= max_steps:
            raise ValueError(
                f"Checkpoint has completed {start_step} steps, "
                f"but max_steps is {max_steps}"
            )

        print(
            f"resuming from: {resume_path}"
        )
        print(
            f"completed steps: {start_step}"
        )

    else:
        # Model construction consumes a different number of random
        # values for different architectures. Reset before evaluation
        # and training so matched experiments see identical batches
        # and dropout streams.
        seed_everything(train_config["seed"])

    if best_validation_loss is not None:
        checkpoint_metadata["best_validation_loss"] = (
            best_validation_loss
        )

    if best_step is not None:
        checkpoint_metadata["best_step"] = best_step

    print(f"device: {device}")
    print(f"parameters: {parameter_count:,}")
    print(f"tokenizer: {tokenizer.to_dict()['type']}")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print(
        f"training tokens: {len(train_data):,}"
    )
    print(
        f"validation tokens: {len(val_data):,}"
    )
    print(
        "training UTF-8 bytes: "
        f"{len(training_text.encode('utf-8')):,}"
    )
    print(
        "validation UTF-8 bytes: "
        f"{len(validation_text.encode('utf-8')):,}"
    )
    print(
        "training bytes/token: "
        f"{len(training_text.encode('utf-8')) / len(train_data):.3f}"
    )
    print(
        "validation bytes/token: "
        f"{len(validation_text.encode('utf-8')) / len(val_data):.3f}"
    )

    block_size = data_config["block_size"]
    batch_size = data_config["batch_size"]
    eval_interval = train_config["eval_interval"]
    eval_iters = train_config["eval_iters"]
    save_interval = checkpoint_config["save_interval"]
    best_path = (
        Path(checkpoint_config["dir"])
        / "best.pt"
    )

    peak_memory = current_memory_bytes(device)
    tokens_since_log = 0

    synchronize_device(device)
    timing_started = time.perf_counter()

    last_loss = None
    last_gradient_norm = None

    for step in range(start_step, max_steps):
        learning_rate = learning_rate_for_step(
            step=step,
            max_steps=max_steps,
            maximum_rate=maximum_rate,
            minimum_rate=minimum_rate,
            warmup_steps=warmup_steps,
        )

        set_learning_rate(
            optimizer,
            learning_rate,
        )

        evaluation_already_completed = (
            evaluation_completed_step == step
        )

        if (
            step % eval_interval == 0
            and not evaluation_already_completed
        ):
            synchronize_device(device)

            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )

            synchronize_device(device)

            print(
                f"update {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {learning_rate:.6g}"
            )

            if validation_loss_improved(
                losses["val"],
                best_validation_loss,
            ):
                best_validation_loss = losses["val"]
                best_step = step
                checkpoint_metadata[
                    "best_validation_loss"
                ] = best_validation_loss
                checkpoint_metadata["best_step"] = best_step

                save_best_validation_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    checkpoint_metadata=checkpoint_metadata,
                    validation_loss=best_validation_loss,
                )

                print(
                    "new best: "
                    f"val loss {best_validation_loss:.4f} "
                    f"at update {best_step}"
                )

            # Exclude evaluation time from throughput.
            tokens_since_log = 0
            timing_started = time.perf_counter()

        if evaluation_already_completed:
            evaluation_completed_step = None

        inputs, targets = get_batch(
            train_data,
            block_size,
            batch_size,
            device,
        )

        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError(
                "Model did not return training loss"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if gradient_clip > 0.0:
            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )
            )
        else:
            gradient_norm = torch.tensor(
                float("nan")
            )

        optimizer.step()

        last_loss = loss
        last_gradient_norm = gradient_norm
        tokens_since_log += inputs.numel()

        peak_memory = max(
            peak_memory,
            current_memory_bytes(device),
        )

        completed_steps = step + 1

        if completed_steps % log_interval == 0:
            synchronize_device(device)

            elapsed = (
                time.perf_counter()
                - timing_started
            )

            tokens_per_second = (
                tokens_since_log / elapsed
                if elapsed > 0
                else 0.0
            )

            memory_mebibytes = (
                current_memory_bytes(device)
                / (1024**2)
            )

            gradient_value = (
                last_gradient_norm.item()
                if last_gradient_norm is not None
                else float("nan")
            )

            print(
                f"step {completed_steps}: "
                f"loss {last_loss.item():.4f}, "
                f"lr {learning_rate:.6g}, "
                f"grad {gradient_value:.4f}, "
                f"tokens/s {tokens_per_second:,.0f}, "
                f"memory {memory_mebibytes:.1f} MiB"
            )

            tokens_since_log = 0
            timing_started = time.perf_counter()

        if completed_steps % save_interval == 0:
            synchronize_device(device)

            save_checkpoint(
                Path(checkpoint_config["dir"])
                / f"step_{completed_steps}.pt",
                model,
                optimizer,
                completed_steps,
                extra=checkpoint_metadata,
            )

            # Exclude checkpoint writing from throughput.
            synchronize_device(device)
            tokens_since_log = 0
            timing_started = time.perf_counter()

    final_losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        block_size=block_size,
        batch_size=batch_size,
        eval_iters=eval_iters,
        device=device,
    )

    final_path = (
        Path(checkpoint_config["dir"])
        / "final.pt"
    )

    if validation_loss_improved(
        final_losses["val"],
        best_validation_loss,
    ):
        best_validation_loss = final_losses["val"]
        best_step = max_steps
        checkpoint_metadata["best_validation_loss"] = (
            best_validation_loss
        )
        checkpoint_metadata["best_step"] = best_step

        save_best_validation_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            step=max_steps,
            checkpoint_metadata=checkpoint_metadata,
            validation_loss=best_validation_loss,
        )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        max_steps,
        extra=checkpoint_metadata,
    )

    print(
        f"final: train loss "
        f"{final_losses['train']:.4f}, "
        f"val loss {final_losses['val']:.4f}"
    )
    print(
        f"observed peak tensor memory: "
        f"{peak_memory / (1024**2):.1f} MiB"
    )
    print(
        f"best: val loss {best_validation_loss:.4f} "
        f"at update {best_step}"
    )
    print(f"best checkpoint: {best_path}")
    print(f"checkpoint: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training configuration.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint from which to resume.",
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
