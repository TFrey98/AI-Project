"""Train the isolated Phase 31 typed whole-span resolver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import (
    load_model_warm_start,
    read_checkpoint,
    save_checkpoint,
)
from story_model.models import build_model
from story_model.provenance import training_fingerprints
from story_model.runtime import resolve_device, seed_everything
from story_model.train import (
    learning_rate_for_step,
    set_learning_rate,
    tokenizer_for_warm_start,
)
from story_model.typed_span_resolver import (
    ACTION_TO_INDEX,
    RESOLVE_ACTION,
    TYPED_SPAN_CONTROL_TOKENS,
    TypedSpanResolver,
    encode_typed_span_records,
    load_typed_span_records,
    typed_span_batch,
)


NEW_PARAMETER_NAMES = (
    "action_head.weight",
    "action_head.bias",
    "candidate_score.weight",
    "candidate_score.bias",
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be an object")
    if config.get("data", {}).get("type") != "typed_span_resolver":
        raise ValueError("Phase 31 requires data.type=typed_span_resolver")
    special_tokens = tuple(config.get("tokenizer", {}).get("special_tokens", ()))
    if special_tokens[-len(TYPED_SPAN_CONTROL_TOKENS) :] != TYPED_SPAN_CONTROL_TOKENS:
        raise ValueError("typed-span control tokens must be appended in canonical order")
    return config


def _random_batch(examples, batch_size: int, device: torch.device):
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    return typed_span_batch(examples, indices, device)


@torch.no_grad()
def _evaluate(model, examples, batch_size, batches, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    action_correct = 0
    resolve_correct = 0
    resolve_total = 0
    examples_seen = 0
    for _ in range(batches):
        batch = _random_batch(examples, batch_size, device)
        output = model(*batch[:5], batch[5], batch[6])
        assert output.loss is not None
        total_loss += float(output.loss)
        action_predictions = output.action_logits.argmax(dim=-1)
        candidate_predictions = output.candidate_logits.argmax(dim=-1)
        action_correct += int((action_predictions == batch[5]).sum())
        resolve = batch[6] != -100
        resolve_total += int(resolve.sum())
        resolve_correct += int(
            (
                (action_predictions == ACTION_TO_INDEX[RESOLVE_ACTION])
                & resolve
                & (candidate_predictions == batch[6])
            ).sum()
        )
        examples_seen += batch_size
    model.train()
    return {
        "loss": total_loss / batches,
        "action_accuracy": action_correct / examples_seen,
        "resolve_accuracy": (
            resolve_correct / resolve_total if resolve_total else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    checkpoint_config = config["checkpoint"]
    seed_everything(int(train_config.get("seed", 1337)))
    device = torch.device(resolve_device(train_config.get("device", "auto")))

    source_checkpoint = read_checkpoint(args.warm_start, map_location="cpu")
    source_config = source_checkpoint.get("extra", {}).get("config", {})
    source_model_config = source_config.get("model")
    if source_model_config != model_config:
        raise ValueError(
            "Phase 31 must preserve the foundation model configuration"
        )
    source_tokenizer, tokenizer = tokenizer_for_warm_start(
        source_checkpoint, config.get("tokenizer")
    )
    block_size = int(data_config["block_size"])
    backbone = build_model(
        model_config,
        vocabulary_size=tokenizer.vocab_size,
        block_size=block_size,
    )
    expanded = load_model_warm_start(
        backbone,
        source_checkpoint,
        source_vocabulary_size=source_tokenizer.vocab_size,
        destination_vocabulary_size=tokenizer.vocab_size,
    )
    model = TypedSpanResolver(backbone).to(device)

    train_records = load_typed_span_records(data_config["train_path"])
    validation_records = load_typed_span_records(data_config["val_path"])
    train_examples = encode_typed_span_records(
        train_records, tokenizer, block_size
    )
    validation_examples = encode_typed_span_records(
        validation_records, tokenizer, block_size
    )

    manifest_path = Path(data_config["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    max_steps = int(train_config["max_steps"])
    warmup_steps = int(train_config.get("warmup_steps", 0))
    maximum_rate = float(train_config["learning_rate"])
    minimum_rate = float(train_config.get("min_learning_rate", maximum_rate))
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    batch_size = int(data_config["batch_size"])
    eval_interval = int(train_config.get("eval_interval", 50))
    eval_batches = int(train_config.get("eval_batches", train_config.get("eval_iters", 20)))
    log_interval = int(train_config.get("log_interval", 25))
    patience_value = train_config.get("early_stopping_patience")
    patience = None if patience_value is None else int(patience_value)
    clip = float(train_config.get("gradient_clip", 1.0))
    checkpoint_dir = Path(checkpoint_config["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "architecture": "typed_span_resolver",
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "manifest": manifest,
        "warm_start": str(args.warm_start),
        **training_fingerprints(tokenizer.to_dict(), manifest),
    }
    print(f"startup: device {device}")
    print(f"startup: loading warm-start checkpoint {args.warm_start}")
    print(f"source completed steps: {source_checkpoint.get('step', 0)}")
    print("expanded vocabulary parameters: " + ", ".join(expanded))
    print("new architecture parameters: " + ", ".join(NEW_PARAMETER_NAMES))
    print(f"device: {device}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"tokenizer: byte_bpe")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print("loss objective: action+whole_candidate_selection")
    print(f"training examples: {len(train_examples):,}")
    print(f"validation examples: {len(validation_examples):,}")

    best_loss: float | None = None
    best_step = 0
    without_improvement = 0

    def evaluate_and_save(step: int) -> bool:
        nonlocal best_loss, best_step, without_improvement
        train_metrics = _evaluate(
            model, train_examples, batch_size, eval_batches, device
        )
        val_metrics = _evaluate(
            model, validation_examples, batch_size, eval_batches, device
        )
        print(
            f"update {step}: train loss {train_metrics['loss']:.4f}, "
            f"val loss {val_metrics['loss']:.4f}, "
            f"train resolve {train_metrics['resolve_accuracy']:.3f}, "
            f"val resolve {val_metrics['resolve_accuracy']:.3f}"
        )
        if math.isfinite(val_metrics["loss"]) and (
            best_loss is None or val_metrics["loss"] < best_loss
        ):
            best_loss = val_metrics["loss"]
            best_step = step
            without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                step,
                extra={
                    **metadata,
                    "best_validation_loss": best_loss,
                    "best_step": best_step,
                },
            )
            print(f"new best: val loss {best_loss:.4f} at update {step}")
        else:
            without_improvement += 1
        return patience is not None and without_improvement >= patience

    evaluate_and_save(0)
    completed_steps = 0
    for step in range(1, max_steps + 1):
        learning_rate = learning_rate_for_step(
            step - 1, max_steps, maximum_rate, minimum_rate, warmup_steps
        )
        set_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(accumulation):
            batch = _random_batch(train_examples, batch_size, device)
            output = model(*batch[:5], batch[5], batch[6])
            assert output.loss is not None
            accumulated_loss += float(output.loss.detach())
            (output.loss / accumulation).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        completed_steps = step
        if step % log_interval == 0:
            print(
                f"step {step}: loss {accumulated_loss / accumulation:.4f}, "
                f"lr {learning_rate:.6g}, grad {float(gradient_norm):.4f}"
            )
        if step % eval_interval == 0 and evaluate_and_save(step):
            print(f"early stopping at update {step}")
            break

    if completed_steps % eval_interval:
        evaluate_and_save(completed_steps)
    save_checkpoint(
        checkpoint_dir / "final.pt",
        model,
        optimizer,
        completed_steps,
        extra={
            **metadata,
            "best_validation_loss": best_loss,
            "best_step": best_step,
        },
    )
    print(f"best: val loss {best_loss:.4f} at update {best_step}")
    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"checkpoint: {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
