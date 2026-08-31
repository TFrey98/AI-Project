"""Continue a Phase 31 resolver on the Phase 32 direct-span family."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import read_checkpoint, save_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    ACTION_TO_INDEX,
    EXPANDED_CONTROL_TOKENS,
    RESOLVE_ACTION,
    ExpandedTypedSpanResolver,
    encode_expanded_records,
    expanded_batch,
    load_expanded_records,
)
from story_model.models import build_model
from story_model.provenance import training_fingerprints
from story_model.runtime import resolve_device, seed_everything
from story_model.train import learning_rate_for_step, set_learning_rate


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be an object")
    if config.get("data", {}).get("type") != "expanded_typed_span_resolver":
        raise ValueError("Phase 32 requires data.type=expanded_typed_span_resolver")
    if tuple(config.get("tokenizer", {}).get("special_tokens", ())) != tuple(
        EXPANDED_CONTROL_TOKENS
    ):
        raise ValueError("Phase 32 must preserve the Phase 31 tokenizer controls")
    return config


def _random_batch(examples, batch_size: int, device: torch.device):
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    return expanded_batch(examples, indices, device)


def _balanced_training_batch(
    phase31_examples,
    phase32_examples,
    batch_size: int,
    device: torch.device,
):
    if batch_size < 2:
        raise ValueError("Phase 32 batch_size must be at least two")
    phase31_count = batch_size // 2
    phase32_count = batch_size - phase31_count
    combined = phase31_examples + phase32_examples
    indices = torch.randint(
        0, len(phase31_examples), (phase31_count,)
    ).tolist()
    indices.extend(
        len(phase31_examples) + index
        for index in torch.randint(
            0, len(phase32_examples), (phase32_count,)
        ).tolist()
    )
    return expanded_batch(combined, indices, device)


@torch.no_grad()
def _evaluate(model, examples, batch_size, batches, device) -> dict[str, float]:
    model.eval()
    loss = 0.0
    action_correct = 0
    resolve_correct = 0
    resolve_total = 0
    total = 0
    for _ in range(batches):
        batch = _random_batch(examples, batch_size, device)
        output = model(*batch[:5], batch[5], batch[6])
        assert output.loss is not None
        loss += float(output.loss)
        actions = output.action_logits.argmax(dim=-1)
        candidates = output.candidate_logits.argmax(dim=-1)
        action_correct += int((actions == batch[5]).sum())
        resolve = batch[6] != -100
        resolve_total += int(resolve.sum())
        resolve_correct += int(
            (
                resolve
                & (actions == ACTION_TO_INDEX[RESOLVE_ACTION])
                & (candidates == batch[6])
            ).sum()
        )
        total += batch_size
    model.train()
    return {
        "loss": loss / batches,
        "action_accuracy": action_correct / total,
        "resolve_accuracy": resolve_correct / resolve_total if resolve_total else 0.0,
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
    seed_everything(int(train_config.get("seed", 1337)))
    device = torch.device(resolve_device(train_config.get("device", "auto")))

    checkpoint = read_checkpoint(args.warm_start, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "typed_span_resolver":
        raise ValueError("Phase 32 must warm-start from a Phase 31 resolver")
    source_config = extra.get("config")
    if not isinstance(source_config, dict) or source_config.get("model") != model_config:
        raise ValueError("Phase 31 and Phase 32 model configurations differ")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 32 requires the Phase 31 byte-BPE tokenizer")
    configured_tokens = tuple(config["tokenizer"]["special_tokens"])
    if tokenizer.special_tokens != configured_tokens:
        raise ValueError("Phase 32 tokenizer does not exactly match Phase 31")

    block_size = int(data_config["block_size"])
    backbone = build_model(model_config, tokenizer.vocab_size, block_size)
    model = ExpandedTypedSpanResolver(backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    phase31_train = load_expanded_records(data_config["phase31_train_path"])
    phase31_val = load_expanded_records(data_config["phase31_val_path"])
    phase32_train = load_expanded_records(data_config["train_path"])
    phase32_val = load_expanded_records(data_config["val_path"])
    phase31_train_examples = encode_expanded_records(
        phase31_train, tokenizer, block_size
    )
    phase32_train_examples = encode_expanded_records(
        phase32_train, tokenizer, block_size
    )
    train_examples = phase31_train_examples + phase32_train_examples
    phase31_val_examples = encode_expanded_records(
        phase31_val, tokenizer, block_size
    )
    phase32_val_examples = encode_expanded_records(
        phase32_val, tokenizer, block_size
    )

    phase31_manifest = json.loads(
        Path(data_config["phase31_manifest_path"]).read_text(encoding="utf-8")
    )
    phase32_manifest = json.loads(
        Path(data_config["manifest_path"]).read_text(encoding="utf-8")
    )
    combined_manifest = {
        "phase31": phase31_manifest,
        "phase32": phase32_manifest,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    maximum_rate = float(train_config["learning_rate"])
    minimum_rate = float(train_config.get("min_learning_rate", maximum_rate))
    max_steps = int(train_config["max_steps"])
    warmup_steps = int(train_config.get("warmup_steps", 0))
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    batch_size = int(data_config["batch_size"])
    eval_interval = int(train_config.get("eval_interval", 50))
    eval_batches = int(train_config.get("eval_batches", 20))
    log_interval = int(train_config.get("log_interval", 25))
    clip = float(train_config.get("gradient_clip", 1.0))
    patience_value = train_config.get("early_stopping_patience")
    patience = None if patience_value is None else int(patience_value)
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "expanded_typed_span_resolver",
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "manifest": combined_manifest,
        "parent_phase31_checkpoint": str(args.warm_start),
        **training_fingerprints(tokenizer.to_dict(), combined_manifest),
    }

    print(f"startup: device {device}")
    print(f"startup: loading Phase 31 resolver {args.warm_start}")
    print(f"source completed steps: {checkpoint.get('step', 0)}")
    print("new architecture parameters: none")
    print("candidate widths: 2, 3, and 4")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print("loss objective: action+whole_candidate_selection")
    print(
        f"training examples: {len(train_examples):,} "
        f"(Phase 31 {len(phase31_train):,} + Phase 32 {len(phase32_train):,})"
    )
    print(
        f"validation examples: {len(phase31_val_examples) + len(phase32_val_examples):,} "
        f"(Phase 31 {len(phase31_val):,} + Phase 32 {len(phase32_val):,})"
    )

    best_loss: float | None = None
    best_step = 0
    without_improvement = 0

    def evaluate_and_save(step: int) -> bool:
        nonlocal best_loss, best_step, without_improvement
        train_metrics = _evaluate(
            model, train_examples, batch_size, eval_batches, device
        )
        phase31_metrics = _evaluate(
            model, phase31_val_examples, batch_size, eval_batches, device
        )
        phase32_metrics = _evaluate(
            model, phase32_val_examples, batch_size, eval_batches, device
        )
        balanced_validation_loss = (
            phase31_metrics["loss"] + phase32_metrics["loss"]
        ) / 2.0
        print(
            f"update {step}: train loss {train_metrics['loss']:.4f}, "
            f"balanced val loss {balanced_validation_loss:.4f}, "
            f"Phase 31 val {phase31_metrics['loss']:.4f}, "
            f"Phase 32 val {phase32_metrics['loss']:.4f}, "
            f"train resolve {train_metrics['resolve_accuracy']:.3f}, "
            f"Phase 31 resolve {phase31_metrics['resolve_accuracy']:.3f}, "
            f"Phase 32 resolve {phase32_metrics['resolve_accuracy']:.3f}"
        )
        if math.isfinite(balanced_validation_loss) and (
            best_loss is None or balanced_validation_loss < best_loss
        ):
            best_loss = balanced_validation_loss
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
            print(
                f"new best: balanced val loss {best_loss:.4f} "
                f"at update {step}"
            )
        else:
            without_improvement += 1
        return patience is not None and without_improvement >= patience

    evaluate_and_save(0)
    completed_steps = 0
    for step in range(1, max_steps + 1):
        learning_rate = learning_rate_for_step(
            step - 1,
            max_steps,
            maximum_rate,
            minimum_rate,
            warmup_steps,
        )
        set_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(accumulation):
            batch = _balanced_training_batch(
                phase31_train_examples,
                phase32_train_examples,
                batch_size,
                device,
            )
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
    print(f"best: balanced val loss {best_loss:.4f} at update {best_step}")
    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"checkpoint: {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
