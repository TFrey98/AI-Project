"""Train count-conditioned unified resolve/clarify routing."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import read_checkpoint, save_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    ACTION_TO_INDEX,
    CLARIFY_ACTION,
    EXPANDED_CONTROL_TOKENS,
    RESOLVE_ACTION,
    load_expanded_records,
)
from story_model.models import build_model
from story_model.provenance import training_fingerprints
from story_model.runtime import resolve_device, seed_everything
from story_model.train import learning_rate_for_step, set_learning_rate
from story_model.unified_typed_span_resolver import (
    MODE_TO_INDEX,
    NO_SUPPORT_OPTION_INDEX,
    SUPPORT_MASK_VERSION,
    STRUCTURED_MODE,
    UnifiedTypedSpanResolver,
    encode_unified_records,
    unified_batch,
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be an object")
    if config.get("data", {}).get("type") != "unified_typed_span_resolver":
        raise ValueError("Phase 33 requires data.type=unified_typed_span_resolver")
    if tuple(config.get("tokenizer", {}).get("special_tokens", ())) != tuple(
        EXPANDED_CONTROL_TOKENS
    ):
        raise ValueError("Phase 33 must preserve the Phase 32 tokenizer controls")
    return config


def _balanced_training_batch(
    phase31_examples,
    phase32_examples,
    batch_size: int,
    device: torch.device,
):
    if batch_size < 2:
        raise ValueError("Phase 33 batch_size must be at least two")
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
    return unified_batch(combined, indices, device)


def _stratified_indices(records, examples_per_cell: int) -> tuple[int, ...]:
    """Choose a fixed panel covering each skill/action/case/width cell."""

    if examples_per_cell < 1:
        raise ValueError("eval_examples_per_cell must be positive")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            record.skill,
            record.expected_action,
            record.case,
            len(record.candidates),
        )
        groups[key].append(index)
    selected = []
    for key in sorted(groups):
        indices = groups[key]
        count = min(examples_per_cell, len(indices))
        if count == 1:
            selected.append(indices[0])
            continue
        selected.extend(
            indices[round(position * (len(indices) - 1) / (count - 1))]
            for position in range(count)
        )
    return tuple(sorted(selected))


@torch.no_grad()
def _evaluate(
    model,
    examples,
    records,
    indices,
    batch_size,
    device,
) -> dict[str, float]:
    model.eval()
    mode_loss_sum = 0.0
    mode_loss_total = 0
    ambiguity_action_loss_sum = 0.0
    ambiguity_action_loss_total = 0
    raw_candidate_loss_sum = 0.0
    raw_candidate_loss_total = 0
    mode_correct = 0
    structured_correct = 0
    structured_total = 0
    resolve_option_correct = 0
    raw_candidate_correct = 0
    resolve_total = 0
    clarify_sentinel_correct = 0
    clarify_total = 0
    multi_action_correct = 0
    multi_action_total = 0
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = unified_batch(examples, batch_indices, device)
        output = model(*batch[:5], batch[5], batch[6], batch[7])
        assert output.loss is not None
        if output.mode_loss is not None:
            mode_loss_sum += float(output.mode_loss) * len(batch_indices)
            mode_loss_total += len(batch_indices)
        structured = batch[6] != -100
        eligible_counts = batch[4][:, :NO_SUPPORT_OPTION_INDEX].sum(dim=-1)
        multi_eligible = structured & (eligible_counts >= 2)
        multi_count = int(multi_eligible.sum())
        if output.ambiguity_action_loss is not None:
            ambiguity_action_loss_sum += (
                float(output.ambiguity_action_loss) * multi_count
            )
            ambiguity_action_loss_total += multi_count
        modes = output.mode_logits.argmax(dim=-1)
        options = output.option_logits.argmax(dim=-1)
        candidate_positions = torch.arange(
            NO_SUPPORT_OPTION_INDEX, device=device
        ).unsqueeze(0)
        inventory_valid = candidate_positions < batch[7].unsqueeze(1)
        raw_candidate_logits = output.raw_option_logits[
            :, :NO_SUPPORT_OPTION_INDEX
        ].masked_fill(
            ~inventory_valid,
            torch.finfo(output.raw_option_logits.dtype).min,
        )
        raw_candidates = raw_candidate_logits.argmax(dim=-1)
        mode_correct += int((modes == batch[5]).sum())
        structured_total += int(structured.sum())
        structured_correct += int(
            (
                structured
                & (modes == MODE_TO_INDEX[STRUCTURED_MODE])
                & (options == batch[6])
            ).sum()
        )
        resolve = torch.tensor(
            [
                records[index].expected_action == RESOLVE_ACTION
                for index in batch_indices
            ],
            dtype=torch.bool,
            device=device,
        )
        clarify = structured & ~resolve
        expected_multi_actions = torch.where(
            resolve,
            torch.full_like(batch[6], ACTION_TO_INDEX[RESOLVE_ACTION]),
            torch.full_like(batch[6], ACTION_TO_INDEX[CLARIFY_ACTION]),
        )
        predicted_multi_actions = output.legacy_action_logits[:, :2].argmax(dim=-1)
        multi_action_total += multi_count
        multi_action_correct += int(
            (
                multi_eligible
                & (predicted_multi_actions == expected_multi_actions)
            ).sum()
        )
        resolve_total += int(resolve.sum())
        resolve_count = int(resolve.sum())
        if output.raw_candidate_loss is not None:
            raw_candidate_loss_sum += (
                float(output.raw_candidate_loss) * resolve_count
            )
            raw_candidate_loss_total += resolve_count
        clarify_total += int(clarify.sum())
        resolve_option_correct += int((resolve & (options == batch[6])).sum())
        raw_candidate_correct += int(
            (resolve & (raw_candidates == batch[6])).sum()
        )
        clarify_sentinel_correct += int(
            (clarify & (options == NO_SUPPORT_OPTION_INDEX)).sum()
        )
        total += len(batch_indices)
    model.train()
    loss = mode_loss_sum / mode_loss_total
    if ambiguity_action_loss_total:
        loss += ambiguity_action_loss_sum / ambiguity_action_loss_total
    if raw_candidate_loss_total:
        loss += raw_candidate_loss_sum / raw_candidate_loss_total
    return {
        "loss": loss,
        "mode_accuracy": mode_correct / total,
        "structured_accuracy": (
            structured_correct / structured_total if structured_total else 0.0
        ),
        "resolve_option_accuracy": (
            resolve_option_correct / resolve_total if resolve_total else 0.0
        ),
        "raw_candidate_top1_accuracy": (
            raw_candidate_correct / resolve_total if resolve_total else 0.0
        ),
        "clarify_sentinel_accuracy": (
            clarify_sentinel_correct / clarify_total if clarify_total else 0.0
        ),
        "multi_candidate_action_accuracy": (
            multi_action_correct / multi_action_total
            if multi_action_total
            else 1.0
        ),
        "multi_candidate_action_examples": multi_action_total,
    }


def _checkpoint_selection(
    phase31_metrics: dict[str, float],
    balanced_validation_loss: float,
    structured_floor: float,
    resolve_option_floor: float,
    sentinel_floor: float,
    multi_action_floor: float,
    mode_floor: float,
) -> tuple[bool, tuple[float, ...]]:
    """Rank eligible checkpoints by loss and reject Phase 31 regressions."""

    eligible = (
        phase31_metrics["structured_accuracy"] >= structured_floor
        and phase31_metrics["resolve_option_accuracy"] >= resolve_option_floor
        and phase31_metrics["raw_candidate_top1_accuracy"]
        >= resolve_option_floor
        and phase31_metrics["clarify_sentinel_accuracy"] >= sentinel_floor
        and phase31_metrics["multi_candidate_action_accuracy"]
        >= multi_action_floor
        and phase31_metrics["mode_accuracy"] >= mode_floor
    )
    if eligible:
        key = (
            1.0,
            -balanced_validation_loss,
            phase31_metrics["structured_accuracy"],
            phase31_metrics["resolve_option_accuracy"],
            phase31_metrics["raw_candidate_top1_accuracy"],
            phase31_metrics["clarify_sentinel_accuracy"],
            phase31_metrics["multi_candidate_action_accuracy"],
            phase31_metrics["mode_accuracy"],
        )
    else:
        key = (
            0.0,
            phase31_metrics["structured_accuracy"],
            phase31_metrics["resolve_option_accuracy"],
            phase31_metrics["raw_candidate_top1_accuracy"],
            phase31_metrics["clarify_sentinel_accuracy"],
            phase31_metrics["multi_candidate_action_accuracy"],
            phase31_metrics["mode_accuracy"],
            -balanced_validation_loss,
        )
    return eligible, key


def _checkpoint_slot(
    eligible: bool,
    selection_key: tuple[float, ...],
    best_key: tuple[float, ...] | None,
    diagnostic_key: tuple[float, ...] | None,
) -> str | None:
    """Keep eligible and diagnostic checkpoints in distinct slots."""

    if eligible:
        return "best" if best_key is None or selection_key > best_key else None
    return (
        "diagnostic"
        if diagnostic_key is None or selection_key > diagnostic_key
        else None
    )


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
    if extra.get("architecture") != "expanded_typed_span_resolver":
        raise ValueError("Phase 33 must warm-start from a Phase 32 resolver")
    source_config = extra.get("config")
    if not isinstance(source_config, dict) or source_config.get("model") != model_config:
        raise ValueError("Phase 32 and Phase 33 model configurations differ")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 33 requires the Phase 32 byte-BPE tokenizer")
    configured_tokens = tuple(config["tokenizer"]["special_tokens"])
    if tokenizer.special_tokens != configured_tokens:
        raise ValueError("Phase 33 tokenizer does not exactly match Phase 32")

    block_size = int(data_config["block_size"])
    backbone = build_model(model_config, tokenizer.vocab_size, block_size)
    model = UnifiedTypedSpanResolver(backbone)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    phase31_train = load_expanded_records(data_config["phase31_train_path"])
    phase31_val = load_expanded_records(data_config["phase31_val_path"])
    phase32_train = load_expanded_records(data_config["train_path"])
    phase32_val = load_expanded_records(data_config["val_path"])
    phase31_train_examples = encode_unified_records(
        phase31_train, tokenizer, block_size
    )
    phase32_train_examples = encode_unified_records(
        phase32_train, tokenizer, block_size
    )
    train_examples = phase31_train_examples + phase32_train_examples
    train_records = phase31_train + phase32_train
    phase31_val_examples = encode_unified_records(
        phase31_val, tokenizer, block_size
    )
    phase32_val_examples = encode_unified_records(
        phase32_val, tokenizer, block_size
    )

    phase31_manifest = json.loads(
        Path(data_config["phase31_manifest_path"]).read_text(encoding="utf-8")
    )
    phase32_manifest = json.loads(
        Path(data_config["manifest_path"]).read_text(encoding="utf-8")
    )
    combined_manifest = {"phase31": phase31_manifest, "phase32": phase32_manifest}
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
    eval_batch_size = int(data_config.get("eval_batch_size", batch_size))
    eval_interval = int(train_config.get("eval_interval", 50))
    eval_examples_per_cell = int(
        train_config.get("eval_examples_per_cell", 4)
    )
    phase31_structured_floor = float(
        train_config.get("phase31_structured_floor", 0.95)
    )
    phase31_resolve_option_floor = float(
        train_config.get("phase31_resolve_option_floor", 0.995)
    )
    phase31_sentinel_floor = float(
        train_config.get("phase31_sentinel_floor", 0.95)
    )
    phase31_multi_action_floor = float(
        train_config.get("phase31_multi_action_floor", 0.95)
    )
    phase31_mode_floor = float(train_config.get("phase31_mode_floor", 0.98))
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    for name, value in (
        ("phase31_structured_floor", phase31_structured_floor),
        ("phase31_resolve_option_floor", phase31_resolve_option_floor),
        ("phase31_sentinel_floor", phase31_sentinel_floor),
        ("phase31_multi_action_floor", phase31_multi_action_floor),
        ("phase31_mode_floor", phase31_mode_floor),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    log_interval = int(train_config.get("log_interval", 25))
    clip = float(train_config.get("gradient_clip", 1.0))
    patience_value = train_config.get("early_stopping_patience")
    patience = None if patience_value is None else int(patience_value)
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "unified_typed_span_resolver",
        "support_mask_version": SUPPORT_MASK_VERSION,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "manifest": combined_manifest,
        "parent_phase32_checkpoint": str(args.warm_start),
        **training_fingerprints(tokenizer.to_dict(), combined_manifest),
    }
    train_eval_indices = _stratified_indices(
        train_records, eval_examples_per_cell
    )
    phase31_val_indices = _stratified_indices(
        phase31_val, eval_examples_per_cell
    )
    phase32_val_indices = _stratified_indices(
        phase32_val, eval_examples_per_cell
    )

    print(f"startup: device {device}")
    print(f"startup: loading Phase 32 resolver {args.warm_start}")
    print(f"source completed steps: {checkpoint.get('step', 0)}")
    print("new architecture parameters: none")
    print("structured options: up to 4 real candidates + no-support sentinel")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print(
        "loss objective: generate_mode+multi_candidate_action"
        "+raw_candidate_ranking"
    )
    print(
        f"training examples: {len(train_examples):,} "
        f"(Phase 31 {len(phase31_train):,} + Phase 32 {len(phase32_train):,})"
    )
    print(
        f"validation examples: {len(phase31_val_examples) + len(phase32_val_examples):,} "
        f"(Phase 31 {len(phase31_val):,} + Phase 32 {len(phase32_val):,})"
    )
    print(
        "fixed validation panels: "
        f"train {len(train_eval_indices):,}, "
        f"Phase 31 {len(phase31_val_indices):,}, "
        f"Phase 32 {len(phase32_val_indices):,}; "
        f"eval batch size {eval_batch_size}"
    )
    print(
        "Phase 31 checkpoint floors: "
        f"structured {phase31_structured_floor:.3f}, "
        f"resolve option {phase31_resolve_option_floor:.3f}, "
        f"sentinel {phase31_sentinel_floor:.3f}, "
        f"multi-candidate action {phase31_multi_action_floor:.3f}, "
        f"mode {phase31_mode_floor:.3f}"
    )

    best_loss: float | None = None
    best_step: int | None = None
    best_selection_key = None
    diagnostic_loss: float | None = None
    diagnostic_step: int | None = None
    diagnostic_selection_key = None
    last_eligible = False
    without_improvement = 0

    def evaluate_and_save(step: int) -> bool:
        nonlocal best_loss, best_step, best_selection_key
        nonlocal diagnostic_loss, diagnostic_step, diagnostic_selection_key
        nonlocal last_eligible, without_improvement
        train_metrics = _evaluate(
            model,
            train_examples,
            train_records,
            train_eval_indices,
            eval_batch_size,
            device,
        )
        phase31_metrics = _evaluate(
            model,
            phase31_val_examples,
            phase31_val,
            phase31_val_indices,
            eval_batch_size,
            device,
        )
        phase32_metrics = _evaluate(
            model,
            phase32_val_examples,
            phase32_val,
            phase32_val_indices,
            eval_batch_size,
            device,
        )
        balanced_validation_loss = (
            phase31_metrics["loss"] + phase32_metrics["loss"]
        ) / 2.0
        print(
            f"update {step}: train loss {train_metrics['loss']:.4f}, "
            f"balanced val loss {balanced_validation_loss:.4f}, "
            f"Phase 31 val {phase31_metrics['loss']:.4f}, "
            f"Phase 32 val {phase32_metrics['loss']:.4f}, "
            f"train structured {train_metrics['structured_accuracy']:.3f}, "
            f"Phase 31 structured {phase31_metrics['structured_accuracy']:.3f}, "
            f"Phase 32 structured {phase32_metrics['structured_accuracy']:.3f}, "
            f"Phase 31 option {phase31_metrics['resolve_option_accuracy']:.3f}, "
            "Phase 31 raw candidate "
            f"{phase31_metrics['raw_candidate_top1_accuracy']:.3f}, "
            f"Phase 31 sentinel {phase31_metrics['clarify_sentinel_accuracy']:.3f}, "
            "Phase 31 multi action "
            f"{phase31_metrics['multi_candidate_action_accuracy']:.3f}, "
            f"Phase 31 mode {phase31_metrics['mode_accuracy']:.3f}, "
            f"Phase 32 sentinel {phase32_metrics['clarify_sentinel_accuracy']:.3f}"
        )
        eligible, selection_key = _checkpoint_selection(
            phase31_metrics,
            balanced_validation_loss,
            phase31_structured_floor,
            phase31_resolve_option_floor,
            phase31_sentinel_floor,
            phase31_multi_action_floor,
            phase31_mode_floor,
        )
        print(f"checkpoint eligible: {'yes' if eligible else 'no'}")
        last_eligible = eligible
        slot = None
        if math.isfinite(balanced_validation_loss):
            slot = _checkpoint_slot(
                eligible,
                selection_key,
                best_selection_key,
                diagnostic_selection_key,
            )
        if slot == "best":
            best_loss = balanced_validation_loss
            best_step = step
            best_selection_key = selection_key
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
                    "checkpoint_eligible": True,
                },
            )
            print(
                f"new best: balanced val loss {best_loss:.4f} "
                f"at update {step}; eligible=True"
            )
        else:
            without_improvement += 1
            if slot == "diagnostic":
                diagnostic_loss = balanced_validation_loss
                diagnostic_step = step
                diagnostic_selection_key = selection_key
                save_checkpoint(
                    checkpoint_dir / "diagnostic-ineligible.pt",
                    model,
                    optimizer,
                    step,
                    extra={
                        **metadata,
                        "diagnostic_validation_loss": diagnostic_loss,
                        "diagnostic_step": diagnostic_step,
                        "checkpoint_eligible": False,
                    },
                )
                print(
                    "new diagnostic-ineligible: balanced val loss "
                    f"{diagnostic_loss:.4f} at update {step}"
                )
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
            output = model(*batch[:5], batch[5], batch[6], batch[7])
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
            "best_checkpoint_eligible": best_step is not None,
            "checkpoint_eligible": last_eligible,
        },
    )
    if best_loss is None:
        print("best eligible checkpoint: none")
    else:
        print(
            f"best eligible: balanced val loss {best_loss:.4f} "
            f"at update {best_step}"
        )
        print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    if diagnostic_loss is not None:
        print(
            "diagnostic ineligible: balanced val loss "
            f"{diagnostic_loss:.4f} at update {diagnostic_step}"
        )
        print(
            "diagnostic checkpoint: "
            f"{checkpoint_dir / 'diagnostic-ineligible.pt'}"
        )
    print(f"checkpoint: {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
