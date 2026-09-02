"""Train the Phase 34 proposer, optionally with Phase 34d boundary loss."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from story_model.checkpoint import read_checkpoint, save_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_CONTROL_TOKENS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_OBJECTIVE_VERSION,
    EXCLUDED_PROPOSER_CASES,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    ExplicitOffsetCandidateProposer,
    decode_proposed_spans,
    encode_proposal_records,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
)
from story_model.models import build_model
from story_model.provenance import training_fingerprints
from story_model.runtime import resolve_device, seed_everything
from story_model.train import learning_rate_for_step, set_learning_rate
from story_model.unified_typed_span_resolver import (
    SUPPORT_MASK_VERSION,
    UnifiedTypedSpanResolver,
)
try:
    from scripts.phase34c_tag_confusion_decision import (
        add_tag_sequence,
        new_tag_audit,
        summarize_tag_audit,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34c_tag_confusion_decision import (  # type: ignore[no-redef]
        add_tag_sequence,
        new_tag_audit,
        summarize_tag_audit,
    )


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be an object")
    if config.get("data", {}).get("type") != "explicit_offset_candidate_proposer":
        raise ValueError(
            "Phase 34 requires data.type=explicit_offset_candidate_proposer"
        )
    if tuple(config.get("tokenizer", {}).get("special_tokens", ())) != tuple(
        EXPANDED_CONTROL_TOKENS
    ):
        raise ValueError("Phase 34 must preserve the Phase 33c tokenizer")
    boundary_loss_weight = float(
        config.get("train", {}).get("boundary_loss_weight", 0.0)
    )
    if boundary_loss_weight < 0.0 or not math.isfinite(boundary_loss_weight):
        raise ValueError("boundary_loss_weight must be finite and nonnegative")
    return config


def _eligible_records(path: str | Path):
    return tuple(
        record
        for record in load_expanded_records(path)
        if proposal_record_is_eligible(record)
    )


def _balanced_batch(
    phase31_examples,
    phase32_examples,
    batch_size: int,
    tokenizer: ByteBPETokenizer,
    source_width: int,
    device: torch.device,
):
    if batch_size < 2:
        raise ValueError("Phase 34 batch size must be at least two")
    phase31_count = batch_size // 2
    phase32_count = batch_size - phase31_count
    combined = phase31_examples + phase32_examples
    indices = torch.randint(0, len(phase31_examples), (phase31_count,)).tolist()
    indices.extend(
        len(phase31_examples) + index
        for index in torch.randint(
            0, len(phase32_examples), (phase32_count,)
        ).tolist()
    )
    return proposal_batch(combined, indices, tokenizer, source_width, device)


def _stratified_indices(records, examples_per_cell: int) -> tuple[int, ...]:
    if examples_per_cell < 1:
        raise ValueError("eval_examples_per_cell must be positive")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[
            (
                record.skill,
                record.expected_action,
                record.case,
                len(record.candidates),
            )
        ].append(index)
    selected = []
    for key in sorted(groups):
        indices = groups[key]
        count = min(examples_per_cell, len(indices))
        if count == 1:
            selected.append(indices[0])
        else:
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
    tokenizer,
    batch_size,
    device,
    boundary_loss_weight,
) -> dict:
    model.eval()
    loss_sum = 0.0
    loss_total = 0
    tag_loss_sum = 0.0
    start_loss_sum = 0.0
    start_positions = 0
    end_loss_sum = 0.0
    end_positions = 0
    rows = []
    tag_audit = new_tag_audit()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = proposal_batch(
            examples,
            batch_indices,
            tokenizer,
            model.source_width,
            device,
        )
        output = model(*batch, boundary_loss_weight=boundary_loss_weight)
        assert output.loss is not None
        supervised = int((batch[3] != -100).sum())
        loss_sum += float(output.loss) * supervised
        loss_total += supervised
        assert output.tag_loss is not None
        assert output.boundary_start_loss is not None
        assert output.boundary_end_loss is not None
        tag_loss_sum += float(output.tag_loss) * supervised
        start_loss_sum += (
            float(output.boundary_start_loss)
            * output.boundary_start_positions
        )
        start_positions += output.boundary_start_positions
        end_loss_sum += (
            float(output.boundary_end_loss) * output.boundary_end_positions
        )
        end_positions += output.boundary_end_positions
        logits = output.tag_logits.detach().cpu()
        targets = batch[3].detach().cpu()
        for offset, index in enumerate(batch_indices):
            supervised_mask = targets[offset] != -100
            byte_logits = logits[offset][supervised_mask].float()
            byte_targets = targets[offset][supervised_mask]
            byte_predictions = byte_logits.argmax(dim=-1)
            byte_nll = F.cross_entropy(
                byte_logits, byte_targets, reduction="none"
            )
            add_tag_sequence(
                tag_audit,
                byte_targets.tolist(),
                byte_predictions.tolist(),
                byte_nll.tolist(),
            )
            predicted = decode_proposed_spans(
                records[index].prompt,
                examples[index],
                logits[offset],
                tokenizer,
            )
            rows.append(
                proposal_metric_row(
                    records[index], examples[index].gold_spans, predicted
                )
            )
    metrics = proposal_metrics(rows)
    metrics["loss"] = loss_sum / loss_total if loss_total else 0.0
    metrics["tag_loss"] = tag_loss_sum / loss_total if loss_total else 0.0
    metrics["boundary_start_loss"] = (
        start_loss_sum / start_positions if start_positions else 0.0
    )
    metrics["boundary_end_loss"] = (
        end_loss_sum / end_positions if end_positions else 0.0
    )
    boundary_components = []
    if start_positions:
        boundary_components.append(metrics["boundary_start_loss"])
    if end_positions:
        boundary_components.append(metrics["boundary_end_loss"])
    metrics["boundary_loss"] = (
        sum(boundary_components) / len(boundary_components)
        if boundary_components
        else 0.0
    )
    metrics["boundary_start_positions"] = start_positions
    metrics["boundary_end_positions"] = end_positions
    tag_metrics = summarize_tag_audit(tag_audit)
    for name in (
        "gold_outside_as_inside_rate",
        "gold_begin_as_inside_rate",
        "pre_start_bleed_rate",
        "end_spill_rate",
        "positive_type_accuracy",
    ):
        metrics[name] = tag_metrics[name]
    model.train()
    return metrics


def _selection(
    metrics: dict,
    precision_floor: float,
    recall_floor: float,
    answer_floor: float,
    type_floor: float,
) -> tuple[bool, tuple[float, ...]]:
    eligible = (
        metrics["exact_span_precision"] >= precision_floor
        and metrics["exact_span_recall"] >= recall_floor
        and metrics["answer_candidate_recall"] >= answer_floor
        and metrics["boundary_type_accuracy"] >= type_floor
        and metrics["offset_validity_rate"] == 1.0
        and metrics["proposal_overflow_rate"] == 0.0
    )
    if eligible:
        return True, (
            1.0,
            -metrics["loss"],
            metrics["answer_candidate_recall"],
            metrics["exact_span_f1"],
            metrics["boundary_type_accuracy"],
        )
    return False, (
        0.0,
        metrics["answer_candidate_recall"],
        metrics["exact_span_f1"],
        metrics["boundary_type_accuracy"],
        metrics["offset_validity_rate"],
        -metrics["proposal_overflow_rate"],
        -metrics["loss"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    data_config = config["data"]
    train_config = config["train"]
    seed_everything(int(train_config.get("seed", 1337)))
    device = torch.device(resolve_device(train_config.get("device", "auto")))

    checkpoint = read_checkpoint(args.warm_start, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "unified_typed_span_resolver":
        raise ValueError("Phase 34 must warm-start from Phase 33c")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("warm-start checkpoint predates Phase 33c routing")
    if extra.get("checkpoint_eligible") is not True:
        raise ValueError("Phase 34 requires an eligible Phase 33c best checkpoint")
    source_config = extra.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("Phase 33c checkpoint has no config")
    if source_config.get("model") != config.get("model"):
        raise ValueError("Phase 33c and Phase 34 model configurations differ")
    if int(source_config.get("data", {}).get("block_size", -1)) != int(
        data_config["block_size"]
    ):
        raise ValueError("Phase 33c and Phase 34 block sizes differ")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 34 requires byte-BPE")
    if tokenizer.special_tokens != tuple(config["tokenizer"]["special_tokens"]):
        raise ValueError("Phase 34 tokenizer differs from Phase 33c")

    block_size = int(data_config["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    resolver = UnifiedTypedSpanResolver(backbone)
    resolver.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = ExplicitOffsetCandidateProposer(resolver, tokenizer).to(device)

    phase31_train = _eligible_records(data_config["phase31_train_path"])
    phase31_val = _eligible_records(data_config["phase31_val_path"])
    phase32_train = _eligible_records(data_config["train_path"])
    phase32_val = _eligible_records(data_config["val_path"])
    phase31_train_examples = encode_proposal_records(
        phase31_train, tokenizer, block_size
    )
    phase32_train_examples = encode_proposal_records(
        phase32_train, tokenizer, block_size
    )
    phase31_val_examples = encode_proposal_records(
        phase31_val, tokenizer, block_size
    )
    phase32_val_examples = encode_proposal_records(
        phase32_val, tokenizer, block_size
    )
    train_examples = phase31_train_examples + phase32_train_examples
    train_records = phase31_train + phase32_train
    val_examples = phase31_val_examples + phase32_val_examples
    val_records = phase31_val + phase32_val

    optimizer = torch.optim.AdamW(
        tuple(model.proposer_parameters()),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    max_steps = int(train_config["max_steps"])
    maximum_rate = float(train_config["learning_rate"])
    minimum_rate = float(train_config.get("min_learning_rate", maximum_rate))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    batch_size = int(data_config["batch_size"])
    eval_batch_size = int(data_config.get("eval_batch_size", batch_size))
    eval_interval = int(train_config.get("eval_interval", 50))
    log_interval = int(train_config.get("log_interval", 25))
    clip = float(train_config.get("gradient_clip", 1.0))
    patience_value = train_config.get("early_stopping_patience")
    patience = None if patience_value is None else int(patience_value)
    examples_per_cell = int(train_config.get("eval_examples_per_cell", 4))
    train_indices = _stratified_indices(train_records, examples_per_cell)
    val_indices = _stratified_indices(val_records, examples_per_cell)
    precision_floor = float(train_config.get("span_precision_floor", 0.98))
    recall_floor = float(train_config.get("span_recall_floor", 0.98))
    answer_floor = float(train_config.get("answer_candidate_floor", 0.995))
    type_floor = float(train_config.get("type_accuracy_floor", 0.99))
    boundary_loss_weight = float(
        train_config.get("boundary_loss_weight", 0.0)
    )
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "phase31": json.loads(
            Path(data_config["phase31_manifest_path"]).read_text(encoding="utf-8")
        ),
        "phase32": json.loads(
            Path(data_config["manifest_path"]).read_text(encoding="utf-8")
        ),
    }
    metadata = {
        "architecture": "explicit_offset_candidate_proposer",
        "explicit_offset_proposer_version": EXPLICIT_OFFSET_PROPOSER_VERSION,
        "support_mask_version": SUPPORT_MASK_VERSION,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "manifest": manifests,
        "parent_phase33_checkpoint": str(args.warm_start),
        "parent_phase33_step": checkpoint.get("step", 0),
        "boundary_objective_version": (
            BOUNDARY_OBJECTIVE_VERSION if boundary_loss_weight else None
        ),
        "boundary_loss_weight": boundary_loss_weight,
        "excluded_proposer_cases": list(EXCLUDED_PROPOSER_CASES),
        **training_fingerprints(tokenizer.to_dict(), manifests),
    }

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(f"startup: device {device}")
    print(f"startup: loading frozen Phase 33c resolver {args.warm_start}")
    print(f"source completed steps: {checkpoint.get('step', 0)}")
    print("frozen architecture parameters: backbone, action_head, candidate_score")
    print("new architecture parameters: " + ", ".join(trainable_names))
    print(f"proposal byte width: {model.source_width}")
    if boundary_loss_weight:
        print("loss objective: typed_byte_BIO+boundary_transition")
        print(f"boundary loss weight: {boundary_loss_weight:g}")
        print("boundary terms: gold_begin + first_gold_outside_after_span")
    else:
        print("loss objective: typed_byte_BIO")
    print("excluded proposer cases: wrong_type (retained in oracle gate)")
    print(
        f"training examples: {len(train_examples):,} "
        f"(Phase 31 {len(phase31_train):,} + Phase 32 {len(phase32_train):,})"
    )
    print(
        f"fixed panels: train {len(train_indices):,}, val {len(val_indices):,}"
    )

    best_key = None
    best_loss = None
    best_step = None
    diagnostic_key = None
    without_improvement = 0
    last_eligible = False

    def evaluate_and_save(step: int) -> bool:
        nonlocal best_key, best_loss, best_step, diagnostic_key
        nonlocal without_improvement, last_eligible
        train_metrics = _evaluate(
            model,
            train_examples,
            train_records,
            train_indices,
            tokenizer,
            eval_batch_size,
            device,
            boundary_loss_weight,
        )
        val_metrics = _evaluate(
            model,
            val_examples,
            val_records,
            val_indices,
            tokenizer,
            eval_batch_size,
            device,
            boundary_loss_weight,
        )
        eligible, selection_key = _selection(
            val_metrics,
            precision_floor,
            recall_floor,
            answer_floor,
            type_floor,
        )
        last_eligible = eligible
        print(
            f"update {step}: train loss {train_metrics['loss']:.4f}, "
            f"val loss {val_metrics['loss']:.4f}, "
            f"start loss {val_metrics['boundary_start_loss']:.4f}, "
            f"end loss {val_metrics['boundary_end_loss']:.4f}, "
            f"B->I {val_metrics['gold_begin_as_inside_rate']:.3f}, "
            f"end spill {val_metrics['end_spill_rate']:.3f}, "
            f"span precision {val_metrics['exact_span_precision']:.3f}, "
            f"span recall {val_metrics['exact_span_recall']:.3f}, "
            f"answer recall {val_metrics['answer_candidate_recall']:.3f}, "
            f"type {val_metrics['boundary_type_accuracy']:.3f}, "
            f"overflow {val_metrics['proposal_overflow_rate']:.3f}, "
            f"checkpoint eligible: {'yes' if eligible else 'no'}"
        )
        slot = None
        if math.isfinite(val_metrics["loss"]):
            if eligible and (best_key is None or selection_key > best_key):
                slot = "best"
            elif not eligible and (
                diagnostic_key is None or selection_key > diagnostic_key
            ):
                slot = "diagnostic"
        if slot == "best":
            best_key = selection_key
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
                    "checkpoint_eligible": True,
                },
            )
            print(f"new best: val loss {best_loss:.4f} at update {step}")
        elif slot == "diagnostic":
            diagnostic_key = selection_key
            without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "diagnostic-ineligible.pt",
                model,
                optimizer,
                step,
                extra={
                    **metadata,
                    "diagnostic_validation_loss": val_metrics["loss"],
                    "diagnostic_step": step,
                    "checkpoint_eligible": False,
                },
            )
            print(f"new diagnostic-ineligible at update {step}")
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
        accumulated_tag_loss = 0.0
        accumulated_start_loss = 0.0
        accumulated_end_loss = 0.0
        accumulated_start_positions = 0
        accumulated_end_positions = 0
        for _ in range(accumulation):
            batch = _balanced_batch(
                phase31_train_examples,
                phase32_train_examples,
                batch_size,
                tokenizer,
                model.source_width,
                device,
            )
            output = model(
                *batch, boundary_loss_weight=boundary_loss_weight
            )
            assert output.loss is not None
            assert output.tag_loss is not None
            assert output.boundary_start_loss is not None
            assert output.boundary_end_loss is not None
            accumulated_loss += float(output.loss.detach())
            accumulated_tag_loss += float(output.tag_loss.detach())
            accumulated_start_loss += float(
                output.boundary_start_loss.detach()
            )
            accumulated_end_loss += float(output.boundary_end_loss.detach())
            accumulated_start_positions += output.boundary_start_positions
            accumulated_end_positions += output.boundary_end_positions
            (output.loss / accumulation).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            tuple(model.proposer_parameters()), clip
        )
        optimizer.step()
        completed_steps = step
        if step % log_interval == 0:
            print(
                f"step {step}: loss {accumulated_loss / accumulation:.4f}, "
                f"tag {accumulated_tag_loss / accumulation:.4f}, "
                f"start {accumulated_start_loss / accumulation:.4f}, "
                f"end {accumulated_end_loss / accumulation:.4f}, "
                f"boundary positions "
                f"{accumulated_start_positions}/{accumulated_end_positions}, "
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
    if best_step is None:
        print("best eligible checkpoint: none")
    else:
        print(f"best eligible: val loss {best_loss:.4f} at update {best_step}")
        print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"checkpoint: {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
