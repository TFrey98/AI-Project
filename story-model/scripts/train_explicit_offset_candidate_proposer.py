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

from story_model.boundary_counterbalance import (
    BOUNDARY_COUNTERBALANCE_VERSION,
    validate_counterbalance_manifest,
)
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
    FACTORIZED_BOUNDARY_TYPE_VERSION,
    OUTSIDE_TAG,
    TOKEN_END_GEOMETRY_VERSION,
    TOKEN_WIDTH_GEOMETRY_VERSION,
    ExplicitOffsetCandidateProposer,
    decode_proposed_spans,
    encode_proposal_records,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
)
from story_model.models import build_model
from story_model.provenance import canonical_json_sha256, training_fingerprints
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
    geometry_version = config.get("train", {}).get(
        "token_width_geometry_version"
    )
    end_geometry_version = config.get("train", {}).get(
        "token_end_geometry_version"
    )
    factorized_version = config.get("train", {}).get(
        "factorized_boundary_type_version"
    )
    counterbalance_version = config.get("train", {}).get(
        "boundary_counterbalance_version"
    )
    if sum(
        value is not None
        for value in (
            geometry_version,
            end_geometry_version,
            factorized_version,
        )
    ) > 1:
        raise ValueError(
            "token-width, token-end, and factorized heads are mutually "
            "exclusive"
        )
    if geometry_version is not None:
        if type(geometry_version) is not int or (
            geometry_version != TOKEN_WIDTH_GEOMETRY_VERSION
        ):
            raise ValueError("unsupported token_width_geometry_version")
        if abs(boundary_loss_weight - 1.0) > 1.0e-9:
            raise ValueError(
                "token-width geometry requires boundary_loss_weight=1.0"
            )
    if end_geometry_version is not None:
        if type(end_geometry_version) is not int or (
            end_geometry_version != TOKEN_END_GEOMETRY_VERSION
        ):
            raise ValueError("unsupported token_end_geometry_version")
        if abs(boundary_loss_weight - 1.0) > 1.0e-9:
            raise ValueError(
                "token-end geometry requires boundary_loss_weight=1.0"
            )
        if not config.get("train", {}).get("token_end_premise_path"):
            raise ValueError(
                "token-end geometry requires token_end_premise_path"
            )
    if factorized_version is not None:
        if type(factorized_version) is not int or (
            factorized_version != FACTORIZED_BOUNDARY_TYPE_VERSION
        ):
            raise ValueError("unsupported factorized_boundary_type_version")
        if abs(boundary_loss_weight - 1.0) > 1.0e-9:
            raise ValueError(
                "factorized boundary/type requires boundary_loss_weight=1.0"
            )
    if counterbalance_version is not None:
        if type(counterbalance_version) is not int or (
            counterbalance_version != BOUNDARY_COUNTERBALANCE_VERSION
        ):
            raise ValueError("unsupported boundary_counterbalance_version")
        if abs(boundary_loss_weight - 1.0) > 1.0e-9:
            raise ValueError(
                "boundary counterbalance requires boundary_loss_weight=1.0"
            )
        if any(
            value is not None
            for value in (
                geometry_version,
                end_geometry_version,
                factorized_version,
            )
        ):
            raise ValueError(
                "boundary counterbalance cannot change proposer architecture"
            )
        required = (
            "counterbalance_train_path",
            "counterbalance_val_path",
            "counterbalance_manifest_path",
        )
        missing = [key for key in required if key not in config.get("data", {})]
        if missing:
            raise ValueError(
                "boundary counterbalance data paths are missing: "
                + ", ".join(missing)
            )
        accumulation = int(
            config.get("train", {}).get("gradient_accumulation_steps", 1)
        )
        counterbalance_batches = int(
            config.get("train", {}).get(
                "counterbalance_microbatches_per_update", 1
            )
        )
        if not 0 < counterbalance_batches < accumulation:
            raise ValueError(
                "counterbalance microbatches must be between zero and the "
                "gradient accumulation count"
            )
    return config


def _read_token_end_premise(train_config: dict):
    if train_config.get("token_end_geometry_version") is None:
        return None, None
    path = Path(train_config["token_end_premise_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("token_end_premise_version") != 1:
        raise ValueError("token-end premise report has an unsupported version")
    decision = report.get("decision", {})
    if (
        decision.get("branch") != "token_end_geometry_indicated"
        or decision.get("training_authorized") is not True
    ):
        raise ValueError(
            "token-end premise audit does not authorize Phase 34g training"
        )
    return path, report


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


def _sample_batch(
    examples,
    batch_size: int,
    tokenizer: ByteBPETokenizer,
    source_width: int,
    device: torch.device,
):
    if batch_size < 1:
        raise ValueError("sample batch size must be positive")
    indices = torch.randint(0, len(examples), (batch_size,)).tolist()
    return proposal_batch(examples, indices, tokenizer, source_width, device)


def _counterbalance_slots(
    step: int,
    accumulation: int,
    count: int,
) -> tuple[int, ...]:
    """Rotate counterbalance microbatches without changing update size."""

    if accumulation < 1 or not 0 < count < accumulation:
        raise ValueError("invalid counterbalance accumulation schedule")
    start = (step - 1) % accumulation
    return tuple((start + offset) % accumulation for offset in range(count))


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


def _focus_stratified_indices(
    examples,
    focus_token_ids,
    examples_per_token: int,
    tokenizer: ByteBPETokenizer,
) -> tuple[int, ...]:
    """Build a fixed panel that represents every counterbalanced identity."""

    if examples_per_token < 1:
        raise ValueError("counterbalance eval examples per token must be positive")
    groups = {int(token_id): [] for token_id in focus_token_ids}
    for index, example in enumerate(examples):
        present = set()
        byte_cursor = 0
        for position in range(example.prompt_token_count):
            token_id = example.input_ids[position]
            width = tokenizer.token_byte_length(token_id)
            if token_id in groups and any(
                tag != OUTSIDE_TAG
                for tag in example.prompt_byte_tags[
                    byte_cursor : byte_cursor + width
                ]
            ):
                present.add(token_id)
            byte_cursor += width
        for token_id in present:
            groups[token_id].append(index)
    selected = []
    for token_id in sorted(groups):
        indices = groups[token_id]
        if not indices:
            raise ValueError(
                f"counterbalance panel has no examples for token {token_id}"
            )
        count = min(examples_per_token, len(indices))
        if count == 1:
            selected.append(indices[0])
        else:
            selected.extend(
                indices[round(position * (len(indices) - 1) / (count - 1))]
                for position in range(count)
            )
    return tuple(sorted(set(selected)))


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
    boundary_class_loss_sum = 0.0
    boundary_class_positions = 0
    type_loss_sum = 0.0
    type_positions = 0
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
        if output.boundary_class_loss is not None:
            boundary_class_loss_sum += (
                float(output.boundary_class_loss) * supervised
            )
            boundary_class_positions += supervised
        if output.type_loss is not None:
            type_loss_sum += float(output.type_loss) * output.type_positions
            type_positions += output.type_positions
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
    if boundary_class_positions:
        metrics["boundary_class_loss"] = (
            boundary_class_loss_sum / boundary_class_positions
        )
    if type_positions:
        metrics["type_loss"] = type_loss_sum / type_positions
    elif boundary_class_positions:
        metrics["type_loss"] = 0.0
    metrics["type_positions"] = type_positions
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
        "row_normalized_confusion_matrix",
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


def _combined_selection(
    retained_metrics: dict,
    counterbalance_metrics: dict,
    precision_floor: float,
    recall_floor: float,
    answer_floor: float,
    type_floor: float,
) -> tuple[bool, tuple[float, ...]]:
    retained_eligible, _ = _selection(
        retained_metrics,
        precision_floor,
        recall_floor,
        answer_floor,
        type_floor,
    )
    counterbalance_eligible, _ = _selection(
        counterbalance_metrics,
        precision_floor,
        recall_floor,
        answer_floor,
        type_floor,
    )
    eligible = retained_eligible and counterbalance_eligible
    return eligible, (
        1.0 if eligible else 0.0,
        min(
            retained_metrics["answer_candidate_recall"],
            counterbalance_metrics["answer_candidate_recall"],
        ),
        min(
            retained_metrics["exact_span_f1"],
            counterbalance_metrics["exact_span_f1"],
        ),
        min(
            retained_metrics["boundary_type_accuracy"],
            counterbalance_metrics["boundary_type_accuracy"],
        ),
        -(retained_metrics["loss"] + counterbalance_metrics["loss"]) / 2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    data_config = config["data"]
    train_config = config["train"]
    token_end_premise_path, token_end_premise = _read_token_end_premise(
        train_config
    )
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
    geometry_version = train_config.get("token_width_geometry_version")
    end_geometry_version = train_config.get("token_end_geometry_version")
    factorized_version = train_config.get(
        "factorized_boundary_type_version"
    )
    counterbalance_version = train_config.get(
        "boundary_counterbalance_version"
    )
    model = ExplicitOffsetCandidateProposer(
        resolver,
        tokenizer,
        token_width_geometry=geometry_version is not None,
        token_end_geometry=end_geometry_version is not None,
        factorized_boundary_type=factorized_version is not None,
    ).to(device)

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

    counterbalance_manifest = None
    counterbalance_pools = None
    counterbalance_train = ()
    counterbalance_val = ()
    counterbalance_train_examples = ()
    counterbalance_val_examples = ()
    if counterbalance_version is not None:
        manifest_path = Path(data_config["counterbalance_manifest_path"])
        counterbalance_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        counterbalance_pools = validate_counterbalance_manifest(
            counterbalance_manifest,
            manifest_path.parent,
            tokenizer,
        )
        expected_counterbalance_paths = {
            "counterbalance_train_path": manifest_path.parent / "train.jsonl",
            "counterbalance_val_path": manifest_path.parent / "val.jsonl",
        }
        for key, expected_path in expected_counterbalance_paths.items():
            if Path(data_config[key]).resolve() != expected_path.resolve():
                raise ValueError(
                    f"{key} must name the file covered by the Phase 35 manifest"
                )
        counterbalance_train = _eligible_records(
            data_config["counterbalance_train_path"]
        )
        counterbalance_val = _eligible_records(
            data_config["counterbalance_val_path"]
        )
        if not all(
            record.source_phase == "phase35"
            for record in counterbalance_train + counterbalance_val
        ):
            raise ValueError("counterbalance rows do not declare Phase 35")
        counterbalance_train_examples = encode_proposal_records(
            counterbalance_train, tokenizer, block_size
        )
        counterbalance_val_examples = encode_proposal_records(
            counterbalance_val, tokenizer, block_size
        )

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
    counterbalance_train_indices = ()
    counterbalance_val_indices = ()
    counterbalance_microbatches = 0
    if counterbalance_version is not None:
        assert counterbalance_pools is not None
        focus_examples = int(
            train_config.get("counterbalance_eval_examples_per_token", 4)
        )
        counterbalance_train_indices = _focus_stratified_indices(
            counterbalance_train_examples,
            counterbalance_pools["train"],
            focus_examples,
            tokenizer,
        )
        counterbalance_val_indices = _focus_stratified_indices(
            counterbalance_val_examples,
            counterbalance_pools["train"],
            focus_examples,
            tokenizer,
        )
        counterbalance_microbatches = int(
            train_config.get("counterbalance_microbatches_per_update", 1)
        )
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
    if counterbalance_manifest is not None:
        manifests["phase35"] = counterbalance_manifest
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
        "token_width_geometry_version": geometry_version,
        "token_end_geometry_version": end_geometry_version,
        "factorized_boundary_type_version": factorized_version,
        "boundary_counterbalance_version": counterbalance_version,
        "counterbalance_microbatches_per_update": (
            counterbalance_microbatches
            if counterbalance_version is not None
            else None
        ),
        "counterbalance_manifest_sha256": (
            canonical_json_sha256(counterbalance_manifest)
            if counterbalance_manifest is not None
            else None
        ),
        "token_end_premise_path": (
            str(token_end_premise_path) if token_end_premise_path else None
        ),
        "token_end_premise_input_sha256": (
            token_end_premise.get("input_sha256")
            if token_end_premise is not None
            else None
        ),
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
    if geometry_version is not None:
        print(f"token-width geometry: version {geometry_version}")
    if end_geometry_version is not None:
        print(f"token-end geometry: version {end_geometry_version}")
        print(f"token-end premise: {token_end_premise_path}")
    if factorized_version is not None:
        print(
            "factorized boundary/type heads: "
            f"version {factorized_version}"
        )
        print(
            "loss objective: shared_byte_BIO+positive_byte_type"
            "+boundary_transition"
        )
        print("boundary class weights: O=0.05, B=1.0, I=0.5")
        print("type loss: positive bytes only, weight 1")
        print(f"boundary loss weight: {boundary_loss_weight:g}")
        print("boundary terms: gold_begin + first_gold_outside_after_span")
    elif boundary_loss_weight:
        print("loss objective: typed_byte_BIO+boundary_transition")
        print(f"boundary loss weight: {boundary_loss_weight:g}")
        print("boundary terms: gold_begin + first_gold_outside_after_span")
    else:
        print("loss objective: typed_byte_BIO")
    if counterbalance_version is not None:
        assert counterbalance_pools is not None
        print(
            "boundary counterbalance: version "
            f"{counterbalance_version}, data-only"
        )
        print(
            "counterbalance schedule: "
            f"{counterbalance_microbatches}/{accumulation} microbatches per "
            "update"
        )
        print(
            "focus-token pools: train "
            f"{len(counterbalance_pools['train'])}, heldout "
            f"{len(counterbalance_pools['heldout'])}, overlap 0"
        )
    print("excluded proposer cases: wrong_type (retained in oracle gate)")
    print(
        f"training examples: {len(train_examples):,} "
        f"(Phase 31 {len(phase31_train):,} + Phase 32 {len(phase32_train):,})"
    )
    print(
        f"fixed panels: train {len(train_indices):,}, val {len(val_indices):,}"
    )
    if counterbalance_version is not None:
        print(
            "counterbalance examples: train "
            f"{len(counterbalance_train_examples):,}, val "
            f"{len(counterbalance_val_examples):,}"
        )
        print(
            "counterbalance fixed panels: train "
            f"{len(counterbalance_train_indices):,}, val "
            f"{len(counterbalance_val_indices):,}"
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
        counterbalance_train_metrics = None
        counterbalance_val_metrics = None
        if counterbalance_version is not None:
            counterbalance_train_metrics = _evaluate(
                model,
                counterbalance_train_examples,
                counterbalance_train,
                counterbalance_train_indices,
                tokenizer,
                eval_batch_size,
                device,
                boundary_loss_weight,
            )
            counterbalance_val_metrics = _evaluate(
                model,
                counterbalance_val_examples,
                counterbalance_val,
                counterbalance_val_indices,
                tokenizer,
                eval_batch_size,
                device,
                boundary_loss_weight,
            )
            eligible, selection_key = _combined_selection(
                val_metrics,
                counterbalance_val_metrics,
                precision_floor,
                recall_floor,
                answer_floor,
                type_floor,
            )
        else:
            eligible, selection_key = _selection(
                val_metrics,
                precision_floor,
                recall_floor,
                answer_floor,
                type_floor,
            )
        last_eligible = eligible
        factorized_eval = ""
        if factorized_version is not None:
            factorized_eval = (
                f"boundary-class "
                f"{val_metrics['boundary_class_loss']:.4f}, "
                f"type loss {val_metrics['type_loss']:.4f}, "
                f"type positions {val_metrics['type_positions']}, "
            )
        print(
            f"update {step}: train loss {train_metrics['loss']:.4f}, "
            f"val loss {val_metrics['loss']:.4f}, "
            f"{factorized_eval}"
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
        if (
            counterbalance_train_metrics is not None
            and counterbalance_val_metrics is not None
        ):
            print(
                f"update {step}: counterbalance train loss "
                f"{counterbalance_train_metrics['loss']:.4f}, val loss "
                f"{counterbalance_val_metrics['loss']:.4f}, "
                "B->I "
                f"{counterbalance_val_metrics['gold_begin_as_inside_rate']:.3f}, "
                "I->B "
                f"{counterbalance_val_metrics['row_normalized_confusion_matrix']['I']['B']:.3f}, "
                "span precision "
                f"{counterbalance_val_metrics['exact_span_precision']:.3f}, "
                "span recall "
                f"{counterbalance_val_metrics['exact_span_recall']:.3f}, "
                "answer recall "
                f"{counterbalance_val_metrics['answer_candidate_recall']:.3f}"
            )
        slot = None
        selection_loss = val_metrics["loss"]
        if counterbalance_val_metrics is not None:
            selection_loss = (
                selection_loss + counterbalance_val_metrics["loss"]
            ) / 2.0
        if math.isfinite(selection_loss):
            if eligible and (best_key is None or selection_key > best_key):
                slot = "best"
            elif not eligible and (
                diagnostic_key is None or selection_key > diagnostic_key
            ):
                slot = "diagnostic"
        if slot == "best":
            best_key = selection_key
            best_loss = selection_loss
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
                    "retained_validation_metrics": val_metrics,
                    "counterbalance_validation_metrics": (
                        counterbalance_val_metrics
                    ),
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
                    "diagnostic_validation_loss": selection_loss,
                    "diagnostic_step": step,
                    "retained_validation_metrics": val_metrics,
                    "counterbalance_validation_metrics": (
                        counterbalance_val_metrics
                    ),
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
        accumulated_boundary_class_loss = 0.0
        accumulated_type_loss = 0.0
        accumulated_type_positions = 0
        accumulated_start_loss = 0.0
        accumulated_end_loss = 0.0
        accumulated_start_positions = 0
        accumulated_end_positions = 0
        accumulated_counterbalance_rows = 0
        counterbalance_slots = set()
        if counterbalance_version is not None:
            counterbalance_slots = set(
                _counterbalance_slots(
                    step, accumulation, counterbalance_microbatches
                )
            )
        for microbatch in range(accumulation):
            if microbatch in counterbalance_slots:
                batch = _sample_batch(
                    counterbalance_train_examples,
                    batch_size,
                    tokenizer,
                    model.source_width,
                    device,
                )
                accumulated_counterbalance_rows += batch_size
            else:
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
            if factorized_version is not None:
                assert output.boundary_class_loss is not None
                assert output.type_loss is not None
                accumulated_boundary_class_loss += float(
                    output.boundary_class_loss.detach()
                )
                accumulated_type_loss += float(output.type_loss.detach())
                accumulated_type_positions += output.type_positions
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
            factorized_log = ""
            if factorized_version is not None:
                factorized_log = (
                    f"boundary-class "
                    f"{accumulated_boundary_class_loss / accumulation:.4f}, "
                    f"type {accumulated_type_loss / accumulation:.4f}, "
                    f"type positions {accumulated_type_positions}, "
                )
            counterbalance_log = ""
            if counterbalance_version is not None:
                counterbalance_log = (
                    f"counterbalance rows {accumulated_counterbalance_rows}/"
                    f"{batch_size * accumulation}, "
                )
            print(
                f"step {step}: loss {accumulated_loss / accumulation:.4f}, "
                f"tag {accumulated_tag_loss / accumulation:.4f}, "
                f"{factorized_log}"
                f"{counterbalance_log}"
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
