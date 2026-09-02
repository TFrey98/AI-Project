"""Phase 34e no-training B-logit calibration and BPE geometry audit."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_OBJECTIVE_VERSION,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    PROPOSAL_TYPES,
    ExplicitOffsetCandidateProposer,
    begin_tag,
    decode_proposed_spans,
    encode_proposal_records,
    inside_tag,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
    proposal_record_is_eligible,
)
from story_model.models import build_model
from story_model.runtime import resolve_device
from story_model.unified_typed_span_resolver import (
    SUPPORT_MASK_VERSION,
    UnifiedTypedSpanResolver,
)
try:
    from scripts.phase34c_tag_confusion_decision import (
        add_tag_sequence,
        collapsed_tag,
        merge_tag_audit,
        new_tag_audit,
        summarize_tag_audit,
    )
    from scripts.phase34e_begin_calibration_decision import (
        begin_calibration_decision,
        select_begin_bias,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34c_tag_confusion_decision import (  # type: ignore[no-redef]
        add_tag_sequence,
        collapsed_tag,
        merge_tag_audit,
        new_tag_audit,
        summarize_tag_audit,
    )
    from phase34e_begin_calibration_decision import (  # type: ignore[no-redef]
        begin_calibration_decision,
        select_begin_bias,
    )


BEGIN_CALIBRATION_ABLATION_VERSION = 1
BEGIN_BIASES = tuple(index * 0.25 for index in range(17))
CALIBRATION_EXAMPLES_PER_CELL = 8
BEGIN_TAG_IDS = tuple(begin_tag(value_type) for value_type in PROPOSAL_TYPES)
SECTION_MARKERS = (
    "<|character|>",
    "<|relationship|>",
    "<|scene|>",
    "<|world_facts|>",
    "<|memories|>",
    "<|conversation|>",
)


def _load_model(path: Path, device: torch.device):
    checkpoint = read_checkpoint(path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "explicit_offset_candidate_proposer":
        raise ValueError("checkpoint is not a Phase 34 candidate proposer")
    if (
        extra.get("explicit_offset_proposer_version")
        != EXPLICIT_OFFSET_PROPOSER_VERSION
    ):
        raise ValueError("checkpoint has an unsupported proposer version")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("checkpoint does not contain the Phase 33c router")
    if extra.get("boundary_objective_version") != BOUNDARY_OBJECTIVE_VERSION:
        raise ValueError("Phase 34e requires a Phase 34d checkpoint")
    if abs(float(extra.get("boundary_loss_weight", -1.0)) - 1.0) > 1.0e-9:
        raise ValueError("Phase 34e requires boundary loss weight 1.0")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training config")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 34e requires byte-BPE")
    block_size = int(config["data"]["block_size"])
    backbone = build_model(config["model"], tokenizer.vocab_size, block_size)
    resolver = UnifiedTypedSpanResolver(backbone)
    model = ExplicitOffsetCandidateProposer(resolver, tokenizer)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, block_size, checkpoint


def _stratified_indices(records, examples_per_cell: int) -> tuple[int, ...]:
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


def _biased_logits(logits: torch.Tensor, bias: float) -> torch.Tensor:
    if bias == 0.0:
        return logits
    adjusted = logits.clone()
    adjusted[..., list(BEGIN_TAG_IDS)] += bias
    return adjusted


def _flat_prompt_logits(logits, example, tokenizer):
    pieces = []
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        pieces.append(logits[token_position, :width].float())
    flat = torch.cat(pieces, dim=0)
    targets = torch.tensor(example.prompt_byte_tags, dtype=torch.long)
    if len(flat) != len(targets):
        raise RuntimeError("prompt logits and byte targets do not align")
    return flat, targets


def _add_tag_predictions(audit, flat_logits, targets, bias):
    adjusted = _biased_logits(flat_logits, bias)
    predictions = adjusted.argmax(dim=-1)
    add_tag_sequence(
        audit,
        targets.tolist(),
        predictions.tolist(),
        (0.0,) * len(targets),
    )


def _policy_summary(rows, audit) -> dict:
    metrics = proposal_metrics(rows)
    tag_metrics = summarize_tag_audit(audit)
    metrics.update(
        {
            "gold_begin_as_inside_rate": tag_metrics[
                "gold_begin_as_inside_rate"
            ],
            "gold_begin_as_inside_count": tag_metrics[
                "gold_begin_as_inside_count"
            ],
            "gold_inside_as_begin_rate": tag_metrics[
                "row_normalized_confusion_matrix"
            ]["I"]["B"],
            "gold_outside_as_begin_rate": tag_metrics[
                "row_normalized_confusion_matrix"
            ]["O"]["B"],
            "end_spill_rate": tag_metrics["end_spill_rate"],
            "positive_type_accuracy": tag_metrics[
                "positive_type_accuracy"
            ],
        }
    )
    return metrics


@torch.no_grad()
def calibrate_bias(
    model,
    tokenizer,
    block_size,
    data_directories,
    device,
    batch_size,
):
    states = {
        bias: {"rows": [], "audit": new_tag_audit()}
        for bias in BEGIN_BIASES
    }
    cell_states = defaultdict(
        lambda: {
            bias: {"rows": [], "audit": new_tag_audit()}
            for bias in BEGIN_BIASES
        }
    )
    panel_counts = {}
    for dataset_name, data_dir in data_directories:
        for split in ("train", "val"):
            records = tuple(
                record
                for record in load_expanded_records(
                    data_dir / f"{split}.jsonl"
                )
                if proposal_record_is_eligible(record)
            )
            examples = encode_proposal_records(records, tokenizer, block_size)
            indices = _stratified_indices(
                records, CALIBRATION_EXAMPLES_PER_CELL
            )
            panel_counts[f"{dataset_name}/{split}"] = len(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = proposal_batch(
                    examples,
                    batch_indices,
                    tokenizer,
                    model.source_width,
                    device,
                )
                logits = model(batch[0], batch[1], batch[2]).tag_logits.cpu()
                for offset, index in enumerate(batch_indices):
                    record = records[index]
                    example_logits = logits[
                        offset, : examples[index].prompt_token_count
                    ]
                    flat_logits, targets = _flat_prompt_logits(
                        example_logits, examples[index], tokenizer
                    )
                    for bias, state in states.items():
                        biased = _biased_logits(example_logits, bias)
                        predicted = decode_proposed_spans(
                            records[index].prompt,
                            examples[index],
                            biased,
                            tokenizer,
                        )
                        row = proposal_metric_row(
                            record,
                            examples[index].gold_spans,
                            predicted,
                        )
                        row_audit = new_tag_audit()
                        _add_tag_predictions(row_audit, flat_logits, targets, bias)
                        state["rows"].append(row)
                        merge_tag_audit(state["audit"], row_audit)
                        for label in (
                            f"{dataset_name}/{split}",
                            f"{dataset_name}/{split}/{record.skill}",
                        ):
                            cell_state = cell_states[label][bias]
                            cell_state["rows"].append(row)
                            merge_tag_audit(cell_state["audit"], row_audit)
    metrics = {
        f"{bias:.2f}": _policy_summary(state["rows"], state["audit"])
        for bias, state in states.items()
    }
    metrics_by_cell = {
        label: {
            f"{bias:.2f}": _policy_summary(
                state["rows"], state["audit"]
            )
            for bias, state in bias_states.items()
        }
        for label, bias_states in sorted(cell_states.items())
    }
    selection = select_begin_bias(metrics, metrics_by_cell)
    return {
        **selection,
        "bias_grid": list(BEGIN_BIASES),
        "examples_per_cell": CALIBRATION_EXAMPLES_PER_CELL,
        "panel_counts": panel_counts,
        "metrics_by_bias": metrics,
        "metrics_by_cell": metrics_by_cell,
    }


def _preceding_character_class(prompt: str, byte_start: int) -> str:
    if byte_start == 0:
        return "prompt_start"
    preceding = prompt.encode("utf-8")[:byte_start].decode("utf-8")[-1]
    if preceding.isspace():
        return "whitespace"
    if preceding.isalnum():
        return "alphanumeric"
    return "punctuation"


def _prompt_section(prompt: str, byte_start: int) -> str:
    prompt_bytes = prompt.encode("utf-8")
    located = []
    for marker in SECTION_MARKERS:
        position = prompt_bytes.rfind(marker.encode("utf-8"), 0, byte_start + 1)
        if position >= 0:
            located.append((position, marker[2:-2]))
    return max(located)[1] if located else "unmarked"


def _span_length_bucket(length: int) -> str:
    if length <= 4:
        return "1-4"
    if length <= 8:
        return "5-8"
    if length <= 16:
        return "9-16"
    if length <= 32:
        return "17-32"
    return "33+"


def _start_rows(
    dataset_name,
    record,
    example,
    logits,
    tokenizer,
    selected_bias,
):
    token_ranges = []
    cursor = 0
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        token_bytes = tokenizer.token_bytes(token_id)
        token_ranges.append(
            (cursor, cursor + len(token_bytes), token_position, token_bytes)
        )
        cursor += len(token_bytes)
    occurrence_counts = Counter()
    rows = []
    for span in example.gold_spans:
        for token_start, token_end, token_position, token_bytes in token_ranges:
            if token_start <= span.byte_start < token_end:
                byte_offset = span.byte_start - token_start
                break
        else:
            raise RuntimeError("gold start is outside prompt token ranges")
        vector = logits[token_position, byte_offset].float()
        baseline_prediction = int(vector.argmax())
        adjusted = _biased_logits(vector, selected_bias)
        calibrated_prediction = int(adjusted.argmax())
        b_tag = begin_tag(span.value_type)
        i_tag = inside_tag(span.value_type)
        key = (span.text, span.value_type)
        ordinal = occurrence_counts[key]
        occurrence_counts[key] += 1
        rows.append(
            {
                "record_id": record.record_id,
                "dataset": dataset_name,
                "split": record.split,
                "skill": record.skill,
                "case": record.case,
                "source_phase": record.source_phase,
                "span_text": span.text,
                "value_type": span.value_type,
                "byte_start": span.byte_start,
                "byte_end": span.byte_end,
                "span_byte_length": span.byte_end - span.byte_start,
                "span_byte_length_bucket": _span_length_bucket(
                    span.byte_end - span.byte_start
                ),
                "span_word_count": len(span.text.split()),
                "occurrence_ordinal": ordinal,
                "prompt_section": _prompt_section(
                    record.prompt, span.byte_start
                ),
                "preceding_character_class": _preceding_character_class(
                    record.prompt, span.byte_start
                ),
                "token_width": len(token_bytes),
                "token_byte_offset": byte_offset,
                "token_alignment": (
                    "at_token_start" if byte_offset == 0 else "inside_token"
                ),
                "token_prefix": token_bytes[:byte_offset].decode(
                    "utf-8", errors="replace"
                ),
                "correct_type_begin_minus_inside_margin": float(
                    vector[b_tag] - vector[i_tag]
                ),
                "calibrated_begin_minus_inside_margin": float(
                    vector[b_tag] + selected_bias - vector[i_tag]
                ),
                "baseline_predicted_class": collapsed_tag(
                    baseline_prediction
                ),
                "calibrated_predicted_class": collapsed_tag(
                    calibrated_prediction
                ),
                "baseline_exact_begin_tag": baseline_prediction == b_tag,
                "calibrated_exact_begin_tag": calibrated_prediction == b_tag,
            }
        )
    return rows


def _start_summary(rows) -> dict:
    rows = tuple(rows)
    margins = sorted(
        row["correct_type_begin_minus_inside_margin"] for row in rows
    )
    baseline_inside = sum(
        row["baseline_predicted_class"] == "I" for row in rows
    )
    calibrated_inside = sum(
        row["calibrated_predicted_class"] == "I" for row in rows
    )
    return {
        "gold_begin_count": len(rows),
        "baseline_begin_as_inside_count": baseline_inside,
        "baseline_begin_as_inside_rate": (
            baseline_inside / len(rows) if rows else 0.0
        ),
        "calibrated_begin_as_inside_count": calibrated_inside,
        "calibrated_begin_as_inside_rate": (
            calibrated_inside / len(rows) if rows else 0.0
        ),
        "baseline_exact_begin_rate": (
            sum(row["baseline_exact_begin_tag"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "calibrated_exact_begin_rate": (
            sum(row["calibrated_exact_begin_tag"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "mean_begin_minus_inside_margin": (
            statistics.fmean(margins) if margins else 0.0
        ),
        "median_begin_minus_inside_margin": (
            statistics.median(margins) if margins else 0.0
        ),
        "minimum_begin_minus_inside_margin": margins[0] if margins else 0.0,
        "maximum_begin_minus_inside_margin": margins[-1] if margins else 0.0,
    }


def _geometry_summary(rows) -> dict:
    rows = tuple(rows)
    dimensions = {}
    for field in (
        "token_alignment",
        "token_byte_offset",
        "token_width",
        "preceding_character_class",
        "span_word_count",
        "span_byte_length_bucket",
        "prompt_section",
    ):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        dimensions[field] = {
            value: _start_summary(group) for value, group in sorted(grouped.items())
        }
    return {"overall": _start_summary(rows), "dimensions": dimensions}


@torch.no_grad()
def evaluate_split(
    model,
    tokenizer,
    block_size,
    records,
    dataset_name,
    selected_bias,
    device,
    batch_size,
    label,
    progress_every,
):
    records = tuple(
        record for record in records if proposal_record_is_eligible(record)
    )
    examples = encode_proposal_records(records, tokenizer, block_size)
    states = {
        "baseline": {"rows": [], "audit": new_tag_audit()},
        "calibrated": {"rows": [], "audit": new_tag_audit()},
    }
    skill_states = defaultdict(
        lambda: {
            "baseline": {"rows": [], "audit": new_tag_audit()},
            "calibrated": {"rows": [], "audit": new_tag_audit()},
        }
    )
    start_rows = []
    started = time.monotonic()
    batch_total = (len(examples) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(
        range(0, len(examples), batch_size), start=1
    ):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        batch = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        logits = model(batch[0], batch[1], batch[2]).tag_logits.cpu()
        for offset, index in enumerate(indices):
            record = records[index]
            example = examples[index]
            example_logits = logits[
                offset, : example.prompt_token_count
            ]
            flat_logits, targets = _flat_prompt_logits(
                example_logits, example, tokenizer
            )
            for policy, bias in (
                ("baseline", 0.0),
                ("calibrated", selected_bias),
            ):
                biased = _biased_logits(example_logits, bias)
                predicted = decode_proposed_spans(
                    record.prompt, example, biased, tokenizer
                )
                row = proposal_metric_row(
                    record, example.gold_spans, predicted
                )
                states[policy]["rows"].append(row)
                skill_states[record.skill][policy]["rows"].append(row)
                row_audit = new_tag_audit()
                _add_tag_predictions(
                    row_audit, flat_logits, targets, bias
                )
                merge_tag_audit(states[policy]["audit"], row_audit)
                merge_tag_audit(
                    skill_states[record.skill][policy]["audit"], row_audit
                )
            start_rows.extend(
                _start_rows(
                    dataset_name,
                    record,
                    example,
                    example_logits,
                    tokenizer,
                    selected_bias,
                )
            )
        if progress_every and (
            batch_number % progress_every == 0 or batch_number == batch_total
        ):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            completed = min(start + batch_size, len(examples))
            print(
                f"{label}: {completed:,}/{len(examples):,} rows "
                f"({completed / elapsed:.1f} rows/s; one forward/two decodes)",
                flush=True,
            )
    policies = {
        policy: _policy_summary(state["rows"], state["audit"])
        for policy, state in states.items()
    }
    per_skill = {
        skill: {
            "policies": {
                policy: _policy_summary(state["rows"], state["audit"])
                for policy, state in policy_states.items()
            }
        }
        for skill, policy_states in sorted(skill_states.items())
    }
    return policies, per_skill, start_rows, len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument(
        "--phase32-data-dir",
        type=Path,
        default=Path("data/character/expanded_typed_span_resolver"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")

    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size, checkpoint = _load_model(
        args.checkpoint, device
    )
    data_directories = (
        ("phase31_regression", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    )
    print("calibrating B-logit bias on deterministic train/val panels")
    calibration = calibrate_bias(
        model,
        tokenizer,
        block_size,
        data_directories,
        device,
        args.batch_size,
    )
    selected_bias = float(calibration["selected_bias"])
    print(f"selected train/val-safe B-logit bias: {selected_bias:.2f}")

    extra = checkpoint.get("extra", {})
    summary = {
        "begin_calibration_ablation_version": (
            BEGIN_CALIBRATION_ABLATION_VERSION
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", 0),
        "checkpoint_eligible": extra.get("checkpoint_eligible"),
        "checkpoint_boundary_objective_version": extra.get(
            "boundary_objective_version"
        ),
        "checkpoint_boundary_loss_weight": extra.get(
            "boundary_loss_weight"
        ),
        "device": str(device),
        "logit_reuse": "one proposer forward; baseline and calibrated decode",
        "calibration_uses": "deterministic train and val panels only",
        "calibration": calibration,
        "datasets": {
            "phase31_regression": {"splits": {}},
            "phase32": {"splits": {}},
        },
    }
    all_start_rows = []
    for dataset_name, data_dir in data_directories:
        for split in EXPANDED_SPLITS:
            label = f"{dataset_name}/{split}"
            all_records = load_expanded_records(data_dir / f"{split}.jsonl")
            policies, per_skill, start_rows, eligible_rows = evaluate_split(
                model,
                tokenizer,
                block_size,
                all_records,
                dataset_name,
                selected_bias,
                device,
                args.batch_size,
                label,
                args.progress_every,
            )
            all_start_rows.extend(start_rows)
            summary["datasets"][dataset_name]["splits"][split] = {
                "rows": len(all_records),
                "eligible_rows": eligible_rows,
                "excluded_rows": len(all_records) - eligible_rows,
                "policies": policies,
                "per_skill": per_skill,
            }
            baseline = policies["baseline"]
            calibrated = policies["calibrated"]
            print(
                f"{label}: B->I "
                f"{baseline['gold_begin_as_inside_rate']:.3f}->"
                f"{calibrated['gold_begin_as_inside_rate']:.3f}, "
                f"precision {baseline['exact_span_precision']:.3f}->"
                f"{calibrated['exact_span_precision']:.3f}, "
                f"recall {baseline['exact_span_recall']:.3f}->"
                f"{calibrated['exact_span_recall']:.3f}",
                flush=True,
            )
    target_rows = tuple(
        row
        for row in all_start_rows
        if row["dataset"] == "phase31_regression"
        and row["split"] in {"lexical", "transfer"}
        and row["skill"] == "scene_route"
    )
    summary["target_geometry"] = _geometry_summary(target_rows)
    summary["decision"] = begin_calibration_decision(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_start_rows
        ),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    print(f"next action: {decision['next_action']}")
    print(f"start rows: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
