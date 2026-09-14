"""Audit Phase 36c's begin-only positional ablation of Phase 36b.

Compares three frozen checkpoints on the unchanged Phase 36a crossed-identity
panel: Phase 35 (pre-registration baseline for complete-span quality), Phase
36b (rejected begin-and-inside pairing, baseline for identity transfer), and
Phase 36c (the begin-only candidate). It also independently recomputes, from
the frozen training data, whether restricting pairing to gold ``B:route``
positions could have reduced how many rows are swap-eligible per update.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from story_model.boundary_counterbalance import validate_counterbalance_manifest
from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import encode_proposal_records
from story_model.paired_identity_invariance import (
    BEGIN_ONLY_POSITION_MODE,
    eligible_swap_candidates,
    registered_identity_ids,
    validate_identity_swap_pool,
)
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device

try:
    from scripts.audit_crossed_boundary_identity import (
        TARGET_SPLITS,
        _read_json,
        _sha256,
        build_crossed_identity_panel,
        canonical_focus_identities,
        evaluate_focus_geometry,
    )
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.train_explicit_offset_candidate_proposer import (
        _eligible_records,
        _evaluate,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities, evaluate_focus_geometry,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from train_explicit_offset_candidate_proposer import (  # type: ignore
        _eligible_records, _evaluate,
    )


BEGIN_ONLY_IDENTITY_INVARIANCE_AUDIT_VERSION = 1


def _checkpoint_metadata(path: Path, checkpoint: dict, tokenizer) -> dict:
    extra = checkpoint.get("extra", {})
    return {
        "path": str(path), "sha256": _sha256(path),
        "step": checkpoint.get("step", 0),
        "checkpoint_eligible": extra.get("checkpoint_eligible"),
        "architecture": extra.get("architecture"),
        "explicit_offset_proposer_version": extra.get("explicit_offset_proposer_version"),
        "boundary_objective_version": extra.get("boundary_objective_version"),
        "boundary_loss_weight": extra.get("boundary_loss_weight"),
        "boundary_counterbalance_version": extra.get("boundary_counterbalance_version"),
        "counterbalance_manifest_sha256": extra.get("counterbalance_manifest_sha256"),
        "paired_identity_invariance_version": extra.get("paired_identity_invariance_version"),
        "identity_invariance_loss_weight": extra.get("identity_invariance_loss_weight"),
        "identity_invariance_position_mode": extra.get("identity_invariance_position_mode"),
        "identity_invariance_swap_pool_sha256": extra.get("identity_invariance_swap_pool_sha256"),
        "identity_invariance_total_swap_position_counts": extra.get(
            "identity_invariance_total_swap_position_counts"
        ),
        "token_width_geometry_version": extra.get("token_width_geometry_version"),
        "token_end_geometry_version": extra.get("token_end_geometry_version"),
        "factorized_boundary_type_version": extra.get("factorized_boundary_type_version"),
        "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
        "parent_phase33_checkpoint": extra.get("parent_phase33_checkpoint"),
        "parent_phase33_step": extra.get("parent_phase33_step"),
    }


def count_eligible_rows(
    records,
    tokenizer,
    block_size: int,
    pool_by_width: dict,
    begin_only: bool,
) -> int:
    """Count rows with at least one eligible swap candidate under a filter."""
    examples = encode_proposal_records(records, tokenizer, block_size)
    count = 0
    for example in examples:
        tokens = torch.tensor(example.input_ids, dtype=torch.long)
        # prompt_byte_tags is byte-level; reconstruct per-token first-byte tags.
        first_byte_tags = []
        cursor = 0
        for position in range(example.prompt_token_count):
            width = tokenizer.token_byte_length(example.input_ids[position])
            first_byte_tags.append(example.prompt_byte_tags[cursor])
            cursor += width
        targets = torch.tensor(
            [[tag] for tag in first_byte_tags], dtype=torch.long
        )
        candidates = eligible_swap_candidates(
            tokens[: example.prompt_token_count],
            example.prompt_token_count,
            targets,
            tokenizer,
            pool_by_width,
            begin_only,
        )
        if candidates:
            count += 1
    return count


def audit_swap_opportunity(
    data_config: dict,
    tokenizer,
    block_size: int,
    pool_by_width: dict,
) -> dict:
    """Independently prove begin-only restriction did not shrink pairing rows."""
    report = {}
    sources = {
        "phase31_train": Path(data_config["phase31_train_path"]),
        "phase32_train": Path(data_config["train_path"]),
        "counterbalance_train": Path(data_config["counterbalance_train_path"]),
    }
    for label, path in sources.items():
        records = _eligible_records(path)
        begin_and_inside = count_eligible_rows(
            records, tokenizer, block_size, pool_by_width, begin_only=False
        )
        begin_only = count_eligible_rows(
            records, tokenizer, block_size, pool_by_width, begin_only=True
        )
        report[label] = {
            "rows": len(records),
            "begin_and_inside_eligible_rows": begin_and_inside,
            "begin_only_eligible_rows": begin_only,
            "eligible_rows_did_not_fall": begin_only >= begin_and_inside,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase35-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36b-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36c-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36a-summary", type=Path, required=True)
    parser.add_argument("--phase31-data-dir", type=Path,
                        default=Path("data/character/typed_span_resolver"))
    parser.add_argument("--phase32-data-dir", type=Path,
                        default=Path("data/character/expanded_typed_span_resolver"))
    parser.add_argument("--phase35-data-dir", type=Path,
                        default=Path("data/character/boundary_counterbalance"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase36a_summary = _read_json(args.phase36a_summary)

    device = torch.device(resolve_device(args.device))
    paths = {
        "phase35": args.phase35_checkpoint,
        "phase36b": args.phase36b_checkpoint,
        "phase36c": args.phase36c_checkpoint,
    }
    models, tokenizers, checkpoints, block_sizes = {}, {}, {}, {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label], tokenizers[label] = model, tokenizer
        checkpoints[label], block_sizes[label] = checkpoint, block_size
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 35/36b/36c block sizes differ")
    tokenizer, block_size = tokenizers["phase36c"], block_sizes["phase36c"]
    tokenizer_hashes = {
        canonical_json_sha256(t.to_dict()) for t in tokenizers.values()
    }
    if len(tokenizer_hashes) != 1:
        raise ValueError("Phase 35/36b/36c tokenizers differ")

    extras = {label: checkpoint.get("extra", {}) for label, checkpoint in checkpoints.items()}
    if extras["phase35"].get("paired_identity_invariance_version") is not None:
        raise ValueError("Phase 35 baseline already used identity invariance")
    if extras["phase36b"].get("paired_identity_invariance_version") != 1:
        raise ValueError("phase36b-checkpoint is not the registered Phase 36b run")
    if extras["phase36c"].get("paired_identity_invariance_version") != 2:
        raise ValueError("phase36c-checkpoint is not a Phase 36c begin-only run")
    if extras["phase36c"].get("identity_invariance_position_mode") != BEGIN_ONLY_POSITION_MODE:
        raise ValueError("Phase 36c checkpoint is not in begin-only position mode")
    counts = extras["phase36c"].get("identity_invariance_total_swap_position_counts") or {}
    if int(counts.get("I", -1)) != 0:
        raise ValueError("Phase 36c checkpoint recorded an I:route pair")
    if int(counts.get("B", 0)) <= 0:
        raise ValueError("Phase 36c checkpoint recorded no B:route pairs")
    for field in (
        "architecture", "boundary_objective_version", "boundary_loss_weight",
        "boundary_counterbalance_version", "counterbalance_manifest_sha256",
        "parent_phase33_checkpoint", "parent_phase33_step",
    ):
        if extras["phase36b"].get(field) != extras["phase36c"].get(field):
            raise ValueError(f"Phase 36b/36c {field} differs")
    if extras["phase36b"].get("identity_invariance_swap_pool_sha256") != extras[
        "phase36c"
    ].get("identity_invariance_swap_pool_sha256"):
        raise ValueError("Phase 36b/36c swap-pool hash differs")

    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    pools = validate_counterbalance_manifest(manifest, args.phase35_data_dir, tokenizer)
    excluded_ids = registered_identity_ids(manifest)
    swap_pool = extras["phase36c"].get("identity_invariance_swap_pool")
    if not isinstance(swap_pool, dict):
        raise ValueError("Phase 36c checkpoint has no swap pool")
    pool_by_width = validate_identity_swap_pool(swap_pool, tokenizer, excluded_ids)

    identities = canonical_focus_identities(manifest, tokenizer)
    expected_identities = {
        group: [{"token_id": token_id, "text": text} for token_id, text in values]
        for group, values in identities.items()
    }
    if phase36a_summary.get("focus_identities") != expected_identities:
        raise ValueError("Phase 36a identities changed")

    source_records = {}
    for split in TARGET_SPLITS:
        path = args.phase31_data_dir / f"{split}.jsonl"
        records = load_expanded_records(path)
        expected = manifest["source_phase31_files"][split]
        if expected["sha256"] != _sha256(path) or expected["rows"] != len(records):
            raise ValueError(f"Phase 31 {split} data changed")
        source_records[split] = records
    panel, validation = build_crossed_identity_panel(
        source_records, identities, tokenizer, block_size
    )
    if validation != phase36a_summary.get("panel_validation"):
        raise ValueError("Phase 36a crossed panel changed")

    report_models = {}
    for label, model in models.items():
        split_reports = {}
        for split in TARGET_SPLITS:
            identity_reports = {}
            for token_id in sorted(panel[split]):
                value = panel[split][token_id]
                records = value["records"]
                examples = encode_proposal_records(records, tokenizer, block_size)
                metrics = _evaluate(
                    model, examples, records, tuple(range(len(records))),
                    tokenizer, args.batch_size, device, 1.0,
                )
                focus, geometry = evaluate_focus_geometry(
                    model, tokenizer, examples, token_id, args.batch_size, device
                )
                identity_reports[str(token_id)] = {
                    "group": value["group"], "text": value["text"],
                    "rows": len(records), "metrics": metrics,
                    "focus": focus, "focus_geometry": geometry,
                }
            split_reports[split] = {"identities": identity_reports}
            print(f"{label}/{split}: evaluated {len(identity_reports)} identities", flush=True)
        report_models[label] = {
            "checkpoint": _checkpoint_metadata(paths[label], checkpoints[label], tokenizer),
            "splits": split_reports,
        }

    config = checkpoints["phase36c"].get("extra", {}).get("config", {})
    data_config = config.get("data", {})
    swap_opportunity = audit_swap_opportunity(data_config, tokenizer, block_size, pool_by_width)

    summary = {
        "begin_only_identity_invariance_audit_version": (
            BEGIN_ONLY_IDENTITY_INVARIANCE_AUDIT_VERSION
        ),
        "training_changes": "begin_only_position_restriction",
        "decoder_changes": "none",
        "architecture_changes": "none",
        "phase36a_summary_path": str(args.phase36a_summary),
        "phase36a_summary_sha256": _sha256(args.phase36a_summary),
        "identity_swap_pool_sha256": extras["phase36c"].get(
            "identity_invariance_swap_pool_sha256"
        ),
        "focus_identities": expected_identities,
        "panel_validation": validation,
        "phase36c_position_counts": counts,
        "swap_opportunity_audit": swap_opportunity,
        "models": report_models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
