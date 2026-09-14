"""Audit Phase 36b with the frozen Phase 36a crossed-identity panel."""

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
    from scripts.phase36a_crossed_identity_decision import crossed_identity_decision
    from scripts.train_explicit_offset_candidate_proposer import _evaluate
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities, evaluate_focus_geometry,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from phase36a_crossed_identity_decision import crossed_identity_decision  # type: ignore
    from train_explicit_offset_candidate_proposer import _evaluate  # type: ignore


PAIRED_IDENTITY_INVARIANCE_AUDIT_VERSION = 1


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
        "identity_invariance_swap_pool_sha256": extra.get("identity_invariance_swap_pool_sha256"),
        "token_width_geometry_version": extra.get("token_width_geometry_version"),
        "token_end_geometry_version": extra.get("token_end_geometry_version"),
        "factorized_boundary_type_version": extra.get("factorized_boundary_type_version"),
        "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
        "parent_phase33_checkpoint": extra.get("parent_phase33_checkpoint"),
        "parent_phase33_step": extra.get("parent_phase33_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase35-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36b-checkpoint", type=Path, required=True)
    parser.add_argument("--phase36a-summary", type=Path, required=True)
    parser.add_argument("--phase36a-decision", type=Path, required=True)
    parser.add_argument("--phase31-data-dir", type=Path,
                        default=Path("data/character/typed_span_resolver"))
    parser.add_argument("--phase35-data-dir", type=Path,
                        default=Path("data/character/boundary_counterbalance"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase36a_summary = _read_json(args.phase36a_summary)
    phase36a_decision = _read_json(args.phase36a_decision)
    if phase36a_decision != crossed_identity_decision(phase36a_summary):
        raise ValueError("Phase 36a decision does not match its summary")
    if phase36a_decision.get("branch") != "trained_identity_memorization_dominant":
        raise ValueError("Phase 36a did not authorize paired invariance")

    device = torch.device(resolve_device(args.device))
    paths = {"phase35": args.phase35_checkpoint, "phase36b": args.phase36b_checkpoint}
    models, tokenizers, checkpoints, block_sizes = {}, {}, {}, {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label], tokenizers[label] = model, tokenizer
        checkpoints[label], block_sizes[label] = checkpoint, block_size
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 35 and Phase 36b block sizes differ")
    tokenizer, block_size = tokenizers["phase36b"], block_sizes["phase36b"]
    if canonical_json_sha256(tokenizers["phase35"].to_dict()) != canonical_json_sha256(tokenizer.to_dict()):
        raise ValueError("Phase 35 and Phase 36b tokenizers differ")

    phase35_extra = checkpoints["phase35"].get("extra", {})
    candidate_extra = checkpoints["phase36b"].get("extra", {})
    if phase35_extra.get("paired_identity_invariance_version") is not None:
        raise ValueError("Phase 35 baseline already used identity invariance")
    if candidate_extra.get("paired_identity_invariance_version") != 1:
        raise ValueError("candidate is not a Phase 36b checkpoint")
    for field in ("architecture", "boundary_objective_version", "boundary_loss_weight",
                  "boundary_counterbalance_version", "parent_phase33_checkpoint",
                  "parent_phase33_step"):
        if phase35_extra.get(field) != candidate_extra.get(field):
            raise ValueError(f"Phase 35/36b {field} differs")

    phase35_sha256 = _sha256(args.phase35_checkpoint)
    if phase36a_summary["models"]["phase35"]["checkpoint"]["sha256"] != phase35_sha256:
        raise ValueError("Phase 36a and Phase 36b use different baselines")
    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    validate_counterbalance_manifest(manifest, args.phase35_data_dir, tokenizer)
    manifest_sha256 = canonical_json_sha256(manifest)
    for label, checkpoint in checkpoints.items():
        if checkpoint.get("extra", {}).get("counterbalance_manifest_sha256") != manifest_sha256:
            raise ValueError(f"{label} checkpoint and manifest differ")

    identities = canonical_focus_identities(manifest, tokenizer)
    expected_identities = {
        group: [{"token_id": token_id, "text": text} for token_id, text in values]
        for group, values in identities.items()
    }
    if phase36a_summary.get("focus_identities") != expected_identities:
        raise ValueError("Phase 36a identities changed")
    swap_pool = candidate_extra.get("identity_invariance_swap_pool")
    if not isinstance(swap_pool, dict):
        raise ValueError("Phase 36b checkpoint has no swap pool")
    validate_identity_swap_pool(swap_pool, tokenizer, registered_identity_ids(manifest))
    swap_pool_sha256 = canonical_json_sha256(swap_pool)
    if candidate_extra.get("identity_invariance_swap_pool_sha256") != swap_pool_sha256:
        raise ValueError("Phase 36b swap-pool hash changed")

    source_records, data_files = {}, {}
    for split in TARGET_SPLITS:
        path = args.phase31_data_dir / f"{split}.jsonl"
        records = load_expanded_records(path)
        expected = manifest["source_phase31_files"][split]
        if expected["sha256"] != _sha256(path) or expected["rows"] != len(records):
            raise ValueError(f"Phase 31 {split} data changed")
        source_records[split] = records
        data_files[split] = {"path": str(path), "sha256": _sha256(path), "rows": len(records)}
    panel, validation = build_crossed_identity_panel(source_records, identities, tokenizer, block_size)
    if validation != phase36a_summary.get("panel_validation"):
        raise ValueError("Phase 36a crossed panel changed")

    candidate_splits = {}
    for split in TARGET_SPLITS:
        reports = {}
        for token_id in sorted(panel[split]):
            value, model = panel[split][token_id], models["phase36b"]
            records = value["records"]
            examples = encode_proposal_records(records, tokenizer, block_size)
            metrics = _evaluate(model, examples, records, tuple(range(len(records))),
                                tokenizer, args.batch_size, device, 1.0)
            focus, geometry = evaluate_focus_geometry(
                model, tokenizer, examples, token_id, args.batch_size, device
            )
            reports[str(token_id)] = {
                "group": value["group"], "text": value["text"],
                "rows": len(records), "metrics": metrics,
                "focus": focus, "focus_geometry": geometry,
            }
        candidate_splits[split] = {"identities": reports}
        print(f"phase36b/{split}: evaluated {len(reports)} identities", flush=True)

    baseline_model = deepcopy(phase36a_summary["models"]["phase35"])
    baseline_model["checkpoint"] = _checkpoint_metadata(
        args.phase35_checkpoint, checkpoints["phase35"], tokenizer
    )
    summary = {
        "paired_identity_invariance_audit_version": 1,
        "training_changes": "paired_identity_invariance_only",
        "decoder_changes": "none", "architecture_changes": "none",
        "phase36a_inputs": {
            "summary_path": str(args.phase36a_summary),
            "summary_sha256": _sha256(args.phase36a_summary),
            "decision_path": str(args.phase36a_decision),
            "decision_sha256": _sha256(args.phase36a_decision),
            "decision_branch": phase36a_decision["branch"],
            "phase35_checkpoint_sha256": phase35_sha256,
            "checkpoint_promotion_authorized": False,
            "full_phase34_evaluation_authorized": False,
        },
        "identity_swap_pool": swap_pool,
        "identity_swap_pool_sha256": swap_pool_sha256,
        "focus_identities": expected_identities,
        "data_files": data_files, "panel_validation": validation,
        "models": {
            "phase35": baseline_model,
            "phase36b": {
                "checkpoint": _checkpoint_metadata(args.phase36b_checkpoint,
                                                   checkpoints["phase36b"], tokenizer),
                "splits": candidate_splits,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
