"""Audit Phase 36d's clean-anchor objective against the Phase 35 baseline.

Phase 36c proved the begin-only ablation did not localize the span damage:
even with zero inside-position pairs, exact-span precision fell in all 28
identity/split cells and the gold-I margin shifted 1.33 logits toward B.
Phase 36d restores begin-and-inside pairing but removes every supervised
tag/type/boundary contribution from the swapped copy, so only the original
("anchor") rows receive direct supervision; the swapped copy participates
only through the JSD consistency term. This audit checks all 14 registered
identities (Phase 36c's failure was isolated to legacy `ro`, which a 12-of-14
check would have caught, but a 14-of-14 check is the more rigorous default
going forward) on the unchanged Phase 36a crossed-identity panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from story_model.boundary_counterbalance import validate_counterbalance_manifest
from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import encode_proposal_records
from story_model.paired_identity_invariance import (
    CLEAN_ANCHOR_SUPERVISION_MODE,
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
    from scripts.train_explicit_offset_candidate_proposer import _evaluate
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_crossed_boundary_identity import (  # type: ignore
        TARGET_SPLITS, _read_json, _sha256, build_crossed_identity_panel,
        canonical_focus_identities, evaluate_focus_geometry,
    )
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from train_explicit_offset_candidate_proposer import _evaluate  # type: ignore


CLEAN_ANCHOR_IDENTITY_INVARIANCE_AUDIT_VERSION = 1


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
        "identity_invariance_supervision_mode": extra.get("identity_invariance_supervision_mode"),
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
    parser.add_argument("--phase36d-checkpoint", type=Path, required=True)
    parser.add_argument("--phase31-data-dir", type=Path,
                        default=Path("data/character/typed_span_resolver"))
    parser.add_argument("--phase35-data-dir", type=Path,
                        default=Path("data/character/boundary_counterbalance"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    paths = {"phase35": args.phase35_checkpoint, "phase36d": args.phase36d_checkpoint}
    models, tokenizers, checkpoints, block_sizes = {}, {}, {}, {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label], tokenizers[label] = model, tokenizer
        checkpoints[label], block_sizes[label] = checkpoint, block_size
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 35/36d block sizes differ")
    tokenizer, block_size = tokenizers["phase36d"], block_sizes["phase36d"]
    if canonical_json_sha256(tokenizers["phase35"].to_dict()) != canonical_json_sha256(
        tokenizer.to_dict()
    ):
        raise ValueError("Phase 35 and Phase 36d tokenizers differ")

    extras = {label: checkpoint.get("extra", {}) for label, checkpoint in checkpoints.items()}
    if extras["phase35"].get("paired_identity_invariance_version") is not None:
        raise ValueError("Phase 35 baseline already used identity invariance")
    if extras["phase36d"].get("paired_identity_invariance_version") != 3:
        raise ValueError("phase36d-checkpoint is not a Phase 36d clean-anchor run")
    if extras["phase36d"].get("identity_invariance_supervision_mode") != (
        CLEAN_ANCHOR_SUPERVISION_MODE
    ):
        raise ValueError("Phase 36d checkpoint is not in clean-anchor supervision mode")

    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    validate_counterbalance_manifest(manifest, args.phase35_data_dir, tokenizer)
    excluded_ids = registered_identity_ids(manifest)
    swap_pool = extras["phase36d"].get("identity_invariance_swap_pool")
    if not isinstance(swap_pool, dict):
        raise ValueError("Phase 36d checkpoint has no swap pool")
    validate_identity_swap_pool(swap_pool, tokenizer, excluded_ids)

    identities = canonical_focus_identities(manifest, tokenizer)
    expected_identities = {
        group: [{"token_id": token_id, "text": text} for token_id, text in values]
        for group, values in identities.items()
    }

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

    summary = {
        "clean_anchor_identity_invariance_audit_version": (
            CLEAN_ANCHOR_IDENTITY_INVARIANCE_AUDIT_VERSION
        ),
        "training_changes": "clean_anchor_supervision_masking_only",
        "decoder_changes": "none",
        "architecture_changes": "none",
        "focus_identities": expected_identities,
        "panel_validation": validation,
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
