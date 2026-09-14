"""Run the frozen Phase 36a crossed identity-context boundary audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from story_model.boundary_counterbalance import (
    BOUNDARY_COUNTERBALANCE_VERSION,
    _route_values,
    _transform_pair,
    focus_boundary_counts,
    scene_route_pairs,
    validate_counterbalance_manifest,
)
from story_model.expanded_typed_span_resolver import load_expanded_records
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_OBJECTIVE_VERSION,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    begin_tag,
    encode_proposal_records,
    inside_tag,
    proposal_batch,
)
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device

try:
    from scripts.audit_boundary_counterbalance import (
        summarize_focus_predictions,
    )
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.phase34c_tag_confusion_decision import collapsed_tag
    from scripts.phase36a_crossed_identity_decision import (
        CROSSED_IDENTITY_AUDIT_VERSION,
        crossed_identity_decision,
    )
    from scripts.train_explicit_offset_candidate_proposer import _evaluate
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_boundary_counterbalance import (  # type: ignore[no-redef]
        summarize_focus_predictions,
    )
    from audit_explicit_offset_tag_confusion import (  # type: ignore[no-redef]
        _load_model,
    )
    from phase34c_tag_confusion_decision import (  # type: ignore[no-redef]
        collapsed_tag,
    )
    from phase36a_crossed_identity_decision import (  # type: ignore[no-redef]
        CROSSED_IDENTITY_AUDIT_VERSION,
        crossed_identity_decision,
    )
    from train_explicit_offset_candidate_proposer import (  # type: ignore[no-redef]
        _evaluate,
    )


TARGET_SPLITS = ("lexical", "transfer")
CROSSED_ROUTE_VARIANT = 0
MINIMUM_CLASS_SUPPORT = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def canonical_focus_identities(manifest: dict, tokenizer) -> dict[str, tuple]:
    """Recover the registered 8/4 identities plus legacy `or`/`ro`."""

    result = {}
    for group in ("train", "heldout"):
        values = []
        for entry in manifest.get("focus_tokens", {}).get(group, ()):
            token_id = int(entry["token_id"])
            text = entry["text"]
            if tokenizer.encode(text) != [token_id]:
                raise ValueError(f"token {token_id} is not canonical for {text!r}")
            values.append((token_id, text))
        result["trained" if group == "train" else group] = tuple(values)
    legacy = []
    premise = manifest.get("phase34k_premise", {})
    for text, token_id in sorted(premise.get("focus_token_ids", {}).items()):
        token_id = int(token_id)
        if tokenizer.encode(text) != [token_id]:
            raise ValueError(f"legacy token {text!r} is not canonical")
        legacy.append((token_id, text))
    result["legacy"] = tuple(legacy)
    expected = {"trained": 8, "heldout": 4, "legacy": 2}
    for group, size in expected.items():
        if len(result.get(group, ())) != size:
            raise ValueError(f"Phase 36a requires {size} {group} identities")
    all_ids = [token_id for values in result.values() for token_id, _ in values]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Phase 36a focus identity groups overlap")
    return result


def _first_byte_tags(example, tokenizer) -> tuple[int, ...]:
    tags = []
    cursor = 0
    for position in range(example.prompt_token_count):
        tags.append(example.prompt_byte_tags[cursor])
        cursor += tokenizer.token_byte_length(example.input_ids[position])
    if cursor != len(example.prompt_byte_tags):
        raise RuntimeError("prompt tags do not align to token bytes")
    return tuple(tags)


def build_crossed_identity_panel(
    source_records: dict,
    identities: dict[str, tuple],
    tokenizer,
    block_size: int,
    minimum_class_support: int = MINIMUM_CLASS_SUPPORT,
) -> tuple[dict, dict]:
    """Cross every identity through identical lexical/transfer contexts."""

    flat = tuple(
        (group, token_id, text)
        for group, values in identities.items()
        for token_id, text in values
    )
    all_ids = {token_id for _, token_id, _ in flat}
    panel = {}
    validation = {}
    for split in TARGET_SPLITS:
        pairs = scene_route_pairs(source_records[split])
        by_identity = {}
        encoded_by_identity = {}
        for group, token_id, text in flat:
            route_values = _route_values(text, split, CROSSED_ROUTE_VARIANT)
            if any(value.count(text) != 1 for value in route_values):
                raise ValueError(
                    f"Phase 36a token {token_id} collides with crossed filler"
                )
            records = []
            for pair_index, pair in enumerate(pairs):
                transformed = _transform_pair(
                    pair,
                    split,
                    pair_index,
                    token_id,
                    text,
                    CROSSED_ROUTE_VARIANT,
                )
                for side, record in enumerate(transformed):
                    records.append(
                        replace(
                            record,
                            record_id=(
                                f"phase36a_{split}_{pair_index:05d}_"
                                f"token_{token_id}:{side}"
                            ),
                            source_context_id=(
                                f"phase36a_{split}_{pair_index:05d}:"
                                f"{side}"
                            ),
                            conversation_id=(
                                f"phase36a_{split}_{pair_index:05d}_"
                                f"token_{token_id}"
                            ),
                            source_phase="phase36a",
                        )
                    )
            records = tuple(records)
            counts = focus_boundary_counts(
                records, tokenizer, block_size, token_id
            )
            if (
                counts["B"] < minimum_class_support
                or counts["I"] < minimum_class_support
                or counts["B"] != counts["I"]
            ):
                raise ValueError(
                    f"Phase 36a {split} token {token_id} lacks balanced support"
                )
            by_identity[token_id] = {
                "group": group,
                "text": text,
                "records": records,
                "focus_counts": counts,
            }
            encoded_by_identity[token_id] = encode_proposal_records(
                records, tokenizer, block_size
            )

        invalid_contexts = []
        reference_id = flat[0][1]
        for row_index in range(len(by_identity[reference_id]["records"])):
            examples = {
                token_id: encoded[row_index]
                for token_id, encoded in encoded_by_identity.items()
            }
            lengths = {
                (example.sequence_tokens, example.prompt_token_count)
                for example in examples.values()
            }
            if len(lengths) != 1:
                invalid_contexts.append(
                    {"row": row_index, "reason": "sequence lengths differ"}
                )
                continue
            prompt_tokens = next(iter(lengths))[1]
            varying = []
            for position in range(prompt_tokens):
                values = {
                    example.input_ids[position] for example in examples.values()
                }
                if len(values) > 1:
                    varying.append(position)
                    if values != all_ids:
                        invalid_contexts.append(
                            {
                                "row": row_index,
                                "position": position,
                                "reason": "non-focus token changed",
                            }
                        )
            if not varying:
                invalid_contexts.append(
                    {"row": row_index, "reason": "no focus position changed"}
                )
                continue
            for token_id, example in examples.items():
                first_tags = _first_byte_tags(example, tokenizer)
                if any(
                    example.input_ids[position] != token_id
                    or first_tags[position]
                    not in (begin_tag("route"), inside_tag("route"))
                    for position in varying
                ):
                    invalid_contexts.append(
                        {
                            "row": row_index,
                            "token_id": token_id,
                            "reason": "focus substitution changed labels",
                        }
                    )
        if invalid_contexts:
            raise ValueError(
                f"Phase 36a {split} has {len(invalid_contexts)} invalid contexts"
            )
        panel[split] = by_identity
        serialized = [
            record.to_dict()
            for token_id in sorted(by_identity)
            for record in by_identity[token_id]["records"]
        ]
        validation[split] = {
            "source_pairs": len(pairs),
            "source_rows": len(pairs) * 2,
            "identities": len(flat),
            "rows_per_identity": len(pairs) * 2,
            "total_rows": len(serialized),
            "matched_contexts": len(pairs) * 2,
            "invalid_contexts": 0,
            "canonical_rows_sha256": canonical_json_sha256(serialized),
        }
    return panel, validation


@torch.no_grad()
def evaluate_focus_geometry(
    model,
    tokenizer,
    examples,
    token_id: int,
    batch_size: int,
    device,
) -> tuple[dict, dict]:
    """Measure focus confusion and the typed route B-minus-I margin."""

    prediction_rows = []
    margins = defaultdict(list)
    route_begin = begin_tag("route")
    route_inside = inside_tag("route")
    model.eval()
    for start in range(0, len(examples), batch_size):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        batch = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        output = model(batch[0], batch[1], batch[2])
        logits = output.tag_logits.detach().cpu()
        predictions = logits.argmax(dim=-1)
        targets = batch[3].detach().cpu()
        for offset, index in enumerate(indices):
            example = examples[index]
            for position in range(example.prompt_token_count):
                if example.input_ids[position] != token_id:
                    continue
                target = int(targets[offset, position, 0])
                if target not in (route_begin, route_inside):
                    continue
                gold = "B" if target == route_begin else "I"
                prediction_rows.append(
                    {
                        "token_id": token_id,
                        "gold": gold,
                        "predicted": collapsed_tag(
                            int(predictions[offset, position, 0])
                        ),
                    }
                )
                margins[gold].append(
                    float(
                        logits[offset, position, 0, route_begin]
                        - logits[offset, position, 0, route_inside]
                    )
                )
    focus = summarize_focus_predictions(prediction_rows)
    geometry = {}
    for gold in ("B", "I"):
        values = margins[gold]
        geometry[f"gold_{gold}"] = {
            "observations": len(values),
            "mean_begin_minus_inside_margin": (
                sum(values) / len(values) if values else 0.0
            ),
            "minimum_begin_minus_inside_margin": min(values, default=0.0),
            "maximum_begin_minus_inside_margin": max(values, default=0.0),
        }
    return focus, geometry


def _checkpoint_metadata(path: Path, checkpoint: dict, tokenizer) -> dict:
    extra = checkpoint.get("extra", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "step": checkpoint.get("step", 0),
        "checkpoint_eligible": extra.get("checkpoint_eligible"),
        "architecture": extra.get("architecture"),
        "explicit_offset_proposer_version": extra.get(
            "explicit_offset_proposer_version"
        ),
        "boundary_objective_version": extra.get("boundary_objective_version"),
        "boundary_loss_weight": extra.get("boundary_loss_weight"),
        "boundary_counterbalance_version": extra.get(
            "boundary_counterbalance_version"
        ),
        "token_width_geometry_version": extra.get(
            "token_width_geometry_version"
        ),
        "token_end_geometry_version": extra.get("token_end_geometry_version"),
        "factorized_boundary_type_version": extra.get(
            "factorized_boundary_type_version"
        ),
        "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
    }


def _validate_checkpoints(checkpoints: dict, tokenizers: dict) -> None:
    for label, checkpoint in checkpoints.items():
        extra = checkpoint.get("extra", {})
        if extra.get("architecture") != "explicit_offset_candidate_proposer":
            raise ValueError(f"{label} is not an explicit-offset proposer")
        if extra.get("explicit_offset_proposer_version") != (
            EXPLICIT_OFFSET_PROPOSER_VERSION
        ):
            raise ValueError(f"{label} proposer version changed")
        if extra.get("boundary_objective_version") != BOUNDARY_OBJECTIVE_VERSION:
            raise ValueError(f"{label} lost the boundary objective")
        if abs(float(extra.get("boundary_loss_weight", -1.0)) - 1.0) > 1.0e-9:
            raise ValueError(f"{label} boundary loss weight changed")
        if any(
            extra.get(field) is not None
            for field in (
                "token_width_geometry_version",
                "token_end_geometry_version",
                "factorized_boundary_type_version",
            )
        ):
            raise ValueError(f"{label} is not the legacy Phase 34d architecture")
    phase35_extra = checkpoints["phase35"].get("extra", {})
    if phase35_extra.get("boundary_counterbalance_version") != (
        BOUNDARY_COUNTERBALANCE_VERSION
    ):
        raise ValueError("Phase 35 checkpoint has wrong counterbalance version")
    hashes = {
        canonical_json_sha256(tokenizer.to_dict())
        for tokenizer in tokenizers.values()
    }
    if len(hashes) != 1:
        raise ValueError("Phase 34d and Phase 35 tokenizers differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase34d-checkpoint", type=Path, required=True)
    parser.add_argument("--phase35-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase35-retained-audit",
        type=Path,
        default=Path("runs/phase35-retained-tag-audit.summary.json"),
    )
    parser.add_argument(
        "--phase35-counterbalance-audit",
        type=Path,
        default=Path("runs/phase35-boundary-counterbalance-audit.summary.json"),
    )
    parser.add_argument(
        "--phase35-decision",
        type=Path,
        default=Path("runs/phase35-boundary-counterbalance-decision.json"),
    )
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument(
        "--phase35-data-dir",
        type=Path,
        default=Path("data/character/boundary_counterbalance"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    retained = _read_json(args.phase35_retained_audit)
    counterbalance = _read_json(args.phase35_counterbalance_audit)
    phase35_decision = _read_json(args.phase35_decision)
    if phase35_decision.get("branch") != "boundary_counterbalance_rejected":
        raise ValueError("Phase 36a requires the rejected Phase 35 branch")
    for field in (
        "checkpoint_promotion_authorized",
        "full_phase34_evaluation_authorized",
    ):
        if phase35_decision.get(field) is not False:
            raise ValueError(f"Phase 35 decision has invalid {field}")

    device = torch.device(resolve_device(args.device))
    paths = {
        "phase34d": args.phase34d_checkpoint,
        "phase35": args.phase35_checkpoint,
    }
    models = {}
    tokenizers = {}
    checkpoints = {}
    block_sizes = {}
    for label, path in paths.items():
        model, tokenizer, block_size, checkpoint = _load_model(path, device)
        models[label] = model
        tokenizers[label] = tokenizer
        checkpoints[label] = checkpoint
        block_sizes[label] = block_size
    _validate_checkpoints(checkpoints, tokenizers)
    if len(set(block_sizes.values())) != 1:
        raise ValueError("Phase 34d and Phase 35 block sizes differ")
    tokenizer = tokenizers["phase35"]
    block_size = block_sizes["phase35"]

    phase35_sha256 = _sha256(args.phase35_checkpoint)
    if retained.get("checkpoint_sha256") != phase35_sha256:
        raise ValueError("retained audit and Phase 35 checkpoint differ")
    if counterbalance.get("checkpoint", {}).get("sha256") != phase35_sha256:
        raise ValueError("counterbalance audit and Phase 35 checkpoint differ")
    if retained.get("checkpoint_eligible") is not False:
        raise ValueError("Phase 35 retained audit eligibility changed")
    if counterbalance.get("checkpoint", {}).get("eligible") is not False:
        raise ValueError("Phase 35 counterbalance eligibility changed")

    manifest_path = args.phase35_data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    pools = validate_counterbalance_manifest(
        manifest, args.phase35_data_dir, tokenizer
    )
    manifest_sha256 = canonical_json_sha256(manifest)
    if counterbalance.get("checkpoint", {}).get("manifest_sha256") != (
        manifest_sha256
    ):
        raise ValueError("Phase 35 audit and manifest differ")
    if checkpoints["phase35"].get("extra", {}).get(
        "counterbalance_manifest_sha256"
    ) != manifest_sha256:
        raise ValueError("Phase 35 checkpoint and manifest differ")
    if counterbalance.get("focus_token_ids") != {
        group: list(token_ids) for group, token_ids in pools.items()
    }:
        raise ValueError("Phase 35 audit focus identities changed")

    identities = canonical_focus_identities(manifest, tokenizer)
    source_records = {}
    data_files = {}
    for split in TARGET_SPLITS:
        path = args.phase31_data_dir / f"{split}.jsonl"
        records = load_expanded_records(path)
        source_records[split] = records
        expected = manifest.get("source_phase31_files", {}).get(split, {})
        if expected.get("sha256") != _sha256(path):
            raise ValueError(f"Phase 31 {split} data changed after Phase 35")
        if int(expected.get("rows", -1)) != len(records):
            raise ValueError(f"Phase 31 {split} row count changed")
        data_files[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(records),
        }
    panel, panel_validation = build_crossed_identity_panel(
        source_records, identities, tokenizer, block_size
    )

    report_models = {}
    for model_label, model in models.items():
        split_reports = {}
        for split in TARGET_SPLITS:
            identity_reports = {}
            for token_id in sorted(panel[split]):
                value = panel[split][token_id]
                records = value["records"]
                examples = encode_proposal_records(
                    records, tokenizer, block_size
                )
                metrics = _evaluate(
                    model,
                    examples,
                    records,
                    tuple(range(len(records))),
                    tokenizer,
                    args.batch_size,
                    device,
                    1.0,
                )
                focus, geometry = evaluate_focus_geometry(
                    model,
                    tokenizer,
                    examples,
                    token_id,
                    args.batch_size,
                    device,
                )
                identity_reports[str(token_id)] = {
                    "group": value["group"],
                    "text": value["text"],
                    "rows": len(records),
                    "metrics": metrics,
                    "focus": focus,
                    "focus_geometry": geometry,
                }
            split_reports[split] = {"identities": identity_reports}
            print(
                f"{model_label}/{split}: evaluated "
                f"{len(identity_reports)} crossed identities",
                flush=True,
            )
        report_models[model_label] = {
            "checkpoint": _checkpoint_metadata(
                paths[model_label], checkpoints[model_label], tokenizer
            ),
            "splits": split_reports,
        }

    original_heldout = {}
    for token_id in pools["heldout"]:
        original_heldout[str(token_id)] = {
            split: counterbalance.get("splits", {})
            .get(split, {})
            .get("focus", {})
            .get("per_token", {})
            .get(str(token_id), {})
            .get("begin_as_inside_rate")
            for split in TARGET_SPLITS
        }
    summary = {
        "crossed_boundary_identity_audit_version": (
            CROSSED_IDENTITY_AUDIT_VERSION
        ),
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint_changes": "none",
        "target": (
            "Phase 31 lexical+transfer scene_route pairs crossed with every "
            "Phase 35 and legacy focus identity"
        ),
        "crossed_route_variant": CROSSED_ROUTE_VARIANT,
        "phase35_inputs": {
            "decision_path": str(args.phase35_decision),
            "decision_sha256": _sha256(args.phase35_decision),
            "decision_branch": phase35_decision["branch"],
            "checkpoint_sha256": phase35_sha256,
            "checkpoint_eligible": retained["checkpoint_eligible"],
            "checkpoint_promotion_authorized": phase35_decision[
                "checkpoint_promotion_authorized"
            ],
            "full_phase34_evaluation_authorized": phase35_decision[
                "full_phase34_evaluation_authorized"
            ],
            "retained_audit_sha256": _sha256(args.phase35_retained_audit),
            "counterbalance_audit_sha256": _sha256(
                args.phase35_counterbalance_audit
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "original_heldout_focus": original_heldout,
        },
        "focus_identities": {
            group: [
                {"token_id": token_id, "text": text}
                for token_id, text in values
            ]
            for group, values in identities.items()
        },
        "data_files": data_files,
        "panel_validation": panel_validation,
        "models": report_models,
    }
    summary["decision"] = crossed_identity_decision(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision = summary["decision"]
    print(f"decision: {decision['branch']}")
    print(f"group results: {decision['group_results']}")
    print(f"span regressions: {len(decision['complete_span_regressions'])}")
    print(f"next action: {decision['next_action']}")
    print(f"summary: {args.output}")
    if decision["branch"] == "invalid_crossed_identity_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
