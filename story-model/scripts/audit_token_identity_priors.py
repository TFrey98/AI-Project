"""Audit token identity and train-time boundary priors for Phase 34h."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_OBJECTIVE_VERSION,
    EXPLICIT_OFFSET_PROPOSER_VERSION,
    OUTSIDE_TAG,
    begin_tag,
    gold_evidence_spans,
    inside_tag,
    proposal_record_is_eligible,
    tag_is_begin,
    tag_type,
)
from story_model.provenance import canonical_json_sha256

try:
    from scripts.phase34h_token_identity_decision import (
        token_identity_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34h_token_identity_decision import (  # type: ignore[no-redef]
        token_identity_decision,
    )


TOKEN_IDENTITY_AUDIT_VERSION = 1
DATASETS = ("phase31_regression", "phase32")
REFERENCE_SPLITS = ("train", "val")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tokenizer(checkpoint_path: Path):
    checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "explicit_offset_candidate_proposer":
        raise ValueError("checkpoint is not a Phase 34 candidate proposer")
    if (
        extra.get("explicit_offset_proposer_version")
        != EXPLICIT_OFFSET_PROPOSER_VERSION
    ):
        raise ValueError("checkpoint has an unsupported proposer version")
    if extra.get("boundary_objective_version") != BOUNDARY_OBJECTIVE_VERSION:
        raise ValueError("Phase 34h requires the Phase 34d objective")
    if abs(float(extra.get("boundary_loss_weight", -1.0)) - 1.0) > 1.0e-9:
        raise ValueError("Phase 34h requires boundary loss weight 1.0")
    if extra.get("token_width_geometry_version") is not None:
        raise ValueError("Phase 34h requires the pre-geometry Phase 34d model")
    if extra.get("token_end_geometry_version") is not None:
        raise ValueError("Phase 34h requires the pre-geometry Phase 34d model")
    tokenizer_data = extra.get("tokenizer", {})
    tokenizer = tokenizer_from_dict(tokenizer_data)
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 34h requires byte-BPE")
    return checkpoint, tokenizer, canonical_json_sha256(tokenizer_data)


def _token_layout(prompt: str, tokenizer: ByteBPETokenizer) -> tuple[dict, ...]:
    rows = []
    cursor = 0
    pieces = []
    for token_position, token_id in enumerate(tokenizer.encode(prompt)):
        token_bytes = tokenizer.token_bytes(token_id)
        pieces.append(token_bytes)
        rows.append(
            {
                "token_position": token_position,
                "token_id": token_id,
                "token_bytes": token_bytes,
                "byte_start": cursor,
                "byte_end": cursor + len(token_bytes),
            }
        )
        cursor += len(token_bytes)
    if b"".join(pieces) != prompt.encode("utf-8"):
        raise RuntimeError("token layout does not reproduce the prompt")
    return tuple(rows)


def _token_at(layout, byte_start: int) -> tuple[dict, int]:
    for token in layout:
        if token["byte_start"] <= byte_start < token["byte_end"]:
            return token, byte_start - token["byte_start"]
    raise ValueError(f"byte position {byte_start} is outside the prompt")


def _token_utf8(token_bytes: bytes):
    try:
        return token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _enrich_start_rows(rows, records, tokenizer):
    layout_cache = {}
    enriched = []
    for index, row in enumerate(rows):
        key = (row.get("dataset"), row.get("split"), row.get("record_id"))
        if key not in records:
            raise ValueError(f"start row {index} has no matching dataset record")
        record = records[key]
        for field in ("skill", "case", "source_phase"):
            if row.get(field) != getattr(record, field):
                raise ValueError(
                    f"start row {index} {field} does not match its record"
                )
        layout = layout_cache.get(record.prompt)
        if layout is None:
            layout = _token_layout(record.prompt, tokenizer)
            layout_cache[record.prompt] = layout
        token, offset = _token_at(layout, int(row["byte_start"]))
        token_bytes = token["token_bytes"]
        if int(row["token_width"]) != len(token_bytes):
            raise ValueError(f"start row {index} token width changed")
        if int(row["token_byte_offset"]) != offset:
            raise ValueError(f"start row {index} token offset changed")
        enriched.append(
            {
                **row,
                "token_id": token["token_id"],
                "token_bytes_hex": token_bytes.hex(),
                "token_utf8": _token_utf8(token_bytes),
                "token_byte_values": list(token_bytes),
                "first_byte": token_bytes[0],
                "second_byte": (
                    token_bytes[1] if len(token_bytes) > 1 else None
                ),
            }
        )
    return tuple(enriched)


def _new_panel() -> dict:
    return {
        "prompt_token_occurrences": 0,
        "labels": Counter(),
        "typed_labels": Counter(),
    }


def _collapsed_label(tag: int) -> str:
    if tag == OUTSIDE_TAG:
        return "O"
    return "B" if tag_is_begin(tag) else "I"


def _record_offset_zero_labels(record, layout):
    prompt_bytes = record.prompt.encode("utf-8")
    tags = [OUTSIDE_TAG] * len(prompt_bytes)
    for span in gold_evidence_spans(record):
        tags[span.byte_start] = begin_tag(span.value_type)
        inside = inside_tag(span.value_type)
        for position in range(span.byte_start + 1, span.byte_end):
            tags[position] = inside
    for token in layout:
        tag = tags[token["byte_start"]]
        label = _collapsed_label(tag)
        value_type = tag_type(tag)
        typed_label = f"{label}:{value_type}" if value_type else None
        yield token["token_id"], label, typed_label


def _add_reference_panels(records, tokenizer, focus_ids):
    panels = {
        token_id: {
            "train": _new_panel(),
            "val": _new_panel(),
            "combined": _new_panel(),
        }
        for token_id in focus_ids
    }
    layout_cache = {}
    for (dataset, split, _record_id), record in records.items():
        if dataset not in DATASETS or split not in REFERENCE_SPLITS:
            continue
        if not proposal_record_is_eligible(record):
            continue
        layout = layout_cache.get(record.prompt)
        if layout is None:
            layout = _token_layout(record.prompt, tokenizer)
            layout_cache[record.prompt] = layout
        for token_id, label, typed_label in _record_offset_zero_labels(
            record, layout
        ):
            if token_id not in panels:
                continue
            for panel_name in (split, "combined"):
                panel = panels[token_id][panel_name]
                panel["prompt_token_occurrences"] += 1
                panel["labels"][label] += 1
                if typed_label is not None:
                    panel["typed_labels"][typed_label] += 1
    return panels


def _summarize_panel(panel: dict) -> dict:
    labels = {label: int(panel["labels"].get(label, 0)) for label in "OBI"}
    typed = {
        label: int(count)
        for label, count in sorted(panel["typed_labels"].items())
    }
    positive = labels["B"] + labels["I"]
    route_begin = typed.get("B:route", 0)
    route_inside = typed.get("I:route", 0)
    route_positive = route_begin + route_inside
    return {
        "prompt_token_occurrences": int(panel["prompt_token_occurrences"]),
        "labels": labels,
        "typed_labels": typed,
        "inside_share_among_positive": (
            labels["I"] / positive if positive else 0.0
        ),
        "route_inside_share_among_positive": (
            route_inside / route_positive if route_positive else 0.0
        ),
    }


def _familiar_metrics(rows, token_id: int) -> dict:
    familiar = tuple(
        row
        for row in rows
        if row["dataset"] == "phase31_regression"
        and row["split"] in REFERENCE_SPLITS
        and row["skill"] == "scene_route"
        and int(row["token_id"]) == token_id
        and int(row["token_byte_offset"]) == 0
    )
    inside = sum(row["baseline_predicted_class"] == "I" for row in familiar)
    exact = sum(bool(row["baseline_exact_begin_tag"]) for row in familiar)
    return {
        "gold_begin_count": len(familiar),
        "begin_as_inside_count": inside,
        "begin_as_inside_rate": inside / len(familiar) if familiar else 0.0,
        "exact_begin_rate": exact / len(familiar) if familiar else 0.0,
    }


def _token_statistics(enriched_rows, reference_panels, tokenizer):
    result = {}
    for token_id, panels in sorted(reference_panels.items()):
        token_bytes = tokenizer.token_bytes(token_id)
        result[str(token_id)] = {
            "token_id": token_id,
            "token_bytes_hex": token_bytes.hex(),
            "token_utf8": _token_utf8(token_bytes),
            "token_byte_values": list(token_bytes),
            "reference": {
                name: _summarize_panel(panel)
                for name, panel in panels.items()
            },
            "familiar_phase31_train_val": _familiar_metrics(
                enriched_rows, token_id
            ),
        }
    return result


def _load_records(phase31_dir: Path, phase32_dir: Path):
    records = {}
    files = {}
    for dataset, directory in (
        ("phase31_regression", phase31_dir),
        ("phase32", phase32_dir),
    ):
        for split in EXPANDED_SPLITS:
            path = directory / f"{split}.jsonl"
            loaded = load_expanded_records(path)
            files[f"{dataset}/{split}"] = {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(loaded),
            }
            for record in loaded:
                key = (dataset, split, record.record_id)
                if key in records:
                    raise ValueError(f"duplicate record key: {key}")
                records[key] = record
    return records, files


def _validate_start_summary(summary: dict, checkpoint: dict) -> None:
    if summary.get("begin_calibration_ablation_version") != 1:
        raise ValueError("start summary is not a Phase 34e ablation")
    if summary.get("decision", {}).get("branch") != (
        "bpe_geometry_representation_indicated"
    ):
        raise ValueError("Phase 34e did not select the BPE-geometry branch")
    if int(summary.get("checkpoint_step", -1)) != int(
        checkpoint.get("step", 0)
    ):
        raise ValueError("start rows and checkpoint have different steps")
    if summary.get("checkpoint_boundary_objective_version") != (
        BOUNDARY_OBJECTIVE_VERSION
    ):
        raise ValueError("start summary lacks the Phase 34d objective")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start-rows", type=Path, required=True)
    parser.add_argument("--start-summary", type=Path)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start_summary_path = args.start_summary or args.start_rows.with_suffix(
        ".summary.json"
    )

    checkpoint, tokenizer, tokenizer_sha256 = _load_tokenizer(
        args.checkpoint
    )
    start_summary = json.loads(start_summary_path.read_text(encoding="utf-8"))
    _validate_start_summary(start_summary, checkpoint)
    start_rows_raw = args.start_rows.read_bytes()
    start_rows = tuple(
        json.loads(line)
        for line in start_rows_raw.decode("utf-8").splitlines()
        if line.strip()
    )
    records, data_files = _load_records(
        args.phase31_data_dir, args.phase32_data_dir
    )
    enriched_rows = _enrich_start_rows(start_rows, records, tokenizer)
    focus_ids = {
        int(row["token_id"])
        for row in enriched_rows
        if row["dataset"] == "phase31_regression"
        and row["split"] in {"lexical", "transfer"}
        and row["skill"] == "scene_route"
        and int(row["token_width"]) == 2
        and int(row["token_byte_offset"]) == 0
    }
    reference_panels = _add_reference_panels(
        records, tokenizer, focus_ids
    )
    token_statistics = _token_statistics(
        enriched_rows, reference_panels, tokenizer
    )
    decision = token_identity_decision(enriched_rows, token_statistics)
    target_rows = tuple(
        row
        for row in enriched_rows
        if row["dataset"] == "phase31_regression"
        and row["split"] in {"lexical", "transfer"}
        and row["skill"] == "scene_route"
    )
    target_path = args.output.with_name(
        f"{args.output.stem}.target.jsonl"
    )
    report = {
        "token_identity_audit_version": TOKEN_IDENTITY_AUDIT_VERSION,
        "training_changes": "none",
        "decoder_changes": "none",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", 0),
        "checkpoint_eligible": checkpoint.get("extra", {}).get(
            "checkpoint_eligible"
        ),
        "checkpoint_boundary_objective_version": checkpoint.get(
            "extra", {}
        ).get("boundary_objective_version"),
        "tokenizer_sha256": tokenizer_sha256,
        "start_rows": {
            "path": str(args.start_rows),
            "sha256": hashlib.sha256(start_rows_raw).hexdigest(),
            "rows": len(start_rows),
            "summary_path": str(start_summary_path),
            "summary_sha256": _sha256(start_summary_path),
        },
        "data_files": data_files,
        "reference_panel": (
            "Phase 31 and Phase 32 train+val prompt-token labels at "
            "byte offset zero"
        ),
        "target_rows": str(target_path),
        "token_statistics": token_statistics,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in target_rows
        ),
        encoding="utf-8",
    )

    print(f"decision: {decision['branch']}")
    if "width_2_offset_0_target" in decision:
        target = decision["width_2_offset_0_target"]
        prior = decision["reference_inside_prior"]["contrast"]
        token = decision["lexical_to_transfer_token_confirmation"]
        byte = decision["lexical_to_transfer_first_byte_confirmation"]
        print(
            "width-2/offset-0 B->I: "
            f"{target['begin_as_inside_rate']:.3f} "
            f"({target['begin_as_inside_count']}/"
            f"{target['gold_begin_count']})"
        )
        print(
            "training inside-prior capture and rate ratio: "
            f"{prior['error_capture']:.3f}, "
            f"{prior['error_rate_ratio']:.3f}"
        )
        print(
            "lexical->transfer token confirmation: "
            f"capture {token['error_capture']:.3f}, "
            f"ratio {token['error_rate_ratio']:.3f}"
        )
        print(
            "lexical->transfer first-byte confirmation: "
            f"capture {byte['error_capture']:.3f}, "
            f"ratio {byte['error_rate_ratio']:.3f}"
        )
    for reason in decision.get("invalid_reasons", []):
        print(f"- {reason}")
    print(f"next action: {decision['next_action']}")
    print(f"report: {args.output}")
    print(f"target rows: {target_path}")
    if decision["branch"] == "invalid_token_identity_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
