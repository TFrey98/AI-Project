"""Audit Phase 35 on all refreshed boundary-counterbalance rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from story_model.boundary_counterbalance import (
    BOUNDARY_COUNTERBALANCE_VERSION,
    FOCUS_SPLIT_KIND,
    validate_counterbalance_manifest,
)
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    BOUNDARY_OBJECTIVE_VERSION,
    encode_proposal_records,
    proposal_batch,
)
from story_model.provenance import canonical_json_sha256
from story_model.runtime import resolve_device

try:
    from scripts.audit_explicit_offset_tag_confusion import _load_model
    from scripts.phase34c_tag_confusion_decision import collapsed_tag
    from scripts.train_explicit_offset_candidate_proposer import _evaluate
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_explicit_offset_tag_confusion import _load_model  # type: ignore
    from phase34c_tag_confusion_decision import collapsed_tag  # type: ignore
    from train_explicit_offset_candidate_proposer import _evaluate  # type: ignore


BOUNDARY_COUNTERBALANCE_AUDIT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_focus_predictions(rows) -> dict:
    """Summarize exact focus-token B/I confusion with per-token support."""

    grouped = defaultdict(Counter)
    for row in rows:
        token_id = int(row["token_id"])
        gold = row["gold"]
        predicted = row["predicted"]
        if gold not in {"B", "I"} or predicted not in {"O", "B", "I"}:
            raise ValueError("focus prediction has an invalid tag class")
        grouped[token_id][f"gold_{gold}"] += 1
        grouped[token_id][f"{gold}_as_{predicted}"] += 1

    def metrics(counter: Counter) -> dict:
        begin = int(counter["gold_B"])
        inside = int(counter["gold_I"])
        return {
            "gold_begin_count": begin,
            "gold_inside_count": inside,
            "begin_as_inside_count": int(counter["B_as_I"]),
            "begin_as_inside_rate": (
                float(counter["B_as_I"]) / begin if begin else 0.0
            ),
            "inside_as_begin_count": int(counter["I_as_B"]),
            "inside_as_begin_rate": (
                float(counter["I_as_B"]) / inside if inside else 0.0
            ),
            "begin_as_outside_rate": (
                float(counter["B_as_O"]) / begin if begin else 0.0
            ),
            "inside_as_outside_rate": (
                float(counter["I_as_O"]) / inside if inside else 0.0
            ),
        }

    aggregate = Counter()
    for counter in grouped.values():
        aggregate.update(counter)
    return {
        "overall": metrics(aggregate),
        "per_token": {
            str(token_id): metrics(counter)
            for token_id, counter in sorted(grouped.items())
        },
    }


@torch.no_grad()
def evaluate_focus_predictions(
    model,
    tokenizer,
    block_size,
    records,
    focus_token_ids,
    batch_size,
    device,
) -> dict:
    examples = encode_proposal_records(records, tokenizer, block_size)
    focus_ids = {int(token_id) for token_id in focus_token_ids}
    rows = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        indices = tuple(range(start, min(start + batch_size, len(examples))))
        batch = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        output = model(batch[0], batch[1], batch[2])
        predictions = output.tag_logits.detach().cpu().argmax(dim=-1)
        targets = batch[3].detach().cpu()
        for offset, index in enumerate(indices):
            example = examples[index]
            for position in range(example.prompt_token_count):
                token_id = example.input_ids[position]
                if token_id not in focus_ids:
                    continue
                gold = collapsed_tag(int(targets[offset, position, 0]))
                if gold not in {"B", "I"}:
                    continue
                rows.append(
                    {
                        "token_id": token_id,
                        "gold": gold,
                        "predicted": collapsed_tag(
                            int(predictions[offset, position, 0])
                        ),
                    }
                )
    result = summarize_focus_predictions(rows)
    observed = {int(token_id) for token_id in result["per_token"]}
    if observed != focus_ids:
        missing = sorted(focus_ids - observed)
        raise ValueError(f"focus audit omitted token IDs {missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/character/boundary_counterbalance"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    device = torch.device(resolve_device(args.device))
    model, tokenizer, block_size, checkpoint = _load_model(
        args.checkpoint, device
    )
    extra = checkpoint.get("extra", {})
    if extra.get("boundary_counterbalance_version") != (
        BOUNDARY_COUNTERBALANCE_VERSION
    ):
        raise ValueError("checkpoint is not a Phase 35 counterbalance run")
    if extra.get("boundary_objective_version") != BOUNDARY_OBJECTIVE_VERSION:
        raise ValueError("Phase 35 checkpoint lost the boundary objective")
    if abs(float(extra.get("boundary_loss_weight", -1.0)) - 1.0) > 1.0e-9:
        raise ValueError("Phase 35 boundary loss weight is not 1.0")
    if any(
        extra.get(field) is not None
        for field in (
            "token_width_geometry_version",
            "token_end_geometry_version",
            "factorized_boundary_type_version",
        )
    ):
        raise ValueError("Phase 35 checkpoint changed proposer architecture")

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pools = validate_counterbalance_manifest(manifest, args.data_dir, tokenizer)
    manifest_sha256 = canonical_json_sha256(manifest)
    if extra.get("counterbalance_manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint did not train on this Phase 35 manifest")
    if block_size != int(manifest.get("block_size", -1)):
        raise ValueError("checkpoint and Phase 35 block sizes differ")

    summary = {
        "boundary_counterbalance_audit_version": (
            BOUNDARY_COUNTERBALANCE_AUDIT_VERSION
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "step": checkpoint.get("step", 0),
            "eligible": extra.get("checkpoint_eligible"),
            "boundary_objective_version": extra.get(
                "boundary_objective_version"
            ),
            "boundary_loss_weight": extra.get("boundary_loss_weight"),
            "boundary_counterbalance_version": extra.get(
                "boundary_counterbalance_version"
            ),
            "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
            "manifest_sha256": manifest_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "phase34k_premise": manifest["phase34k_premise"],
        },
        "focus_token_ids": {
            kind: list(token_ids) for kind, token_ids in pools.items()
        },
        "splits": {},
    }
    for split in EXPANDED_SPLITS:
        records = load_expanded_records(args.data_dir / f"{split}.jsonl")
        examples = encode_proposal_records(records, tokenizer, block_size)
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
        pool_kind = FOCUS_SPLIT_KIND[split]
        focus = evaluate_focus_predictions(
            model,
            tokenizer,
            block_size,
            records,
            pools[pool_kind],
            args.batch_size,
            device,
        )
        summary["splits"][split] = {
            "rows": len(records),
            "focus_pool": pool_kind,
            "metrics": metrics,
            "focus": focus,
        }
        overall = focus["overall"]
        print(
            f"{split}: B->I {overall['begin_as_inside_rate']:.3f}, "
            f"I->B {overall['inside_as_begin_rate']:.3f}, "
            f"span precision {metrics['exact_span_precision']:.3f}, "
            f"span recall {metrics['exact_span_recall']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
