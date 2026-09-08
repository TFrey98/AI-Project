"""Build the tokenizer-aware Phase 35 boundary counterbalance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.boundary_counterbalance import (
    DEFAULT_HELDOUT_FOCUS_TOKENS,
    DEFAULT_TRAIN_FOCUS_TOKENS,
    build_boundary_counterbalance_dataset,
)
from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    load_expanded_records,
)
from story_model.unified_typed_span_resolver import SUPPORT_MASK_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase33-checkpoint", type=Path, required=True)
    parser.add_argument("--phase34k-summary", type=Path, required=True)
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/character/boundary_counterbalance"),
    )
    parser.add_argument(
        "--train-focus-tokens",
        type=int,
        default=DEFAULT_TRAIN_FOCUS_TOKENS,
    )
    parser.add_argument(
        "--heldout-focus-tokens",
        type=int,
        default=DEFAULT_HELDOUT_FOCUS_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    checkpoint = read_checkpoint(args.phase33_checkpoint, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "unified_typed_span_resolver":
        raise ValueError("Phase 35 requires the Phase 33c resolver checkpoint")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("Phase 35 parent predates Phase 33c routing")
    if extra.get("checkpoint_eligible") is not True:
        raise ValueError("Phase 35 requires an eligible Phase 33c checkpoint")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 35 requires byte-BPE")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("Phase 33c checkpoint has no configuration")
    block_size = int(config.get("data", {}).get("block_size", 0))
    if block_size < 1:
        raise ValueError("Phase 33c checkpoint has an invalid block size")

    phase34k_summary = json.loads(
        args.phase34k_summary.read_text(encoding="utf-8")
    )
    source_files = {
        split: args.phase31_data_dir / f"{split}.jsonl"
        for split in EXPANDED_SPLITS
    }
    source_records = {
        split: load_expanded_records(path)
        for split, path in source_files.items()
    }
    manifest = build_boundary_counterbalance_dataset(
        source_records,
        tokenizer,
        phase34k_summary,
        args.output_dir,
        source_files,
        args.phase34k_summary,
        train_focus_tokens=args.train_focus_tokens,
        heldout_focus_tokens=args.heldout_focus_tokens,
        block_size=block_size,
        seed=args.seed,
    )

    print(
        "focus tokens: train "
        f"{len(manifest['focus_tokens']['train'])}, heldout "
        f"{len(manifest['focus_tokens']['heldout'])}, overlap 0"
    )
    print(
        "Phase 34k focus identities excluded: "
        + ", ".join(
            f"{text}={token_id}"
            for text, token_id in sorted(
                manifest["phase34k_premise"]["focus_token_ids"].items()
            )
        )
    )
    for split in EXPANDED_SPLITS:
        report = manifest["splits"][split]
        minimum = min(
            counts["B"]
            for counts in report["focus_label_counts"].values()
        )
        print(
            f"{split}: {report['rows']:,} rows, {report['pairs']:,} pairs, "
            f"{report['focus_pool']} focus pool, minimum B/I support "
            f"{minimum}/{minimum}"
        )
    print(f"output: {args.output_dir}")
    print("Phase 35 boundary counterbalance dataset: passed")


if __name__ == "__main__":
    main()
