"""Apply Phase 28 semantic-transfer acceptance thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.semantic_evaluation import (
    SEMANTIC_TRANSFER_THRESHOLDS,
    semantic_transfer_acceptance_failures,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-summary", required=True)
    parser.add_argument("--semantic-summary", required=True)
    args = parser.parse_args()
    generation_path = Path(args.generation_summary)
    semantic_path = Path(args.semantic_summary)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    summaries = {}

    for split in SEMANTIC_TRANSFER_THRESHOLDS:
        summaries[split] = {
            **generation.get(split, {}),
            **semantic.get(split, {}),
        }

    failures = semantic_transfer_acceptance_failures(summaries)
    print(f"generation summary: {generation_path}")
    print(f"semantic summary: {semantic_path}")

    for split, thresholds in SEMANTIC_TRANSFER_THRESHOLDS.items():
        print(f"{split} thresholds:")

        for metric, threshold in thresholds.items():
            comparator = "<=" if metric.endswith("_max") else ">="
            shown_metric = metric.removesuffix("_max")
            print(f"- {shown_metric}: {comparator} {threshold:.1%}")

    if failures:
        print("semantic transfer gate: failed")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("semantic transfer gate: passed")


if __name__ == "__main__":
    main()
