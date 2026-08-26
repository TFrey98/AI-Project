"""Apply the Phase 27 stop/go thresholds to generation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.counterfactual_probe import (
    COUNTERFACTUAL_ACCEPTANCE_THRESHOLDS,
    counterfactual_acceptance_failures,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    path = Path(args.summary)
    summaries = json.loads(path.read_text(encoding="utf-8"))
    failures = counterfactual_acceptance_failures(summaries)

    print(f"summary: {path}")

    for split, thresholds in COUNTERFACTUAL_ACCEPTANCE_THRESHOLDS.items():
        print(f"{split} thresholds:")

        for metric, threshold in thresholds.items():
            comparator = "<=" if metric.endswith("_max") else ">="
            shown_metric = metric.removesuffix("_max")
            print(f"- {shown_metric}: {comparator} {threshold:.1%}")

    if failures:
        print("counterfactual probe gate: failed")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("counterfactual probe gate: passed")


if __name__ == "__main__":
    main()
