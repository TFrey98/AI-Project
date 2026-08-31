"""Apply the bounded Phase 31 typed-span acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "train": (0.95, 0.90),
    "val": (0.95, 0.90),
    "lexical": (0.95, 0.90),
    "paraphrase": (0.95, 0.90),
    "transfer": (0.90, 0.80),
}


def gate_failures(summary: dict) -> tuple[str, ...]:
    splits = summary.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("resolver summary has no splits object")
    failures = []
    for split, (resolve_minimum, pair_minimum) in THRESHOLDS.items():
        if split not in splits:
            failures.append(f"missing split {split}")
            continue
        metrics = splits[split]
        checks = (
            (
                "end_to_end_resolve_accuracy",
                float(metrics["end_to_end_resolve_accuracy"]),
                resolve_minimum,
                "minimum",
            ),
            (
                "counterfactual_pair_resolve_accuracy",
                float(metrics["counterfactual_pair_resolve_accuracy"]),
                pair_minimum,
                "minimum",
            ),
            (
                "clarify_accuracy",
                float(metrics["clarify_accuracy"]),
                0.95,
                "minimum",
            ),
            (
                "generate_accuracy",
                float(metrics["generate_accuracy"]),
                0.95,
                "minimum",
            ),
            (
                "value_missing_rate",
                float(metrics["value_missing_rate"]),
                0.02,
                "maximum",
            ),
            (
                "wrong_alternative_rate",
                float(metrics["wrong_alternative_rate"]),
                0.02,
                "maximum",
            ),
        )
        for name, actual, threshold, direction in checks:
            failed = (
                actual < threshold
                if direction == "minimum"
                else actual > threshold
            )
            if failed:
                comparator = "below" if direction == "minimum" else "above"
                failures.append(
                    f"{split} {name} {actual:.3f} is {comparator} "
                    f"{threshold:.3f}"
                )
    return tuple(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    failures = gate_failures(summary)
    for split, (resolve_minimum, pair_minimum) in THRESHOLDS.items():
        print(
            f"{split}: resolve >= {resolve_minimum:.0%}, "
            f"pairs >= {pair_minimum:.0%}, clarify >= 95%, "
            "generate >= 95%, missing <= 2%, wrong <= 2%"
        )
    if failures:
        print("typed-span resolver gate: failed")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("typed-span resolver gate: passed")


if __name__ == "__main__":
    main()
