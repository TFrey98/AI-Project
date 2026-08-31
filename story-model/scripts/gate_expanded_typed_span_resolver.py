"""Gate Phase 32 per skill while preserving every Phase 31 split."""

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


def _metric_failures(label: str, metrics: dict, resolve_min: float, pair_min: float):
    checks = (
        ("end_to_end_resolve_accuracy", resolve_min, "minimum"),
        ("counterfactual_pair_resolve_accuracy", pair_min, "minimum"),
        ("clarify_accuracy", 0.95, "minimum"),
        ("generate_accuracy", 0.95, "minimum"),
        ("value_missing_rate", 0.02, "maximum"),
        ("wrong_alternative_rate", 0.02, "maximum"),
    )
    failures = []
    for name, threshold, direction in checks:
        actual = float(metrics[name])
        failed = actual < threshold if direction == "minimum" else actual > threshold
        if failed:
            comparator = "below" if direction == "minimum" else "above"
            failures.append(
                f"{label} {name} {actual:.3f} is {comparator} {threshold:.3f}"
            )
    return failures


def _selection_failures(
    label: str, metrics: dict, resolve_min: float, pair_min: float
):
    checks = (
        ("end_to_end_resolve_accuracy", resolve_min, "minimum"),
        ("counterfactual_pair_resolve_accuracy", pair_min, "minimum"),
        ("value_missing_rate", 0.02, "maximum"),
        ("wrong_alternative_rate", 0.02, "maximum"),
    )
    failures = []
    for name, threshold, direction in checks:
        actual = float(metrics[name])
        failed = actual < threshold if direction == "minimum" else actual > threshold
        if failed:
            comparator = "below" if direction == "minimum" else "above"
            failures.append(
                f"{label} {name} {actual:.3f} is {comparator} {threshold:.3f}"
            )
    return failures


def gate_failures(summary: dict) -> tuple[str, ...]:
    failures = []
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary.get(dataset_name)
        if not isinstance(dataset, dict) or not isinstance(dataset.get("splits"), dict):
            failures.append(f"missing dataset {dataset_name}")
            continue
        for split, (resolve_min, pair_min) in THRESHOLDS.items():
            metrics = dataset["splits"].get(split)
            if not isinstance(metrics, dict):
                failures.append(f"missing split {dataset_name}/{split}")
                continue
            failures.extend(
                _metric_failures(
                    f"{dataset_name}/{split}", metrics, resolve_min, pair_min
                )
            )
            per_skill = metrics.get("per_skill")
            if not isinstance(per_skill, dict) or not per_skill:
                failures.append(f"missing per-skill metrics {dataset_name}/{split}")
                continue
            for skill, skill_metrics in sorted(per_skill.items()):
                failures.extend(
                    _metric_failures(
                        f"{dataset_name}/{split}/{skill}",
                        skill_metrics,
                        resolve_min,
                        pair_min,
                    )
                )
            if dataset_name == "phase32":
                widths = metrics.get("per_candidate_width")
                if not isinstance(widths, dict) or set(widths) != {"2", "3", "4"}:
                    failures.append(
                        f"missing candidate widths {dataset_name}/{split}"
                    )
                else:
                    for width, width_metrics in sorted(widths.items()):
                        failures.extend(
                            _selection_failures(
                                f"{dataset_name}/{split}/width-{width}",
                                width_metrics,
                                resolve_min,
                                pair_min,
                            )
                        )
                cells = metrics.get("per_skill_candidate_width")
                if not isinstance(cells, dict):
                    failures.append(
                        f"missing skill-width metrics {dataset_name}/{split}"
                    )
                else:
                    for skill, width_groups in sorted(cells.items()):
                        if set(width_groups) != {"2", "3", "4"}:
                            failures.append(
                                f"missing widths {dataset_name}/{split}/{skill}"
                            )
                            continue
                        for width, width_metrics in sorted(width_groups.items()):
                            failures.extend(
                                _selection_failures(
                                    f"{dataset_name}/{split}/{skill}/width-{width}",
                                    width_metrics,
                                    resolve_min,
                                    pair_min,
                                )
                            )
    return tuple(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    failures = gate_failures(summary)
    print("Phase 31 regression and Phase 32 per-skill thresholds:")
    for split, (resolve_min, pair_min) in THRESHOLDS.items():
        print(
            f"- {split}: resolve >= {resolve_min:.0%}, pairs >= {pair_min:.0%}, "
            "clarify/generate >= 95%, missing/wrong <= 2%"
        )
    if failures:
        print("expanded typed-span resolver gate: failed")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("expanded typed-span resolver gate: passed")


if __name__ == "__main__":
    main()
