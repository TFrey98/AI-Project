"""Gate Phase 33 routing while preserving every Phase 31/32 capability."""

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


def _check(
    failures: list[str],
    label: str,
    metrics: dict,
    name: str,
    threshold: float,
    minimum: bool,
) -> None:
    if name not in metrics:
        failures.append(f"{label} is missing {name}")
        return
    actual = float(metrics[name])
    failed = actual < threshold if minimum else actual > threshold
    if failed:
        comparator = "below" if minimum else "above"
        failures.append(
            f"{label} {name} {actual:.3f} is {comparator} {threshold:.3f}"
        )


def _check_multi_candidate_action(
    failures: list[str], label: str, metrics: dict
) -> None:
    count_name = "multi_candidate_action_examples"
    accuracy_name = "multi_candidate_action_accuracy"
    if count_name not in metrics:
        failures.append(f"{label} is missing {count_name}")
        return
    if accuracy_name not in metrics:
        failures.append(f"{label} is missing {accuracy_name}")
        return
    if int(metrics[count_name]) > 0:
        _check(failures, label, metrics, accuracy_name, 0.95, True)


def _full_failures(
    label: str,
    metrics: dict,
    resolve_min: float,
    pair_min: float,
) -> list[str]:
    failures: list[str] = []
    checks = (
        ("end_to_end_resolve_accuracy", resolve_min, True),
        ("counterfactual_pair_resolve_accuracy", pair_min, True),
        ("real_candidate_top1_accuracy", 0.995, True),
        ("clarify_accuracy", 0.95, True),
        ("clarify_sentinel_accuracy", 0.95, True),
        ("generate_accuracy", 0.99, True),
        ("mode_accuracy", 0.98, True),
        ("value_missing_rate", 0.02, False),
        ("wrong_alternative_rate", 0.02, False),
        ("no_support_false_positive_rate", 0.02, False),
    )
    for name, threshold, minimum in checks:
        _check(failures, label, metrics, name, threshold, minimum)
    _check_multi_candidate_action(failures, label, metrics)
    return failures


def _selection_failures(
    label: str,
    metrics: dict,
    resolve_min: float,
    pair_min: float,
) -> list[str]:
    failures: list[str] = []
    checks = (
        ("end_to_end_resolve_accuracy", resolve_min, True),
        ("counterfactual_pair_resolve_accuracy", pair_min, True),
        ("real_candidate_top1_accuracy", 0.995, True),
        ("value_missing_rate", 0.02, False),
        ("wrong_alternative_rate", 0.02, False),
        ("no_support_false_positive_rate", 0.02, False),
    )
    for name, threshold, minimum in checks:
        _check(failures, label, metrics, name, threshold, minimum)
    _check_multi_candidate_action(failures, label, metrics)
    return failures


def gate_failures(summary: dict) -> tuple[str, ...]:
    failures: list[str] = []
    for dataset_name in ("phase31_regression", "phase32"):
        dataset = summary.get(dataset_name)
        if not isinstance(dataset, dict) or not isinstance(dataset.get("splits"), dict):
            failures.append(f"missing dataset {dataset_name}")
            continue
        for split, (resolve_min, pair_min) in THRESHOLDS.items():
            metrics = dataset["splits"].get(split)
            label = f"{dataset_name}/{split}"
            if not isinstance(metrics, dict):
                failures.append(f"missing split {label}")
                continue
            failures.extend(_full_failures(label, metrics, resolve_min, pair_min))

            per_skill = metrics.get("per_skill")
            if not isinstance(per_skill, dict) or not per_skill:
                failures.append(f"missing per-skill metrics {label}")
                continue
            for skill, skill_metrics in sorted(per_skill.items()):
                failures.extend(
                    _full_failures(
                        f"{label}/{skill}", skill_metrics, resolve_min, pair_min
                    )
                )

            widths = metrics.get("per_candidate_width")
            expected_widths = {"2", "3", "4"} if dataset_name == "phase32" else {"2"}
            if not isinstance(widths, dict) or set(widths) != expected_widths:
                failures.append(f"missing candidate widths {label}")
            else:
                for width, width_metrics in sorted(widths.items()):
                    failures.extend(
                        _selection_failures(
                            f"{label}/width-{width}",
                            width_metrics,
                            resolve_min,
                            pair_min,
                        )
                    )

            cells = metrics.get("per_skill_candidate_width")
            if not isinstance(cells, dict):
                failures.append(f"missing skill-width metrics {label}")
                continue
            for skill, width_groups in sorted(cells.items()):
                if set(width_groups) != expected_widths:
                    failures.append(f"missing widths {label}/{skill}")
                    continue
                for width, width_metrics in sorted(width_groups.items()):
                    failures.extend(
                        _selection_failures(
                            f"{label}/{skill}/width-{width}",
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
    print("Phase 33 unified routing thresholds:")
    for split, (resolve_min, pair_min) in THRESHOLDS.items():
        print(
            f"- {split}: resolve >= {resolve_min:.0%}, pairs >= {pair_min:.0%}, "
            "real-candidate >= 99.5%, sentinel >= 95%, generate >= 99%, "
            "multi-candidate action >= 95% where applicable, "
            "missing/wrong/no-support false-positive <= 2%"
        )
    if failures:
        print("unified typed-span resolver gate: failed")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("unified typed-span resolver gate: passed")


if __name__ == "__main__":
    main()
