"""Gate Phase 34 without weakening the complete Phase 33c oracle gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.gate_unified_typed_span_resolver import (
        THRESHOLDS,
        gate_failures as phase33_gate_failures,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from gate_unified_typed_span_resolver import (  # type: ignore[no-redef]
        THRESHOLDS,
        gate_failures as phase33_gate_failures,
    )


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


def _proposal_failures(
    label: str, metrics: dict, *, per_skill: bool = False
) -> list[str]:
    failures: list[str] = []
    precision_recall_floor = 0.95 if per_skill else 0.98
    answer_floor = 0.98 if per_skill else 0.995
    for name, threshold, minimum in (
        ("exact_span_precision", precision_recall_floor, True),
        ("exact_span_recall", precision_recall_floor, True),
        ("answer_candidate_recall", answer_floor, True),
        ("boundary_type_accuracy", 0.99, True),
        ("offset_validity_rate", 1.0, True),
        ("proposal_overflow_rate", 0.0, False),
    ):
        _check(failures, label, metrics, name, threshold, minimum)
    return failures


def _end_to_end_failures(
    label: str,
    metrics: dict,
    resolve_min: float,
    pair_min: float,
) -> list[str]:
    failures: list[str] = []
    for name, threshold, minimum in (
        ("candidate_inventory_recall", 0.995, True),
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
    ):
        _check(failures, label, metrics, name, threshold, minimum)
    return failures


def gate_failures(phase34_summary: dict, phase33_summary: dict) -> tuple[str, ...]:
    failures = [
        f"Phase 33c oracle regression: {failure}"
        for failure in phase33_gate_failures(phase33_summary)
    ]
    if phase34_summary.get("checkpoint_eligible") is not True:
        failures.append("Phase 34 checkpoint is not marked eligible")
    if phase34_summary.get("excluded_cases") != ["wrong_type"]:
        failures.append("Phase 34 must exclude only wrong_type proposer rows")

    for dataset_name in ("phase31_regression", "phase32"):
        dataset = phase34_summary.get(dataset_name)
        if not isinstance(dataset, dict) or not isinstance(
            dataset.get("splits"), dict
        ):
            failures.append(f"missing Phase 34 dataset {dataset_name}")
            continue
        if int(dataset.get("excluded_rows", 0)) <= 0:
            failures.append(
                f"{dataset_name} did not report excluded wrong_type rows"
            )
        for split, (resolve_min, pair_min) in THRESHOLDS.items():
            metrics = dataset["splits"].get(split)
            label = f"{dataset_name}/{split}"
            if not isinstance(metrics, dict):
                failures.append(f"missing Phase 34 split {label}")
                continue
            proposer = metrics.get("proposer")
            if not isinstance(proposer, dict):
                failures.append(f"missing proposer metrics {label}")
            else:
                failures.extend(_proposal_failures(label, proposer))
            failures.extend(
                _end_to_end_failures(label, metrics, resolve_min, pair_min)
            )
            per_skill = metrics.get("per_skill")
            if not isinstance(per_skill, dict) or not per_skill:
                failures.append(f"missing per-skill Phase 34 metrics {label}")
                continue
            for skill, skill_metrics in sorted(per_skill.items()):
                skill_label = f"{label}/{skill}"
                skill_proposer = skill_metrics.get("proposer")
                if not isinstance(skill_proposer, dict):
                    failures.append(f"missing proposer metrics {skill_label}")
                else:
                    failures.extend(
                        _proposal_failures(
                            skill_label, skill_proposer, per_skill=True
                        )
                    )
                failures.extend(
                    _end_to_end_failures(
                        skill_label, skill_metrics, resolve_min, pair_min
                    )
                )
    return tuple(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phase33-summary", type=Path, required=True)
    args = parser.parse_args()
    phase34_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    phase33_summary = json.loads(
        args.phase33_summary.read_text(encoding="utf-8")
    )
    failures = gate_failures(phase34_summary, phase33_summary)
    print("Phase 34 explicit-offset proposal thresholds:")
    print("- aggregate span precision/recall >= 98%")
    print("- aggregate answer inventory recall and real top-1 >= 99.5%")
    print("- UTF-8 offset validity = 100%; candidate overflow = 0%")
    print("- predicted-candidate resolver behavior retains Phase 33 thresholds")
    print("- the complete Phase 33c oracle-candidate gate must still pass")
    if failures:
        print("explicit-offset candidate proposer gate: failed")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("explicit-offset candidate proposer gate: passed")


if __name__ == "__main__":
    main()
