"""Gate Phase 35 from frozen retained and refreshed-token audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase35_boundary_counterbalance_decision import (
        boundary_counterbalance_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase35_boundary_counterbalance_decision import (  # type: ignore
        boundary_counterbalance_decision,
    )


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--counterbalance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision = boundary_counterbalance_decision(
        _read(args.baseline),
        _read(args.candidate),
        _read(args.counterbalance),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    for category in (
        "invalid_reasons",
        "regression_failures",
        "generalization_failures",
    ):
        for reason in decision[category]:
            print(f"- {reason}")
    print(
        "full Phase 34 evaluation authorized: "
        f"{decision['full_phase34_evaluation_authorized']}"
    )
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] != "boundary_counterbalance_diagnostic_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
