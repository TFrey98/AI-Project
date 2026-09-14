"""Recompute and export the registered Phase 36d decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase36d_clean_anchor_identity_invariance_decision import (
        clean_anchor_identity_invariance_decision,
    )
except ModuleNotFoundError:
    from phase36d_clean_anchor_identity_invariance_decision import (  # type: ignore
        clean_anchor_identity_invariance_decision,
    )


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossed-summary", type=Path, required=True)
    parser.add_argument("--baseline-retained", type=Path, required=True)
    parser.add_argument("--candidate-retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = clean_anchor_identity_invariance_decision(
        _read(args.crossed_summary),
        _read(args.baseline_retained),
        _read(args.candidate_retained),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    for category in ("invalid_reasons", "transfer_failures", "span_regressions_vs_phase35",
                      "retained_regression_failures"):
        for reason in decision[category]:
            print(f"- {reason}")
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] != "clean_anchor_identity_invariance_diagnostic_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
