"""Recompute and export the registered Phase 36g decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase36g_consistency_weight_decision import (
        consistency_weight_decision,
    )
except ModuleNotFoundError:
    from phase36g_consistency_weight_decision import (  # type: ignore
        consistency_weight_decision,
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
    decision = consistency_weight_decision(
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
    print(f"comparison step: {decision['comparison_step']}")
    for category in (
        "invalid_reasons",
        "transfer_failures",
        "span_regressions_vs_phase35",
        "retained_regression_failures",
    ):
        for reason in decision[category][:20]:
            print(f"- {reason}")
        if len(decision[category]) > 20:
            print(f"  ... and {len(decision[category]) - 20} more {category}")
    # Reported independently, never folded into the branch name.
    print(f"transfer_ok: {decision['transfer_ok']}")
    print(f"span_ok: {decision['span_ok']}")
    print(f"retained_ok: {decision['retained_ok']}")
    print(f"candidate_checkpoint_eligible: {decision['candidate_checkpoint_eligible']}")
    print(
        "full Phase 34 evaluation authorized: "
        f"{decision['full_phase34_evaluation_authorized']}"
    )
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] == "invalid_consistency_weight_comparison":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
