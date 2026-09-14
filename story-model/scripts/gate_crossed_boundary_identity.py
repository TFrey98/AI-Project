"""Recompute and export the frozen Phase 36a crossed-identity decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase36a_crossed_identity_decision import (
        crossed_identity_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase36a_crossed_identity_decision import (  # type: ignore
        crossed_identity_decision,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Phase 36a summary must be a JSON object")
    decision = crossed_identity_decision(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    for reason in decision["invalid_reasons"]:
        print(f"- {reason}")
    print(f"group results: {decision['group_results']}")
    print(f"span regressions: {len(decision['complete_span_regressions'])}")
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] == "invalid_crossed_identity_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
