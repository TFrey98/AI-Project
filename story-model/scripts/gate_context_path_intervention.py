"""Recompute the Phase 34k decision from a frozen audit summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase34k_context_path_decision import context_path_decision
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34k_context_path_decision import (  # type: ignore[no-redef]
        context_path_decision,
    )


EXPECTED_AUDIT_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("context_path_intervention_version") != (
        EXPECTED_AUDIT_VERSION
    ):
        raise ValueError("input is not a Phase 34k context-path audit")
    decision = context_path_decision(summary)
    embedded = summary.get("decision")
    if embedded is not None and embedded != decision:
        raise ValueError("embedded Phase 34k decision is stale or inconsistent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    for reason in decision["invalid_reasons"]:
        print(f"- {reason}")
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] == "invalid_context_path_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
