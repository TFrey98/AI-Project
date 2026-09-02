"""Gate Phase 34f against the frozen Phase 34d tag audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase34f_token_width_decision import (
        token_width_geometry_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34f_token_width_decision import (  # type: ignore[no-redef]
        token_width_geometry_decision,
    )


def _read_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("tag_confusion_audit_version") != 1:
        raise ValueError(f"{path} is not a Phase 34c tag-confusion summary")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision = token_width_geometry_decision(
        _read_summary(args.baseline),
        _read_summary(args.candidate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    print(
        "diagnostic gates: "
        f"target={decision['target_gate_passed']}, "
        f"safety={decision['safety_gate_passed']}, "
        f"eligible={decision['candidate_checkpoint_eligible']}"
    )
    print(
        "target token-width-2 B->I: "
        f"{decision['target_token_width_2_begin_as_inside_rate']}"
    )
    for category in (
        "provenance_failures",
        "target_failures",
        "safety_failures",
    ):
        for failure in decision[category]:
            print(f"- {failure}")
    print(f"next action: {decision['next_action']}")
    print(f"decision report: {args.output}")
    if decision["branch"] != "token_width_geometry_diagnostic_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
