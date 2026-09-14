"""Recompute and export the registered Phase 36b decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.phase36b_identity_invariance_decision import identity_invariance_decision
except ModuleNotFoundError:
    from phase36b_identity_invariance_decision import identity_invariance_decision  # type: ignore


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
    decision = identity_invariance_decision(
        _read(args.crossed_summary), _read(args.baseline_retained),
        _read(args.candidate_retained),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"decision: {decision['branch']}")
    print(f"trained passing: {decision['trained_identities_passing']}/8")
    print(f"previously severe passing: {decision['previously_severe_identities_passing']}/4")
    print(f"next action: {decision['next_action']}")
    if decision["branch"] != "identity_invariance_diagnostic_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
