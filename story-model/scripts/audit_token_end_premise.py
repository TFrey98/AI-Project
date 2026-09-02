"""Audit Phase 34e start rows before authorizing Phase 34g training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.phase34g_token_end_premise import (
        token_end_premise_decision,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from phase34g_token_end_premise import (  # type: ignore[no-redef]
        token_end_premise_decision,
    )


TOKEN_END_PREMISE_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.start_rows.read_bytes()
    rows = tuple(
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    decision = token_end_premise_decision(rows)
    report = {
        "token_end_premise_version": TOKEN_END_PREMISE_VERSION,
        "input": str(args.start_rows),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision: {decision['branch']}")
    if "target" in decision:
        print(
            "target error capture at token end: "
            f"{decision['target_error_token_end_capture']:.3f}"
        )
        print(
            "width-2 error capture at token end: "
            f"{decision['width_two_error_token_end_capture']:.3f}"
        )
        print(
            "token-end error-rate ratio: "
            f"{decision['token_end_error_rate_ratio']:.3f}"
        )
    for category in ("invalid_reasons", "failures"):
        for failure in decision[category]:
            print(f"- {failure}")
    print(f"next action: {decision['next_action']}")
    print(f"report: {args.output}")
    if decision["branch"] != "token_end_geometry_indicated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
