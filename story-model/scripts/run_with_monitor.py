"""Run any long project command with persistent resource telemetry."""

from __future__ import annotations

import argparse
import sys

from story_model.run_monitor import run_monitored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--status-interval", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=120.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("a command is required after --")

    result = run_monitored(
        command=command,
        name=args.name,
        output_root=args.output_root,
        sample_interval=args.sample_interval,
        status_interval=args.status_interval,
        stale_seconds=args.stale_seconds,
    )
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
