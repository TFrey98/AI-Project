"""Summarize a run_with_monitor.py telemetry directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tail", type=int, default=12)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    samples = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    cpu_values = [
        sample["cpu_percent"]
        for sample in samples
        if "cpu_percent" in sample
    ]
    rss_values = [
        sample["rss_bytes"]
        for sample in samples
        if "rss_bytes" in sample
    ]
    maximum_output_age = max(
        (
            sample.get("last_output_age_seconds", 0.0)
            for sample in samples
        ),
        default=0.0,
    )
    hitch_count = sum(
        bool(sample.get("monitor_hitch")) for sample in samples
    )
    stale_count = sum(
        bool(sample.get("stale_output")) for sample in samples
    )
    print(f"name: {metadata['name']}")
    print(f"state: {status['state']}")
    print(f"exit code: {status.get('exit_code', 'running')}")
    print(f"elapsed: {status['elapsed_seconds'] / 60:.1f} minutes")
    print(f"git commit: {metadata.get('git_commit', 'unknown')}")
    print(f"git dirty: {metadata.get('git_dirty', 'unknown')}")
    print(f"samples: {len(samples):,}")
    print(
        "peak child CPU: "
        + (
            f"{max(cpu_values):.1f}%"
            if cpu_values
            else "unavailable"
        )
    )
    print(
        "peak child RSS: "
        + (
            f"{max(rss_values) / (1024 ** 3):.2f} GiB"
            if rss_values
            else "unavailable"
        )
    )
    print(f"maximum output silence: {maximum_output_age:.1f}s")
    print(f"monitor hitches: {hitch_count}")
    print(f"stale-output samples: {stale_count}")

    if args.tail > 0:
        lines = (run_dir / "output.log").read_text(
            encoding="utf-8"
        ).splitlines()
        print("last output:")

        for line in lines[-args.tail:]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
