"""External command telemetry for long training and evaluation runs."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


RUN_LOG_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_run_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    return slug or "run"


def parse_ps_snapshot(output: str) -> dict[str, Any] | None:
    """Parse the portable columns requested from macOS/Linux ``ps``."""

    line = output.strip()

    if not line:
        return None

    fields = line.split()

    if len(fields) != 7:
        return None

    try:
        return {
            "pid": int(fields[0]),
            "ppid": int(fields[1]),
            "state": fields[2],
            "process_elapsed": fields[3],
            "cpu_percent": float(fields[4]),
            "memory_percent": float(fields[5]),
            "rss_bytes": int(fields[6]) * 1024,
        }
    except ValueError:
        return None


def process_snapshot(pid: int) -> dict[str, Any] | None:
    result = subprocess.run(
        [
            "ps",
            "-o",
            "pid=,ppid=,state=,etime=,%cpu=,%mem=,rss=",
            "-p",
            str(pid),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        parsed = parse_ps_snapshot(result.stdout)

        if parsed is not None:
            return parsed

    if sys.platform.startswith("linux"):
        try:
            fields = {}

            for line in Path(f"/proc/{pid}/status").read_text(
                encoding="utf-8"
            ).splitlines():
                if ":" in line:
                    name, value = line.split(":", 1)
                    fields[name] = value.strip()

            rss_kib = int(fields.get("VmRSS", "0 kB").split()[0])
            return {
                "pid": pid,
                "ppid": int(fields.get("PPid", "0")),
                "state": fields.get("State", "unknown").split()[0],
                "rss_bytes": rss_kib * 1024,
            }
        except (OSError, ValueError, IndexError):
            pass

    return None


def system_memory_snapshot() -> dict[str, int]:
    """Return available/total memory without third-party dependencies."""

    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}

        try:
            for line in Path("/proc/meminfo").read_text(
                encoding="utf-8"
            ).splitlines():
                name, raw_value = line.split(":", 1)
                first_value = raw_value.strip().split()[0]
                values[name] = int(first_value) * 1024
        except (OSError, ValueError, IndexError):
            return {}

        output = {}

        if "MemTotal" in values:
            output["system_memory_total_bytes"] = values["MemTotal"]

        if "MemAvailable" in values:
            output["system_memory_available_bytes"] = values[
                "MemAvailable"
            ]

        return output

    if sys.platform == "darwin":
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return {}

        page_match = re.search(
            r"page size of\s+(\d+) bytes",
            result.stdout,
        )

        if page_match is None:
            return {}

        page_size = int(page_match.group(1))
        pages: dict[str, int] = {}

        for line in result.stdout.splitlines()[1:]:
            match = re.match(r"([^:]+):\s+(\d+)\.", line)

            if match is not None:
                pages[match.group(1)] = int(match.group(2))

        available_pages = sum(
            pages.get(name, 0)
            for name in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
            )
        )
        output = {
            "system_memory_available_bytes": available_pages * page_size,
        }

        try:
            total_pages = os.sysconf("SC_PHYS_PAGES")
            physical_page_size = os.sysconf("SC_PAGE_SIZE")
            output["system_memory_total_bytes"] = (
                int(total_pages) * int(physical_page_size)
            )
        except (ValueError, OSError, KeyError):
            pass

        return output

    return {}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_metadata(directory: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")

    if commit is None:
        return {}

    status = run_git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
    }


@dataclass(frozen=True)
class MonitoredRunResult:
    exit_code: int
    run_dir: Path
    state: str


def run_monitored(
    command: list[str],
    name: str,
    output_root: str | Path = "runs",
    sample_interval: float = 10.0,
    status_interval: float = 60.0,
    stale_seconds: float = 120.0,
    cwd: str | Path | None = None,
    emit: Callable[[str], None] = print,
) -> MonitoredRunResult:
    """Run a child command with tee output and an external heartbeat."""

    if not command:
        raise ValueError("monitored command cannot be empty")

    if sample_interval <= 0.0:
        raise ValueError("sample_interval must be positive")

    if status_interval <= 0.0:
        raise ValueError("status_interval must be positive")

    if stale_seconds <= 0.0:
        raise ValueError("stale_seconds must be positive")

    working_directory = Path(cwd or Path.cwd()).resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}-{slugify_run_name(name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "output.log"
    metrics_path = run_dir / "metrics.jsonl"
    metadata_path = run_dir / "metadata.json"
    status_path = run_dir / "status.json"
    started_wall = utc_now()
    started_monotonic = time.monotonic()
    metadata = {
        "schema_version": RUN_LOG_VERSION,
        "name": name,
        "command": command,
        "cwd": str(working_directory),
        "started_at": started_wall,
        "platform": platform.platform(),
        "python": sys.version,
        **git_metadata(working_directory),
    }
    write_json_atomic(metadata_path, metadata)
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    last_output_monotonic = started_monotonic
    last_output_at = started_wall
    last_line = ""
    output_lock = threading.Lock()

    emit(f"[monitor] run directory: {run_dir}")
    emit(f"[monitor] child pid: {process.pid}")

    def read_output() -> None:
        nonlocal last_output_monotonic, last_output_at, last_line
        assert process.stdout is not None

        with output_path.open("w", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                now_monotonic = time.monotonic()
                now_wall = utc_now()

                with output_lock:
                    last_output_monotonic = now_monotonic
                    last_output_at = now_wall
                    last_line = line.rstrip("\n")[-500:]
                    log.write(line)

                print(line, end="", flush=True)

    reader = threading.Thread(
        target=read_output,
        name="run-output-reader",
        daemon=True,
    )
    reader.start()
    previous_sample_at = started_monotonic
    last_status_at = started_monotonic - status_interval
    last_stale_warning_at: float | None = None
    interrupted = False
    final_sample: dict[str, Any] | None = None

    try:
        with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics:
            while process.poll() is None:
                now = time.monotonic()
                elapsed = now - started_monotonic

                with output_lock:
                    output_age = now - last_output_monotonic
                    current_last_output_at = last_output_at
                    current_last_line = last_line

                sample_gap = now - previous_sample_at
                snapshot = process_snapshot(process.pid) or {
                    "pid": process.pid,
                }
                disk = shutil.disk_usage(run_dir)
                sample: dict[str, Any] = {
                    "schema_version": RUN_LOG_VERSION,
                    "timestamp": utc_now(),
                    "elapsed_seconds": elapsed,
                    "sample_gap_seconds": sample_gap,
                    "monitor_hitch": sample_gap > sample_interval * 2.5,
                    "last_output_age_seconds": output_age,
                    "stale_output": output_age >= stale_seconds,
                    "disk_free_bytes": disk.free,
                    **snapshot,
                    **system_memory_snapshot(),
                }
                loads = os.getloadavg()
                sample.update(
                    {
                        "load_1m": loads[0],
                        "load_5m": loads[1],
                        "load_15m": loads[2],
                    }
                )
                metrics.write(json.dumps(sample, sort_keys=True) + "\n")
                final_sample = sample
                previous_sample_at = now

                status = {
                    "schema_version": RUN_LOG_VERSION,
                    "state": "running",
                    "pid": process.pid,
                    "updated_at": sample["timestamp"],
                    "elapsed_seconds": elapsed,
                    "last_output_at": current_last_output_at,
                    "last_output_age_seconds": output_age,
                    "last_line": current_last_line,
                    "latest_metrics": sample,
                }
                write_json_atomic(status_path, status)

                if now - last_status_at >= status_interval:
                    cpu = snapshot.get("cpu_percent")
                    rss = snapshot.get("rss_bytes")
                    cpu_text = "unknown" if cpu is None else f"{cpu:.1f}%"
                    rss_text = (
                        "unknown"
                        if rss is None
                        else f"{rss / (1024 ** 3):.2f} GiB"
                    )
                    emit(
                        "[monitor] "
                        f"elapsed {elapsed / 60:.1f}m; "
                        f"CPU {cpu_text}; RSS {rss_text}; "
                        f"last output {output_age:.0f}s ago"
                    )
                    last_status_at = now

                if (
                    output_age >= stale_seconds
                    and (
                        last_stale_warning_at is None
                        or now - last_stale_warning_at >= stale_seconds
                    )
                ):
                    emit(
                        "[monitor] WARNING: no child output for "
                        f"{output_age:.0f}s; process state "
                        f"{snapshot.get('state', 'unknown')}"
                    )
                    last_stale_warning_at = now

                if sample["monitor_hitch"]:
                    emit(
                        "[monitor] WARNING: telemetry sample gap was "
                        f"{sample_gap:.1f}s; the system may have slept "
                        "or hitched"
                    )

                deadline = time.monotonic() + sample_interval

                while process.poll() is None:
                    remaining = deadline - time.monotonic()

                    if remaining <= 0.0:
                        break

                    time.sleep(min(0.25, remaining))

    except KeyboardInterrupt:
        interrupted = True
        emit("[monitor] interrupt received; forwarding SIGINT to child")
        os.killpg(process.pid, signal.SIGINT)

        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            emit("[monitor] child did not stop; forwarding SIGTERM")
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)

    exit_code = process.wait()
    reader.join(timeout=5.0)
    elapsed = time.monotonic() - started_monotonic
    state = (
        "interrupted"
        if interrupted
        else "completed" if exit_code == 0 else "failed"
    )

    with output_lock:
        final_last_output_at = last_output_at
        final_last_line = last_line

    final_status = {
        "schema_version": RUN_LOG_VERSION,
        "state": state,
        "pid": process.pid,
        "updated_at": utc_now(),
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "last_output_at": final_last_output_at,
        "last_line": final_last_line,
        "latest_metrics": final_sample,
    }
    write_json_atomic(status_path, final_status)
    emit(
        f"[monitor] {state}; exit {exit_code}; "
        f"elapsed {elapsed / 60:.1f}m"
    )
    emit(f"[monitor] logs: {run_dir}")
    return MonitoredRunResult(
        exit_code=exit_code,
        run_dir=run_dir,
        state=state,
    )
