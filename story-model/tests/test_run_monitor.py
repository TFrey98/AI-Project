import json
import sys

from story_model.run_monitor import (
    parse_ps_snapshot,
    process_snapshot,
    run_monitored,
    slugify_run_name,
)


def test_parse_ps_snapshot_normalizes_rss_to_bytes():
    snapshot = parse_ps_snapshot(
        "123 1 R+ 00:17 98.5 4.2 1048576\n"
    )

    assert snapshot == {
        "pid": 123,
        "ppid": 1,
        "state": "R+",
        "process_elapsed": "00:17",
        "cpu_percent": 98.5,
        "memory_percent": 4.2,
        "rss_bytes": 1_073_741_824,
    }


def test_parse_ps_snapshot_rejects_unexpected_output():
    assert parse_ps_snapshot("") is None
    assert parse_ps_snapshot("not enough columns") is None


def test_current_process_snapshot_has_identity_and_memory():
    import os

    snapshot = process_snapshot(os.getpid())

    assert snapshot is not None
    assert snapshot["pid"] == os.getpid()
    assert snapshot["rss_bytes"] > 0


def test_run_name_is_filesystem_safe():
    assert slugify_run_name("Foundation v2: val/new") == (
        "Foundation-v2-val-new"
    )


def test_monitored_run_persists_output_metrics_and_status(tmp_path):
    messages = []
    result = run_monitored(
        command=[
            sys.executable,
            "-c",
            (
                "import time; "
                "print('stage: start', flush=True); "
                "time.sleep(0.15); "
                "print('stage: done', flush=True)"
            ),
        ],
        name="monitor test",
        output_root=tmp_path,
        sample_interval=0.03,
        status_interval=0.05,
        stale_seconds=1.0,
        emit=messages.append,
    )

    assert result.exit_code == 0
    assert result.state == "completed"
    assert "stage: start" in (
        result.run_dir / "output.log"
    ).read_text(encoding="utf-8")
    assert (result.run_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).strip()
    status = json.loads(
        (result.run_dir / "status.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (result.run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    assert metadata["command"][0] == sys.executable
    assert any("run directory" in message for message in messages)
