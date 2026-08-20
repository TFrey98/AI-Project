import pytest

from story_model.progress import ProgressReporter, format_duration


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_format_duration_is_human_readable():
    assert format_duration(9.6) == "10s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3661) == "1h 01m 01s"
    assert format_duration(None) == "unknown"


def test_progress_is_throttled_but_always_reports_completion():
    clock = FakeClock()
    messages = []
    reporter = ProgressReporter(
        label="score val",
        total=100,
        unit="tokens",
        interval_seconds=10.0,
        emit=messages.append,
        clock=clock,
    )
    clock.now = 5.0
    reporter.update(50)
    assert messages == []

    reporter.update(100)
    assert len(messages) == 1
    assert "100/100 tokens (100.0%)" in messages[0]
    assert "ETA 0s" in messages[0]


def test_progress_rejects_invalid_settings():
    with pytest.raises(ValueError, match="total"):
        ProgressReporter("test", 0, "items")

    with pytest.raises(ValueError, match="interval"):
        ProgressReporter("test", 1, "items", interval_seconds=-1)
