"""Tests for src/ui/notifications.py."""

from __future__ import annotations

from src.ui.notifications import (
    NotificationManager,
    NotificationType,
    format_process_disappeared,
    format_transcription_complete,
)


def _recording_backend(records: list[tuple[str, str, str]]):
    """Backend that captures every (title, subtitle, message) sent."""

    def _b(title: str, subtitle: str, message: str) -> None:
        records.append((title, subtitle, message))

    return _b


# ----------------------------------------------------------- delivery + content
def test_notify_calls_backend_with_title_and_message() -> None:
    sent: list[tuple[str, str, str]] = []
    nm = NotificationManager(backend=_recording_backend(sent))
    ok = nm.notify(NotificationType.MEETING_DETECTED, "Meeting", "Zoom")
    assert ok is True
    assert sent == [("Meeting", "", "Zoom")]


def test_notify_with_subtitle() -> None:
    sent: list[tuple[str, str, str]] = []
    nm = NotificationManager(backend=_recording_backend(sent))
    nm.notify(NotificationType.MEETING_APPROACHING, "title", "msg", subtitle="14:00")
    assert sent == [("title", "14:00", "msg")]


# ---------------------------------------------------------------- rate limiting
def test_rate_limits_same_type() -> None:
    sent: list = []
    fake_clock = {"t": 0.0}
    nm = NotificationManager(
        backend=_recording_backend(sent),
        rate_limit_seconds=30.0,
        clock=lambda: fake_clock["t"],
    )
    assert nm.notify(NotificationType.MEETING_DETECTED, "a", "b") is True
    assert nm.notify(NotificationType.MEETING_DETECTED, "c", "d") is False
    assert len(sent) == 1


def test_rate_limit_releases_after_window() -> None:
    sent: list = []
    fake_clock = {"t": 0.0}
    nm = NotificationManager(
        backend=_recording_backend(sent),
        rate_limit_seconds=30.0,
        clock=lambda: fake_clock["t"],
    )
    nm.notify(NotificationType.MEETING_DETECTED, "a", "b")
    fake_clock["t"] = 31.0
    nm.notify(NotificationType.MEETING_DETECTED, "c", "d")
    assert len(sent) == 2


def test_rate_limit_is_per_type() -> None:
    """Different notification types don't share a rate-limit window."""
    sent: list = []
    fake_clock = {"t": 0.0}
    nm = NotificationManager(
        backend=_recording_backend(sent),
        rate_limit_seconds=30.0,
        clock=lambda: fake_clock["t"],
    )
    nm.notify(NotificationType.MEETING_APPROACHING, "1", "")
    nm.notify(NotificationType.MEETING_DETECTED, "2", "")
    nm.notify(NotificationType.RECORDING_STARTED, "3", "")
    assert len(sent) == 3


def test_force_bypasses_rate_limit() -> None:
    sent: list = []
    fake_clock = {"t": 0.0}
    nm = NotificationManager(
        backend=_recording_backend(sent),
        rate_limit_seconds=30.0,
        clock=lambda: fake_clock["t"],
    )
    nm.notify(NotificationType.ERROR, "first", "")
    assert nm.notify(NotificationType.ERROR, "second", "", force=True) is True
    assert [s[0] for s in sent] == ["first", "second"]


def test_reset_clears_history() -> None:
    sent: list = []
    fake_clock = {"t": 0.0}
    nm = NotificationManager(
        backend=_recording_backend(sent),
        rate_limit_seconds=30.0,
        clock=lambda: fake_clock["t"],
    )
    nm.notify(NotificationType.MEETING_DETECTED, "a", "b")
    nm.reset()
    assert nm.notify(NotificationType.MEETING_DETECTED, "c", "d") is True


def test_backend_failure_returns_false_does_not_raise() -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("notification center disabled")

    nm = NotificationManager(backend=boom)
    assert nm.notify(NotificationType.ERROR, "title", "body") is False


# ----------------------------------------------------------------- formatters
def test_format_transcription_complete() -> None:
    title, body = format_transcription_complete("Standup", duration_minutes=12.4)
    assert title == "Transcript ready"
    assert body == "Standup (12 min)"


def test_format_transcription_complete_rounds_zero_for_short() -> None:
    _t, body = format_transcription_complete("ad-hoc", duration_minutes=0.3)
    assert body == "ad-hoc (0 min)"


def test_format_process_disappeared() -> None:
    title, body = format_process_disappeared("zoom.us")
    assert "exited" in title.lower()
    assert "zoom.us" in body
