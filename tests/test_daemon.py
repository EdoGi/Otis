"""Tests for src/daemon.py.

The daemon glues together pieces that have unit tests of their own; here we
verify the gluing only — that signals from a fake detector drive the recorder
through the right lifecycle, and that shutdown is clean.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.config import Config
from src.daemon import OtisDaemon
from src.detection.calendar_poller import CalendarEvent
from src.detection.detector import MeetingContext, MeetingDetector, MeetingState


# ============================================================================
# Fakes
# ============================================================================
class _FakeRecorder:
    """Minimal stand-in for DualStreamRecorder."""

    def __init__(self, audio_dir: str = "/tmp/otis-test") -> None:
        self.audio_dir = audio_dir
        self.session_id: str | None = None
        self.started = False
        self.stopped = False

    def start(self, session_id: str | None = None) -> str:
        self.started = True
        self.session_id = session_id or "fake-session"
        return self.session_id

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {
            "session_id": self.session_id,
            "mic_wav": f"{self.session_id}_mic.wav",
            "system_wav": f"{self.session_id}_system.wav",
        }


class _FakeProcessMonitor:
    def __init__(self) -> None:
        self.detected_cbs: list = []
        self.ended_cbs: list = []
        self.started = False
        self.stopped = False

    def on_meeting_detected(self, cb): self.detected_cbs.append(cb)
    def on_meeting_ended(self, cb): self.ended_cbs.append(cb)
    def start(self): self.started = True
    def stop(self): self.stopped = True

    def fire_detected(self, name: str) -> None:
        for cb in self.detected_cbs:
            cb(name)

    def fire_ended(self, name: str) -> None:
        for cb in self.ended_cbs:
            cb(name)


# ============================================================================
# Fixtures
# ============================================================================
def _make_config() -> Config:
    return Config({
        "app": {"name": "Otis"},
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "mic_device": None,
            "system_audio_device": "BlackHole 2ch",
        },
        "storage": {
            "audio_dir": "/tmp/otis-test-daemon",
            "transcript_dir": "/tmp/otis-test-daemon-transcripts",
        },
        "detection": {
            "process_monitor": {"enabled": False},  # we wire our own
            "calendar": {"enabled": False},
        },
    })


def _make_daemon(detector: MeetingDetector) -> tuple[OtisDaemon, list[str], _FakeRecorder]:
    """Build a daemon with an injected detector + fake recorder, capturing notify()."""
    notifications: list[str] = []
    fake_recorder = _FakeRecorder()
    daemon = OtisDaemon(
        config=_make_config(),
        notify=notifications.append,
        recorder_factory=lambda _cfg: fake_recorder,
        detector=detector,
    )
    return daemon, notifications, fake_recorder


# ============================================================================
# Tests
# ============================================================================
def test_meeting_detected_starts_recorder() -> None:
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)

    pm.fire_detected("zoom.us")
    assert rec.started, "recorder should auto-start on DETECTED"
    assert detector.get_state() == MeetingState.RECORDING
    assert any("Detected" in n for n in notes)
    assert any("Recording session" in n for n in notes)


def test_user_stop_via_meeting_ended_stops_recorder() -> None:
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)

    pm.fire_detected("zoom.us")
    detector.user_stopped_recording()
    assert rec.stopped, "user_stopped_recording must stop the recorder"
    assert any("Saved" in n for n in notes)


def test_process_disappearing_keeps_recording_but_warns() -> None:
    """The advisory path: recording continues, the daemon warns the user."""
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)

    pm.fire_detected("zoom.us")
    assert rec.started and not rec.stopped

    pm.fire_ended("zoom.us")
    # Recording continues — daemon should NOT have stopped it.
    assert not rec.stopped, "recording must continue after process exit"
    assert detector.get_state() == MeetingState.RECORDING
    # And we surfaced a warning to the user.
    assert any("exited" in n.lower() and "continues" in n.lower() for n in notes)


def test_request_stop_unblocks_run_and_cleans_up() -> None:
    """Calling request_stop() makes run() return, stopping any active recorder."""
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)

    # Start a recording, then ask the daemon to stop in a worker thread.
    pm.fire_detected("zoom.us")
    assert rec.started

    t = threading.Thread(target=daemon.run, daemon=True)
    t.start()
    # Give run() a moment to enter its wait loop.
    threading.Event().wait(0.05)
    daemon.request_stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "daemon.run should return after request_stop"
    assert rec.stopped, "shutdown must stop the recorder"


def test_double_detection_does_not_start_two_recorders() -> None:
    """Idempotent: a second DETECTED while already RECORDING is a no-op."""
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, _notes, rec = _make_daemon(detector)

    pm.fire_detected("zoom.us")
    assert rec.started

    # Re-fire while still recording — must not create a second recorder.
    saved_session = rec.session_id
    pm.fire_detected("zoom.us")
    assert rec.session_id == saved_session
    assert daemon.recorder is rec  # same instance


def test_calendar_approaching_does_not_start_recording() -> None:
    """Only DETECTED triggers the recorder; APPROACHING is just a heads-up."""
    detector = MeetingDetector(
        process_monitor=None,
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)

    # Drive the detector directly through its calendar handler.
    ev = CalendarEvent(
        id="evt-1",
        title="Future meeting",
        start=datetime(2026, 5, 9, 12, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 9, 12, 31, tzinfo=timezone.utc),
        meeting_link="https://meet.google.com/abc",
    )
    detector._handle_calendar_upcoming(ev)  # type: ignore[attr-defined]

    assert not rec.started
    assert detector.get_state() == MeetingState.APPROACHING
    assert any("<2 min" in n or "<2min" in n for n in notes)


def test_recorder_factory_failure_is_reported_not_crashed() -> None:
    """If building the recorder raises, daemon notifies and stays alive."""
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )

    notes: list[str] = []

    def boom(_cfg):
        raise RuntimeError("simulated recorder failure")

    daemon = OtisDaemon(
        config=_make_config(),
        notify=notes.append,
        recorder_factory=boom,
        detector=detector,
    )
    pm.fire_detected("zoom.us")
    # No recorder, but the daemon didn't crash — we got an error message.
    assert daemon.recorder is None
    assert any("Could not start" in n for n in notes)
