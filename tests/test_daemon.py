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


# ============================================================================
# Transcription-on-stop (shared pipeline)
# ============================================================================
def _make_pipeline(tmp_path, processor):
    from src.pipeline import TranscriptionPipeline
    from src.storage.transcript_store import TranscriptStore
    from src.transcription.whisper_engine import WhisperEngine

    return TranscriptionPipeline(
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        store=TranscriptStore(tmp_path / "transcripts"),
        engine=WhisperEngine(model_name="tiny", transcribe_fn=lambda *_a, **_k: {}),
        processor=processor,
    )


class _RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process(self, session, *, meeting=None, language=None, on_progress=None):
        self.calls.append({"session": session, "meeting": meeting, "language": language})


def _make_daemon_with_pipeline(detector, tmp_path, processor):
    notifications: list[str] = []
    fake_recorder = _FakeRecorder()
    from src.pipeline import TranscriptionPipeline  # noqa: F401

    pipeline = _make_pipeline(tmp_path, processor)
    daemon = OtisDaemon(
        config=_make_config(),
        notify=notifications.append,
        recorder_factory=lambda _cfg: fake_recorder,
        detector=detector,
        transcription_pipeline=pipeline,
    )
    return daemon, notifications, fake_recorder, pipeline


def _join_transcription(daemon: OtisDaemon) -> None:
    thread = daemon._transcribe_thread
    if thread is not None:
        thread.join(timeout=5.0)


def test_meeting_end_triggers_transcription_with_meeting_context(tmp_path) -> None:
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    processor = _RecordingProcessor()
    daemon, notes, rec, _pipeline = _make_daemon_with_pipeline(detector, tmp_path, processor)

    pm.fire_detected("zoom.us")
    detector.user_stopped_recording()
    _join_transcription(daemon)

    assert rec.stopped
    assert len(processor.calls) == 1
    call = processor.calls[0]
    assert call["session"].session_id == "fake-session"
    assert call["meeting"].app == "zoom.us"
    assert any("Transcript saved" in n for n in notes)
    # Detector cycled through PROCESSING back to IDLE.
    assert detector.get_state() == MeetingState.IDLE


def test_transcription_failure_saves_placeholder_and_daemon_survives(tmp_path) -> None:
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )

    class _Boom:
        def process(self, *_a, **_k):
            raise RuntimeError("model exploded")

    daemon, notes, _rec, pipeline = _make_daemon_with_pipeline(detector, tmp_path, _Boom())

    pm.fire_detected("zoom.us")
    detector.user_stopped_recording()
    _join_transcription(daemon)

    assert any("Transcription failed" in n for n in notes)
    placeholders = [
        fm for fm in pipeline.store.list_transcripts(tag="failed")
        if fm["id"] == "fake-session"
    ]
    assert len(placeholders) == 1
    assert detector.get_state() == MeetingState.IDLE  # state machine recovered


def test_no_pipeline_means_record_only(tmp_path) -> None:
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    daemon, notes, rec = _make_daemon(detector)
    pm.fire_detected("zoom.us")
    detector.user_stopped_recording()
    assert rec.stopped
    assert daemon._transcribe_thread is None  # no worker spawned


# ============================================================================
# Working-hours schedule
# ============================================================================
def _daemon_with_clock(now: datetime):
    pm = _FakeProcessMonitor()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[],
        clock=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )
    notes: list[str] = []
    clock = {"now": now}
    fake_recorder = _FakeRecorder()
    daemon = OtisDaemon(
        config=_make_config(),
        notify=notes.append,
        recorder_factory=lambda _cfg: fake_recorder,
        detector=detector,
        clock=lambda: clock["now"],
    )
    return daemon, detector, pm, notes, clock, fake_recorder


def test_apply_schedule_starts_detector_inside_window() -> None:
    monday_noon = datetime(2026, 6, 8, 12, 0)  # Monday, inside 08-20
    daemon, detector, pm, notes, _clock, _rec = _daemon_with_clock(monday_noon)
    daemon._apply_schedule()
    assert pm.started
    assert any("detection active" in n.lower() for n in notes)


def test_apply_schedule_keeps_detector_off_outside_window() -> None:
    sunday = datetime(2026, 6, 7, 12, 0)  # Sunday — not a working day
    daemon, _detector, pm, _notes, clock, _rec = _daemon_with_clock(sunday)
    daemon._apply_schedule()
    assert not pm.started

    # Window opens Monday morning → detector comes up on the next tick.
    clock["now"] = datetime(2026, 6, 8, 9, 0)
    daemon._apply_schedule()
    assert pm.started


def test_window_closing_stops_detector_but_never_mid_recording() -> None:
    monday_noon = datetime(2026, 6, 8, 12, 0)
    daemon, detector, pm, notes, clock, rec = _daemon_with_clock(monday_noon)
    daemon._apply_schedule()
    assert pm.started

    # A meeting starts; then the window closes while recording.
    pm.fire_detected("zoom.us")
    assert rec.started
    clock["now"] = datetime(2026, 6, 8, 22, 0)  # after hours
    daemon._apply_schedule()
    assert not pm.stopped, "active recording must never be interrupted"

    # Recording ends → next tick pauses detection.
    detector.user_stopped_recording()
    daemon._apply_schedule()
    assert pm.stopped
    assert any("detection paused" in n.lower() for n in notes)
