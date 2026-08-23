"""Cover for audit C5's UI half — start/stop ran on the rumps main thread.

Measured on this Mac with the real audio stack: ``recorder.start()`` takes
5344 ms the first time in a process (cold CoreAudio device enumeration) and
196-266 ms after that; ``recorder.stop()`` takes ~250 ms and can reach ~12 s
if a writer is wedged. A responsive main thread has roughly 50 ms.

So the click now only sets a transient state and hands the work to a worker;
the result comes back through the same event queue everything else uses.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rumps")

from src.audio.recorder import RecorderState
from src.config import Config
from src.detection.detector import MeetingDetector
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager


class FakeRecorder:
    """Stands in for DualStreamRecorder, with a controllable start/stop."""

    def __init__(self, *, start_gate=None, stop_gate=None,
                 start_exc=None, stop_exc=None) -> None:
        self.state = RecorderState.IDLE
        self.session_id = "sess-abcdef123456"
        self.on_device_error = None
        self._start_gate = start_gate
        self._stop_gate = stop_gate
        self._start_exc = start_exc
        self._stop_exc = stop_exc
        self.stopped = threading.Event()

    def start(self) -> str:
        if self._start_gate is not None:
            self._start_gate.wait(5.0)
        if self._start_exc is not None:
            raise self._start_exc
        self.state = RecorderState.RECORDING
        return self.session_id

    def stop(self) -> dict[str, Any]:
        if self._stop_gate is not None:
            self._stop_gate.wait(5.0)
        self.stopped.set()
        if self._stop_exc is not None:
            raise self._stop_exc
        self.state = RecorderState.STOPPED
        return {"session_id": self.session_id, "mic_wav": "m.wav"}


@pytest.fixture()
def app(tmp_path: Path):
    cfg = Config({
        "app": {"working_days": [0, 1, 2, 3, 4],
                "working_hours": {"start": "00:00", "end": "23:59"}},
        "transcription": {"model": "small", "language": None,
                          "defer_while_in_call": False},
        "storage": {"audio_dir": str(tmp_path / "audio"),
                    "transcript_dir": str(tmp_path / "transcripts")},
        "detection": {"process_monitor": {"whitelisted_apps": ["zoom.us"]}},
        "web": {"host": "127.0.0.1", "port": 8765},
    })
    notes: list[tuple] = []
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: FakeRecorder(),
        notifications=NotificationManager(backend=lambda *a: notes.append(a)),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_notes = notes  # type: ignore[attr-defined]
    return bar


def _settle(app, name: str, timeout: float = 5.0) -> None:
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout)
    app._drain_main_queue(None)


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
def test_start_does_not_block_the_caller(app) -> None:
    gate = threading.Event()
    app._recorder_factory = lambda _c: FakeRecorder(start_gate=gate)

    t0 = time.perf_counter()
    app._do_start_recording()
    elapsed = time.perf_counter() - t0

    # The fake holds start() for up to 5 s. Returning at all means we did not
    # wait for it — asserting the state alone would pass either way, since a
    # synchronous version also leaves STARTING on screen while it blocks.
    assert elapsed < 0.5, f"start blocked the caller for {elapsed*1000:.0f} ms"
    assert app._snapshot.state == UiState.STARTING
    gate.set()
    _settle(app, "otis-recorder-start")
    assert app._snapshot.state == UiState.RECORDING


def test_a_successful_start_lands_on_the_main_thread(app) -> None:
    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    assert app._snapshot.state == UiState.RECORDING
    assert app._recorder is not None
    assert app._starting is False
    assert any("Recording started" in str(n) for n in app._test_notes)


def test_a_second_click_while_starting_is_ignored(app) -> None:
    """Two recorders would fight over the same devices and the same files."""
    gate = threading.Event()
    built: list[FakeRecorder] = []

    def factory(_c):
        r = FakeRecorder(start_gate=gate)
        built.append(r)
        return r

    app._recorder_factory = factory
    app._do_start_recording()
    app._do_start_recording()
    app._do_start_recording()
    gate.set()
    _settle(app, "otis-recorder-start")

    assert len(built) == 1, f"{len(built)} recorders built"


def test_a_failed_start_returns_to_the_previous_state(app) -> None:
    """Not blindly IDLE — a meeting may still be detected."""
    app._set_state(UiState.DETECTED)
    app._recorder_factory = lambda _c: FakeRecorder(
        start_exc=RuntimeError("no microphone")
    )

    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    assert app._snapshot.state == UiState.DETECTED
    assert app._recorder is None
    assert app._starting is False
    assert any("no microphone" in str(n) for n in app._test_notes)


def test_start_can_be_retried_after_a_failure(app) -> None:
    app._recorder_factory = lambda _c: FakeRecorder(start_exc=RuntimeError("busy"))
    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    app._recorder_factory = lambda _c: FakeRecorder()
    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    assert app._snapshot.state == UiState.RECORDING


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------
def test_stop_does_not_block_the_caller(app, monkeypatch) -> None:
    # Pin the handoff: the real handler would run, fail fast against the empty
    # metadata, and queue its own force_state → IDLE into the same drain.
    monkeypatch.setattr(MenuBarApp, "_run_transcription", lambda self, meta: None)
    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    gate = threading.Event()
    app._recorder._stop_gate = gate

    t0 = time.perf_counter()
    app._do_stop_and_transcribe()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"stop blocked the caller for {elapsed*1000:.0f} ms"
    assert app._snapshot.state == UiState.SAVING
    gate.set()
    _settle(app, "otis-recorder-stop")
    assert app._snapshot.state == UiState.PROCESSING


def test_a_failed_stop_does_not_leave_the_ui_saving(app) -> None:
    app._recorder_factory = lambda _c: FakeRecorder(stop_exc=OSError("disk full"))
    app._do_start_recording()
    _settle(app, "otis-recorder-start")

    app._do_stop_and_transcribe()
    _settle(app, "otis-recorder-stop")

    assert app._snapshot.state == UiState.IDLE
    assert any("disk full" in str(n) for n in app._test_notes)


def test_stopping_hands_the_metadata_to_transcription(app, monkeypatch) -> None:
    handed: list[dict] = []
    monkeypatch.setattr(MenuBarApp, "_run_transcription",
                        lambda self, meta: handed.append(meta))
    app._do_start_recording()
    _settle(app, "otis-recorder-start")
    app._do_stop_and_transcribe()
    _settle(app, "otis-recorder-stop")
    _settle(app, "otis-transcription")

    assert len(handed) == 1
    assert handed[0]["session_id"] == "sess-abcdef123456"
    assert "_meeting" in handed[0] and "_language" in handed[0]


# ---------------------------------------------------------------------------
# The transient states must not be clickable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", [UiState.STARTING, UiState.SAVING])
def test_start_and_stop_are_both_hidden_while_in_flight(app, state) -> None:
    app._set_state(state)
    assert app._mi["start"].hidden is True
    assert app._mi["stop"].hidden is True


@pytest.mark.parametrize(
    "state,expected",
    [(UiState.STARTING, "starting…"), (UiState.SAVING, "saving…")],
)
def test_the_transient_states_say_what_is_happening(app, state, expected) -> None:
    app._set_state(state)
    assert app._snapshot.title_text() == expected


# ---------------------------------------------------------------------------
# Quit still flushes synchronously
# ---------------------------------------------------------------------------
def test_teardown_stops_the_recorder_synchronously(app) -> None:
    """On quit there is no event loop left to deliver a worker's result."""
    app._do_start_recording()
    _settle(app, "otis-recorder-start")
    recorder = app._recorder
    assert recorder is not None

    app._teardown()

    assert recorder.stopped.is_set()
    assert app._recorder is None
