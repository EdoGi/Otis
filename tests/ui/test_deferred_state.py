"""Cover for audit finding B5 — the deferral bricked the app for up to 2 hours.

The wait for a live call to end ran *inside* PROCESSING. For as long as
``DEFER_MAX_SECONDS`` (7200 s), with nothing actually running, the app:

* hid Start — and the reason it was waiting is that another call was live, so
  Otis was unrecordable during exactly the meeting you would want it for;
* dropped detector approaching/detected events entirely;
* refused Generate Transcript as "Busy";
* warned on quit that "a transcription is running" and would be discarded,
  which was false, frightening the user into waiting out a job that had not
  begun.

DEFERRED is that wait, named. Nothing is running in it, so everything above
stays available, and "Transcribe Now" ends it on demand.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingContext, MeetingDetector, MeetingState
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager


@pytest.fixture()
def app(tmp_path: Path):
    cfg = Config({
        "app": {"working_days": [0, 1, 2, 3, 4, 5, 6],
                "working_hours": {"start": "00:00", "end": "23:59"}},
        "transcription": {"model": "small", "language": None,
                          "defer_while_in_call": True},
        "storage": {"audio_dir": str(tmp_path / "audio"),
                    "transcript_dir": str(tmp_path / "transcripts")},
        "detection": {"process_monitor": {"whitelisted_apps": ["zoom.us"]}},
        "web": {"host": "127.0.0.1", "port": 8765},
    })
    notes: list[tuple] = []
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: notes.append(a)),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_notes = notes  # type: ignore[attr-defined]
    return bar


# ---------------------------------------------------------------------------
# Everything PROCESSING took away, DEFERRED gives back
# ---------------------------------------------------------------------------
def test_start_stays_available_while_deferred(app) -> None:
    """The wait is triggered by a live call — the case Start is needed for."""
    app._set_state(UiState.DEFERRED)
    assert app._mi["start"].hidden is False


def test_start_is_still_hidden_while_processing(app) -> None:
    app._set_state(UiState.PROCESSING)
    assert app._mi["start"].hidden is True


def test_detection_events_are_not_dropped_while_deferred(app) -> None:
    app._set_state(UiState.DEFERRED)
    ctx = MeetingContext(state=MeetingState.DETECTED, app="zoom.us", title="Standup")

    app._handle_event("detected", ctx)

    assert app._snapshot.state == UiState.DETECTED
    assert any("Meeting detected" in str(n) for n in app._test_notes)


def test_detection_events_are_still_suppressed_while_processing(app) -> None:
    app._set_state(UiState.PROCESSING)
    ctx = MeetingContext(state=MeetingState.DETECTED, app="zoom.us", title="Standup")

    app._handle_event("detected", ctx)

    assert app._snapshot.state == UiState.PROCESSING


def test_generate_transcript_is_not_refused_while_deferred(app, monkeypatch) -> None:
    started: list = []
    monkeypatch.setattr(MenuBarApp, "_repair_then_transcribe",
                        lambda self, *a: started.append(a))
    app._set_state(UiState.DEFERRED)

    sender = type("S", (), {})()
    sender._otis_session_id = "sess-a"
    sender._otis_metadata_path = None
    sender._otis_mic_path = "/tmp/x_mic.wav"
    sender._otis_system_path = None
    app._on_generate_clicked(sender)

    assert started, "refused as Busy while nothing was running"
    assert not any("Busy" in str(n) for n in app._test_notes)


def test_generate_transcript_is_still_refused_while_processing(app) -> None:
    app._set_state(UiState.PROCESSING)
    sender = type("S", (), {})()
    sender._otis_session_id = "sess-a"
    sender._otis_metadata_path = None
    sender._otis_mic_path = "/tmp/x_mic.wav"
    sender._otis_system_path = None
    app._on_generate_clicked(sender)
    assert any("Busy" in str(n) for n in app._test_notes)


def test_the_schedule_tick_leaves_a_deferred_job_alone(app) -> None:
    """Falling out of working hours must not clear a pending transcription."""
    app._set_state(UiState.DEFERRED)
    app._on_schedule_tick(None)
    assert app._snapshot.state == UiState.DEFERRED


def test_deferred_says_what_it_is_doing(app) -> None:
    app._set_state(UiState.DEFERRED)
    assert app._snapshot.title_text() == "waiting"


# ---------------------------------------------------------------------------
# Transcribe Now
# ---------------------------------------------------------------------------
def test_transcribe_now_is_offered_only_while_deferred(app) -> None:
    app._set_state(UiState.DEFERRED)
    assert app._mi["transcribe_now"].hidden is False
    for state in (UiState.IDLE, UiState.RECORDING, UiState.PROCESSING):
        app._set_state(state)
        assert app._mi["transcribe_now"].hidden is True


def test_transcribe_now_ends_the_wait(app, monkeypatch) -> None:
    """The user would rather have the transcript than a cool laptop."""
    from src import pipeline as pipeline_mod

    polled = threading.Event()

    def never_quiet() -> bool:
        polled.set()          # the wait is definitely running now
        return True

    monkeypatch.setattr(pipeline_mod, "make_call_probe",
                        lambda *a, **k: never_quiet)
    # The real poll interval is DEFER_POLL_SECONDS; don't sit through it.
    real_wait = pipeline_mod.wait_for_call_to_end
    monkeypatch.setattr(
        pipeline_mod, "wait_for_call_to_end",
        lambda **kw: real_wait(**{**kw, "poll_seconds": 0.01, "max_seconds": 20.0}),
    )
    done = threading.Event()
    t = threading.Thread(
        target=lambda: (app._defer_while_in_call(), done.set()), daemon=True
    )
    t.start()

    # Set the override only once the wait has started: _defer_while_in_call
    # clears the flag on entry, so an early set would be swallowed.
    assert polled.wait(5.0), "the deferral never started"
    app._on_transcribe_now_clicked(None)

    assert done.wait(10.0), "the override did not end the wait"
    t.join(1.0)


def test_the_override_does_not_leak_into_the_next_transcription(app, monkeypatch) -> None:
    """A stale flag would silently disable deferral from then on."""
    from src import pipeline as pipeline_mod

    calls: list[int] = []

    def probe():
        calls.append(1)
        return False

    monkeypatch.setattr(pipeline_mod, "make_call_probe", lambda *a, **k: probe)
    app._skip_deferral.set()
    app._defer_while_in_call()
    assert app._skip_deferral.is_set() is False, "cleared for the next job"


def test_the_wait_really_happens_in_deferred_not_processing(app, monkeypatch) -> None:
    """The transition itself, not just what each state looks like.

    Setting the state by hand in a test passes just as happily when the worker
    still defers inside PROCESSING, which is the bug.
    """
    from src import pipeline as pipeline_mod

    busy_for = [True, True, False]

    def probe() -> bool:
        return busy_for.pop(0) if busy_for else False

    monkeypatch.setattr(pipeline_mod, "make_call_probe", lambda *a, **k: probe)
    real_wait = pipeline_mod.wait_for_call_to_end
    monkeypatch.setattr(
        pipeline_mod, "wait_for_call_to_end",
        lambda **kw: real_wait(**{**kw, "poll_seconds": 0.01, "max_seconds": 5.0}),
    )
    monkeypatch.setattr(MenuBarApp, "_transcribe", lambda self, meta: None,
                        raising=False)
    app._transcribe = lambda _meta: None

    app._run_transcription({"session_id": "sess-a", "mic_wav": "x.wav",
                            "_meeting": {"title": None}, "_language": None})

    states = []
    while not app._events.empty():
        kind, payload = app._events.get_nowait()
        if kind == "force_state":
            states.append(payload)

    assert UiState.DEFERRED in states, "the wait ran without its own state"
    assert states.index(UiState.DEFERRED) < states.index(UiState.PROCESSING), (
        "PROCESSING must not be entered until Whisper actually starts"
    )
