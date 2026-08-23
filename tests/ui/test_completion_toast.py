"""Cover for audit finding E4 — the completion toast always said "Recording".

``_meeting_title_for`` read ``metadata["title"]``. Recorder metadata has no
such key — verified against every metadata.json on this machine, which carry
session_id, mic_wav, pauses, anchors and nothing resembling a title. The
meeting title lives at ``metadata["_meeting"]["title"]``, and an
auto-suggested one only ever exists on the saved transcript.

So the one notification the user actually waits for named the meeting
"Recording", every time, for every meeting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.storage.transcript_store import TranscriptStore
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager


@pytest.fixture()
def app(tmp_path: Path):
    store = TranscriptStore(tmp_path / "transcripts")
    cfg = Config({"storage": {"audio_dir": str(tmp_path / "audio"),
                              "transcript_dir": str(store.root)}})
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        transcript_store=store,
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_store = store  # type: ignore[attr-defined]
    return bar


def test_real_recorder_metadata_no_longer_yields_recording(app) -> None:
    """The exact shape recorder.stop() writes — no `title` anywhere."""
    metadata = {
        "session_id": "sess-a", "mic_wav": "a_mic.wav", "sample_rate": 16000,
        "start_wall_clock": "2026-08-07T10:01:48+00:00",
        "_meeting": {"title": "Sprint Planning", "app": "zoom.us"},
    }
    assert app._meeting_title_for(metadata) == "Sprint Planning"


def test_the_saved_transcript_title_wins(app) -> None:
    """It is what Recent and the web UI show, and the only place an
    auto-suggested title exists."""
    app._test_store.save(
        {"id": "sess-a", "title": "Onboarding with Acme", "date": "2026-08-07",
         "start_time": "12:01"}, "# Onboarding\n\nbody")
    metadata = {"session_id": "sess-a", "_meeting": {"title": "Ad-hoc Recording"}}

    assert app._meeting_title_for(metadata) == "Onboarding with Acme"


def test_the_calendar_title_is_used_when_no_transcript_is_found(app) -> None:
    metadata = {"session_id": "missing", "_meeting": {"title": "Standup"}}
    assert app._meeting_title_for(metadata) == "Standup"


def test_a_nameless_recording_is_named_by_when_it_happened(app) -> None:
    """Better than "Recording": it identifies which one."""
    metadata = {"session_id": "sess-x", "_meeting": {"title": None},
                "start_wall_clock": "2026-08-07T10:01:48+00:00"}
    title = app._meeting_title_for(metadata)
    assert title.startswith("Recording — 2026-08-07")


def test_a_blank_calendar_title_does_not_win_over_the_timestamp(app) -> None:
    metadata = {"session_id": "sess-x", "_meeting": {"title": "   "},
                "start_wall_clock": "2026-08-07T10:01:48+00:00"}
    assert "2026-08-07" in app._meeting_title_for(metadata)


def test_with_nothing_at_all_it_falls_back_gracefully(app) -> None:
    assert app._meeting_title_for({}) == "Recording"


def test_an_unreadable_store_does_not_break_the_toast(app, monkeypatch) -> None:
    def boom(_sid):
        raise OSError("transcripts unreadable")

    monkeypatch.setattr(app._store, "get_transcript", boom)
    metadata = {"session_id": "sess-a", "_meeting": {"title": "Standup"}}
    assert app._meeting_title_for(metadata) == "Standup"


# ---------------------------------------------------------------------------
# --log-level scoping
# ---------------------------------------------------------------------------
def test_log_level_applies_to_otis_not_to_every_library() -> None:
    """DEBUG used to turn on urllib3 and googleapiclient too, and the 1 MB
    rotation then threw away the Otis lines the flag was set to read."""
    import logging

    from src.main import _configure_logging

    try:
        _configure_logging("DEBUG")
        assert logging.getLogger("src").level == logging.DEBUG
        assert logging.getLogger("src.detection.calendar_poller").isEnabledFor(
            logging.DEBUG
        )
        assert not logging.getLogger("urllib3").isEnabledFor(logging.DEBUG)
        assert not logging.getLogger("googleapiclient").isEnabledFor(logging.INFO)
    finally:
        _configure_logging("INFO")
        logging.getLogger("src").setLevel(logging.INFO)
