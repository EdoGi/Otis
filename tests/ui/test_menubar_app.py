"""Behavioural tests for MenuBarApp itself (not just its pure helpers).

These construct the real rumps-backed app headlessly (menus and timers are
created but the event loop never runs), so they need macOS + rumps and are
skipped elsewhere — the rest of the suite stays platform-independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager


@pytest.fixture
def app(tmp_path: Path):
    cfg = Config(
        {
            "app": {
                "working_days": [0, 1, 2, 3, 4],
                "working_hours": {"start": "08:00", "end": "20:00"},
            },
            "transcription": {"model": "small", "language": None},
            "storage": {
                "audio_dir": str(tmp_path / "audio"),
                "transcript_dir": str(tmp_path / "transcripts"),
            },
            "detection": {"process_monitor": {"whitelisted_apps": ["zoom.us"]}},
            "web": {"host": "127.0.0.1", "port": 8765},
        }
    )
    detector = MeetingDetector(process_monitor=None, calendar_pollers=[])
    notifications: list[tuple[str, str, str]] = []
    bar = MenuBarApp(
        config=cfg,
        detector=detector,
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(
            backend=lambda t, s, m: notifications.append((t, s, m))
        ),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_notifications = notifications  # type: ignore[attr-defined]
    return bar


# ---------------------------------------------------------------------------
# Transcription progress in the title
# ---------------------------------------------------------------------------
def test_progress_shown_while_processing(app) -> None:
    app._set_state(UiState.PROCESSING)
    app.notify_transcription_progress(42)
    app._drain_main_queue(None)
    assert app._app.title == "42%"


def test_progress_ignored_outside_processing(app) -> None:
    app._set_state(UiState.IDLE)
    app.notify_transcription_progress(42)
    app._drain_main_queue(None)
    assert app._app.title == ""


def test_stale_progress_after_completion_does_not_resurrect_title(app) -> None:
    app._set_state(UiState.PROCESSING)
    app.notify_transcription_progress(99)
    app._set_state(UiState.IDLE)  # transcription finished first
    app._drain_main_queue(None)
    assert app._app.title == ""


# ---------------------------------------------------------------------------
# Working-hours editor + live config application
# ---------------------------------------------------------------------------
def test_working_hour_pick_persists_and_applies_live(app, tmp_path: Path) -> None:
    item = app._hour_items[("start", "09:00")]
    app._on_working_hour_picked(item)

    # Live config updated (no restart needed).
    assert app._config.get("app", "working_hours", "start") == "09:00"
    # Persisted to the user config file.
    assert "09:00" in (tmp_path / "config.yaml").read_text()
    # Menu reflects the choice.
    assert app._mi["settings_hours_root"].title == "Working Hours: 09:00 → 20:00"
    assert app._hour_items[("start", "09:00")].state
    assert not app._hour_items[("start", "08:00")].state


def test_day_toggle_updates_live_config(app) -> None:
    saturday_item = app._day_items[5]
    assert not saturday_item.state
    app._on_day_toggled(saturday_item)
    assert 5 in app._config.get("app", "working_days")


def test_working_hours_menu_shows_current_config(app) -> None:
    assert app._mi["settings_hours_root"].title == "Working Hours: 08:00 → 20:00"
    assert app._hour_items[("end", "20:00")].state


# ---------------------------------------------------------------------------
# Device-error event → notification, no state change
# ---------------------------------------------------------------------------
def test_device_error_event_notifies_without_stopping(app) -> None:
    app._set_state(UiState.RECORDING)
    app._cb_device_error("mic", RuntimeError("device unplugged"))
    app._drain_main_queue(None)
    titles = [t for t, _s, _m in app._test_notifications]
    assert any("Mic stream failed" in t for t in titles)
    assert app.snapshot.state == UiState.RECORDING  # recording untouched
