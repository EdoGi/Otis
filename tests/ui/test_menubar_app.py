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
            # defer_while_in_call off: tests must never block on the real mic.
            "transcription": {"model": "small", "language": None,
                              "defer_while_in_call": False},
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


# ---------------------------------------------------------------------------
# Defer-while-in-call wiring
# ---------------------------------------------------------------------------
def test_transcription_defers_until_call_ends(app, monkeypatch) -> None:
    import src.pipeline as pipeline_mod

    captured: dict = {}

    def fake_wait(*, is_busy, on_first_wait=None, should_abort=None, **_kw):
        captured["is_busy"] = is_busy
        captured["on_first_wait"] = on_first_wait
        return 0.0

    monkeypatch.setattr(pipeline_mod, "wait_for_call_to_end", fake_wait)
    monkeypatch.setattr(pipeline_mod, "make_call_probe", lambda _cfg: lambda: False)

    # The fixture disables deferral so other tests never touch the real
    # mic; this test exercises it with the wait fully faked.
    app._config.apply_overrides({"transcription": {"defer_while_in_call": True}})

    app._defer_while_in_call()
    assert "is_busy" in captured, "deferral gate was never consulted"

    # An active recorder makes the gate report busy regardless of the mic.
    app._recorder = object()
    assert captured["is_busy"]() is True
    app._recorder = None
    assert captured["is_busy"]() is False

    # The first-wait callback sends a user-facing notification.
    captured["on_first_wait"]()
    titles = [t for t, _s, _m in app._test_notifications]
    assert any("deferred" in t.lower() for t in titles)


def test_deferral_can_be_disabled_in_config(app, monkeypatch) -> None:
    import src.pipeline as pipeline_mod

    called = {"n": 0}
    monkeypatch.setattr(
        pipeline_mod, "wait_for_call_to_end",
        lambda **_kw: called.__setitem__("n", called["n"] + 1) or 0.0,
    )
    app._config.apply_overrides({"transcription": {"defer_while_in_call": False}})
    app._defer_while_in_call()
    assert called["n"] == 0
