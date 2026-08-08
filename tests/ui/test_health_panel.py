"""Tests for the menu-bar Status panel.

This is the surface whose absence let three failures run silently for months.
The contract that matters: a component transition raises exactly one
notification, repeated identical reports raise none, and the repair actions
only appear when they would actually do something.

Like test_menubar_app.py these build the real rumps app headlessly, so they
need macOS + rumps and skip elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import CALENDAR, SYSTEM_AUDIO, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


@pytest.fixture
def app(tmp_path: Path, registry: HealthRegistry):
    cfg = Config(
        {
            "app": {
                "working_days": [0, 1, 2, 3, 4],
                "working_hours": {"start": "08:00", "end": "20:00"},
            },
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
    sent: list[tuple[str, str, str]] = []
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(
            backend=lambda t, s, m: sent.append((t, s, m))
        ),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=registry,
    )
    bar._sent = sent  # type: ignore[attr-defined]
    return bar


def drain(app) -> None:
    """Run the queue drain the 100 ms rumps timer would run."""
    app._drain_main_queue(None)


def titles_for(app, needle: str) -> list[tuple[str, str, str]]:
    return [n for n in app._sent if needle in n[0]]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_panel_lists_every_reported_component(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                    detail="routed via Otis")
    registry.report(CALENDAR, label="Calendar", state=HealthState.FAIL,
                    detail="disconnected")
    drain(app)

    entries = list(app._mi["health_root"].keys())
    assert any("System audio" in t for t in entries)
    assert any("Calendar" in t for t in entries)


def test_panel_title_flags_degradation(app, registry) -> None:
    assert app._mi["health_root"].title == "Status"
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed")
    drain(app)
    assert app._mi["health_root"].title != "Status"


def test_empty_registry_shows_a_placeholder(app) -> None:
    entries = list(app._mi["health_root"].keys())
    assert len(entries) == 1
    assert "no checks" in entries[0].lower()


# ---------------------------------------------------------------------------
# Notifications — one per transition, never per report
# ---------------------------------------------------------------------------
def test_a_failure_notifies_once_with_the_fix(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="captured silence", hint="Set output to Otis.")
    drain(app)

    sent = titles_for(app, "System audio")
    assert len(sent) == 1
    assert "captured silence" in sent[0][2]
    assert "Set output to Otis." in sent[0][2]


def test_repeating_the_same_state_never_re_notifies(app, registry) -> None:
    """A 60s poll loop reports health every cycle; that must stay silent."""
    for _ in range(20):
        registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                        detail="captured silence")
        drain(app)
    assert len(titles_for(app, "System audio")) == 1


def test_detail_updates_do_not_re_notify(app, registry) -> None:
    """Live level meters update the detail line on every duration tick."""
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="no signal for 60s")
    drain(app)
    for i in range(10):
        registry.touch(SYSTEM_AUDIO, detail=f"meter {i}")
        drain(app)
    assert len(titles_for(app, "System audio")) == 1


def test_recovery_is_announced(app, registry) -> None:
    registry.report(CALENDAR, label="Calendar", state=HealthState.FAIL, detail="dead")
    drain(app)
    registry.report(CALENDAR, label="Calendar", state=HealthState.OK,
                    detail="2 calendars")
    drain(app)

    assert any("recovered" in n[0].lower() for n in titles_for(app, "Calendar"))


# ---------------------------------------------------------------------------
# Repair actions appear only when actionable
# ---------------------------------------------------------------------------
def test_reconnect_calendar_hidden_while_healthy(app, registry) -> None:
    registry.report(CALENDAR, label="Calendar", state=HealthState.OK)
    drain(app)
    assert app._mi["reconnect_calendar"].hidden is True


def test_reconnect_calendar_appears_when_auth_dies(app, registry) -> None:
    registry.report(CALENDAR, label="Calendar", state=HealthState.FAIL,
                    detail="disconnected")
    drain(app)
    assert app._mi["reconnect_calendar"].hidden is False


def test_fix_audio_appears_when_routing_breaks(app, registry) -> None:
    assert app._mi["fix_audio"].hidden is True
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed")
    drain(app)
    assert app._mi["fix_audio"].hidden is False


# ---------------------------------------------------------------------------
# The ⚠ badge
# ---------------------------------------------------------------------------
def test_badge_shows_while_idle_and_degraded(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed")
    drain(app)
    assert app._app.title == "⚠"


def test_badge_clears_on_recovery(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL)
    drain(app)
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK)
    drain(app)
    assert app._app.title == ""


def test_badge_never_overwrites_the_recording_timer(app, registry) -> None:
    """During a call the MM:SS timer matters more than the badge."""
    app._set_state(UiState.RECORDING)
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL)
    drain(app)
    assert app._app.title != "⚠"
