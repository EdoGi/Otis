"""Cover for audit finding E1 — the App Whitelist was a one-way door.

The submenu was built *from the whitelist itself*, with every entry ticked.
So unticking Slack removed it from the config, and on the next launch it was
no longer in the list — unreachable except by hand-editing YAML. There was no
way to add an app either, and the change needed a restart regardless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.detection.process_monitor import ProcessMonitor
from src.ui.menubar import (
    KNOWN_MEETING_APPS,
    MAX_RUNNING_CANDIDATES,
    MenuBarApp,
    whitelist_candidates,
)
from src.ui.notifications import NotificationManager, NotificationType


# ---------------------------------------------------------------------------
# What the editor offers
# ---------------------------------------------------------------------------
def test_an_unticked_app_is_still_offered() -> None:
    """The bug, stated directly: turning Slack off must not hide Slack."""
    assert "Slack" in whitelist_candidates([])


def test_enabled_apps_come_first_and_are_never_dropped() -> None:
    out = whitelist_candidates(["My Custom Conferencer", "zoom.us"])
    assert out[0] == "My Custom Conferencer"
    assert "zoom.us" in out


def test_running_apps_can_be_added_without_leaving_the_menu() -> None:
    out = whitelist_candidates([], running=["Around", "Brave Browser"])
    assert "Around" in out and "Brave Browser" in out


def test_duplicates_are_collapsed_case_insensitively() -> None:
    out = whitelist_candidates(["Slack"], running=["slack", "SLACK"])
    assert [n for n in out if n.lower() == "slack"] == ["Slack"]


def test_the_running_list_cannot_swamp_the_menu() -> None:
    """93 apps run on this Mac; a menu is not a process table."""
    running = [f"App {i:03d}" for i in range(200)]
    out = whitelist_candidates([], running=running)
    assert len(out) <= len(KNOWN_MEETING_APPS) + MAX_RUNNING_CANDIDATES


def test_blank_names_are_ignored() -> None:
    assert whitelist_candidates(["", "  "], running=[""]) == list(KNOWN_MEETING_APPS)


# ---------------------------------------------------------------------------
# The menu reflects state, rather than assuming it
# ---------------------------------------------------------------------------
@pytest.fixture()
def app(tmp_path: Path):
    monitor = ProcessMonitor(whitelisted_apps=["zoom.us"], process_iter=lambda: [])
    cfg = Config({
        "app": {"working_days": [0], "working_hours": {"start": "00:00",
                                                       "end": "23:59"}},
        "transcription": {"model": "small"},
        "storage": {"audio_dir": str(tmp_path / "a"),
                    "transcript_dir": str(tmp_path / "t")},
        "detection": {"process_monitor": {"whitelisted_apps": ["zoom.us"]}},
    })
    notes: list[tuple] = []
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=monitor, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: notes.append(a)),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_notes = notes  # type: ignore[attr-defined]
    bar._test_monitor = monitor  # type: ignore[attr-defined]
    return bar


def test_only_the_enabled_apps_are_ticked(app) -> None:
    items = app._whitelist_items
    assert bool(items["zoom.us"].state) is True
    assert bool(items["Slack"].state) is False, "everything used to be ticked"


def test_unticking_then_reticking_works_in_one_session(app) -> None:
    item = app._whitelist_items["zoom.us"]

    app._on_whitelist_toggled(item)
    assert bool(item.state) is False
    assert app._config.get("detection", "process_monitor",
                           "whitelisted_apps") == []

    app._on_whitelist_toggled(item)
    assert bool(item.state) is True
    assert app._config.get("detection", "process_monitor",
                           "whitelisted_apps") == ["zoom.us"]


def test_a_change_applies_without_a_restart(app) -> None:
    app._on_whitelist_toggled(app._whitelist_items["Slack"])
    assert "slack" in app._test_monitor.whitelist


def test_saving_is_not_reported_as_an_error(app, monkeypatch) -> None:
    """It arrived as NotificationType.ERROR — a success dressed as a failure."""
    kinds: list[NotificationType] = []
    real = app._notifications.notify
    monkeypatch.setattr(
        app._notifications, "notify",
        lambda kind, *a, **kw: (kinds.append(kind), real(kind, *a, **kw))[1],
    )

    app._on_whitelist_toggled(app._whitelist_items["Slack"])

    assert NotificationType.SETTING_SAVED in kinds
    assert NotificationType.ERROR not in kinds


def test_a_detector_without_a_monitor_still_saves(tmp_path: Path) -> None:
    """Calendar-only setups have no ProcessMonitor to update."""
    cfg = Config({"storage": {"audio_dir": str(tmp_path / "a"),
                              "transcript_dir": str(tmp_path / "t")},
                  "detection": {"process_monitor": {"whitelisted_apps": []}}})
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._on_whitelist_toggled(bar._whitelist_items["Slack"])
    assert bar._config.get("detection", "process_monitor",
                           "whitelisted_apps") == ["Slack"]


# ---------------------------------------------------------------------------
# The monitor half
# ---------------------------------------------------------------------------
def test_the_monitor_watches_the_new_list_immediately() -> None:
    procs = [{"pid": 1, "name": "Around", "bundle_id": "co.around.app"}]
    monitor = ProcessMonitor(whitelisted_apps=["zoom.us"],
                             process_iter=lambda: procs)
    assert monitor.poll_once() == set()

    monitor.set_whitelist(["Around"])

    assert monitor.poll_once() == {"around"}


def test_removing_an_app_stops_matching_it() -> None:
    procs = [{"pid": 1, "name": "Around", "bundle_id": "co.around.app"}]
    monitor = ProcessMonitor(whitelisted_apps=["Around"],
                             process_iter=lambda: procs)
    assert monitor.poll_once() == {"around"}
    monitor.set_whitelist([])
    assert monitor.poll_once() == set()
