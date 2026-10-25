"""Regression: launched outside working hours, the panel hid the calendar.

Calendar health was reported only from inside the poll loop, and the poll
loop starts only in working hours. So the Login Item at boot on a Sunday (or
before 08:00) showed no calendar rows and no 'Reconnect Calendar…' item —
although both of this user's Google tokens have been missing since
2026-09-17. A sibling of the e3f7d70 bug: the repair item could not appear
in the very state that needed it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.calendar_poller import GoogleCalendarPoller
from src.detection.detector import MeetingDetector
from src.health import get_health, reset_health
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_health()
    yield
    reset_health()


def _off_hours_app(tmp_path: Path, pollers: list[GoogleCalendarPoller]):
    import rumps

    class NoLoopApp(rumps.App):
        def run(self, **kwargs):
            return None

    return MenuBarApp(
        config=Config({"app": {"working_days": []}}),  # never a working day
        detector=MeetingDetector(process_monitor=None, calendar_pollers=pollers),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        audio_dir=tmp_path / "audio",  # run() must not scan the user's ~/Otis/audio
        health=get_health(),
        rumps_app_factory=NoLoopApp,
    )


def _poller(tmp_path: Path, label: str, *, token: bool) -> GoogleCalendarPoller:
    token_path = tmp_path / f"{label}_token.json"
    if token:
        token_path.write_text("{}", encoding="utf-8")
    return GoogleCalendarPoller(credentials_path=tmp_path / "credentials.json",
                                token_path=token_path, label=label)


def test_missing_tokens_show_up_at_an_off_hours_launch(tmp_path: Path) -> None:
    import rumps

    work = _poller(tmp_path, "work", token=False)
    app = _off_hours_app(tmp_path, [work])
    app.run()
    rumps.events.before_quit.unregister(app._on_before_quit)
    app._drain_main_queue(None)

    rep = get_health().get("calendar:work")
    assert rep is not None and "disconnected" in rep.detail
    assert app._mi["reconnect_calendar"].hidden is False
    assert not work.is_running(), "off hours: report, but do not poll"


def test_an_account_with_a_token_gets_no_row_until_it_polls(tmp_path: Path) -> None:
    """Present is all we can say offline; an UNKNOWN 'paused' row would turn
    into a 'Calendar recovered' toast at 08:00 every morning."""
    import rumps

    personal = _poller(tmp_path, "personal", token=True)
    app = _off_hours_app(tmp_path, [personal])
    app.run()
    rumps.events.before_quit.unregister(app._on_before_quit)
    assert get_health().get("calendar:personal") is None


def test_the_offline_check_makes_no_request(tmp_path: Path, monkeypatch) -> None:
    work = _poller(tmp_path, "work", token=False)
    monkeypatch.setattr(work, "_build_service",
                        lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert work.report_credentials_offline() is False
    assert _poller(tmp_path, "ok", token=True).report_credentials_offline() is True
