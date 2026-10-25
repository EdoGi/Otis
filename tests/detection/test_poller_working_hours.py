"""Regression: the calendar pollers ran at night and on a Sunday.

Working hours were enforced only by the menu bar's 60 s schedule timer, which
skips four UI states and stops entirely while the main loop is stuck. From
2026-09-10 evening to 09-14 morning a frozen copy of Otis kept both pollers
going: Google Calendar requests at 20:41, 22:18 and 23:15 on Sunday 09-13 and
through the night after, plus "upcoming meeting" alerts nobody saw. Each
poller now checks the window itself before touching the network.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.config import Config
from src.detection.calendar_poller import GoogleCalendarPoller
from src.detection.detector import MeetingDetector


def _poller(tmp_path: Path) -> GoogleCalendarPoller:
    return GoogleCalendarPoller(credentials_path=tmp_path / "c.json",
                                token_path=tmp_path / "t.json",
                                poll_interval_seconds=0.01, label="work")


def _cycles_in(poller: GoogleCalendarPoller, seconds: float) -> int:
    count = [0]

    def cycle() -> None:
        count[0] += 1

    poller._poll_cycle = cycle  # type: ignore[method-assign]
    stop = threading.Event()
    t = threading.Thread(target=poller._run, args=(stop,))
    t.start()
    time.sleep(seconds)
    stop.set()
    t.join(5)
    return count[0]


def test_an_inactive_window_makes_no_poll(tmp_path: Path) -> None:
    poller = _poller(tmp_path)
    poller.is_active = lambda: False
    assert _cycles_in(poller, 0.3) == 0


def test_an_active_window_polls(tmp_path: Path) -> None:
    poller = _poller(tmp_path)
    poller.is_active = lambda: True
    assert _cycles_in(poller, 0.3) > 0


def test_no_gate_means_always_poll(tmp_path: Path) -> None:
    assert _cycles_in(_poller(tmp_path), 0.3) > 0


def test_a_raising_gate_does_not_kill_the_poller(tmp_path: Path) -> None:
    poller = _poller(tmp_path)

    def broken() -> bool:
        raise RuntimeError("bad config")

    poller.is_active = broken
    assert _cycles_in(poller, 0.3) > 0, "fail open: a broken gate must not stop meetings"


def test_the_menu_bar_gates_its_pollers_on_its_working_hours(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    poller = _poller(tmp_path)
    MenuBarApp(
        config=Config({"app": {"working_days": []}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[poller]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=HealthRegistry(),
    )
    assert poller.is_active is not None
    assert poller.is_active() is False


def test_the_daemon_gates_its_pollers_too(tmp_path: Path) -> None:
    from src.daemon import OtisDaemon

    poller = _poller(tmp_path)
    OtisDaemon(
        config=Config({"app": {"working_days": []},
                       "storage": {"queue_dir": str(tmp_path / "q")},
                       "transcription": {"defer_while_in_call": False}}),
        notify=lambda _m: None, recorder_factory=lambda _c: None,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[poller]),
    )
    assert poller.is_active is not None
    assert poller.is_active() is False
