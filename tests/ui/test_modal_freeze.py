"""Regression: a menu dialog froze the whole app — twice for good.

rumps schedules every Timer in NSDefaultRunLoopMode only, and an alert runs
the loop in NSModalPanelRunLoopMode, so while any rumps.alert was open the
100 ms drain, the schedule tick and the duration tick all stopped (probe: 11
fires per second -> 0). 'Fix System Audio…' opened such an alert before it
updated the panel. In the logs it was never answered on 2026-09-10 14:03:40
and 2026-10-14 15:16:53: each time the instance kept port 8765 while its menu
stopped reacting, and the user launched more copies. After the 09-10 one the
main loop stayed dead for three days while detection kept running.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src import single_instance
from src.audio.blackhole_check import RoutingState, RoutingStatus
from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


@pytest.fixture
def app(tmp_path: Path, registry: HealthRegistry, monkeypatch):
    import rumps

    def no_modal(*a, **k):
        raise AssertionError("a menu action opened a modal alert")

    monkeypatch.setattr(rumps, "alert", no_modal)
    sent: list[tuple[str, str, str]] = []
    bar = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=registry,
        surface_request_path=tmp_path / single_instance.SURFACE_NAME,
    )
    bar._sent = sent  # type: ignore[attr-defined]
    opened: list[str] = []
    monkeypatch.setattr(bar, "_open_url", lambda url: opened.append(url))
    monkeypatch.setattr(bar, "_open_app", lambda name: opened.append(f"app:{name}"))
    bar._opened = opened  # type: ignore[attr-defined]
    return bar


def _routing(monkeypatch, status: RoutingStatus) -> None:
    import src.audio.blackhole_check as bc

    monkeypatch.setattr(bc, "check_output_routing", lambda probe=None: status)


NOT_ROUTED = RoutingStatus(
    state=RoutingState.NOT_ROUTED, default_output_name="Speakers", suggested_device="Otis"
)
ROUTED = RoutingStatus(
    state=RoutingState.OK, default_output_name="Otis", includes_blackhole=True,
    is_aggregate=True, member_names=["Speakers", "BlackHole 2ch"],
)


def test_fix_audio_reports_without_a_modal(app, registry, monkeypatch) -> None:
    _routing(monkeypatch, NOT_ROUTED)
    app._on_fix_audio(None)
    assert registry.get(SYSTEM_AUDIO).state is HealthState.FAIL
    assert any("Speakers" in m for _t, _s, m in app._sent)
    assert any("Sound" in url for url in app._opened), "should open Sound settings"


def test_fix_audio_repaints_the_panel_immediately(app, registry, monkeypatch) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed (output: Speakers)")
    app._drain_main_queue(None)

    _routing(monkeypatch, ROUTED)
    app._on_fix_audio(None)
    # No drain call here: the panel must already show the new state.
    rows = list(app._mi["health_root"].keys())
    assert any("routed via Otis" in r for r in rows)
    recovered = [n for n in app._sent if "recovered" in n[0]]
    assert len(recovered) == 1, "one toast for one fix, not two"


def test_fix_audio_when_nothing_changed_still_answers(app, registry, monkeypatch) -> None:
    _routing(monkeypatch, NOT_ROUTED)
    app._on_fix_audio(None)
    before = len(app._sent)
    app._on_fix_audio(None)  # FAIL -> FAIL: the registry stays quiet
    assert len(app._sent) == before + 1


def test_fix_audio_inaudible_opens_audio_midi_setup(app, monkeypatch) -> None:
    _routing(monkeypatch, RoutingStatus(
        state=RoutingState.INAUDIBLE, default_output_name="Otis BT",
        includes_blackhole=True, is_aggregate=True, member_names=["BlackHole 2ch"],
    ))
    app._on_fix_audio(None)
    assert "app:Audio MIDI Setup" in app._opened


def test_reconnect_calendar_starts_without_a_modal(app, monkeypatch) -> None:
    started: list[bool] = []
    monkeypatch.setattr(app, "_run_calendar_reauth", lambda: started.append(True))
    app._on_reconnect_calendar(None)
    deadline = time.monotonic() + 2
    while not started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started == [True]
    assert any("browser" in m.lower() for _t, _s, m in app._sent)


def test_timers_keep_firing_while_a_modal_is_open(app, tmp_path) -> None:
    """Spin the run loop the way NSAlert.runModal does and watch the drain."""
    from AppKit import NSModalPanelRunLoopMode
    from Foundation import NSDate, NSRunLoop

    surface = tmp_path / single_instance.SURFACE_NAME
    single_instance.request_surface(surface)
    loop = NSRunLoop.currentRunLoop()
    deadline = time.monotonic() + 1.0
    while surface.exists() and time.monotonic() < deadline:
        loop.runMode_beforeDate_(NSModalPanelRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))
    assert not surface.exists(), "the drain timer did not fire in modal-panel mode"
