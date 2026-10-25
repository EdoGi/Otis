"""Regressions in how the Status panel renders while recording.

1. The live level meter was written into the health row's ``detail`` with
   touch(), the only place the reason lives: '✕ System audio — not routed
   (…)' became '✕ System audio — ▃▃▃·····', and stayed that way after the
   recording until something re-reported (29 minutes on 2026-10-21).
2. Meters were only drawn for streams that already had a health row, so the
   microphone — which nothing reports until a silence or a transcription —
   had no meter at all during a meeting.
3. The whole Status submenu was deleted and rebuilt every second while
   recording, and rumps keeps every MenuItem it ever registered: +18,000
   registry entries and 8-16 MB per recording hour, never freed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager


class _Recorder:
    def __init__(self) -> None:
        self.bars = {"mic": "▃▃······", "system": "▅▅▅·····"}

    def level_meters(self, width: int = 8) -> dict[str, str]:
        return dict(self.bars)


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


@pytest.fixture
def app(tmp_path: Path, registry: HealthRegistry):
    return MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=registry,
    )


def rows(app) -> list[str]:
    return [item.title for item in app._mi["health_root"].values() if hasattr(item, "title")]


def _recording(app, recorder: _Recorder) -> None:
    app._recorder = recorder
    app._snapshot.state = UiState.RECORDING


def test_the_reason_survives_the_meter(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed (output: Speakers)")
    _recording(app, _Recorder())
    app._refresh_level_meters()
    shown = rows(app)
    assert any("not routed (output: Speakers)" in r for r in shown)
    assert any("▅▅▅" in r for r in shown)
    assert registry.get(SYSTEM_AUDIO).detail == "not routed (output: Speakers)"


def test_both_meters_show_even_without_health_rows(app) -> None:
    _recording(app, _Recorder())
    app._refresh_level_meters()
    shown = rows(app)
    assert any("Mic" in r and "▃▃" in r for r in shown)
    assert any("System" in r and "▅▅▅" in r for r in shown)


def test_meters_disappear_when_recording_stops(app, registry) -> None:
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                    detail="signal ok")
    _recording(app, _Recorder())
    app._refresh_level_meters()
    app._snapshot.state = UiState.IDLE
    app._on_duration_tick(None)
    shown = rows(app)
    assert not any("▅" in r or "▃" in r for r in shown)
    assert any("signal ok" in r for r in shown)


def test_an_hour_of_meter_ticks_registers_no_new_menu_items(app, registry) -> None:
    import rumps.rumps as rumps_impl

    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK, detail="ok")
    recorder = _Recorder()
    _recording(app, recorder)
    app._refresh_level_meters()
    before = len(rumps_impl.NSApp._ns_to_py_and_callback)
    for i in range(3600):
        recorder.bars = {"mic": "▃" * (i % 8), "system": "▅" * ((i + 3) % 8)}
        app._refresh_level_meters()
    after = len(rumps_impl.NSApp._ns_to_py_and_callback)
    assert after - before < 10, f"{after - before} menu items leaked in one hour"


def test_rows_that_change_are_rebuilt_without_leaking(app, registry) -> None:
    import rumps.rumps as rumps_impl

    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK, detail="ok")
    app._refresh_health_menu()
    before = len(rumps_impl.NSApp._ns_to_py_and_callback)
    for i in range(200):
        state = HealthState.FAIL if i % 2 else HealthState.OK
        registry.report(SYSTEM_AUDIO, label="System audio", state=state, detail=f"d{i}")
        app._refresh_health_menu()
    assert len(rumps_impl.NSApp._ns_to_py_and_callback) - before < 10
    assert any("d199" in r for r in rows(app))
