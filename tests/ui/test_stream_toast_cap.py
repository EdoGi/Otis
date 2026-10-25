"""Regression: stream-health toasts flapped all through a meeting.

The system stream is flagged after 60 s of digital silence, which is what a
muted or quiet far side sounds like. Every flip raised a forced toast with a
sound: 12 in 21 minutes on 2026-09-24, 6 on 2026-10-21 — each time the
signal came back on its own, so the routing was fine. During a recording a
stream now toasts at most one "problem" and one "recovered"; the panel still
shows every change. A silence that coincides with the Mac's output leaving
BlackHole (earbuds connecting mid-call) is real and always toasts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.audio.blackhole_check import RoutingState, RoutingStatus
from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp, UiState
from src.ui.notifications import NotificationManager

ROUTED = RoutingStatus(state=RoutingState.OK, default_output_name="Otis",
                       includes_blackhole=True, is_aggregate=True)
EARBUDS = RoutingStatus(state=RoutingState.NOT_ROUTED, default_output_name="AirPods",
                        suggested_device="Otis")


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


def _app(tmp_path: Path, registry: HealthRegistry, routing: list[RoutingStatus]):
    sent: list[tuple[str, str, str]] = []
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=registry,
        routing_check=lambda: routing[0],
    )
    app._sent = sent  # type: ignore[attr-defined]
    return app


def _flip(app, registry, states: list[HealthState]) -> None:
    for i, st in enumerate(states):
        registry.report(SYSTEM_AUDIO, label="System audio", state=st, detail=f"step {i}")
        app._drain_main_queue(None)


def _audio_toasts(app) -> list[str]:
    return [t for t, _s, _m in app._sent if t.startswith("System audio")]


def test_a_quiet_far_side_toasts_once_each_way(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, [ROUTED])
    app._begin_recording_toasts()
    app._snapshot.state = UiState.RECORDING
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK] * 6)
    assert _audio_toasts(app) == ["System audio problem", "System audio recovered"]
    assert registry.get(SYSTEM_AUDIO).detail == "step 17", "the panel still follows every change"


def test_losing_the_route_mid_call_always_toasts(tmp_path, registry) -> None:
    routing = [ROUTED]
    app = _app(tmp_path, registry, routing)
    app._begin_recording_toasts()
    app._snapshot.state = UiState.RECORDING
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK])
    routing[0] = EARBUDS  # the user's earbuds connect; macOS moves the output
    _flip(app, registry, [HealthState.FAIL])
    assert _audio_toasts(app).count("System audio problem") == 2


def test_outside_a_recording_nothing_is_capped(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, [ROUTED])
    app._snapshot.state = UiState.IDLE
    _flip(app, registry, [HealthState.FAIL, HealthState.OK, HealthState.FAIL, HealthState.OK])
    assert len(_audio_toasts(app)) == 4


def test_each_recording_gets_its_own_allowance(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, [ROUTED])
    app._begin_recording_toasts()
    app._snapshot.state = UiState.RECORDING
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK, HealthState.FAIL])
    app._begin_recording_toasts()
    _flip(app, registry, [HealthState.OK, HealthState.FAIL])
    assert _audio_toasts(app).count("System audio problem") == 2
