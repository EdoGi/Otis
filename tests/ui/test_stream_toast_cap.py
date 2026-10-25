"""Regression: stream-health toasts flapped all through a meeting.

The system stream is flagged after 60 s of digital silence, which is what a
muted or quiet far side sounds like. Every flip raised a forced toast with a
sound: 12 in 21 minutes on 2026-09-24, 6 on 2026-10-21 — each time the
signal came back on its own. During a recording a stream now toasts at most
one "problem" and one "recovered"; the panel still shows every change.

Exceptions, each a different kind of news:
* the route is lost *during* the recording (earbuds connect mid-call) —
  not a route that was "not routed" all along: this user's default output is
  the speakers while the meeting app plays straight into the Otis device, so
  a live routing check says NOT_ROUTED for every meeting (review finding of
  the first version of this cap, which therefore capped nothing here);
* a deliberate repeat (silence since the first block, or a stream silent for
  ten minutes straight);
* the user clicking Fix System Audio….
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
SPEAKERS = RoutingStatus(state=RoutingState.NOT_ROUTED,
                         default_output_name="Altoparlanti MacBook Pro", suggested_device="Otis")
EARBUDS = RoutingStatus(state=RoutingState.NOT_ROUTED, default_output_name="AirPods",
                        suggested_device="Otis")
NO_SPEAKERS = RoutingStatus(state=RoutingState.INAUDIBLE, default_output_name="Otis BT",
                            includes_blackhole=True, is_aggregate=True)


class _Recorder:
    on_device_error = None
    state = None

    def start(self) -> str:
        return "abcdef12-session"


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


def _app(tmp_path: Path, registry: HealthRegistry, routing: list[RoutingStatus]):
    sent: list[tuple[str, str, str]] = []
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: _Recorder(),
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        audio_dir=tmp_path / "audio",
        health=registry,
        routing_check=lambda: routing[0],
    )
    app._sent = sent  # type: ignore[attr-defined]
    return app


def _start_recording(app) -> None:
    """The real path: the worker re-checks routing, the drain handles the start."""
    app._start_recorder_worker()
    app._drain_main_queue(None)
    assert app._snapshot.state == UiState.RECORDING


def _flip(app, registry, states: list[HealthState], *, force: bool = False) -> None:
    for i, st in enumerate(states):
        registry.report(SYSTEM_AUDIO, label="System audio", state=st, detail=f"step {i}",
                        force_notify=force)
        app._drain_main_queue(None)


def _problems(app) -> int:
    return [t for t, _s, _m in app._sent].count("System audio problem")


def _recoveries(app) -> int:
    return [t for t, _s, _m in app._sent].count("System audio recovered")


@pytest.mark.parametrize("route", [ROUTED, SPEAKERS, NO_SPEAKERS],
                         ids=["routed", "speakers-with-app-into-otis", "no-speakers"])
def test_a_quiet_far_side_toasts_once_each_way(tmp_path, registry, route) -> None:
    app = _app(tmp_path, registry, [route])
    _start_recording(app)
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK, detail="signal ok")
    app._drain_main_queue(None)
    _flip(app, registry, [HealthState.FAIL, HealthState.OK] * 6)
    # Counted from 'Recording started': exactly one of each, whichever report
    # spent the allowance (the first signal, for a route that started FAIL).
    titles = [t for t, _s, _m in app._sent]
    during = titles[titles.index("Recording started"):]
    assert during.count("System audio problem") == 1
    assert during.count("System audio recovered") == 1
    assert registry.get(SYSTEM_AUDIO).detail == "step 11", "the panel follows every change"


def test_losing_the_route_mid_call_always_toasts(tmp_path, registry) -> None:
    routing = [ROUTED]
    app = _app(tmp_path, registry, routing)
    _start_recording(app)
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK])
    routing[0] = EARBUDS  # the user's earbuds connect; macOS moves the output
    _flip(app, registry, [HealthState.FAIL])
    assert _problems(app) == 2


def test_a_deliberate_repeat_is_never_capped(tmp_path, registry) -> None:
    """Silence since the first block, or ten minutes straight: forced on purpose."""
    app = _app(tmp_path, registry, [SPEAKERS])
    _start_recording(app)
    _flip(app, registry, [HealthState.OK])
    start = _problems(app)
    _flip(app, registry, [HealthState.FAIL, HealthState.OK, HealthState.FAIL])
    assert _problems(app) == start + 1
    _flip(app, registry, [HealthState.FAIL], force=True)
    assert _problems(app) == start + 2


def test_fix_system_audio_always_answers_mid_recording(tmp_path, registry, monkeypatch) -> None:
    routing = [ROUTED]
    app = _app(tmp_path, registry, routing)
    monkeypatch.setattr(app, "_open_url", lambda url: None)
    _start_recording(app)
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK, HealthState.FAIL])
    before = len(app._sent)
    app._on_fix_audio(None)  # FAIL -> OK: the "recovered" allowance is spent
    assert len(app._sent) == before + 1, "a click must always get an answer"


def test_outside_a_recording_nothing_is_capped(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, [ROUTED])
    _flip(app, registry, [HealthState.FAIL, HealthState.OK, HealthState.FAIL, HealthState.OK])
    assert _problems(app) == 2 and _recoveries(app) == 2


def test_each_recording_gets_its_own_allowance(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, [ROUTED])
    _start_recording(app)
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK, HealthState.FAIL])
    app._snapshot.state = UiState.IDLE
    _start_recording(app)  # a new recorder_started resets the allowance
    _flip(app, registry, [HealthState.OK, HealthState.FAIL])
    assert _problems(app) == 2


def test_a_forced_alert_after_a_fix_click_still_toasts(tmp_path, registry, monkeypatch) -> None:
    """Review round 2: a Fix click set the row OK in the middle of a dead
    stream, so the forced ten-minute report arrived as OK -> FAIL and was
    capped like an ordinary second "problem"."""
    app = _app(tmp_path, registry, [ROUTED])
    monkeypatch.setattr(app, "_open_url", lambda url: None)
    _start_recording(app)
    _flip(app, registry, [HealthState.OK, HealthState.FAIL])  # first-tier: spends "problem"
    app._on_fix_audio(None)                                    # row -> OK, stream still dead
    start = _problems(app)
    _flip(app, registry, [HealthState.FAIL], force=True)       # the ten-minute report
    assert _problems(app) == start + 1


@pytest.mark.parametrize("route", [ROUTED, SPEAKERS], ids=["routed", "speakers-with-app-into-otis"])
def test_a_forced_first_alert_spends_the_allowance(tmp_path, registry, route) -> None:
    """Review round 3: joining early (nothing heard for 60 s) forces the first
    "problem" toast; it must still count, or the next ordinary quiet stretch
    toasts a second time."""
    app = _app(tmp_path, registry, [route])
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                    detail="signal ok")  # a previous meeting's live signal
    app._drain_main_queue(None)
    _start_recording(app)
    _flip(app, registry, [HealthState.FAIL], force=True)       # silence since the start
    _flip(app, registry, [HealthState.OK, HealthState.FAIL, HealthState.OK, HealthState.FAIL])
    titles = [t for t, _s, _m in app._sent]
    assert titles[titles.index("Recording started"):].count("System audio problem") == 1
