"""Routing was a launch-time snapshot, and one state was mislabelled.

* check_output_routing() ran once at launch (and on a 'Fix System Audio…'
  click). macOS switches the default output whenever earbuds connect — the
  user's logs show 'soundcore Liberty 4 Pro', the built-in speakers and
  'Otis BT' on different launches — so by the time a meeting started the row
  could say the opposite of the truth. The recording now re-checks before
  capture starts.
* INAUDIBLE (the output feeds BlackHole but has no speakers) was reported as
  '✕ not routed', the opposite of what is wrong: Otis records fine and the
  user hears nothing (logged 2026-08-28 and 2026-09-03 with 'Otis BT').
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.audio.blackhole_check import RoutingState, RoutingStatus
from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager

ROUTED = RoutingStatus(state=RoutingState.OK, default_output_name="Otis",
                       includes_blackhole=True, is_aggregate=True,
                       member_names=["Speakers", "BlackHole 2ch"])
EARBUDS = RoutingStatus(state=RoutingState.NOT_ROUTED, default_output_name="soundcore Liberty 4 Pro",
                        suggested_device="Otis")
NO_SPEAKERS = RoutingStatus(state=RoutingState.INAUDIBLE, default_output_name="Otis BT",
                            includes_blackhole=True, is_aggregate=True,
                            member_names=["BlackHole 2ch"])


class _Recorder:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.on_device_error = None

    def start(self) -> str:
        self._calls.append("recorder.start")
        return "session-1234"


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


def _app(tmp_path: Path, registry: HealthRegistry, routing, calls: list[str]):
    def check():
        calls.append("routing")
        if isinstance(routing, Exception):
            raise routing
        return routing

    sent: list[tuple[str, str, str]] = []
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: _Recorder(calls),
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=registry,
        routing_check=check,
    )
    app._sent = sent  # type: ignore[attr-defined]
    return app


def test_no_speakers_is_a_warning_not_not_routed(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, NO_SPEAKERS, [])
    app._report_audio_routing(NO_SPEAKERS)
    rep = registry.get(SYSTEM_AUDIO)
    assert rep.state is HealthState.WARN
    assert "not routed" not in rep.detail
    assert "speakers" in rep.detail


def test_starting_a_recording_rechecks_routing_first(tmp_path, registry) -> None:
    calls: list[str] = []
    app = _app(tmp_path, registry, EARBUDS, calls)
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                    detail="routed via Otis")  # true at launch, before the earbuds
    app._start_recorder_worker()
    assert calls == ["routing", "recorder.start"], "check before capture, so live signal wins"
    assert registry.get(SYSTEM_AUDIO).state is HealthState.FAIL
    app._drain_main_queue(None)
    assert any("soundcore" in m for _t, _s, m in app._sent)


def test_a_fixed_route_clears_the_launch_warning_at_recording_start(tmp_path, registry) -> None:
    app = _app(tmp_path, registry, ROUTED, [])
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                    detail="not routed (output: Speakers)")
    app._start_recorder_worker()
    assert registry.get(SYSTEM_AUDIO).state is HealthState.OK


def test_a_failing_routing_check_never_blocks_recording(tmp_path, registry) -> None:
    calls: list[str] = []
    app = _app(tmp_path, registry, RuntimeError("CoreAudio said no"), calls)
    app._start_recorder_worker()
    assert "recorder.start" in calls
    events = []
    while not app._events.empty():
        events.append(app._events.get_nowait()[0])
    assert "recorder_started" in events


SPEAKERS = RoutingStatus(state=RoutingState.NOT_ROUTED,
                         default_output_name="Altoparlanti MacBook Pro", suggested_device="Otis")


def test_an_unchanged_route_does_not_re_alarm_at_every_meeting(tmp_path, registry) -> None:
    """Review finding: with the speakers as default output and the meeting app
    playing into the Otis device, the launch check says "not routed", the
    first recording's live signal says OK — and every later recording opened
    with a fresh "will record as silence" alarm, then "recovered" seconds
    later. The route had not changed, so there was nothing new to say."""
    app = _app(tmp_path, registry, SPEAKERS, [])
    app._report_audio_routing(SPEAKERS)            # launch: FAIL
    registry.report(SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                    detail="signal ok")            # meeting 1: live signal
    app._drain_main_queue(None)
    sent = len(app._sent)
    app._start_recorder_worker()                   # meeting 2 starts
    app._drain_main_queue(None)
    assert registry.get(SYSTEM_AUDIO).state is HealthState.OK
    assert not any("problem" in t for t, _s, _m in app._sent[sent:])


def test_a_changed_route_is_still_reported_at_recording_start(tmp_path, registry) -> None:
    calls: list[str] = []
    app = _app(tmp_path, registry, EARBUDS, calls)
    app._report_audio_routing(ROUTED)              # launch: routed through Otis
    app._start_recorder_worker()                   # earbuds since then
    assert registry.get(SYSTEM_AUDIO).state is HealthState.FAIL
