"""Regression: the Status panel cried wolf, then swallowed the real alarm.

1. A startup routing check that said "not routed" was only cleared by a
   60-second silence followed by signal: LevelMonitor reports a *recovery*,
   never a healthy start. On 2026-10-21 the panel showed ✕ System audio for
   the first 33 minutes of a meeting whose system stream carried audio from
   its first block (the 76-minute FLAC is 20 MB of real content).
2. The registry notifies only on a state change, so when the row was already
   FAIL from launch, a system stream that captured nothing at all raised no
   alert during the call: FAIL -> FAIL is "no change".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.levels import LevelMonitor
from src.audio.recorder import DualStreamRecorder
from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState, get_health, reset_health


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_health()
    yield
    reset_health()


def _startup_routing_fail() -> None:
    get_health().report(SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                        detail="not routed (output: Speakers)")


def _notifications() -> list[tuple[HealthState, HealthState]]:
    seen: list[tuple[HealthState, HealthState]] = []
    get_health().on_change(lambda rep, prev: seen.append((prev, rep.state)))
    return seen


def test_first_signal_fires_once() -> None:
    heard: list[str] = []
    m = LevelMonitor("system", on_first_signal=heard.append)
    m.observe(0.0, 0.1)
    assert heard == []
    m.observe(0.05, 0.1)
    m.observe(0.05, 0.1)
    assert heard == ["system"]
    assert m.has_heard_signal


def test_a_live_signal_clears_a_stale_startup_failure(tmp_path: Path) -> None:
    _startup_routing_fail()
    meter = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)._make_meter("system")
    meter.observe(0.05, 0.1)
    rep = get_health().get(SYSTEM_AUDIO)
    assert rep.state is HealthState.OK
    assert "signal" in rep.detail


def test_the_mic_row_appears_with_the_first_signal(tmp_path: Path) -> None:
    """No Microphone row existed during a recording until a silence/signal
    transition — none at all in the 76-minute meeting of 2026-10-21."""
    from src.health import MICROPHONE

    meter = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)._make_meter("mic")
    meter.observe(0.2, 0.1)
    assert get_health().get(MICROPHONE).state is HealthState.OK


def test_silence_from_the_start_alerts_even_when_already_failed(tmp_path: Path) -> None:
    _startup_routing_fail()
    seen = _notifications()
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec._silence_alert_seconds = 0.0  # flag on the first quiet block
    meter = rec._make_meter("system")
    meter.observe(0.0, 0.1)
    assert seen == [(HealthState.FAIL, HealthState.FAIL)], "nothing was ever captured: say so"
    assert get_health().get(SYSTEM_AUDIO).detail.startswith("no signal")


def test_a_pause_after_real_signal_is_not_forced(tmp_path: Path) -> None:
    """Far-side pauses already produce problem/recovered pairs; a stream that
    has carried audio must not also bypass the registry's de-duplication."""
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec._silence_alert_seconds = 0.0
    meter = rec._make_meter("system")
    meter.observe(0.05, 0.1)          # heard: row OK
    _startup_routing_fail()           # something else set FAIL meanwhile
    seen = _notifications()
    meter.observe(0.0, 0.1)           # silence after signal
    assert seen == []


def test_force_notify_fires_on_an_unchanged_state() -> None:
    reg = HealthRegistry()
    seen: list[HealthState] = []
    reg.on_change(lambda rep, prev: seen.append(rep.state))
    reg.report("x", label="X", state=HealthState.FAIL)
    reg.report("x", label="X", state=HealthState.FAIL)
    assert seen == [HealthState.FAIL]
    reg.report("x", label="X", state=HealthState.FAIL, force_notify=True)
    assert seen == [HealthState.FAIL, HealthState.FAIL]
