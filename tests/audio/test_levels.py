"""Tests for src/audio/levels.py — catching a dead stream during the call.

The system stream recorded pure silence for two months and nothing noticed
until transcription, by which point the meeting was over. LevelMonitor is the
in-flight check: it flags a stream that has stayed silent long enough that
something is genuinely broken, without crying wolf during a quiet moment.
"""

from __future__ import annotations

import pytest

from src.audio.levels import LevelMonitor, render_meter, rms_from_int16


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


def feed_silence(mon: LevelMonitor, clock: FakeClock, seconds: float, *, step: float = 0.256) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        mon.observe(0.0, step)
        clock.advance(step)
        elapsed += step


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------
def test_silence_is_flagged_only_after_the_full_window(clock) -> None:
    fired: list[tuple[str, float]] = []
    mon = LevelMonitor(
        "system", clock=clock, silence_seconds=60.0,
        on_silence=lambda label, dur: fired.append((label, dur)),
    )

    feed_silence(mon, clock, 30.0)
    assert mon.is_silent is False
    assert fired == []

    feed_silence(mon, clock, 40.0)
    assert mon.is_silent is True
    assert len(fired) == 1
    assert fired[0][0] == "system"
    assert fired[0][1] >= 60.0


def test_silence_fires_once_not_every_block(clock) -> None:
    fired: list[str] = []
    mon = LevelMonitor(
        "system", clock=clock, silence_seconds=10.0,
        on_silence=lambda label, dur: fired.append(label),
    )
    feed_silence(mon, clock, 120.0)
    assert len(fired) == 1, "a dead stream must not re-notify on every 256ms block"


def test_signal_clears_the_flag_and_notifies_recovery(clock) -> None:
    silences: list[str] = []
    signals: list[str] = []
    mon = LevelMonitor(
        "system", clock=clock, silence_seconds=10.0,
        on_silence=lambda label, dur: silences.append(label),
        on_signal=signals.append,
    )

    feed_silence(mon, clock, 20.0)
    assert mon.is_silent is True

    mon.observe(0.05, 0.256)  # someone speaks
    assert mon.is_silent is False
    assert signals == ["system"]
    assert len(silences) == 1


def test_recovery_only_fires_if_we_had_flagged(clock) -> None:
    signals: list[str] = []
    mon = LevelMonitor("mic", clock=clock, silence_seconds=60.0, on_signal=signals.append)
    mon.observe(0.05, 0.256)
    mon.observe(0.06, 0.256)
    assert signals == [], "a healthy stream should not announce recovery"


def test_brief_quiet_does_not_trip_the_flag(clock) -> None:
    fired: list[str] = []
    mon = LevelMonitor(
        "mic", clock=clock, silence_seconds=60.0,
        on_silence=lambda label, dur: fired.append(label),
    )
    for _ in range(5):
        feed_silence(mon, clock, 20.0)
        mon.observe(0.03, 0.256)  # a word every 20 s
    assert fired == [], "pauses in conversation are not a broken device"


def test_silent_for_reports_elapsed(clock) -> None:
    mon = LevelMonitor("system", clock=clock, silence_seconds=60.0)
    assert mon.silent_for() == 0.0
    feed_silence(mon, clock, 10.0)
    assert mon.silent_for() == pytest.approx(10.0, abs=0.5)
    mon.observe(0.1, 0.256)
    assert mon.silent_for() == 0.0


# ---------------------------------------------------------------------------
# Meter
# ---------------------------------------------------------------------------
def test_level_tracks_input_and_peak_is_sticky(clock) -> None:
    mon = LevelMonitor("mic", clock=clock)
    for _ in range(30):
        mon.observe(0.2, 0.256)
    assert mon.level == pytest.approx(0.2, abs=0.02)
    assert mon.peak == pytest.approx(0.2)

    for _ in range(30):
        mon.observe(0.0, 0.256)
    assert mon.level < 0.01
    assert mon.peak == pytest.approx(0.2), "peak should survive a quiet patch"


def test_reset_clears_everything(clock) -> None:
    mon = LevelMonitor("mic", clock=clock, silence_seconds=1.0)
    feed_silence(mon, clock, 5.0)
    assert mon.is_silent is True
    mon.reset()
    assert mon.is_silent is False
    assert mon.level == 0.0
    assert mon.peak == 0.0
    assert mon.observed_seconds == 0.0


@pytest.mark.parametrize("level", [0.0, 0.001, 0.05, 0.5, 1.0])
def test_meter_is_always_the_requested_width(level: float) -> None:
    assert len(render_meter(level, width=10)) == 10


def test_meter_distinguishes_dead_from_quiet() -> None:
    dead = render_meter(0.0, width=8)
    quiet = render_meter(0.01, width=8)
    loud = render_meter(0.6, width=8)
    assert set(dead) == {"·"}, "digital silence should read as empty"
    assert dead != quiet != loud
    assert quiet.count("·") > loud.count("·")


# ---------------------------------------------------------------------------
# RMS helper
# ---------------------------------------------------------------------------
def test_rms_of_silence_is_zero() -> None:
    np = pytest.importorskip("numpy")
    assert rms_from_int16(np.zeros(4096, dtype=np.int16)) == 0.0


def test_rms_of_full_scale_square_is_one() -> None:
    np = pytest.importorskip("numpy")
    block = np.full(4096, 32767, dtype=np.int16)
    assert rms_from_int16(block) == pytest.approx(1.0, abs=0.001)


def test_rms_of_empty_block_is_zero() -> None:
    np = pytest.importorskip("numpy")
    assert rms_from_int16(np.zeros(0, dtype=np.int16)) == 0.0


def test_rms_never_raises_on_garbage() -> None:
    assert rms_from_int16(object()) == 0.0


def test_callback_exception_does_not_reach_the_audio_thread(clock) -> None:
    def boom(*_a):
        raise RuntimeError("health registry exploded")

    mon = LevelMonitor("system", clock=clock, silence_seconds=1.0, on_silence=boom)
    feed_silence(mon, clock, 5.0)  # must not raise
    assert mon.is_silent is True
