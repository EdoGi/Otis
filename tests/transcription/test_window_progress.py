"""Regression: transcription progress froze for 16 minutes and looked like a hang.

Progress was pure wall-clock: elapsed / (duration / RTF), capped at 99 %. The
RTF is one per-model average that real streams miss by 0.6-2.9x, so when a
stream ran slower than the average the bar hit its cap and sat there. On
2026-10-21 the menu title read "44%" for 16.7 minutes and "89%" for 7.9 while
the mic and system streams finished their remaining windows. Nothing was
wrong, but it looked frozen.

Windowed transcription knows exactly how much audio is done after each
10-minute window; progress now follows that, interpolates inside a window
from the speed this run has actually shown, and never runs past the end of
the window it is in.
"""

from __future__ import annotations

from typing import Any

from src.transcription import whisper_engine as we
from src.transcription.whisper_engine import _ProgressEstimator


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_an_optimistic_estimate_stops_at_the_window_edge() -> None:
    clock = _Clock()
    est = _ProgressEstimator(2400.0, lambda _p: None, rtf=10.0, clock=clock)  # 4 windows
    est.start_clock()
    clock.now = 1000.0  # the average said 240 s for everything
    assert est.percent() < 25.0, "must not claim the second window before the first ends"


def test_each_finished_window_moves_the_bar() -> None:
    clock = _Clock()
    est = _ProgressEstimator(2400.0, lambda _p: None, rtf=10.0, clock=clock)
    est.start_clock()
    seen = []
    for i, t in enumerate((400.0, 800.0, 1200.0), start=1):
        clock.now = t
        est.window_done(i * 0.25)
        seen.append(est.percent())
    assert seen == sorted(seen)
    assert seen[0] >= 25.0 and seen[1] >= 50.0 and seen[2] >= 75.0


def test_inside_a_window_it_uses_this_runs_own_speed() -> None:
    clock = _Clock()
    est = _ProgressEstimator(2400.0, lambda _p: None, rtf=10.0, clock=clock)
    est.start_clock()
    clock.now = 400.0
    est.window_done(0.25)        # 600 s of audio in 400 s: 1.5x, not 10x
    clock.now = 600.0            # half of the next window at that speed
    assert 35.0 <= est.percent() <= 40.0


def test_progress_never_goes_backwards() -> None:
    clock = _Clock()
    est = _ProgressEstimator(2400.0, lambda _p: None, rtf=10.0, clock=clock)
    est.start_clock()
    clock.now = 230.0
    high = est.percent()
    clock.now = 231.0
    est.window_done(0.05)  # a window that kept little text
    assert est.percent() >= high


class _Samples:
    """len() and slicing are all _transcribe_windowed needs."""

    def __init__(self, seconds: float) -> None:
        self._n = int(seconds * we.TARGET_SAMPLE_RATE)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, item: slice) -> object:
        return object()


class _FakeMlx:
    def transcribe(self, _chunk: object, **_kw: Any) -> dict[str, Any]:
        return {"language": "fr", "segments": [
            {"start": 0.0, "end": 290.0, "text": "a"},
            {"start": 290.0, "end": 580.0, "text": "b"},
        ]}


def test_windowed_transcription_reports_each_window() -> None:
    reported: list[float] = []
    with we._window_progress(reported.append):
        we._transcribe_windowed(_FakeMlx(), _Samples(25 * 60), language="fr")
    assert reported == sorted(reported)
    assert len(reported) >= 3
    assert reported[-1] == 1.0
    assert 0.0 < reported[0] < 1.0


def test_without_a_listener_nothing_is_reported() -> None:
    # The default: retranscribe.py and the tests call it bare.
    out = we._transcribe_windowed(_FakeMlx(), _Samples(25 * 60), language="fr")
    assert out["segments"]
