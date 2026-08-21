"""Live input-level tracking, so a dead stream is caught while it's happening.

The audit's headline failure — two months of system audio recorded as pure
digital silence — was only ever discovered *after* transcription, because that
is the first point anything looked at the samples. By then the meeting is over.

This module watches the level as the audio arrives. It answers two questions
the menu bar needs:

* how loud is this stream right now (a meter the user can glance at), and
* has it been silent long enough that something is actually broken?

Kept free of ``sounddevice`` and ``numpy`` so it is testable anywhere: the
caller computes one RMS value per capture block and feeds it in.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: RMS below this counts as "no signal". Digital silence is exactly 0.0; real
#: rooms with an open mic sit far above this even when nobody is speaking, so
#: the threshold mainly separates "device is dead" from "nobody spoke".
SILENCE_RMS_THRESHOLD = 1e-4

#: How long a stream must stay silent before we call it broken. Long enough to
#: survive a quiet moment in a meeting, short enough to catch it during the call.
DEFAULT_SILENCE_SECONDS = 60.0

#: Exponential smoothing applied to the displayed meter (0 = frozen, 1 = raw).
_METER_SMOOTHING = 0.3


class LevelMonitor:
    """Rolling level + sustained-silence detector for one capture stream."""

    def __init__(
        self,
        label: str,
        *,
        silence_threshold: float = SILENCE_RMS_THRESHOLD,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        on_silence: Callable[[str, float], None] | None = None,
        on_signal: Callable[[str], None] | None = None,
        gap_tolerance_seconds: float = 2.0,
    ) -> None:
        self.label = label
        self._threshold = float(silence_threshold)
        self._silence_seconds = float(silence_seconds)
        self._clock = clock
        self._on_silence = on_silence
        self._on_signal = on_signal
        # A gap between observations longer than this means capture stopped
        # (pause, sleep) rather than the stream going quiet.
        self._gap_tolerance = float(gap_tolerance_seconds)

        self._lock = threading.Lock()
        self._level = 0.0
        self._peak = 0.0
        self._silent_since: float | None = None
        self._flagged = False
        self._observed_seconds = 0.0
        self._last_observed_at: float | None = None

    # ------------------------------------------------------------------ input
    def observe(self, rms: float, duration_seconds: float) -> None:
        """Feed one capture block's RMS. Called from the audio callback.

        Must stay cheap and must never raise — this runs on CoreAudio's
        realtime thread.
        """
        now = self._clock()
        fire_silence = False
        fire_signal = False
        silent_for = 0.0

        with self._lock:
            # Silence is measured in *observed* time, not wall-clock. The
            # capture callback returns early while a recording is paused, so
            # observe() simply stops being called — and a monotonic clock keeps
            # running. Using wall-clock here meant the first quiet block after
            # resuming reported the entire pause as dead air: a real 25.6-minute
            # pause in a real session would have raised "no signal for 1560s".
            gap = now - self._last_observed_at if self._last_observed_at is not None else 0.0
            if gap > self._gap_tolerance and self._silent_since is not None:
                # Time we did not watch cannot count against the stream.
                self._silent_since += gap
            self._last_observed_at = now
            self._observed_seconds += max(0.0, duration_seconds)
            self._level = (
                _METER_SMOOTHING * rms + (1.0 - _METER_SMOOTHING) * self._level
            )
            self._peak = max(self._peak, rms)

            if rms >= self._threshold:
                if self._flagged:
                    fire_signal = True
                self._silent_since = None
                self._flagged = False
            else:
                if self._silent_since is None:
                    self._silent_since = now
                silent_for = now - self._silent_since
                if not self._flagged and silent_for >= self._silence_seconds:
                    self._flagged = True
                    fire_silence = True

        # Callbacks fire outside the lock — they end up in the health registry,
        # which has its own lock and its own callbacks.
        if fire_silence and self._on_silence is not None:
            self._safe(self._on_silence, self.label, silent_for)
        if fire_signal and self._on_signal is not None:
            self._safe(self._on_signal, self.label)

    @staticmethod
    def _safe(fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("LevelMonitor callback failed")

    # ----------------------------------------------------------------- output
    @property
    def level(self) -> float:
        """Smoothed RMS, for the meter."""
        with self._lock:
            return self._level

    @property
    def peak(self) -> float:
        """Loudest block seen since the last reset."""
        with self._lock:
            return self._peak

    @property
    def is_silent(self) -> bool:
        """True once the stream has been silent past the threshold duration."""
        with self._lock:
            return self._flagged

    @property
    def observed_seconds(self) -> float:
        with self._lock:
            return self._observed_seconds

    def silent_for(self) -> float:
        """Seconds since the stream last carried signal (0 when it has some)."""
        with self._lock:
            if self._silent_since is None:
                return 0.0
            return self._clock() - self._silent_since

    def reset(self) -> None:
        with self._lock:
            self._level = 0.0
            self._peak = 0.0
            self._silent_since = None
            self._flagged = False
            self._observed_seconds = 0.0
            self._last_observed_at = None

    def bar(self, width: int = 10) -> str:
        """A tiny text meter for the menu: ``▁▂▄█``-style blocks."""
        return render_meter(self.level, width=width)


_BLOCKS = "▁▂▃▄▅▆▇█"


def render_meter(level: float, *, width: int = 10) -> str:
    """Render 0.0-1.0 as a fixed-width block meter.

    Uses a cube-root curve: speech RMS lives around 0.01-0.2, so a linear
    meter would look permanently empty.
    """
    if level <= 0.0:
        return "·" * width
    scaled = min(1.0, max(0.0, level) ** (1.0 / 3.0))
    filled = round(scaled * width)
    if filled == 0:
        filled = 1
    idx = min(len(_BLOCKS) - 1, int(scaled * (len(_BLOCKS) - 1)))
    return _BLOCKS[idx] * filled + "·" * (width - filled)


def rms_from_int16(samples: Any) -> float:
    """RMS of an int16 numpy block, normalised to 0.0-1.0.

    Separated out so the audio callback stays a one-liner and the maths is
    testable without a capture device.
    """
    try:
        import numpy as np

        if samples.size == 0:
            return 0.0
        # float32 keeps this cheap; the block is ~4096 samples.
        arr = samples.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(arr * arr))) / 32768.0
    except Exception:  # pragma: no cover (numpy absent / odd dtype)
        return 0.0


__all__ = [
    "DEFAULT_SILENCE_SECONDS",
    "SILENCE_RMS_THRESHOLD",
    "LevelMonitor",
    "render_meter",
    "rms_from_int16",
]
