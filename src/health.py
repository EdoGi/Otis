"""Component health registry — the app's answer to "am I actually working?".

Otis's recurring failure mode was that it *detected* its own breakage and only
wrote it to a log file: system audio captured pure silence for two months, the
Google Calendar token was destroyed and the reauth callback had no subscriber,
and the process monitor fired hundreds of phantom detections. In every case the
signal existed; there was nowhere for it to go.

This module is that somewhere. Subsystems report their state here; the menu bar
renders the result, and a single change-callback turns a *transition* into a
user-facing notification. Reporting the same state repeatedly is free and
silent, which is what lets a 60-second poll loop report health on every cycle
without spamming anyone.

    health = get_health()
    health.report(
        SYSTEM_AUDIO,
        label="System audio",
        state=HealthState.FAIL,
        detail="captured silence for 60s",
        hint="Set your Mac's output device to 'Otis'.",
    )

Design notes
------------
* ``report()`` is idempotent with respect to notifications: callbacks fire only
  when ``state`` actually changes, never when only ``detail`` moves.
* Callbacks are isolated — a raising notifier cannot break a recording.
* The registry is thread-safe; reports arrive from the audio writer thread, the
  calendar poller, the transcription worker and the rumps main thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

logger = logging.getLogger(__name__)


# Canonical component keys. Using constants (rather than bare strings at each
# call site) keeps the menu ordering and the reporters in sync.
SYSTEM_AUDIO = "system_audio"
MICROPHONE = "microphone"
CALENDAR = "calendar"
DISK = "disk"
TRANSCRIPTION = "transcription"

#: Display order for the health panel. Keys not listed here trail, in the
#: order they were first reported.
CANONICAL_ORDER: tuple[str, ...] = (
    SYSTEM_AUDIO,
    MICROPHONE,
    CALENDAR,
    TRANSCRIPTION,
    DISK,
)


class HealthState(str, Enum):
    """How a component is doing. Ordered by severity via :data:`_SEVERITY`."""

    OK = "ok"
    UNKNOWN = "unknown"
    WARN = "warn"
    FAIL = "fail"


_SEVERITY: dict[HealthState, int] = {
    HealthState.OK: 0,
    HealthState.UNKNOWN: 1,
    HealthState.WARN: 2,
    HealthState.FAIL: 3,
}

#: Glyphs used in the menu-bar panel. Deliberately monochrome-friendly.
_GLYPH: dict[HealthState, str] = {
    HealthState.OK: "✓",
    HealthState.UNKNOWN: "·",
    HealthState.WARN: "!",
    HealthState.FAIL: "✕",
}


@dataclass(frozen=True)
class HealthReport:
    """One component's current state."""

    key: str
    label: str
    state: HealthState
    detail: str = ""
    hint: str | None = None
    updated_at: float = 0.0

    @property
    def glyph(self) -> str:
        return _GLYPH.get(self.state, "·")

    def one_line(self) -> str:
        """``"✕ System audio — captured silence"`` for the menu panel."""
        text = f"{self.glyph} {self.label}"
        if self.detail:
            text += f" — {self.detail}"
        return text


ChangeCallback = Callable[[HealthReport, HealthState], None]


class HealthRegistry:
    """Thread-safe map of component key → :class:`HealthReport`."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._reports: dict[str, HealthReport] = {}
        self._callbacks: list[ChangeCallback] = []

    # =====================================================================
    # Reporting
    # =====================================================================
    def report(
        self,
        key: str,
        *,
        label: str,
        state: HealthState,
        detail: str = "",
        hint: str | None = None,
    ) -> HealthReport:
        """Record a component's state; fire callbacks only on a real change.

        Returns the stored report. Safe to call on every poll tick — repeated
        identical states are recorded without notifying anyone.
        """
        rep = HealthReport(
            key=key,
            label=label,
            state=state,
            detail=detail,
            hint=hint,
            updated_at=self._clock(),
        )
        with self._lock:
            previous = self._reports.get(key)
            self._reports[key] = rep
            prev_state = previous.state if previous is not None else HealthState.UNKNOWN
            changed = prev_state is not state
            callbacks = list(self._callbacks) if changed else []

        if changed:
            logger.info(
                "health[%s] %s → %s%s",
                key, prev_state.value, state.value,
                f" ({detail})" if detail else "",
            )
            for cb in callbacks:
                try:
                    cb(rep, prev_state)
                except Exception:
                    logger.exception("Health change callback failed for %s", key)
        return rep

    def clear(self, key: str) -> None:
        with self._lock:
            self._reports.pop(key, None)

    def on_change(self, callback: ChangeCallback) -> None:
        """Subscribe to state transitions. Callbacks must not raise (we catch)."""
        with self._lock:
            self._callbacks.append(callback)

    # =====================================================================
    # Reading
    # =====================================================================
    def get(self, key: str) -> HealthReport | None:
        with self._lock:
            return self._reports.get(key)

    def snapshot(self) -> list[HealthReport]:
        """All components, canonical keys first, then first-report order."""
        with self._lock:
            reports = dict(self._reports)
        ordered: list[HealthReport] = []
        for key in CANONICAL_ORDER:
            if key in reports:
                ordered.append(reports.pop(key))
        ordered.extend(reports.values())
        return ordered

    def overall(self) -> HealthState:
        """The worst state across all components (``OK`` when empty)."""
        with self._lock:
            states = [r.state for r in self._reports.values()]
        if not states:
            return HealthState.OK
        return max(states, key=lambda s: _SEVERITY[s])

    def is_degraded(self) -> bool:
        """True when something needs the user's attention (WARN or FAIL)."""
        return self.overall() in (HealthState.WARN, HealthState.FAIL)

    def touch(self, key: str, *, detail: str) -> None:
        """Update only the detail line of an existing component, never the state."""
        with self._lock:
            existing = self._reports.get(key)
            if existing is None:
                return
            self._reports[key] = replace(
                existing, detail=detail, updated_at=self._clock()
            )


def describe_health(registry: HealthRegistry) -> list[str]:
    """Render a registry as one display line per component."""
    return [rep.one_line() for rep in registry.snapshot()]


#: The fix is the same whoever notices the stream is dead, so the hint lives
#: here rather than being restated at each reporting site.
STREAM_HINTS = {
    "system": (
        "Set your Mac's output device to the Otis Multi-Output device "
        "(System Settings → Sound → Output)."
    ),
    "mic": "Check your input device and macOS microphone permission.",
}

_STREAM_COMPONENTS = {
    "system": (SYSTEM_AUDIO, "System audio"),
    "mic": (MICROPHONE, "Microphone"),
}


def report_stream_health(
    stream: str,
    *,
    silent: bool,
    detail: str,
    registry: HealthRegistry | None = None,
) -> None:
    """Report a capture stream as healthy or silent.

    Called from two places with the same meaning: the recorder's live level
    monitor (during the call) and the transcription silence pre-check (after
    it). A silent *system* stream means the far side of the meeting is being
    lost, which is a hard failure with a known fix; a silent *mic* usually just
    means nobody spoke on this side, so it is only a warning.
    """
    component = _STREAM_COMPONENTS.get(stream)
    if component is None:
        return
    key, label = component
    reg = registry if registry is not None else get_health()
    if silent:
        state = HealthState.FAIL if stream == "system" else HealthState.WARN
        reg.report(key, label=label, state=state, detail=detail,
                   hint=STREAM_HINTS.get(stream))
    else:
        reg.report(key, label=label, state=HealthState.OK, detail=detail)


# ---------------------------------------------------------------------------
# Process-global registry
# ---------------------------------------------------------------------------
# Otis is a single-process app, and the reporters (whisper engine, recorder
# writer thread, calendar poller) sit several layers below the menu bar that
# renders them. Threading a registry through every constructor would be a lot
# of plumbing for no testability gain, so we keep one process-global instance
# and let tests reset it.
_GLOBAL: HealthRegistry | None = None
_GLOBAL_LOCK = threading.Lock()


def get_health() -> HealthRegistry:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = HealthRegistry()
        return _GLOBAL


def reset_health() -> None:
    """Drop the process-global registry (tests, and config reloads)."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        _GLOBAL = None


__all__ = [
    "CALENDAR",
    "CANONICAL_ORDER",
    "DISK",
    "MICROPHONE",
    "SYSTEM_AUDIO",
    "TRANSCRIPTION",
    "HealthRegistry",
    "HealthReport",
    "HealthState",
    "describe_health",
    "get_health",
    "reset_health",
]
