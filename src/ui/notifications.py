"""User-facing macOS notifications with rate limiting.

Wraps :func:`rumps.notification` so the rest of the app can call
``manager.notify(NotificationType.MEETING_DETECTED, "Title", "Body")`` without
worrying about:

* ``rumps`` only being importable on macOS,
* the same event flooding the user (we cap at one notification per type per
  ``rate_limit_seconds`` window),
* failures bubbling up — Notification Center can be disabled per-app and we
  don't want that to crash the daemon.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_SECONDS = 30.0


class NotificationType(str, Enum):
    """Stable identifiers used by the rate-limiter."""

    MEETING_APPROACHING = "meeting_approaching"
    MEETING_DETECTED = "meeting_detected"
    RECORDING_STARTED = "recording_started"
    RECORDING_PAUSED = "recording_paused"
    PROCESS_DISAPPEARED = "process_disappeared"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    ERROR = "error"


class NotificationManager:
    """Send macOS notifications with per-type rate limiting.

    Parameters
    ----------
    rate_limit_seconds:
        Minimum seconds between two notifications of the same type. Default 30.
    backend:
        Optional callable ``(title, subtitle, message) -> None`` overriding the
        default :func:`rumps.notification`. Tests pass a recorder.
    clock:
        Override for ``time.monotonic`` (tests pass a controllable clock).
    """

    def __init__(
        self,
        *,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        backend: Callable[[str, str, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate_limit = float(rate_limit_seconds)
        self._backend = backend or _default_backend
        self._clock = clock
        self._lock = threading.RLock()
        self._last_sent: dict[NotificationType, float] = {}

    def notify(
        self,
        kind: NotificationType,
        title: str,
        message: str,
        *,
        subtitle: str = "",
        force: bool = False,
    ) -> bool:
        """Try to deliver a notification. Return ``True`` if sent, ``False`` if rate-limited."""
        now = self._clock()
        with self._lock:
            last = self._last_sent.get(kind)
            if not force and last is not None and (now - last) < self._rate_limit:
                logger.debug(
                    "Notification %s rate-limited (last %.1fs ago)", kind, now - last
                )
                return False
            self._last_sent[kind] = now
        try:
            self._backend(title, subtitle, message)
            logger.info("Notification[%s]: %s — %s", kind.value, title, message)
            return True
        except Exception as exc:
            logger.warning("Notification backend failed (%s): %s", kind.value, exc)
            return False

    def reset(self) -> None:
        """Forget the last-sent timestamps (useful in tests)."""
        with self._lock:
            self._last_sent.clear()


def _default_backend(title: str, subtitle: str, message: str) -> None:
    """Lazy-imported ``rumps.notification`` so non-macOS environments can import this module."""
    try:
        import rumps
    except Exception as exc:
        raise RuntimeError(
            "rumps is not available — notifications can only be sent on macOS."
        ) from exc
    rumps.notification(title=title, subtitle=subtitle or None, message=message)


# Convenience formatters used by the menu bar app -----------------------------
def format_transcription_complete(meeting_title: str, duration_minutes: float) -> tuple[str, str]:
    """Return the (title, body) pair for a 'transcript ready' notification."""
    duration = max(0, int(round(duration_minutes)))
    return "Transcript ready", f"{meeting_title} ({duration} min)"


def format_process_disappeared(app_name: str) -> tuple[str, str]:
    return "Meeting app exited", f"{app_name} closed — Stop recording?"


def silent_backend(_title: str, _subtitle: str, _message: str) -> None:  # pragma: no cover
    """No-op backend, useful for tests and dry-run modes."""
    return None


__all__ = [
    "DEFAULT_RATE_LIMIT_SECONDS",
    "NotificationManager",
    "NotificationType",
    "format_process_disappeared",
    "format_transcription_complete",
    "silent_backend",
]
