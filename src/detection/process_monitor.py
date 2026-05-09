"""Process-based meeting detection.

Polls running processes against a configurable whitelist (Zoom, Teams, …) and a
blacklist (e.g. SuperWhisper) on a background thread. For browser-based
meetings (Google Meet, Teams web, etc.) we additionally consult CoreAudio: a
whitelisted browser only counts as "in a meeting" while the default microphone
is in use.

Events:
    on_meeting_detected(callback)   -- fired with the matched app name
    on_meeting_ended(callback)      -- fired when a previously-detected app exits

The monitor debounces: an app that was already alerted within
``debounce_seconds`` won't trigger a second ``meeting_detected`` event, so
restarting Zoom doesn't spam listeners.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Browser process names we recognise. The presence of one of these is not
# enough on its own — we additionally require the default mic to be active.
DEFAULT_BROWSER_APPS: tuple[str, ...] = (
    "Google Chrome",
    "Chromium",
    "Safari",
    "Firefox",
    "Arc",
    "Microsoft Edge",
    "Brave Browser",
    "Vivaldi",
)

# Type aliases for clarity.
DetectedCallback = Callable[[str], None]
EndedCallback = Callable[[str], None]
MicProbe = Callable[[], bool]
ProcessIter = Callable[[], Iterable[dict[str, Any]]]


@dataclass
class _AppState:
    """Per-app bookkeeping inside the monitor."""

    name: str
    first_seen_monotonic: float
    last_seen_monotonic: float
    last_alert_monotonic: float | None = None
    pids: set[int] = field(default_factory=set)


class ProcessMonitor:
    """Background poller that detects meeting apps by inspecting the process list.

    Parameters
    ----------
    whitelisted_apps:
        Substrings (case-insensitive) of process names that count as a meeting
        when running. Examples: ``"zoom.us"``, ``"Microsoft Teams"``.
    blacklisted_apps:
        Substrings to ignore even if they would otherwise match. Examples:
        ``"SuperWhisper"`` (don't fire just because the dictation app is open).
    poll_interval_seconds:
        How often to scan ``psutil.process_iter`` (default 5 s).
    debounce_seconds:
        Window during which a re-detection of the same app does *not* fire a
        new ``meeting_detected`` event (default 30 s). Prevents Zoom restarts
        from spamming listeners.
    browser_apps:
        Process names treated as browsers — only fire when their mic is also
        active. Defaults to :data:`DEFAULT_BROWSER_APPS`.
    mic_probe:
        Callable returning ``True`` while some process is capturing from the
        default input device. Defaults to a CoreAudio query via
        :func:`src.audio.coreaudio_probe.is_default_input_running`.
        Override in tests.
    process_iter:
        Callable returning an iterable of ``{"pid", "name"}`` dicts. Defaults
        to ``psutil.process_iter``. Override in tests.
    """

    def __init__(
        self,
        *,
        whitelisted_apps: Iterable[str],
        blacklisted_apps: Iterable[str] = (),
        poll_interval_seconds: float = 5.0,
        debounce_seconds: float = 30.0,
        browser_apps: Iterable[str] | None = None,
        mic_probe: MicProbe | None = None,
        process_iter: ProcessIter | None = None,
    ) -> None:
        self._whitelist = [w.lower() for w in whitelisted_apps]
        self._blacklist = [b.lower() for b in blacklisted_apps]
        self._poll_interval = float(poll_interval_seconds)
        self._debounce = float(debounce_seconds)
        self._browser_apps_lower = [
            b.lower() for b in (browser_apps if browser_apps is not None else DEFAULT_BROWSER_APPS)
        ]
        self._mic_probe = mic_probe or _default_mic_probe
        self._process_iter = process_iter or _default_process_iter

        self._on_detected: list[DetectedCallback] = []
        self._on_ended: list[EndedCallback] = []

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._active: dict[str, _AppState] = {}

    # =====================================================================
    # Public API
    # =====================================================================
    def on_meeting_detected(self, callback: DetectedCallback) -> None:
        """Register a callback for the ``meeting_detected`` event."""
        self._on_detected.append(callback)

    def on_meeting_ended(self, callback: EndedCallback) -> None:
        """Register a callback for the ``meeting_ended`` event."""
        self._on_ended.append(callback)

    def start(self) -> None:
        """Begin polling on a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("ProcessMonitor already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="otis-process-monitor", daemon=True
        )
        self._thread.start()
        logger.info(
            "ProcessMonitor started (poll=%.1fs, whitelist=%s, blacklist=%s).",
            self._poll_interval,
            self._whitelist,
            self._blacklist,
        )

    def stop(self) -> None:
        """Signal the polling thread to stop and wait briefly for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def active_meetings(self) -> list[str]:
        """Snapshot of currently-detected meeting app names (thread-safe)."""
        with self._lock:
            return list(self._active.keys())

    def poll_once(self) -> set[str]:
        """Run a single scan and emit events. Public for testing.

        Returns the set of whitelisted app names detected this cycle.
        """
        try:
            processes = list(self._process_iter())
        except Exception as exc:  # pragma: no cover (defensive)
            logger.warning("Process enumeration failed: %s", exc)
            return set()

        names_seen: set[str] = set()
        pids_by_name: dict[str, set[int]] = {}
        browser_names_seen: set[str] = set()
        browser_pids: dict[str, set[int]] = {}

        for proc in processes:
            try:
                name = proc.get("name") if isinstance(proc, dict) else None
                pid = proc.get("pid") if isinstance(proc, dict) else None
            except Exception:  # pragma: no cover
                continue
            if not name:
                continue
            lname = name.lower()
            if any(b in lname for b in self._blacklist):
                continue
            for white in self._whitelist:
                if white in lname:
                    names_seen.add(white)
                    pids_by_name.setdefault(white, set())
                    if pid is not None:
                        pids_by_name[white].add(int(pid))
                    break
            for bro in self._browser_apps_lower:
                if bro in lname:
                    browser_names_seen.add(bro)
                    browser_pids.setdefault(bro, set())
                    if pid is not None:
                        browser_pids[bro].add(int(pid))
                    break

        # Browsers only count if the default mic is currently in use.
        if browser_names_seen:
            try:
                mic_in_use = bool(self._mic_probe())
            except Exception as exc:
                logger.warning("Mic probe raised %s; treating mic as inactive.", exc)
                mic_in_use = False
            if mic_in_use:
                for bname in browser_names_seen:
                    names_seen.add(bname)
                    pids_by_name.setdefault(bname, set()).update(browser_pids.get(bname, set()))
            else:
                logger.debug(
                    "Browser(s) running but mic is idle; skipping: %s",
                    sorted(browser_names_seen),
                )

        self._reconcile(names_seen, pids_by_name)
        return names_seen

    # =====================================================================
    # Internals
    # =====================================================================
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # pragma: no cover (defensive)
                logger.exception("ProcessMonitor poll cycle crashed; continuing.")
            self._stop_event.wait(self._poll_interval)
        logger.info("ProcessMonitor stopped.")

    def _reconcile(self, seen: set[str], pids_by_name: dict[str, set[int]]) -> None:
        """Update state given the apps detected this cycle and emit events."""
        now = time.monotonic()
        with self._lock:
            # Newly-seen and refresh existing.
            for name in seen:
                state = self._active.get(name)
                if state is None:
                    state = _AppState(
                        name=name,
                        first_seen_monotonic=now,
                        last_seen_monotonic=now,
                        pids=set(pids_by_name.get(name, set())),
                    )
                    self._active[name] = state
                    self._fire_detected_if_needed(state, now)
                else:
                    state.last_seen_monotonic = now
                    state.pids = set(pids_by_name.get(name, state.pids))
                    self._fire_detected_if_needed(state, now)

            # Apps that were active last cycle but not this cycle ⇒ ended.
            ended = [n for n in self._active if n not in seen]
            for name in ended:
                state = self._active.pop(name)
                logger.info("Meeting app exited: %s", name)
                _safe_call_each(self._on_ended, name)

    def _fire_detected_if_needed(self, state: _AppState, now: float) -> None:
        if (
            state.last_alert_monotonic is not None
            and (now - state.last_alert_monotonic) < self._debounce
        ):
            return
        state.last_alert_monotonic = now
        logger.info("Meeting app detected: %s (pids=%s)", state.name, sorted(state.pids))
        _safe_call_each(self._on_detected, state.name)


# ============================================================================
# Defaults — kept as module-level functions so tests can swap them via
# constructor injection rather than monkey-patching.
# ============================================================================
def _default_mic_probe() -> bool:
    try:
        from src.audio.coreaudio_probe import is_default_input_running

        return is_default_input_running()
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Default mic probe unavailable (%s); browser-based meetings won't be detected.",
            exc,
        )
        return False


def _default_process_iter() -> Iterable[dict[str, Any]]:
    """Yield dicts with ``pid`` and ``name`` for every visible process."""
    import psutil

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            yield proc.info  # type: ignore[attr-defined]
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
            continue
        except Exception:  # pragma: no cover
            continue


def _safe_call_each(callbacks: list[Callable[..., None]], *args: Any) -> None:
    for cb in callbacks:
        try:
            cb(*args)
        except Exception:
            logger.exception("Callback %r raised", cb)
