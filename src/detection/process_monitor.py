"""Process-based meeting detection.

Polls running processes against a configurable whitelist (Zoom, Teams, …) and a
blacklist (e.g. SuperWhisper) on a background thread.

Detection is deliberately calendar-first. An earlier version also promoted any
running *browser* to "in a meeting" whenever the default microphone was in use,
which produced 398 phantom detections against 8 real recordings: Apple's
always-running daemons matched the browser names by substring, and any mic use
at all — dictation, a call, or Otis's own recorder — armed the heuristic. That
path is gone. Browser calls (Meet, Whereby) are started with Start Recording.

Events:
    on_meeting_detected(callback)   -- fired with the matched app name
    on_meeting_ended(callback)      -- fired when a previously-detected app exits

The monitor debounces: an app that was already alerted within
``debounce_seconds`` won't trigger a second ``meeting_detected`` event, so
restarting Zoom doesn't spam listeners.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _boundary_pattern(entry: str) -> re.Pattern[str]:
    # Lookarounds, not \b, for two reasons.
    #
    # \b is a transition between a word and a non-word char, so an entry
    # ending in a non-word char can never match: after the ")" of "Microsoft
    # Teams (work or school)" — a real shipped app name — \b demands a
    # following word character, which never comes before a space.
    #
    # And "." must count as part of a name, not a boundary. psutil reports
    # reverse-DNS process names like "com.apple.WebKit.WebContent", where a
    # bare \b would let "arc" match "com.apple.Arc.helper" and reopen the
    # phantom-detection class exact matching was introduced to close.
    return re.compile(rf"(?<![\w.]){re.escape(entry)}(?![\w.])")


def bundle_id_matches(entry: str, bundle_id: str) -> bool:
    """True if ``entry`` names the app owning reverse-DNS ``bundle_id``.

    Bundle ids need their own rule. Word-boundary matching treats every "." as
    a boundary, so ``app_name_matches("arc", "com.apple.Arc.helper")`` is True
    — reopening exactly the phantom-detection class that exact matching closed.

    A bundle id is a dot-separated hierarchy, so the honest test is identity or
    ancestry: ``us.zoom.xos`` matches itself and ``us.zoom.xos.renderer``, and
    nothing matches a bare word buried in the middle.
    """
    if not entry or not bundle_id:
        return False
    entry = entry.lower().strip(".")
    bundle_id = bundle_id.lower()
    if "." not in entry:
        return False  # a bare word is an app name, not a bundle id
    return bundle_id == entry or bundle_id.startswith(entry + ".")


def app_name_matches(entry: str, process_name: str) -> bool:
    """True if ``process_name`` belongs to the app named by ``entry``.

    Matching used to be a raw ``entry in process_name`` substring test, which
    is how ``'arc'`` matched ``searchpartyd`` and ``'safari'`` matched
    ``SafariWidgetExtension`` — Apple daemons that run 24/7 on every Mac. With
    mic-activation on, that made the browser set permanently non-empty and
    produced 398 phantom "Meeting detected" events against 8 real recordings.

    Exact match, or a word-boundary match so genuine helper processes
    ("Slack Helper", "Microsoft Teams Helper (Renderer)") still count as their
    parent app. ``re.escape`` keeps ``zoom.us``'s dot literal.
    """
    if not entry or not process_name:
        return False
    entry = entry.lower()
    name = process_name.lower()
    if entry == name:
        return True
    return _boundary_pattern(entry).search(name) is not None

# Type aliases for clarity.
DetectedCallback = Callable[[str], None]
EndedCallback = Callable[[str], None]
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
        process_iter: ProcessIter | None = None,
    ) -> None:
        self._whitelist = [w.lower() for w in whitelisted_apps]
        self._blacklist = [b.lower() for b in blacklisted_apps]
        self._poll_interval = float(poll_interval_seconds)
        self._debounce = float(debounce_seconds)
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
        # Fresh event per run so a previous thread that's still mid-scan
        # can't be resurrected by clearing a shared event.
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event,),
            name="otis-process-monitor",
            daemon=True,
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
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover (slow process scan)
                logger.info(
                    "ProcessMonitor still finishing its last scan; "
                    "it will exit on its own."
                )
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

        for proc in processes:
            try:
                name = proc.get("name") if isinstance(proc, dict) else None
                pid = proc.get("pid") if isinstance(proc, dict) else None
                bundle_id = proc.get("bundle_id") if isinstance(proc, dict) else None
            except Exception:  # pragma: no cover
                continue
            if not name:
                continue
            # NSWorkspace reports *localized* names ("Aperçu", "Accessibilité"),
            # so a whitelist entry is matched against the bundle id as well.
            # That lets "us.zoom.xos" work on any language of macOS.
            lname = name.lower()
            bid = str(bundle_id).lower() if bundle_id else ""

            def owns(entry: str, _lname: str = lname, _bid: str = bid) -> bool:
                """Does ``entry`` name this app, by display name or bundle id?"""
                return app_name_matches(entry, _lname) or bundle_id_matches(entry, _bid)

            # A deny-list stays deliberately broad: missing something we meant
            # to exclude is worse than excluding a little too much, so it keeps
            # the old substring behaviour ("superwhisper" must still catch
            # "superwhisperd").
            if any(b in lname or (bid and b in bid) for b in self._blacklist):
                continue
            for white in self._whitelist:
                if owns(white):
                    names_seen.add(white)
                    pids_by_name.setdefault(white, set())
                    if pid is not None:
                        pids_by_name[white].add(int(pid))
                    break

        self._reconcile(names_seen, pids_by_name)
        return names_seen

    # =====================================================================
    # Internals
    # =====================================================================
    def _run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # pragma: no cover (defensive)
                logger.exception("ProcessMonitor poll cycle crashed; continuing.")
            stop_event.wait(self._poll_interval)
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
        first_alert = state.last_alert_monotonic is None
        state.last_alert_monotonic = now
        if first_alert:
            logger.info(
                "Meeting app detected: %s (pids=%s)", state.name, sorted(state.pids)
            )
        else:
            # Continuously-running apps re-fire every debounce window (the
            # detector needs that to pick up back-to-back meetings), but a
            # log line every 30s for the entire call is pure noise.
            logger.debug(
                "Meeting app still present: %s (re-fire)", state.name
            )
        _safe_call_each(self._on_detected, state.name)


# ============================================================================
# Defaults — kept as module-level functions so tests can swap them via
# constructor injection rather than monkey-patching.
# ============================================================================
def _default_process_iter() -> Iterable[dict[str, Any]]:
    """Default source: every visible process, via psutil.

    Slower than :func:`nsworkspace_process_iter` but correct everywhere, which
    is why it is the default. See that function for when the faster source is
    safe.
    """
    return _psutil_processes()


def nsworkspace_process_iter() -> Iterable[dict[str, Any]]:
    """Running *applications* via AppKit. Only valid with a live main run loop.

    Measured on this Mac: **0.58 ms for 93 apps versus 24.8 ms for 571
    processes**, and end-to-end ``poll_once`` drops from 24.8 ms to ~1 ms. The
    reason psutil is so expensive is subtle — ``Process.name()`` falls back to
    ``cmdline()`` whenever the name is 15+ characters, and that is a second
    sysctl copying the target's whole argv and environment; 261 of 571
    processes here qualified, so each poll paid ~830 syscalls.

    It is also *more correct* for this job: it reports applications, not every
    process, so the Apple daemons behind the phantom detections
    (searchpartyd, TrialArchivingService, SafariWidgetExtension) never appear.

    **The catch, and why this is not the default.** Cocoa only refreshes
    ``NSWorkspace.runningApplications`` while the main run loop runs. Verified
    on this machine: from a background thread with no ``NSApplication``,
    launching TextEdit never changed the array, and pumping
    ``CFRunLoopRunInMode`` from that thread did not help either. ``otis run``
    is headless and blocks on an Event, so it would have gone permanently
    blind to any app started after launch — while the menu bar, which rumps
    drives with a real run loop, is fine.

    So the front end declares this: ``main.py`` opts in because it runs rumps;
    the daemon keeps psutil. Falls back to psutil if AppKit is unavailable.
    """
    apps = _nsworkspace_apps()
    return apps if apps is not None else _psutil_processes()


def _nsworkspace_apps() -> list[dict[str, Any]] | None:
    """Running applications via AppKit, or ``None`` if unavailable."""
    try:
        from AppKit import NSWorkspace
    except Exception:  # pragma: no cover (non-macOS / pyobjc missing)
        return None
    try:
        out: list[dict[str, Any]] = []
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            name = app.localizedName()
            if not name:
                continue
            out.append({
                "pid": int(app.processIdentifier()),
                "name": str(name),
                # Names are localized ("Aperçu", "Accessibilité"), so a bundle
                # id is the locale-proof way to whitelist an app.
                "bundle_id": str(app.bundleIdentifier() or ""),
            })
        return out
    except Exception:  # pragma: no cover (AppKit hiccup)
        logger.warning("NSWorkspace enumeration failed; falling back to psutil.",
                       exc_info=True)
        return None


def _psutil_processes() -> Iterable[dict[str, Any]]:
    """Fallback source: every visible process. Slower, but always available."""
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
