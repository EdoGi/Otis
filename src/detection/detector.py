"""Detection orchestrator.

Wires :class:`ProcessMonitor` and :class:`GoogleCalendarPoller` into a single
state machine and collapses overlapping signals into one meeting.

State machine
-------------
::

    IDLE --calendar alert-->        APPROACHING
    IDLE --process detected-->      DETECTED
    APPROACHING --process detected->DETECTED
    APPROACHING --user starts-->    DETECTED
    DETECTED --user starts-->       RECORDING
    RECORDING --user stops-->       ENDED
    RECORDING --process exits-->    ENDED
    ENDED --transcription begins--> PROCESSING
    PROCESSING --transcription done->IDLE

Correlation
-----------
A process detection that arrives within ±``correlation_minutes`` (default 5) of
an APPROACHING calendar event is treated as **the same meeting** — its title,
attendees, and link are inherited from the calendar entry, and no second
``meeting_detected`` event is emitted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.detection.calendar_poller import CalendarEvent, GoogleCalendarPoller
from src.detection.process_monitor import ProcessMonitor

logger = logging.getLogger(__name__)


class MeetingState(str, Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    DETECTED = "detected"
    RECORDING = "recording"
    ENDED = "ended"
    PROCESSING = "processing"


@dataclass
class MeetingContext:
    """Everything we know about the currently-live meeting (if any)."""

    state: MeetingState = MeetingState.IDLE
    title: str | None = None
    app: str | None = None  # process name that triggered detection (if any)
    start_time: datetime | None = None  # calendar start, if known
    end_time: datetime | None = None  # calendar end, if known
    participants: list[dict[str, str]] = field(default_factory=list)
    meeting_link: str | None = None
    calendar_event_id: str | None = None
    is_tentative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "title": self.title,
            "app": self.app,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "participants": list(self.participants),
            "meeting_link": self.meeting_link,
            "calendar_event_id": self.calendar_event_id,
            "is_tentative": self.is_tentative,
        }


# Callback signatures — the orchestrator hands a MeetingContext snapshot to
# the *meeting* events. The advisory ``on_process_disappeared`` event passes
# only the app name because it is a soft signal — recording continues.
# each. Callbacks must be quick; they're invoked on the polling threads.
ApproachingCallback = Callable[[MeetingContext], None]
DetectedCallback = Callable[[MeetingContext], None]
EndedCallback = Callable[[MeetingContext], None]
ProcessGoneCallback = Callable[[str], None]


class MeetingDetector:
    """Orchestrates process and calendar signals into a single state machine.

    Parameters
    ----------
    process_monitor:
        Pre-built :class:`ProcessMonitor`. The detector only attaches its own
        callbacks; lifecycle (``start``/``stop``) is owned by us.
    calendar_poller:
        Pre-built :class:`GoogleCalendarPoller` (or ``None`` if calendar
        detection is disabled in config).
    correlation_minutes:
        Time window for matching a process event to a known calendar event.
    clock:
        Override for ``datetime.now(timezone.utc)`` (for tests).
    """

    def __init__(
        self,
        *,
        process_monitor: ProcessMonitor | None,
        calendar_poller: GoogleCalendarPoller | None = None,
        calendar_pollers: Iterable[GoogleCalendarPoller] | None = None,
        correlation_minutes: float = 5.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._process_monitor = process_monitor
        # Accept either a single ``calendar_poller`` (back-compat) or a list
        # of ``calendar_pollers`` (one per Google account, e.g. personal +
        # work). Both forms can be combined; we collapse to one list.
        self._calendar_pollers: list[GoogleCalendarPoller] = []
        if calendar_pollers is not None:
            self._calendar_pollers.extend(calendar_pollers)
        if calendar_poller is not None:
            self._calendar_pollers.append(calendar_poller)
        self._correlation = timedelta(minutes=float(correlation_minutes))
        self._now = clock or _utcnow

        self._lock = threading.RLock()
        self._context = MeetingContext()
        # Calendar events seen as "approaching" but not yet matched to a process.
        # Keyed by ``CalendarEvent.canonical_key`` (iCalUID when available, else
        # the per-calendar event id) so cross-invited meetings — visible from
        # both a personal and a work poller — collapse to one entry.
        self._pending_approaching: dict[str, CalendarEvent] = {}
        # Canonical keys we've already fired ``meeting_approaching`` for in
        # the current cycle. Reset when the meeting transitions out of
        # APPROACHING (process detected, calendar end reached, etc.).
        self._alerted_uids: set[str] = set()

        self._on_approaching: list[ApproachingCallback] = []
        self._on_detected: list[DetectedCallback] = []
        self._on_ended: list[EndedCallback] = []
        self._on_process_gone: list[ProcessGoneCallback] = []

        if self._process_monitor is not None:
            self._process_monitor.on_meeting_detected(self._handle_process_detected)
            self._process_monitor.on_meeting_ended(self._handle_process_ended)
        for poller in self._calendar_pollers:
            poller.on_upcoming_meeting(self._handle_calendar_upcoming)
            poller.on_meeting_should_end(self._handle_calendar_ended)

    @property
    def calendar_pollers(self) -> tuple[GoogleCalendarPoller, ...]:
        """All calendar pollers attached to this detector (read-only)."""
        return tuple(self._calendar_pollers)

    # =====================================================================
    # Public API
    # =====================================================================
    def on_meeting_approaching(self, callback: ApproachingCallback) -> None:
        self._on_approaching.append(callback)

    def on_meeting_detected(self, callback: DetectedCallback) -> None:
        self._on_detected.append(callback)

    def on_meeting_ended(self, callback: EndedCallback) -> None:
        self._on_ended.append(callback)

    def on_process_disappeared(self, callback: ProcessGoneCallback) -> None:
        """Advisory event: a previously-detected meeting app has exited.

        Fired when the process backing an in-progress meeting disappears
        *while the recorder is still running*. The detector does **not**
        change state — recording continues. Subscribers (UI, daemon) decide
        whether to surface a notification, prompt the user, or auto-stop.

        Use ``user_stopped_recording()`` from your UI to actually end the
        recording in response.
        """
        self._on_process_gone.append(callback)

    def start(self) -> None:
        if self._process_monitor is not None:
            self._process_monitor.start()
        for poller in self._calendar_pollers:
            poller.start()
        logger.info(
            "MeetingDetector started (process=%s, calendar_pollers=%d).",
            self._process_monitor is not None,
            len(self._calendar_pollers),
        )

    def stop(self) -> None:
        if self._process_monitor is not None:
            self._process_monitor.stop()
        for poller in self._calendar_pollers:
            poller.stop()
        logger.info("MeetingDetector stopped.")

    def get_state(self) -> MeetingState:
        with self._lock:
            return self._context.state

    def get_current_meeting(self) -> MeetingContext | None:
        with self._lock:
            if self._context.state == MeetingState.IDLE:
                return None
            # Return a copy-via-dict to keep callers from mutating internal state.
            return _copy_context(self._context)

    # ----------------------------- user-driven transitions (called from UI)
    def user_started_recording(self) -> None:
        with self._lock:
            if self._context.state in (MeetingState.RECORDING, MeetingState.PROCESSING):
                return
            if self._context.state == MeetingState.IDLE:
                # User clicked Record without any signal — bare-minimum context.
                self._context = MeetingContext(
                    state=MeetingState.RECORDING,
                    start_time=self._now(),
                )
                logger.info("RECORDING (manual start, no prior signal).")
                return
            self._transition(MeetingState.RECORDING)

    def user_stopped_recording(self) -> None:
        with self._lock:
            if self._context.state != MeetingState.RECORDING:
                return
            self._transition(MeetingState.ENDED)
            ctx = _copy_context(self._context)
        _safe_call_each(self._on_ended, ctx)

    def transcription_started(self) -> None:
        with self._lock:
            if self._context.state in (MeetingState.ENDED, MeetingState.RECORDING):
                self._transition(MeetingState.PROCESSING)

    def transcription_finished(self) -> None:
        with self._lock:
            if self._context.state == MeetingState.PROCESSING:
                self._transition(MeetingState.IDLE)
                self._context = MeetingContext()  # reset
                self._alerted_uids.clear()
                self._pending_approaching.clear()

    # =====================================================================
    # Signal handlers
    # =====================================================================
    def _handle_calendar_upcoming(self, event: CalendarEvent) -> None:
        with self._lock:
            key = event.canonical_key
            already_alerted = key in self._alerted_uids
            self._pending_approaching[key] = event
            if already_alerted:
                # Same logical meeting received from a second poller — drop.
                return
            self._alerted_uids.add(key)

            if self._context.state == MeetingState.IDLE:
                self._context = _context_from_event(event, MeetingState.APPROACHING)
                ctx = _copy_context(self._context)
                logger.info("APPROACHING: %s", event.title)
            elif self._context.state == MeetingState.APPROACHING:
                # Different upcoming event arrived — update the context.
                self._context = _context_from_event(event, MeetingState.APPROACHING)
                ctx = _copy_context(self._context)
            else:
                # We're already DETECTED/RECORDING/etc. Don't overwrite, but
                # remember the event in case we need to enrich later.
                ctx = None
        if ctx is not None:
            _safe_call_each(self._on_approaching, ctx)

    def _handle_calendar_ended(self, event: CalendarEvent) -> None:
        with self._lock:
            key = event.canonical_key
            self._pending_approaching.pop(key, None)
            self._alerted_uids.discard(key)
            # Soft signal: we don't auto-end the recording. UI may surface this
            # to the user. The detector itself only cares to clean up state if
            # we're still APPROACHING (no process arrived).
            if (
                self._context.state == MeetingState.APPROACHING
                and self._context.calendar_event_id == event.id
            ):
                self._transition(MeetingState.IDLE)
                self._context = MeetingContext()
                logger.info("Approaching meeting passed without detection: %s", event.title)

    def _handle_process_detected(self, app_name: str) -> None:
        with self._lock:
            # If a recording is already in progress, the process signal is
            # advisory only — never overwrite RECORDING/PROCESSING state, even
            # if a freshly-arrived calendar event would otherwise correlate.
            # (Without this guard, an upcoming calendar alert that fires mid-
            # recording would silently demote the UI to DETECTED while the
            # recorder thread keeps writing audio, hiding Pause/Stop.)
            if self._context.state in (MeetingState.RECORDING, MeetingState.PROCESSING):
                if self._context.app is None:
                    self._context.app = app_name
                return
            now = self._now()
            matched = self._match_calendar_event(now)
            if matched is not None:
                self._context = _context_from_event(matched, MeetingState.DETECTED)
                self._context.app = app_name
                logger.info(
                    "DETECTED via process %s, correlated to calendar event %s.",
                    app_name,
                    matched.title,
                )
            else:
                # Ad-hoc: process appeared with no calendar correlation.
                # Use the current wall clock as the start_time so the UI / next
                # phases have a usable timestamp (no calendar entry to copy).
                self._context = MeetingContext(
                    state=MeetingState.DETECTED,
                    app=app_name,
                    start_time=now,
                )
                logger.info("DETECTED via process %s (no calendar correlation).", app_name)
            ctx = _copy_context(self._context)
        _safe_call_each(self._on_detected, ctx)

    def _handle_process_ended(self, app_name: str) -> None:
        """React when the process behind the active meeting goes away.

        * RECORDING — stay in RECORDING, fire ``on_process_disappeared`` only.
          The recording continues; the UI / daemon decides whether to stop.
          (Rationale: real meeting apps occasionally crash or get restarted
          mid-call; auto-stopping would split the transcript.)
        * DETECTED / APPROACHING — no recording is in progress yet, so we
          quietly slip back to IDLE.
        * IDLE / PROCESSING — nothing to do.
        """
        with self._lock:
            if self._context.state in (MeetingState.IDLE, MeetingState.PROCESSING):
                return
            if self._context.app != app_name:
                return  # a different app exited; ignore.
            if self._context.state == MeetingState.RECORDING:
                logger.info(
                    "Meeting app %r exited during RECORDING — "
                    "advisory only, state stays RECORDING.",
                    app_name,
                )
                advisory_app: str | None = app_name
                ended_ctx: MeetingContext | None = None
            elif self._context.state in (MeetingState.DETECTED, MeetingState.APPROACHING):
                self._transition(MeetingState.IDLE)
                self._context = MeetingContext()
                advisory_app = None
                ended_ctx = None
            else:
                advisory_app = None
                ended_ctx = None
        if advisory_app is not None:
            _safe_call_each(self._on_process_gone, advisory_app)
        if ended_ctx is not None:
            _safe_call_each(self._on_ended, ended_ctx)

    # =====================================================================
    # Internals
    # =====================================================================
    def _match_calendar_event(self, when: datetime) -> CalendarEvent | None:
        """Return the calendar event whose start is within ±correlation of ``when``."""
        best: CalendarEvent | None = None
        best_delta: timedelta | None = None
        for event in self._pending_approaching.values():
            delta = abs(when - event.start)
            if delta <= self._correlation and (best_delta is None or delta < best_delta):
                best = event
                best_delta = delta
        # Also match against the current context if it's APPROACHING — we may
        # not have stored the event in pending if the same call set it up.
        ctx = self._context
        if (
            ctx.state == MeetingState.APPROACHING
            and ctx.start_time is not None
            and abs(when - ctx.start_time) <= self._correlation
        ):
            ctx_event = self._pending_approaching.get(ctx.calendar_event_id or "")
            if ctx_event is not None and (best is None or abs(when - ctx_event.start) < (best_delta or self._correlation + timedelta(seconds=1))):
                best = ctx_event
        return best

    def _transition(self, new_state: MeetingState) -> None:
        old = self._context.state
        self._context.state = new_state
        logger.info("State: %s -> %s", old.value, new_state.value)


# ============================================================================
# Helpers
# ============================================================================
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _context_from_event(event: CalendarEvent, state: MeetingState) -> MeetingContext:
    return MeetingContext(
        state=state,
        title=event.title,
        start_time=event.start,
        end_time=event.end,
        participants=list(event.attendees),
        meeting_link=event.meeting_link,
        calendar_event_id=event.id,
        is_tentative=event.is_tentative,
    )


def _copy_context(ctx: MeetingContext) -> MeetingContext:
    return MeetingContext(
        state=ctx.state,
        title=ctx.title,
        app=ctx.app,
        start_time=ctx.start_time,
        end_time=ctx.end_time,
        participants=list(ctx.participants),
        meeting_link=ctx.meeting_link,
        calendar_event_id=ctx.calendar_event_id,
        is_tentative=ctx.is_tentative,
    )


def _safe_call_each(callbacks: list[Callable[..., None]], *args: Any) -> None:
    for cb in callbacks:
        try:
            cb(*args)
        except Exception:
            logger.exception("Detector callback %r raised", cb)
