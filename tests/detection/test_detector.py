"""Tests for src/detection/detector.py.

The orchestrator depends only on ``ProcessMonitor`` and
``GoogleCalendarPoller`` interfaces — both are easy to fake. We don't start any
background threads here; we drive the detector directly through the handler
methods that production code reaches via callbacks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.detection.calendar_poller import CalendarEvent
from src.detection.detector import (
    MeetingContext,
    MeetingDetector,
    MeetingState,
)
from src.detection.process_monitor import ProcessMonitor


# ============================================================================
# Fakes
# ============================================================================
class _FakeProcessMonitor:
    def __init__(self) -> None:
        self.detected_cbs: list = []
        self.ended_cbs: list = []
        self.started = False
        self.stopped = False

    def on_meeting_detected(self, cb): self.detected_cbs.append(cb)
    def on_meeting_ended(self, cb): self.ended_cbs.append(cb)
    def start(self): self.started = True
    def stop(self): self.stopped = True

    def fire_detected(self, name: str) -> None:
        for cb in self.detected_cbs:
            cb(name)

    def fire_ended(self, name: str) -> None:
        for cb in self.ended_cbs:
            cb(name)


class _FakeCalendarPoller:
    def __init__(self) -> None:
        self.upcoming_cbs: list = []
        self.ended_cbs: list = []
        self.started = False
        self.stopped = False

    def on_upcoming_meeting(self, cb): self.upcoming_cbs.append(cb)
    def on_meeting_should_end(self, cb): self.ended_cbs.append(cb)
    def on_needs_reauth(self, cb): pass
    def start(self): self.started = True
    def stop(self): self.stopped = True

    def fire_upcoming(self, ev: CalendarEvent) -> None:
        for cb in self.upcoming_cbs:
            cb(ev)

    def fire_ended(self, ev: CalendarEvent) -> None:
        for cb in self.ended_cbs:
            cb(ev)


def _evt(
    when: datetime,
    *,
    eid: str = "e1",
    title: str = "Standup",
    ical_uid: str | None = None,
) -> CalendarEvent:
    return CalendarEvent(
        id=eid,
        title=title,
        start=when,
        end=when + timedelta(minutes=30),
        attendees=[{"email": "a@x", "name": "Alice", "responseStatus": "accepted"}],
        meeting_link="https://meet.google.com/abc",
        ical_uid=ical_uid,
    )


def _make(
    *,
    now: datetime,
    process_monitor=True,
    calendar_poller=True,
    correlation_minutes: float = 5.0,
) -> tuple[MeetingDetector, _FakeProcessMonitor | None, _FakeCalendarPoller | None]:
    pm = _FakeProcessMonitor() if process_monitor else None
    cp = _FakeCalendarPoller() if calendar_poller else None
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_poller=cp,  # type: ignore[arg-type]
        correlation_minutes=correlation_minutes,
        clock=lambda: now,
    )
    return detector, pm, cp


# ============================================================================
# State transitions
# ============================================================================
def test_initial_state_is_idle() -> None:
    detector, _, _ = _make(now=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc))
    assert detector.get_state() == MeetingState.IDLE
    assert detector.get_current_meeting() is None


def test_calendar_alert_drives_idle_to_approaching() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, _, cp = _make(now=now)
    captured: list[MeetingContext] = []
    detector.on_meeting_approaching(captured.append)

    cp.fire_upcoming(_evt(now + timedelta(minutes=2), title="Sync"))

    assert detector.get_state() == MeetingState.APPROACHING
    assert len(captured) == 1
    ctx = detector.get_current_meeting()
    assert ctx is not None
    assert ctx.title == "Sync"
    assert ctx.meeting_link == "https://meet.google.com/abc"


def test_get_current_meeting_for_adhoc_has_now_as_start_time() -> None:
    """get_current_meeting() during an ad-hoc DETECTED meeting matches the spec."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)
    pm.fire_detected("zoom.us")
    ctx = detector.get_current_meeting()
    assert ctx is not None
    assert ctx.title is None
    assert ctx.app == "zoom.us"
    assert ctx.start_time == now
    assert ctx.participants == []


def test_process_alone_drives_idle_to_detected() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)
    captured: list[MeetingContext] = []
    detector.on_meeting_detected(captured.append)

    pm.fire_detected("zoom.us")

    assert detector.get_state() == MeetingState.DETECTED
    assert captured[0].app == "zoom.us"
    assert captured[0].title is None  # no calendar correlation
    # Ad-hoc detection must still stamp a start_time so Phase 4 has something
    # to anchor the audio recording against.
    assert captured[0].start_time == now


def test_calendar_then_correlated_process_is_one_meeting() -> None:
    """An approaching event + a process within ±correlation should NOT fire two events."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, cp = _make(now=now)
    approach: list[MeetingContext] = []
    detected: list[MeetingContext] = []
    detector.on_meeting_approaching(approach.append)
    detector.on_meeting_detected(detected.append)

    # Calendar fires first; the meeting starts in 1 minute.
    cp.fire_upcoming(_evt(now + timedelta(minutes=1), title="Project Sync"))
    # Process arrives now (within the correlation window).
    pm.fire_detected("zoom.us")

    assert len(approach) == 1
    assert len(detected) == 1
    ctx = detector.get_current_meeting()
    assert ctx is not None
    assert ctx.state == MeetingState.DETECTED
    assert ctx.title == "Project Sync"
    assert ctx.app == "zoom.us"


def test_uncorrelated_process_replaces_distant_calendar_context() -> None:
    """When the process arrives too far from the calendar event, treat as ad-hoc."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, cp = _make(now=now, correlation_minutes=2.0)

    cp.fire_upcoming(_evt(now + timedelta(minutes=30), title="Later"))
    assert detector.get_state() == MeetingState.APPROACHING

    pm.fire_detected("zoom.us")
    ctx = detector.get_current_meeting()
    assert ctx is not None
    assert ctx.state == MeetingState.DETECTED
    # Outside correlation window ⇒ ad-hoc, no calendar metadata copied.
    assert ctx.app == "zoom.us"
    assert ctx.title is None


def test_user_starts_and_stops_recording_drives_state() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)

    pm.fire_detected("zoom.us")
    assert detector.get_state() == MeetingState.DETECTED

    detector.user_started_recording()
    assert detector.get_state() == MeetingState.RECORDING

    ended: list[MeetingContext] = []
    detector.on_meeting_ended(ended.append)
    detector.user_stopped_recording()
    assert detector.get_state() == MeetingState.ENDED
    assert ended and ended[0].app == "zoom.us"


def test_manual_start_from_idle_creates_minimal_context() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, _, _ = _make(now=now)
    detector.user_started_recording()
    assert detector.get_state() == MeetingState.RECORDING
    ctx = detector.get_current_meeting()
    assert ctx is not None and ctx.title is None and ctx.app is None


def test_process_exit_during_recording_is_advisory_only() -> None:
    """A meeting app crashing mid-recording must NOT auto-stop the recording.

    Rationale: real-world apps occasionally crash and reconnect; auto-stopping
    would split the transcript. The detector fires ``on_process_disappeared``
    so the UI can warn the user, but state stays RECORDING. The user decides
    when to actually stop.
    """
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)

    pm.fire_detected("zoom.us")
    detector.user_started_recording()
    assert detector.get_state() == MeetingState.RECORDING

    advisory: list[str] = []
    ended: list[MeetingContext] = []
    detector.on_process_disappeared(advisory.append)
    detector.on_meeting_ended(ended.append)

    pm.fire_ended("zoom.us")
    # State unchanged — recording continues.
    assert detector.get_state() == MeetingState.RECORDING
    # Advisory callback fired exactly once.
    assert advisory == ["zoom.us"]
    # No on_meeting_ended yet — that only fires when the user actually stops.
    assert ended == []

    # Now the user explicitly stops; only THEN does on_meeting_ended fire.
    detector.user_stopped_recording()
    assert detector.get_state() == MeetingState.ENDED
    assert ended and ended[0].app == "zoom.us"


def test_process_exit_during_detected_returns_to_idle_silently() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)

    detected_calls: list[MeetingContext] = []
    ended_calls: list[MeetingContext] = []
    detector.on_meeting_detected(detected_calls.append)
    detector.on_meeting_ended(ended_calls.append)

    pm.fire_detected("zoom.us")
    pm.fire_ended("zoom.us")

    assert detector.get_state() == MeetingState.IDLE
    assert detector.get_current_meeting() is None
    assert ended_calls == []  # only fires when we were actually RECORDING


def test_calendar_alert_during_recording_does_not_demote_state() -> None:
    """Regression: an upcoming calendar alert that fires mid-recording, followed
    by a routine process re-detection, must NOT demote state from RECORDING to
    DETECTED. (Pre-fix, the process handler unconditionally overwrote context
    when a calendar event correlated, silently losing the recording state while
    the recorder thread kept writing audio.)
    """
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, cp = _make(now=now)

    # Recording underway via manual start.
    detector.user_started_recording()
    assert detector.get_state() == MeetingState.RECORDING

    detected_calls: list[MeetingContext] = []
    detector.on_meeting_detected(detected_calls.append)

    # A new calendar event's 2-min upcoming alert fires while we're recording.
    cp.fire_upcoming(_evt(now + timedelta(minutes=1), title="Surprise call"))
    # State must stay RECORDING — the upcoming handler is a no-op here.
    assert detector.get_state() == MeetingState.RECORDING

    # The process monitor's routine 30s rescan fires again. With the old bug
    # this would correlate against the just-stored calendar event and rewrite
    # state to DETECTED. Now it must remain RECORDING.
    pm.fire_detected("zoom.us")
    assert detector.get_state() == MeetingState.RECORDING
    # And no on_meeting_detected callback should fire — we're already recording.
    assert detected_calls == []
    # The app name should still be recorded onto the context as a side note.
    ctx = detector.get_current_meeting()
    assert ctx is not None and ctx.app == "zoom.us"


def test_calendar_ended_clears_approaching_when_no_process_came() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, _, cp = _make(now=now)
    ev = _evt(now + timedelta(minutes=1))
    cp.fire_upcoming(ev)
    assert detector.get_state() == MeetingState.APPROACHING
    cp.fire_ended(ev)
    assert detector.get_state() == MeetingState.IDLE


def test_transcription_lifecycle() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)
    pm.fire_detected("zoom.us")
    detector.user_started_recording()
    detector.user_stopped_recording()
    detector.transcription_started()
    assert detector.get_state() == MeetingState.PROCESSING
    detector.transcription_finished()
    assert detector.get_state() == MeetingState.IDLE
    assert detector.get_current_meeting() is None


# ============================================================================
# Lifecycle wiring
# ============================================================================
def test_start_stop_propagates_to_components() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, cp = _make(now=now)
    detector.start()
    assert pm.started and cp.started
    detector.stop()
    assert pm.stopped and cp.stopped


def test_callback_exception_does_not_break_others() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, _, cp = _make(now=now)
    fine: list[MeetingContext] = []

    def bad(_): raise RuntimeError("boom")

    detector.on_meeting_approaching(bad)
    detector.on_meeting_approaching(fine.append)
    cp.fire_upcoming(_evt(now + timedelta(minutes=1)))
    assert len(fine) == 1


def test_get_current_meeting_returns_a_copy() -> None:
    """Mutating the snapshot must not change internal state."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    detector, pm, _ = _make(now=now)
    pm.fire_detected("zoom.us")
    snap = detector.get_current_meeting()
    assert snap is not None
    snap.app = "tampered"
    fresh = detector.get_current_meeting()
    assert fresh is not None and fresh.app == "zoom.us"


def test_multiple_calendar_pollers_share_same_callbacks() -> None:
    """Both pollers should drive the same MeetingDetector state."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    pm = _FakeProcessMonitor()
    personal = _FakeCalendarPoller()
    work = _FakeCalendarPoller()

    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[personal, work],  # type: ignore[arg-type]
        clock=lambda: now,
    )

    approaches: list[MeetingContext] = []
    detector.on_meeting_approaching(approaches.append)

    # Personal account fires first — APPROACHING.
    personal.fire_upcoming(_evt(now + timedelta(minutes=2), eid="p-1", title="Personal sync"))
    assert detector.get_state() == MeetingState.APPROACHING
    assert approaches[-1].title == "Personal sync"

    # Work account fires next — should also be picked up (replacing context
    # because it's still APPROACHING).
    work.fire_upcoming(_evt(now + timedelta(minutes=1), eid="w-1", title="Work sync"))
    assert approaches[-1].title == "Work sync"
    assert len(approaches) == 2


def test_cross_invited_meeting_only_fires_once() -> None:
    """The same iCalUID arriving from two pollers must not double-notify."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    pm = _FakeProcessMonitor()
    personal = _FakeCalendarPoller()
    work = _FakeCalendarPoller()

    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[personal, work],  # type: ignore[arg-type]
        clock=lambda: now,
    )
    approaches: list[MeetingContext] = []
    detector.on_meeting_approaching(approaches.append)

    # Two CalendarEvent objects with different per-calendar ids but the SAME
    # iCalUID — i.e. one logical meeting that both accounts were invited to.
    shared_uid = "shared-meeting-uid@google.com"
    personal.fire_upcoming(
        _evt(now + timedelta(minutes=2), eid="p-9", title="Customer call", ical_uid=shared_uid)
    )
    work.fire_upcoming(
        _evt(now + timedelta(minutes=2), eid="w-9", title="Customer call", ical_uid=shared_uid)
    )

    assert len(approaches) == 1
    assert detector.get_state() == MeetingState.APPROACHING


def test_distinct_meetings_from_two_accounts_both_fire() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    personal = _FakeCalendarPoller()
    work = _FakeCalendarPoller()
    detector = MeetingDetector(
        process_monitor=None,
        calendar_pollers=[personal, work],  # type: ignore[arg-type]
        clock=lambda: now,
    )
    approaches: list[MeetingContext] = []
    detector.on_meeting_approaching(approaches.append)

    personal.fire_upcoming(
        _evt(now + timedelta(minutes=1), eid="p-1", title="Friend lunch",
             ical_uid="uid-friend")
    )
    work.fire_upcoming(
        _evt(now + timedelta(minutes=2), eid="w-1", title="Standup",
             ical_uid="uid-standup")
    )

    assert [a.title for a in approaches] == ["Friend lunch", "Standup"]


def test_detector_start_stop_propagates_to_all_pollers() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    pm = _FakeProcessMonitor()
    a = _FakeCalendarPoller()
    b = _FakeCalendarPoller()
    detector = MeetingDetector(
        process_monitor=pm,  # type: ignore[arg-type]
        calendar_pollers=[a, b],  # type: ignore[arg-type]
        clock=lambda: now,
    )
    detector.start()
    assert pm.started and a.started and b.started
    detector.stop()
    assert pm.stopped and a.stopped and b.stopped


def test_calendar_poller_singular_and_plural_can_combine() -> None:
    """Back-compat: ``calendar_poller=`` and ``calendar_pollers=`` can coexist."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    a = _FakeCalendarPoller()
    b = _FakeCalendarPoller()
    detector = MeetingDetector(
        process_monitor=None,
        calendar_poller=a,  # type: ignore[arg-type]
        calendar_pollers=[b],  # type: ignore[arg-type]
        clock=lambda: now,
    )
    assert len(detector.calendar_pollers) == 2


def test_real_process_monitor_can_attach() -> None:
    """Detector must accept a real ProcessMonitor instance, not just the fake."""
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    real_pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: iter([]),
        mic_probe=lambda: False,
    )
    detector = MeetingDetector(
        process_monitor=real_pm,
        calendar_poller=None,
        clock=lambda: now,
    )
    assert detector.get_state() == MeetingState.IDLE
