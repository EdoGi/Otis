"""Regression cover for audit finding B6 — APPROACHING and ENDED never exit.

Both are absorbing states in practice, because the signal that should leave
them can simply never arrive:

* APPROACHING's exit is a calendar "ended" event, lost on day rollover
  (reset_alerts() runs before the fetch), when the Mac sleeps through the
  meeting's end, when the event is deleted in Google Calendar, and during
  off-hours when the poller is stopped. Result: "Meeting in 2 min" pinned in
  the menu bar indefinitely and a permanently stale get_current_meeting().
* ENDED's only exit is transcription_started(), which the daemon skips
  entirely when there is no pipeline or no metadata. From ENDED every future
  "Meeting in 2 min" alert is silently dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.detection.calendar_poller import CalendarEvent
from src.detection.detector import (
    APPROACHING_GRACE_SECONDS,
    ENDED_MAX_SECONDS,
    MeetingDetector,
    MeetingState,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw) -> None:
        self.now += timedelta(**kw)


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
def detector(clock: Clock) -> MeetingDetector:
    return MeetingDetector(process_monitor=None, calendar_pollers=[], clock=clock)


def event(clock: Clock, *, minutes: int = 30) -> CalendarEvent:
    return CalendarEvent(
        id="evt-1",
        title="Roadmap sync",
        start=clock.now + timedelta(minutes=2),
        end=clock.now + timedelta(minutes=minutes),
        attendees=[],
        ical_uid="evt-1@google.com",
    )


# ---------------------------------------------------------------------------
# APPROACHING
# ---------------------------------------------------------------------------
def test_approaching_survives_until_the_meeting_should_have_ended(detector, clock):
    detector._handle_calendar_upcoming(event(clock))
    assert detector._context.state is MeetingState.APPROACHING

    # Still inside the meeting: the watchdog (which runs on every read) must
    # leave the context alone.
    clock.advance(minutes=20)
    current = detector.get_current_meeting()
    assert current is not None
    assert current.title == "Roadmap sync"

    # And just past the end, but inside the grace window.
    clock.advance(minutes=12)
    assert detector.get_current_meeting() is not None


def test_approaching_is_released_after_the_grace_period(detector, clock):
    detector._handle_calendar_upcoming(event(clock, minutes=30))
    assert detector._context.state is MeetingState.APPROACHING

    clock.advance(seconds=30 * 60 + APPROACHING_GRACE_SECONDS + 60)
    detector._handle_calendar_upcoming(event(clock, minutes=999))  # any signal

    # The stale "Meeting in 2 min" context is gone.
    assert detector._context.title != "Roadmap sync"


def test_stale_approaching_does_not_leak_through_get_current_meeting(detector, clock):
    detector._handle_calendar_upcoming(event(clock, minutes=30))
    assert detector.get_current_meeting() is not None

    clock.advance(hours=6)
    assert detector.get_current_meeting() is None, (
        "a meeting that ended six hours ago is not the current meeting"
    )


def test_approaching_without_an_end_time_is_still_bounded(detector, clock):
    ev = CalendarEvent(
        id="e", title="No end", start=clock.now + timedelta(minutes=2),
        end=clock.now + timedelta(minutes=2), attendees=[], ical_uid="e@g",
    )
    detector._handle_calendar_upcoming(ev)
    clock.advance(hours=5)
    assert detector.get_current_meeting() is None


# ---------------------------------------------------------------------------
# ENDED
# ---------------------------------------------------------------------------
def test_ended_recovers_when_transcription_never_starts(detector, clock):
    """daemon.py returns early with no pipeline, so ENDED was permanent — and
    from ENDED every future heads-up was silently dropped."""
    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    detector.user_stopped_recording()
    assert detector._context.state is MeetingState.ENDED

    clock.advance(seconds=ENDED_MAX_SECONDS + 60)
    detector._handle_calendar_upcoming(event(clock))

    assert detector._context.state is not MeetingState.ENDED


def test_a_meeting_alert_works_again_after_the_watchdog_fires(detector, clock):
    approaching: list = []
    detector.on_meeting_approaching(approaching.append)

    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    detector.user_stopped_recording()

    clock.advance(seconds=ENDED_MAX_SECONDS + 60)
    detector._handle_calendar_upcoming(event(clock))

    assert approaching, "future heads-ups must survive a stuck ENDED"


def test_ended_outlasts_a_full_deferral(detector, clock):
    """The daemon defers transcription *before* transcription_started(), so
    ENDED legitimately persists for the whole 2 h deferral window."""
    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    detector.user_stopped_recording()

    clock.advance(seconds=7200)  # pipeline.DEFER_MAX_SECONDS
    detector._handle_calendar_upcoming(event(clock))
    assert detector._context.state is MeetingState.ENDED, (
        "the watchdog must not cut a legitimate deferral short"
    )


# ---------------------------------------------------------------------------
# The watchdog must never touch live work
# ---------------------------------------------------------------------------
def test_recording_is_never_interrupted_by_the_watchdog(detector, clock):
    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    clock.advance(hours=12)
    detector._handle_calendar_upcoming(event(clock))
    assert detector._context.state is MeetingState.RECORDING


def test_processing_is_never_interrupted_by_the_watchdog(detector, clock):
    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    detector.user_stopped_recording()
    detector.transcription_started()
    assert detector._context.state is MeetingState.PROCESSING

    clock.advance(hours=12)
    detector._handle_calendar_upcoming(event(clock))
    assert detector._context.state is MeetingState.PROCESSING
