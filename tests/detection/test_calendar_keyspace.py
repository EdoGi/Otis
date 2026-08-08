"""Regression cover for audit finding B9 — id vs canonical_key mismatch.

``_pending_approaching`` is keyed by ``CalendarEvent.canonical_key`` (the
iCalUID), but three comparison sites used the per-calendar ``event.id``:

* the correlation fallback looked up ``_pending_approaching`` by id, so for any
  real Google event — which always carries an iCalUID — it returned None 100%
  of the time. That whole branch was dead in production and alive only in
  tests, because the test fixture defaulted ``ical_uid=None``.
* the "is this the meeting we're already tracking" checks compared ids, so a
  meeting invited on *both* a personal and a work account was treated as two
  different meetings depending on which poller spoke.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.detection.calendar_poller import CalendarEvent
from src.detection.detector import MeetingDetector, MeetingState


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)

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


def shared_event(clock: Clock, *, event_id: str) -> CalendarEvent:
    """The same real meeting, as seen by two different accounts.

    Google gives each account its own ``id`` for its copy but the same
    ``iCalUID`` for both — which is exactly what canonical_key exists for.
    """
    return CalendarEvent(
        id=event_id,
        title="Cross-invited standup",
        start=clock.now + timedelta(minutes=2),
        end=clock.now + timedelta(minutes=30),
        attendees=[],
        ical_uid="shared-uid@google.com",
    )


# ---------------------------------------------------------------------------
# Cross-account identity
# ---------------------------------------------------------------------------
def test_the_same_meeting_from_two_accounts_is_one_meeting(detector, clock) -> None:
    approaching: list = []
    detector.on_meeting_approaching(approaching.append)

    detector._handle_calendar_upcoming(shared_event(clock, event_id="personal-9"))
    detector._handle_calendar_upcoming(shared_event(clock, event_id="work-9"))

    assert len(approaching) == 1, (
        "one meeting invited on two accounts must not alert twice"
    )


def test_an_end_signal_from_the_other_account_clears_the_context(
    detector, clock
) -> None:
    """The context held the personal copy's id; the work poller's end signal
    carried a different id, so the APPROACHING context was never cleared."""
    detector._handle_calendar_upcoming(shared_event(clock, event_id="personal-9"))
    assert detector._context.state is MeetingState.APPROACHING

    detector._handle_calendar_ended(shared_event(clock, event_id="work-9"))

    assert detector._context.state is MeetingState.IDLE
    assert detector.get_current_meeting() is None


def test_context_carries_both_identifiers(detector, clock) -> None:
    detector._handle_calendar_upcoming(shared_event(clock, event_id="personal-9"))
    ctx = detector.get_current_meeting()
    assert ctx is not None
    assert ctx.calendar_event_id == "personal-9"
    assert ctx.calendar_key == "shared-uid@google.com"
    assert "calendar_key" in ctx.to_dict()


# ---------------------------------------------------------------------------
# The correlation fallback is live again
# ---------------------------------------------------------------------------
def test_correlation_works_for_events_that_have_an_ical_uid(detector, clock) -> None:
    """This is the case that was 100% dead in production: the lookup used the
    per-calendar id while the dict was keyed by iCalUID."""
    detector._handle_calendar_upcoming(shared_event(clock, event_id="personal-9"))
    clock.advance(minutes=2)

    detected: list = []
    detector.on_meeting_detected(detected.append)
    detector._handle_process_detected("zoom.us")

    assert len(detected) == 1
    assert detected[0].title == "Cross-invited standup", (
        "the process should have inherited the calendar title"
    )
    assert detected[0].calendar_key == "shared-uid@google.com"


def test_events_without_an_ical_uid_still_correlate(detector, clock) -> None:
    """canonical_key falls back to the id, so imported/non-RFC events work."""
    ev = CalendarEvent(
        id="no-uid-1",
        title="Imported meeting",
        start=clock.now + timedelta(minutes=2),
        end=clock.now + timedelta(minutes=30),
        attendees=[],
        ical_uid=None,
    )
    detector._handle_calendar_upcoming(ev)
    clock.advance(minutes=2)

    detected: list = []
    detector.on_meeting_detected(detected.append)
    detector._handle_process_detected("zoom.us")

    assert detected and detected[0].title == "Imported meeting"


def test_a_genuinely_different_meeting_is_still_a_new_alert(detector, clock) -> None:
    approaching: list = []
    detector.on_meeting_approaching(approaching.append)

    detector._handle_calendar_upcoming(shared_event(clock, event_id="personal-9"))
    other = CalendarEvent(
        id="other-1", title="Different meeting",
        start=clock.now + timedelta(minutes=3),
        end=clock.now + timedelta(minutes=40),
        attendees=[], ical_uid="other-uid@google.com",
    )
    detector._handle_calendar_upcoming(other)

    assert len(approaching) == 2
