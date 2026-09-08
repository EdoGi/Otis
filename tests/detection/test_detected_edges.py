"""Regression cover for audit finding A3 (second half) — DETECTED re-fires.

``ProcessMonitor`` deliberately re-fires ``on_detected`` every 30 s for a
still-running app, so that a Zoom left open can re-trigger a meeting after the
previous recording finished. The detector took that heartbeat at face value:
it rebuilt the whole context and re-fired ``on_meeting_detected`` every single
time.

The default whitelist contains Slack, which runs all day. That meant an INFO
line and a UI state change every 30 s — 2 880 log lines a day from one app —
and the detector could effectively never settle.

The heartbeat must stay (back-to-back meetings depend on it) but the detector
has to treat it as news only on a real edge.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.detection.calendar_poller import CalendarEvent
from src.detection.detector import MeetingDetector, MeetingState


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


@pytest.fixture()
def detections(detector: MeetingDetector) -> list:
    seen: list = []
    detector.on_meeting_detected(seen.append)
    return seen


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------
def test_slack_running_all_day_fires_once(detector, detections, clock) -> None:
    """Simulate 8 hours of 30-second heartbeats from a permanently-open app."""
    for _ in range(8 * 60 * 2):  # 960 heartbeats
        detector._handle_process_detected("slack")
        clock.advance(seconds=30)

    assert len(detections) == 1, (
        f"one DETECTED edge expected, got {len(detections)} — "
        "this is the 2 880-lines-a-day bug"
    )


def test_state_stays_detected_across_heartbeats(detector, clock) -> None:
    detector._handle_process_detected("slack")
    assert detector._context.state is MeetingState.DETECTED
    for _ in range(10):
        detector._handle_process_detected("slack")
    assert detector._context.state is MeetingState.DETECTED


def test_a_different_app_is_a_real_edge(detector, detections) -> None:
    detector._handle_process_detected("slack")
    detector._handle_process_detected("zoom.us")
    assert len(detections) == 2
    assert [d.app for d in detections] == ["slack", "zoom.us"]


# ---------------------------------------------------------------------------
# The heartbeat must still do its job
# ---------------------------------------------------------------------------
def test_app_can_retrigger_after_a_recording_ends(detector, detections) -> None:
    """The reason the 30 s re-fire exists: back-to-back meetings."""
    detector._handle_process_detected("zoom.us")
    assert len(detections) == 1

    detector.user_started_recording()
    detector.user_stopped_recording()
    detector.transcription_started()
    detector.transcription_finished()
    assert detector._context.state is MeetingState.IDLE

    # Zoom is still open; the next heartbeat is a genuine new meeting.
    detector._handle_process_detected("zoom.us")
    assert len(detections) == 2


def test_heartbeat_upgrades_an_adhoc_context_when_calendar_catches_up(
    detector, detections, clock
) -> None:
    """A process can be seen before its calendar event arrives. When the
    correlation shows up, that IS news and should fire once more."""
    detector._handle_process_detected("zoom.us")
    assert len(detections) == 1
    assert detections[0].title is None

    event = CalendarEvent(
        id="evt-1",
        title="Roadmap sync",
        start=clock.now,
        end=clock.now + timedelta(minutes=30),
        attendees=[],
        ical_uid="evt-1@google.com",
    )
    detector._handle_calendar_upcoming(event)
    detector._handle_process_detected("zoom.us")

    assert len(detections) == 2
    assert detections[-1].title == "Roadmap sync"


def test_no_further_fire_once_correlated(detector, detections, clock) -> None:
    event = CalendarEvent(
        id="evt-1",
        title="Roadmap sync",
        start=clock.now,
        end=clock.now + timedelta(minutes=30),
        attendees=[],
        ical_uid="evt-1@google.com",
    )
    detector._handle_calendar_upcoming(event)
    detector._handle_process_detected("zoom.us")
    before = len(detections)
    for _ in range(20):
        detector._handle_process_detected("zoom.us")
    assert len(detections) == before


def test_recording_state_is_still_never_demoted(detector, detections) -> None:
    detector._handle_process_detected("zoom.us")
    detector.user_started_recording()
    assert detector._context.state is MeetingState.RECORDING
    for _ in range(10):
        detector._handle_process_detected("slack")
    assert detector._context.state is MeetingState.RECORDING
