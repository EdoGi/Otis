"""Meeting detection (process + calendar)."""

from src.detection.calendar_poller import (
    CalendarAuthError,
    CalendarEvent,
    GoogleCalendarPoller,
    build_pollers_from_config,
    parse_event,
)
from src.detection.detector import MeetingContext, MeetingDetector, MeetingState
from src.detection.process_monitor import DEFAULT_BROWSER_APPS, ProcessMonitor

__all__ = [
    "CalendarAuthError",
    "CalendarEvent",
    "DEFAULT_BROWSER_APPS",
    "GoogleCalendarPoller",
    "MeetingContext",
    "MeetingDetector",
    "MeetingState",
    "ProcessMonitor",
    "build_pollers_from_config",
    "parse_event",
]
