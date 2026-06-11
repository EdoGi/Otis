"""Tests for src/detection/calendar_poller.py.

We exercise the *parsing* surface (a pure function) and the *polling* loop with
a fake calendar service. The OAuth flow itself isn't tested — it requires a
browser round-trip; ``setup_google_cal.sh`` runs a real auth test on the
user's machine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.detection.calendar_poller import (
    CalendarEvent,
    GoogleCalendarPoller,
    _extract_meeting_link,
    build_pollers_from_config,
    parse_event,
)


# ============================================================================
# Event parsing — pure function, fast tests.
# ============================================================================
def _event_with(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "evt-1",
        "summary": "Standup",
        "start": {"dateTime": "2026-05-09T09:00:00+02:00"},
        "end": {"dateTime": "2026-05-09T09:30:00+02:00"},
        "attendees": [
            {"email": "me@example.com", "self": True, "responseStatus": "accepted"},
            {"email": "alice@example.com", "displayName": "Alice", "responseStatus": "accepted"},
        ],
    }
    base.update(overrides)
    return base


def test_parse_event_extracts_basics() -> None:
    ev = parse_event(_event_with())
    assert ev is not None
    assert ev.id == "evt-1"
    assert ev.title == "Standup"
    assert ev.start.tzinfo is not None
    assert ev.end > ev.start
    assert {a["email"] for a in ev.attendees} == {"me@example.com", "alice@example.com"}
    assert ev.is_tentative is False


def test_parse_event_drops_all_day() -> None:
    raw = _event_with(start={"date": "2026-05-09"}, end={"date": "2026-05-10"})
    assert parse_event(raw) is None


def test_parse_event_drops_declined() -> None:
    raw = _event_with(
        attendees=[{"email": "me@example.com", "self": True, "responseStatus": "declined"}]
    )
    assert parse_event(raw) is None


def test_parse_event_keeps_tentative_with_flag() -> None:
    raw = _event_with(
        attendees=[{"email": "me@example.com", "self": True, "responseStatus": "tentative"}]
    )
    ev = parse_event(raw)
    assert ev is not None
    assert ev.is_tentative is True


def test_parse_event_handles_zulu_time() -> None:
    raw = _event_with(
        start={"dateTime": "2026-05-09T09:00:00Z"},
        end={"dateTime": "2026-05-09T09:30:00Z"},
    )
    ev = parse_event(raw)
    assert ev is not None
    assert ev.start.tzinfo == timezone.utc


def test_meeting_link_from_conference_data() -> None:
    raw = _event_with(
        conferenceData={
            "entryPoints": [
                {"entryPointType": "more", "uri": "https://example.com/info"},
                {"entryPointType": "video", "uri": "https://meet.google.com/abc-defg-hij"},
            ]
        }
    )
    ev = parse_event(raw)
    assert ev is not None
    assert ev.meeting_link == "https://meet.google.com/abc-defg-hij"


def test_meeting_link_falls_back_to_description() -> None:
    raw = _event_with(
        description=(
            "Hop in here: https://us02web.zoom.us/j/123456789\n"
            "Or call the bridge: +1 555 0100"
        )
    )
    ev = parse_event(raw)
    assert ev is not None
    assert ev.meeting_link is not None
    assert "zoom.us" in ev.meeting_link


def test_meeting_link_falls_back_to_location() -> None:
    raw = _event_with(location="https://teams.microsoft.com/l/meetup-join/abc")
    ev = parse_event(raw)
    assert ev is not None
    assert ev.meeting_link is not None
    assert "teams.microsoft.com" in ev.meeting_link


def test_meeting_link_none_when_no_url() -> None:
    raw = _event_with(description="Office, room 4B")
    assert parse_event(raw).meeting_link is None  # type: ignore[union-attr]


def test_extract_meeting_link_no_match_returns_none() -> None:
    assert _extract_meeting_link({}) is None


# ============================================================================
# Poll cycle — fake the googleapiclient service.
# ============================================================================
class _FakeService:
    """Stand-in for ``googleapiclient.discovery.build('calendar', ...)`` output.

    Supports either a single item list (legacy) or a per-calendar mapping.
    """

    def __init__(
        self,
        items: list[dict[str, Any]] | None = None,
        items_by_calendar: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._items = items or []
        self._items_by_calendar = items_by_calendar or {}
        self.last_kwargs: dict[str, Any] | None = None
        self.calendars_queried: list[str] = []

    def events(self) -> "_FakeService":
        return self

    def list(self, **kwargs: Any) -> "_FakeService":
        self.last_kwargs = kwargs
        cal = kwargs.get("calendarId")
        if cal is not None:
            self.calendars_queried.append(cal)
        return self

    def execute(self) -> dict[str, Any]:
        cal = (self.last_kwargs or {}).get("calendarId")
        if cal in self._items_by_calendar:
            return {"items": list(self._items_by_calendar[cal])}
        return {"items": list(self._items)}


def _fixed_clock(when: datetime):
    return lambda: when


def _make_poller(items, *, now: datetime) -> tuple[GoogleCalendarPoller, _FakeService]:
    poller = GoogleCalendarPoller(
        credentials_path="/nonexistent/credentials.json",
        token_path="/nonexistent/token.json",
        clock=_fixed_clock(now),
    )
    fake = _FakeService(items)
    poller._service = fake  # type: ignore[attr-defined]
    return poller, fake


def test_fetch_today_events_returns_parsed_events() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    soon = (now + timedelta(minutes=30)).isoformat()
    later = (now + timedelta(minutes=60)).isoformat()
    poller, _ = _make_poller(
        [
            {
                "id": "a",
                "summary": "Sync",
                "start": {"dateTime": soon},
                "end": {"dateTime": later},
            }
        ],
        now=now,
    )
    events = poller.fetch_today_events()
    assert len(events) == 1
    assert events[0].title == "Sync"


def test_upcoming_within_window() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    items = [
        {
            "id": "in-1",
            "summary": "In window",
            "start": {"dateTime": (now + timedelta(minutes=1)).isoformat()},
            "end": {"dateTime": (now + timedelta(minutes=30)).isoformat()},
        },
        {
            "id": "in-2",
            "summary": "Out of window",
            "start": {"dateTime": (now + timedelta(minutes=10)).isoformat()},
            "end": {"dateTime": (now + timedelta(minutes=40)).isoformat()},
        },
    ]
    poller, _ = _make_poller(items, now=now)
    titles = [e.title for e in poller.upcoming_within(2)]
    assert titles == ["In window"]


def test_poll_cycle_fires_upcoming_once_then_end_once() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    items = [
        {
            "id": "evt",
            "summary": "Test",
            "start": {"dateTime": (now + timedelta(minutes=1)).isoformat()},
            "end": {"dateTime": (now + timedelta(minutes=2)).isoformat()},
        }
    ]
    poller, _ = _make_poller(items, now=now)

    upcoming: list[CalendarEvent] = []
    ended: list[CalendarEvent] = []
    poller.on_upcoming_meeting(upcoming.append)
    poller.on_meeting_should_end(ended.append)

    poller._poll_cycle()
    poller._poll_cycle()
    assert len(upcoming) == 1, "upcoming should fire exactly once"
    assert ended == []

    # Advance the clock past the event end and poll again.
    poller._now = _fixed_clock(now + timedelta(minutes=3))  # type: ignore[attr-defined]
    poller._poll_cycle()
    poller._poll_cycle()
    assert len(ended) == 1, "should_end should fire exactly once"


def test_fetch_swallows_api_errors() -> None:
    """A transient API outage must not crash the poller."""

    class _Boom:
        def events(self):
            return self

        def list(self, **kwargs: Any):
            return self

        def execute(self) -> dict[str, Any]:
            raise RuntimeError("HTTP 500")

    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    poller = GoogleCalendarPoller(
        credentials_path="/nope.json",
        clock=_fixed_clock(now),
    )
    poller._service = _Boom()  # type: ignore[attr-defined]
    assert poller.fetch_today_events() == []


def test_fetch_drops_cached_service_when_all_calendars_fail() -> None:
    """A dead service (revoked token etc.) must be rebuilt, not cached forever."""

    class _Boom:
        def events(self):
            return self

        def list(self, **kwargs: Any):
            return self

        def execute(self) -> dict[str, Any]:
            raise RuntimeError("invalid_grant")

    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    poller = GoogleCalendarPoller(
        credentials_path="/nope.json",
        clock=_fixed_clock(now),
    )
    poller._service = _Boom()  # type: ignore[attr-defined]
    assert poller.fetch_today_events() == []
    assert poller._service is None  # type: ignore[attr-defined]


def test_fetch_keeps_service_when_only_one_calendar_fails() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)

    class _PartialService(_FakeService):
        def execute(self) -> dict[str, Any]:
            cal = (self.last_kwargs or {}).get("calendarId")
            if cal == "broken@x.com":
                raise RuntimeError("HTTP 404")
            return super().execute()

    poller = GoogleCalendarPoller(
        credentials_path="/nope.json",
        calendar_ids=["primary", "broken@x.com"],
        clock=_fixed_clock(now),
    )
    fake = _PartialService(
        [
            {
                "id": "a",
                "summary": "Sync",
                "start": {"dateTime": (now + timedelta(minutes=5)).isoformat()},
                "end": {"dateTime": (now + timedelta(minutes=35)).isoformat()},
            }
        ]
    )
    poller._service = fake  # type: ignore[attr-defined]
    events = poller.fetch_today_events()
    assert [e.id for e in events] == ["a"]
    assert poller._service is fake  # type: ignore[attr-defined]


def test_poll_cycle_resets_alert_bookkeeping_on_date_rollover() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
    items = [
        {
            "id": "evt",
            "summary": "Daily",
            "start": {"dateTime": (now + timedelta(minutes=1)).isoformat()},
            "end": {"dateTime": (now + timedelta(minutes=2)).isoformat()},
        }
    ]
    poller, _ = _make_poller(items, now=now)
    poller._poll_cycle()
    assert poller._alerted_upcoming  # type: ignore[attr-defined]

    # Next day: stale ids from yesterday are cleared.
    tomorrow = now + timedelta(days=1)
    poller._now = _fixed_clock(tomorrow)  # type: ignore[attr-defined]
    poller._service = _FakeService([])  # type: ignore[attr-defined]
    poller._poll_cycle()
    assert not poller._alerted_upcoming  # type: ignore[attr-defined]
    assert not poller._alerted_ended  # type: ignore[attr-defined]


def test_fetch_iterates_multiple_calendars_and_dedupes() -> None:
    """When ``calendar_ids`` lists multiple calendars, each is fetched and
    duplicate ids (the same event present on both) appear only once.
    """
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)

    common = {
        "id": "shared-1",
        "summary": "Cross-calendar invite",
        "start": {"dateTime": (now + timedelta(minutes=10)).isoformat()},
        "end": {"dateTime": (now + timedelta(minutes=40)).isoformat()},
    }
    personal_only = {
        "id": "personal-1",
        "summary": "Personal stuff",
        "start": {"dateTime": (now + timedelta(minutes=5)).isoformat()},
        "end": {"dateTime": (now + timedelta(minutes=20)).isoformat()},
    }
    work_only = {
        "id": "work-1",
        "summary": "Standup",
        "start": {"dateTime": (now + timedelta(minutes=60)).isoformat()},
        "end": {"dateTime": (now + timedelta(minutes=90)).isoformat()},
    }

    fake = _FakeService(
        items_by_calendar={
            "primary": [personal_only, common],
            "work@company.com": [work_only, common],
        }
    )
    poller = GoogleCalendarPoller(
        credentials_path="/nope.json",
        calendar_ids=["primary", "work@company.com"],
        clock=_fixed_clock(now),
    )
    poller._service = fake  # type: ignore[attr-defined]
    events = poller.fetch_today_events()

    assert fake.calendars_queried == ["primary", "work@company.com"]
    titles = [e.title for e in events]
    # All three distinct events present, sorted by start time.
    assert titles == ["Personal stuff", "Cross-calendar invite", "Standup"]
    # No duplicate of the common event.
    assert len(events) == 3


def test_fetch_continues_when_one_calendar_fails() -> None:
    now = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)

    class _PartiallyBrokenService:
        def __init__(self):
            self.last = {}

        def events(self):
            return self

        def list(self, **kwargs):
            self.last = kwargs
            return self

        def execute(self):
            cal = self.last.get("calendarId")
            if cal == "broken":
                raise RuntimeError("HTTP 403 forbidden")
            return {
                "items": [
                    {
                        "id": "ok-1",
                        "summary": "Healthy event",
                        "start": {"dateTime": (now + timedelta(minutes=5)).isoformat()},
                        "end": {"dateTime": (now + timedelta(minutes=20)).isoformat()},
                    }
                ]
            }

    poller = GoogleCalendarPoller(
        credentials_path="/nope.json",
        calendar_ids=["primary", "broken"],
        clock=_fixed_clock(now),
    )
    poller._service = _PartiallyBrokenService()  # type: ignore[attr-defined]
    events = poller.fetch_today_events()
    assert len(events) == 1
    assert events[0].title == "Healthy event"


def test_build_pollers_from_config_legacy_single_account() -> None:
    cfg = {
        "poll_interval_seconds": 30,
        "pre_meeting_alert_minutes": 1,
        "credentials_path": "~/.otis/credentials.json",
        "calendar_ids": ["primary", "team@example.com"],
    }
    pollers = build_pollers_from_config(cfg)
    assert len(pollers) == 1
    assert pollers[0]._calendar_ids == ["primary", "team@example.com"]
    assert pollers[0]._poll_interval == 30


def test_build_pollers_from_config_multi_account() -> None:
    cfg = {
        "poll_interval_seconds": 60,
        "pre_meeting_alert_minutes": 2,
        "accounts": [
            {
                "label": "personal",
                "credentials_path": "~/.otis/credentials.json",
                "calendar_ids": ["primary"],
            },
            {
                "label": "work",
                "credentials_path": "~/.otis/credentials.json",
                "token_path": "~/.otis/google_token_work.json",
                "calendar_ids": ["primary", "team@company.com"],
            },
        ],
    }
    pollers = build_pollers_from_config(cfg)
    assert len(pollers) == 2

    personal, work = pollers
    assert personal._calendar_ids == ["primary"]
    assert work._calendar_ids == ["primary", "team@company.com"]
    # Default token path for the personal label, custom for work.
    assert str(personal._token_path).endswith("google_token.json")
    assert str(work._token_path).endswith("google_token_work.json")


def test_build_pollers_from_config_empty_accounts_falls_back_to_legacy() -> None:
    cfg = {"credentials_path": "~/.otis/credentials.json"}
    pollers = build_pollers_from_config(cfg)
    assert len(pollers) == 1
    assert pollers[0]._calendar_ids == ["primary"]


def test_parse_event_extracts_ical_uid() -> None:
    raw = _event_with(iCalUID="abc-123@google.com")
    ev = parse_event(raw)
    assert ev is not None
    assert ev.ical_uid == "abc-123@google.com"
    # canonical_key prefers iCalUID over per-calendar id
    assert ev.canonical_key == "abc-123@google.com"


def test_canonical_key_falls_back_to_id_when_no_ical_uid() -> None:
    raw = _event_with()  # no iCalUID
    ev = parse_event(raw)
    assert ev is not None
    assert ev.ical_uid is None
    assert ev.canonical_key == "evt-1"


def test_event_to_dict_round_trip() -> None:
    raw = _event_with(
        conferenceData={"entryPoints": [{"entryPointType": "video", "uri": "https://x"}]}
    )
    ev = parse_event(raw)
    assert ev is not None
    d = ev.to_dict()
    assert d["title"] == "Standup"
    assert d["meeting_link"] == "https://x"
    assert d["start"].endswith("+02:00")
