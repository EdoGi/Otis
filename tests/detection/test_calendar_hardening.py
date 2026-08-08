"""Regression cover for audit finding B9 — the calendar client's soft spots.

Three separate ways the poller misbehaved:

* ``maxResults=100`` with no ``nextPageToken`` handling silently dropped the
  tail of a busy day — and the dropped events are the later ones, i.e. the
  meetings still to come;
* the "day" window was built from a UTC ``now``, so in Europe/Paris "today"
  ran 02:00→02:00 local and the alert reset fired at 02:00 (17:00 for a US
  user, mid-workday);
* no HTTP timeout anywhere, so one half-open socket wedged the poll thread
  permanently while ``stop()`` gave up after 2 s and the next off-hours cycle
  started another thread on top.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.detection.calendar_poller import (
    API_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    MAX_EVENT_PAGES,
    GoogleCalendarPoller,
    local_day_bounds,
)


# ---------------------------------------------------------------------------
# Local-day boundaries
# ---------------------------------------------------------------------------
def test_day_bounds_start_at_local_midnight() -> None:
    # 00:30 UTC on 8 Aug — in any positive-offset zone that is already the 8th
    # locally, and the window must begin at local midnight, not UTC midnight.
    now = datetime(2026, 8, 8, 0, 30, tzinfo=timezone.utc)
    start, end = local_day_bounds(now)

    assert (start.hour, start.minute, start.second) == (0, 0, 0)
    assert end - start == timedelta(days=1)
    # The bounds are expressed in the machine's local zone, so a caller
    # comparing them against local event times gets the right answer.
    assert start.utcoffset() == now.astimezone().utcoffset()


def test_day_bounds_cover_the_whole_local_day() -> None:
    now = datetime(2026, 8, 8, 15, 45, tzinfo=timezone.utc)
    start, end = local_day_bounds(now)
    local_now = now.astimezone()
    assert start <= local_now < end


def test_an_early_morning_local_meeting_is_inside_the_window() -> None:
    """A 00:30 local meeting used to fall outside a UTC-anchored window."""
    now = datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc).astimezone()
    start, end = local_day_bounds(now)
    early = now.replace(hour=0, minute=30, second=0, microsecond=0)
    assert start <= early < end


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
class PagedService:
    """Fake Calendar service returning ``pages`` successive result pages."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.tokens_seen: list[str | None] = []
        self.retries_seen: list[int] = []
        self._idx = 0

    def events(self) -> PagedService:
        return self

    def list(self, **kwargs: Any) -> PagedService:
        self.tokens_seen.append(kwargs.get("pageToken"))
        token = kwargs.get("pageToken")
        self._idx = 0 if token is None else int(token)
        return self

    def execute(self, **kw: Any) -> dict[str, Any]:
        self.retries_seen.append(kw.get("num_retries", 0))
        items = self.pages[self._idx]
        out: dict[str, Any] = {"items": items}
        if self._idx + 1 < len(self.pages):
            out["nextPageToken"] = str(self._idx + 1)
        return out


def _event(n: int, start: datetime) -> dict[str, Any]:
    return {
        "id": f"evt-{n}",
        "iCalUID": f"evt-{n}@google.com",
        "summary": f"Meeting {n}",
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": (start + timedelta(minutes=30)).isoformat()},
        "attendees": [],
    }


@pytest.fixture()
def poller(tmp_path) -> GoogleCalendarPoller:
    return GoogleCalendarPoller(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        calendar_ids=["primary"],
    )


def test_all_pages_are_followed(poller) -> None:
    base = datetime.now(timezone.utc).astimezone().replace(hour=9, minute=0, second=0,
                                                           microsecond=0)
    pages = [
        [_event(i, base + timedelta(minutes=i)) for i in range(0, 3)],
        [_event(i, base + timedelta(minutes=i)) for i in range(3, 6)],
        [_event(i, base + timedelta(minutes=i)) for i in range(6, 8)],
    ]
    service = PagedService(pages)
    poller._service = service

    events = poller.fetch_today_events()

    assert len(events) == 8, "the tail of a busy day must not be dropped"
    assert service.tokens_seen == [None, "1", "2"]


def test_single_page_makes_one_request(poller) -> None:
    base = datetime.now(timezone.utc).astimezone().replace(hour=9, minute=0, second=0,
                                                           microsecond=0)
    service = PagedService([[_event(0, base)]])
    poller._service = service

    assert len(poller.fetch_today_events()) == 1
    assert service.tokens_seen == [None]


def test_pagination_is_bounded(poller) -> None:
    """A pathological calendar can't spin the poll thread forever."""

    class Endless(PagedService):
        def list(self, **kwargs: Any) -> Endless:
            self.tokens_seen.append(kwargs.get("pageToken"))
            return self

        def execute(self, **kw: Any) -> dict[str, Any]:
            return {"items": [], "nextPageToken": "next"}

    service = Endless([[]])
    poller._service = service
    poller.fetch_today_events()
    assert len(service.tokens_seen) == MAX_EVENT_PAGES


def test_requests_ask_the_client_library_to_retry(poller) -> None:
    base = datetime.now(timezone.utc).astimezone().replace(hour=9, minute=0)
    service = PagedService([[_event(0, base)]])
    poller._service = service
    poller.fetch_today_events()
    assert service.retries_seen == [API_RETRIES]
    assert API_RETRIES >= 1


# ---------------------------------------------------------------------------
# Transport timeout
# ---------------------------------------------------------------------------
def test_authorized_http_carries_a_timeout(poller) -> None:
    """httplib2 defaults to timeout=None — blocking forever on a dead socket."""
    pytest.importorskip("httplib2")
    pytest.importorskip("google_auth_httplib2")

    class Creds:
        def before_request(self, *a, **k):  # pragma: no cover - unused
            pass

    http = poller._authorized_http(Creds())
    underlying = getattr(http, "http", None)
    assert underlying is not None
    assert underlying.timeout == HTTP_TIMEOUT_SECONDS


def test_timeout_is_a_sane_value() -> None:
    assert 1.0 <= HTTP_TIMEOUT_SECONDS <= 60.0
