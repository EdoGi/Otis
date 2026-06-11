"""Google Calendar poller.

Authenticates via OAuth2 (Desktop client credentials), polls the user's primary
calendar every ~60 s, and fires callbacks when:

* an event is starting within ``pre_meeting_alert_minutes`` (default 2)
* a previously-alerted event has reached its scheduled end time

Filters out:
* all-day events (``date`` instead of ``dateTime``)
* events the user has declined (their attendee entry has
  ``responseStatus == "declined"``)

Tentative events are *included* with ``is_tentative=True`` on the parsed event.

Network and token errors degrade gracefully: the poll loop logs a warning,
applies an exponential backoff (capped), and retries — it never crashes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Read-only scope is enough — we never modify the user's calendar.
SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/calendar.readonly",)

# Best-effort regexes to pull a meeting URL out of the event description when
# Google's structured ``conferenceData`` field is empty.
_URL_RE = re.compile(
    r"https?://[^\s>'\"]+",
    re.IGNORECASE,
)
_KNOWN_HOSTS = (
    "meet.google.com",
    "zoom.us",
    "zoom.com",
    "teams.microsoft.com",
    "teams.live.com",
    "webex.com",
    "whereby.com",
    "around.co",
)


@dataclass
class CalendarEvent:
    """Normalised view of a Google Calendar event for our purposes."""

    id: str
    title: str
    start: datetime  # timezone-aware (UTC if Google didn't supply one)
    end: datetime
    attendees: list[dict[str, str]] = field(default_factory=list)
    meeting_link: str | None = None
    is_tentative: bool = False
    ical_uid: str | None = None  # iCalendar UID — same value across all calendars
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_key(self) -> str:
        """Stable key that's the same for cross-invited copies of one meeting.

        Falls back to the per-calendar ``id`` when the API doesn't provide an
        ``iCalUID`` (some imported / non-RFC events).
        """
        return self.ical_uid or self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "attendees": list(self.attendees),
            "meeting_link": self.meeting_link,
            "is_tentative": self.is_tentative,
            "ical_uid": self.ical_uid,
        }


UpcomingCallback = Callable[[CalendarEvent], None]
EndedCallback = Callable[[CalendarEvent], None]
ReauthCallback = Callable[[str], None]


class CalendarAuthError(RuntimeError):
    """Raised when OAuth credentials cannot be loaded or refreshed."""


class GoogleCalendarPoller:
    """Background poller that fires upcoming-meeting events from Google Calendar."""

    def __init__(
        self,
        *,
        credentials_path: str | os.PathLike[str],
        token_path: str | os.PathLike[str] | None = None,
        poll_interval_seconds: float = 60.0,
        pre_meeting_alert_minutes: float = 2.0,
        calendar_id: str = "primary",
        calendar_ids: Iterable[str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._credentials_path = Path(os.path.expanduser(str(credentials_path)))
        self._token_path = (
            Path(os.path.expanduser(str(token_path)))
            if token_path is not None
            else self._credentials_path.with_name("google_token.json")
        )
        self._poll_interval = float(poll_interval_seconds)
        self._pre_alert = float(pre_meeting_alert_minutes)
        # Accept either a single ``calendar_id`` (back-compat) or a list of
        # ``calendar_ids`` (e.g. ["primary", "work@company.com"]). Both cases
        # collapse to one list internally.
        self._calendar_ids: list[str] = list(calendar_ids) if calendar_ids else [calendar_id]
        self._now = clock or _utcnow

        self._on_upcoming: list[UpcomingCallback] = []
        self._on_ended: list[EndedCallback] = []
        self._on_reauth: list[ReauthCallback] = []

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        # Bookkeeping so we don't double-fire.
        self._alerted_upcoming: set[str] = set()
        self._alerted_ended: set[str] = set()
        self._last_poll_date: Any = None  # datetime.date of the last poll cycle
        self._service: Any = None  # googleapiclient resource, lazy-built

    # =====================================================================
    # Public API
    # =====================================================================
    def on_upcoming_meeting(self, callback: UpcomingCallback) -> None:
        self._on_upcoming.append(callback)

    def on_meeting_should_end(self, callback: EndedCallback) -> None:
        self._on_ended.append(callback)

    def on_needs_reauth(self, callback: ReauthCallback) -> None:
        """Fired when the saved token can't be refreshed and the user must re-auth."""
        self._on_reauth.append(callback)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Fresh event per run: a previous thread that's still finishing a
        # blocking network call keeps its own (already-set) event and exits
        # on its own, instead of being resurrected by this clear().
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event,),
            name="otis-calendar-poller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "GoogleCalendarPoller started (poll=%.1fs, alert=%.1fmin).",
            self._poll_interval,
            self._pre_alert,
        )

    def stop(self) -> None:
        """Signal the poll loop to exit. Returns quickly.

        The loop wakes immediately from its wait; the only thing that can
        delay exit is a network fetch already in flight. Don't block the
        caller (often the main/UI thread) on that — the thread is a daemon
        and checks the stop event right after the fetch.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.info(
                    "Calendar poller still finishing its last cycle; "
                    "it will exit on its own."
                )
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ auth
    def authenticate(self, *, headless: bool = False) -> Any:
        """Load (or perform first-run) OAuth credentials and return them.

        ``headless=True`` skips the browser flow when no token exists — useful
        for the polling thread, which should fall back to a re-auth notification
        rather than blocking on a console prompt.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self._credentials_path.exists():
            raise CalendarAuthError(
                f"OAuth credentials not found at {self._credentials_path}. "
                "Run scripts/setup_google_cal.sh and drop the JSON there."
            )

        creds: Credentials | None = None
        if self._token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(self._token_path), list(SCOPES)
                )
            except Exception as exc:
                logger.warning("Could not load saved token (%s); will re-auth.", exc)
                creds = None

        if creds is not None and creds.valid:
            return creds

        if creds is not None and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_token(creds)
                return creds
            except Exception as exc:
                logger.warning("Token refresh failed (%s); deleting saved token.", exc)
                try:
                    self._token_path.unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass

        # No usable credentials. The poller calls us with headless=True so we
        # surface a re-auth request instead of blocking the thread on a browser.
        if headless:
            raise CalendarAuthError(
                "No valid OAuth token. Re-run authentication interactively."
            )

        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._credentials_path), list(SCOPES)
        )
        creds = flow.run_local_server(port=0)
        self._save_token(creds)
        return creds

    def _save_token(self, creds: Any) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
            try:
                os.chmod(self._token_path, 0o600)
            except OSError:  # pragma: no cover
                pass
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not persist token to %s: %s", self._token_path, exc)

    def _build_service(self) -> Any:
        from googleapiclient.discovery import build

        creds = self.authenticate(headless=True)
        return build("calendar", "v3", credentials=creds, cache_discovery=False)

    # --------------------------------------------------------------- fetching
    def fetch_today_events(self) -> list[CalendarEvent]:
        """Pull today's events from every configured calendar.

        Iterates over ``self._calendar_ids`` so users can poll multiple
        calendars (e.g. ``["primary", "work@company.com"]``). Returns events
        merged and de-duplicated by id, sorted by start time.

        A failure on one calendar logs a warning and skips it — the rest still
        return.
        """
        try:
            if self._service is None:
                self._service = self._build_service()
        except CalendarAuthError as exc:
            logger.warning("Calendar auth needs attention: %s", exc)
            _safe_call_each(self._on_reauth, str(exc))
            self._service = None
            return []
        except Exception as exc:
            logger.warning("Calendar service build failed: %s", exc)
            self._service = None
            return []

        now = self._now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        seen_ids: set[str] = set()
        events: list[CalendarEvent] = []
        failed_calendars = 0
        for cal_id in self._calendar_ids:
            try:
                response = (
                    self._service.events()
                    .list(
                        calendarId=cal_id,
                        timeMin=start_of_day.isoformat(),
                        timeMax=end_of_day.isoformat(),
                        singleEvents=True,
                        orderBy="startTime",
                        maxResults=100,
                    )
                    .execute()
                )
            except Exception as exc:
                logger.warning("Calendar %r fetch failed: %s", cal_id, exc)
                failed_calendars += 1
                continue

            items = response.get("items", []) if isinstance(response, dict) else []
            for raw in items:
                ev = parse_event(raw)
                if ev is None or ev.id in seen_ids:
                    continue
                seen_ids.add(ev.id)
                events.append(ev)

        # Every calendar failing usually means the cached service is dead
        # (revoked token, expired credentials, long network change) — drop it
        # so the next cycle rebuilds it through authenticate(), which also
        # re-fires the reauth callback if the user has to intervene.
        if failed_calendars and failed_calendars == len(self._calendar_ids):
            logger.warning(
                "All %d calendar fetch(es) failed; discarding cached service "
                "so the next poll rebuilds it.",
                failed_calendars,
            )
            self._service = None

        events.sort(key=lambda e: e.start)
        return events

    # ----------------------------------------------------------------- state
    def upcoming_within(self, minutes: float) -> list[CalendarEvent]:
        """Return today's events whose start is within ``minutes`` from now."""
        now = self._now()
        horizon = now + timedelta(minutes=minutes)
        events = self.fetch_today_events()
        return [e for e in events if now <= e.start <= horizon]

    def reset_alerts(self) -> None:
        """Forget which events we've already alerted on (e.g. on date change)."""
        with self._lock:
            self._alerted_upcoming.clear()
            self._alerted_ended.clear()

    # =====================================================================
    # Internals
    # =====================================================================
    def _run(self, stop_event: threading.Event) -> None:
        backoff = self._poll_interval
        while not stop_event.is_set():
            try:
                self._poll_cycle()
                backoff = self._poll_interval
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Calendar poll cycle crashed; backing off.")
                backoff = min(backoff * 2, 600.0)
            stop_event.wait(backoff)
        logger.info("GoogleCalendarPoller stopped.")

    def _poll_cycle(self) -> None:
        now = self._now()
        # Day rollover: yesterday's alert bookkeeping is useless today and
        # would otherwise grow unboundedly in a long-running app.
        today = now.date()
        if self._last_poll_date is not None and today != self._last_poll_date:
            logger.info("Date rolled over to %s; resetting alert bookkeeping.", today)
            self.reset_alerts()
        self._last_poll_date = today

        events = self.fetch_today_events()
        horizon = now + timedelta(minutes=self._pre_alert)

        for ev in events:
            if now <= ev.start <= horizon and ev.id not in self._alerted_upcoming:
                with self._lock:
                    self._alerted_upcoming.add(ev.id)
                logger.info("Upcoming meeting in <%.0fmin: %s", self._pre_alert, ev.title)
                _safe_call_each(self._on_upcoming, ev)
            if (
                ev.id in self._alerted_upcoming
                and ev.id not in self._alerted_ended
                and now >= ev.end
            ):
                with self._lock:
                    self._alerted_ended.add(ev.id)
                logger.info("Calendar end reached for: %s", ev.title)
                _safe_call_each(self._on_ended, ev)


# ============================================================================
# Event parsing — pure functions, easy to test.
# ============================================================================
def parse_event(raw: dict[str, Any]) -> CalendarEvent | None:
    """Convert a Google Calendar event dict into our :class:`CalendarEvent`.

    Returns ``None`` when the event should be ignored (all-day, declined, or
    malformed).
    """
    start = raw.get("start") or {}
    end = raw.get("end") or {}

    if "dateTime" not in start or "dateTime" not in end:
        # All-day events have ``date`` but no ``dateTime``.
        return None

    if _user_declined(raw):
        return None

    try:
        start_dt = _parse_dt(start["dateTime"])
        end_dt = _parse_dt(end["dateTime"])
    except ValueError:
        return None

    return CalendarEvent(
        id=str(raw.get("id", "")),
        title=str(raw.get("summary", "(no title)")),
        start=start_dt,
        end=end_dt,
        attendees=_parse_attendees(raw.get("attendees", [])),
        meeting_link=_extract_meeting_link(raw),
        is_tentative=_user_tentative(raw),
        ical_uid=raw.get("iCalUID") or None,
        raw=raw,
    )


def _parse_dt(value: str) -> datetime:
    """Parse an RFC 3339 datetime as returned by the Calendar API."""
    # Python <3.11 doesn't accept the trailing 'Z'.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_attendees(raw_attendees: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in raw_attendees:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "email": str(a.get("email", "")),
                "name": str(a.get("displayName") or a.get("email", "")),
                "responseStatus": str(a.get("responseStatus", "needsAction")),
            }
        )
    return out


def _user_declined(raw: dict[str, Any]) -> bool:
    for a in raw.get("attendees", []) or []:
        if isinstance(a, dict) and a.get("self") and a.get("responseStatus") == "declined":
            return True
    return False


def _user_tentative(raw: dict[str, Any]) -> bool:
    for a in raw.get("attendees", []) or []:
        if isinstance(a, dict) and a.get("self") and a.get("responseStatus") == "tentative":
            return True
    return False


def _extract_meeting_link(raw: dict[str, Any]) -> str | None:
    """Find the conference URL in conferenceData first, then in description."""
    conf = raw.get("conferenceData") or {}
    for ep in conf.get("entryPoints", []) or []:
        if isinstance(ep, dict) and ep.get("entryPointType") == "video" and ep.get("uri"):
            return str(ep["uri"])
    # Fallback: scan description for a known-host URL.
    desc = raw.get("description") or ""
    for url in _URL_RE.findall(desc):
        lower = url.lower()
        if any(host in lower for host in _KNOWN_HOSTS):
            return url
    # Some events put the link in ``location``.
    loc = raw.get("location") or ""
    for url in _URL_RE.findall(loc):
        lower = url.lower()
        if any(host in lower for host in _KNOWN_HOSTS):
            return url
    return None


# ============================================================================
# Helpers
# ============================================================================
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_call_each(callbacks: list[Callable[..., None]], *args: Any) -> None:
    for cb in callbacks:
        try:
            cb(*args)
        except Exception:
            logger.exception("Calendar callback %r raised", cb)


# Re-export a small JSON helper used by tests.
def event_to_json(event: CalendarEvent) -> str:
    return json.dumps(event.to_dict(), indent=2)


def build_pollers_from_config(calendar_cfg: Any) -> list[GoogleCalendarPoller]:
    """Construct one :class:`GoogleCalendarPoller` per configured Google account.

    The config block (``detection.calendar``) can be either:

    * **Single account (legacy)** — ``credentials_path`` + optional
      ``calendar_ids``. One poller is returned.
    * **Multi-account** — ``accounts: [{label, credentials_path, token_path,
      calendar_ids}, ...]``. One poller per entry.
    """
    cfg = _as_plain_dict(calendar_cfg)
    poll_interval = float(cfg.get("poll_interval_seconds", 60))
    pre_alert = float(cfg.get("pre_meeting_alert_minutes", 2))

    accounts = cfg.get("accounts")
    if accounts:
        pollers: list[GoogleCalendarPoller] = []
        for entry in accounts:
            entry_dict = _as_plain_dict(entry)
            label = str(entry_dict.get("label", "default"))
            creds = entry_dict.get("credentials_path") or "~/.otis/credentials.json"
            token = entry_dict.get("token_path") or _default_token_path_for_label(label)
            ids = entry_dict.get("calendar_ids") or ["primary"]
            pollers.append(
                GoogleCalendarPoller(
                    credentials_path=creds,
                    token_path=token,
                    poll_interval_seconds=poll_interval,
                    pre_meeting_alert_minutes=pre_alert,
                    calendar_ids=list(ids),
                )
            )
        return pollers

    # Legacy single-account form.
    creds = cfg.get("credentials_path") or "~/.otis/credentials.json"
    token = cfg.get("token_path")  # may be None — poller picks the default
    ids = cfg.get("calendar_ids") or ["primary"]
    return [
        GoogleCalendarPoller(
            credentials_path=creds,
            token_path=token,
            poll_interval_seconds=poll_interval,
            pre_meeting_alert_minutes=pre_alert,
            calendar_ids=list(ids),
        )
    ]


def _default_token_path_for_label(label: str) -> str:
    """Map an account label to its default cached-token filename.

    Labels ``personal``, ``default``, ``primary`` (and the empty string) all
    use the unsuffixed ``google_token.json`` — this matches the convention in
    ``scripts/setup_google_cal.sh`` so the bundled default config "just works"
    without a label suffix.
    """
    if label in ("personal", "default", "primary", ""):
        return "~/.otis/google_token.json"
    return f"~/.otis/google_token_{label}.json"


def _as_plain_dict(node: Any) -> dict[str, Any]:
    """Coerce a Config / AttrDict / regular dict into a plain dict view."""
    if isinstance(node, dict):
        return dict(node)
    if hasattr(node, "raw"):
        return dict(node.raw)
    if hasattr(node, "_data"):
        return dict(node._data)  # type: ignore[attr-defined]
    return dict(node) if node is not None else {}
