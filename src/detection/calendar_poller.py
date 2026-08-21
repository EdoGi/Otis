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


#: Transport timeout for every Calendar API call. httplib2 defaults to None
#: (block forever), which is how one bad socket wedged the poller permanently.
HTTP_TIMEOUT_SECONDS = 10.0

#: Client-library retries for a single request — rides out 5xx and rate limits
#: instead of failing the whole poll cycle.
API_RETRIES = 3

#: Ceiling for the poll-loop backoff when the calendar is unreachable.
MAX_BACKOFF_SECONDS = 600.0

#: Safety valve on nextPageToken following, so a pathological calendar can't
#: spin the poll thread indefinitely. 250 events/page.
MAX_EVENT_PAGES = 8


class CalendarAuthError(RuntimeError):
    """Raised when OAuth credentials cannot be loaded or refreshed."""


class CalendarUnreachableError(CalendarAuthError):
    """The saved token is fine; Google just wasn't reachable this cycle.

    A subclass of :class:`CalendarAuthError` so existing handlers keep working,
    but callers that care can tell "you need to reconnect" apart from "the
    train went into a tunnel".
    """


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
        label: str | None = None,
    ) -> None:
        # One health row per account. Reporting every poller to a single shared
        # CALENDAR key made one broken account flap the row against a healthy
        # one — ~10 toasts per 5 minutes, the exact spam the health surface was
        # built to stop, moved one layer up.
        self.label = label
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
        # Auth failure is announced once per failure streak, not once per poll.
        # Before this flag existed the poller logged the same warning twice a
        # minute for three months — 2 309 lines, 45% of the log file, and not
        # one of them reached the user.
        self._auth_failure_announced = False
        # Set by fetch_today_events so the poll loop can actually back off.
        self._last_cycle_failed = False
        # Threads that outlived their stop() join. Tracked so a wedged
        # poller can't be silently duplicated on the next start().
        self._lingering: list[threading.Thread] = []

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
        # A thread that outlived its stop() join used to be forgotten (stop()
        # set _thread = None regardless), so every off-hours cycle could stack
        # another poller — each holding its own socket — on top of a wedged
        # one. With an HTTP timeout in place a stuck thread now exits within
        # seconds, so refusing to double up is safe and self-healing.
        self._prune_lingering()
        if self._lingering:
            logger.warning(
                "Not starting a second calendar poller: %d previous thread(s) "
                "still finishing. Will start on the next attempt.",
                len(self._lingering),
            )
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
                # Remember it, so start() knows not to stack another poller
                # on top while this one is still holding a socket.
                self._lingering.append(thread)
            self._thread = None
        self._prune_lingering()

    def _prune_lingering(self) -> None:
        self._lingering = [t for t in self._lingering if t.is_alive()]

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
                # Only an explicit verdict from Google's OAuth server means the
                # token is dead. Everything else — offline wake, captive portal,
                # DNS blip, 503 — is transient, and deleting a working refresh
                # token because the Wi-Fi dropped is exactly how this app lost
                # calendar detection for three months.
                if _token_is_permanently_dead(exc):
                    logger.warning(
                        "Token was revoked by the OAuth server (%s); deleting it.", exc
                    )
                    try:
                        self._token_path.unlink(missing_ok=True)
                    except Exception:  # pragma: no cover
                        pass
                else:
                    logger.warning(
                        "Token refresh failed transiently (%s); keeping the token "
                        "and retrying on the next poll.", exc,
                    )
                    # The token is intact — we simply could not reach Google.
                    # Telling the user to "reconnect your calendar" because a
                    # captive portal ate one request is worse than saying
                    # nothing, so this raises a distinct error the caller
                    # reports as UNKNOWN rather than as a broken account.
                    raise CalendarUnreachableError(str(exc)) from exc

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
        # An explicit transport timeout is the difference between a bad network
        # and a permanently wedged poller. httplib2's default is None, so a
        # half-open socket (captive portal, VPN flap, sleep mid-request) blocks
        # the poll thread forever — stop() then joins for 2 s, gives up, and the
        # next off-hours cycle starts *another* thread on top of it.
        return build(
            "calendar",
            "v3",
            credentials=creds,
            cache_discovery=False,
            http=self._authorized_http(creds),
        )

    def _authorized_http(self, creds: Any) -> Any:
        """An httplib2 transport with a timeout, authorised with ``creds``."""
        import google_auth_httplib2
        import httplib2

        return google_auth_httplib2.AuthorizedHttp(
            creds, http=httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS)
        )

    def _health_key(self) -> str:
        from src.health import CALENDAR

        return f"{CALENDAR}:{self.label}" if self.label else CALENDAR

    def _health_label(self) -> str:
        return f"Calendar ({self.label})" if self.label else "Calendar"

    # ------------------------------------------------------- auth announcements
    def _announce_auth_failure(self, message: str) -> None:
        """Report broken auth once per failure streak — to the user, not the log.

        The poll loop runs every 60 s per account. Announcing on every cycle is
        what produced 2 880 warning lines a day and trained everyone to ignore
        the log entirely, so the transition is the event, not the state.
        """
        from src.health import HealthState, get_health

        with self._lock:
            already = self._auth_failure_announced
            self._auth_failure_announced = True

        get_health().report(
            self._health_key(),
            label=self._health_label(),
            state=HealthState.FAIL,
            detail="disconnected — reconnect to get meeting titles",
            hint="Run scripts/setup_google_cal.sh to reconnect this account.",
        )
        if already:
            logger.debug("Calendar auth still needs attention: %s", message)
            return
        logger.warning("Calendar auth needs attention: %s", message)
        _safe_call_each(self._on_reauth, message)

    def _announce_unreachable(self, message: str) -> None:
        """Network trouble, not an auth problem. Never say "reconnect"."""
        from src.health import HealthState, get_health

        logger.info("Calendar unreachable this cycle: %s", message)
        get_health().report(
            self._health_key(),
            label=self._health_label(),
            state=HealthState.UNKNOWN,
            detail="offline — will retry",
        )

    def notify_auth_recovered(self) -> None:
        """Clear the failure streak so a future break is announced again."""
        from src.health import HealthState, get_health

        with self._lock:
            was_broken = self._auth_failure_announced
            self._auth_failure_announced = False
        get_health().report(
            self._health_key(),
            label=self._health_label(),
            state=HealthState.OK,
            detail=f"{len(self._calendar_ids)} calendar(s)",
        )
        if was_broken:
            logger.info("Calendar auth recovered.")

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
                self.notify_auth_recovered()
        except CalendarUnreachableError as exc:
            self._announce_unreachable(str(exc))
            self._service = None
            self._last_cycle_failed = True
            return []
        except CalendarAuthError as exc:
            self._announce_auth_failure(str(exc))
            self._service = None
            self._last_cycle_failed = True
            return []
        except Exception as exc:
            logger.warning("Calendar service build failed: %s", exc)
            self._service = None
            self._last_cycle_failed = True
            return []

        start_of_day, end_of_day = local_day_bounds(self._now())

        seen_ids: set[str] = set()
        events: list[CalendarEvent] = []
        failed_calendars = 0
        for cal_id in self._calendar_ids:
            try:
                items = self._list_events_paged(cal_id, start_of_day, end_of_day)
            except Exception as exc:
                logger.warning("Calendar %r fetch failed: %s", cal_id, exc)
                failed_calendars += 1
                continue

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

        self._last_cycle_failed = bool(failed_calendars)
        events.sort(key=lambda e: e.start)
        return events

    def _list_events_paged(
        self, cal_id: str, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch every event in the window, following ``nextPageToken``.

        The old call passed ``maxResults=100`` and ignored the page token, so a
        day with more than 100 events silently lost the tail — and the events
        most likely to be dropped are the later ones, i.e. the meetings still
        to come. ``num_retries`` lets the client library ride out a 5xx or a
        rate-limit instead of failing the whole cycle.
        """
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(MAX_EVENT_PAGES):
            response = (
                self._service.events()
                .list(
                    calendarId=cal_id,
                    timeMin=start.isoformat(),
                    timeMax=end.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=250,
                    pageToken=page_token,
                )
                .execute(num_retries=API_RETRIES)
            )
            if not isinstance(response, dict):
                break
            items.extend(response.get("items", []) or [])
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        else:
            logger.warning(
                "Calendar %r has more than %d pages of events; truncating.",
                cal_id, MAX_EVENT_PAGES,
            )
        return items

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
                # The backoff used to grow only when _poll_cycle *raised* — but
                # fetch_today_events swallows every exception, so it never did,
                # and an offline laptop retried at full rate: 8 hours away from
                # the network cost ~960 doomed requests per account and twice
                # that in warning lines. Failures are now reported rather than
                # inferred from an exception that never arrives.
                if self._last_cycle_failed:
                    backoff = min(max(backoff * 2, self._poll_interval), MAX_BACKOFF_SECONDS)
                    logger.debug("Calendar fetch failed; backing off to %.0fs.", backoff)
                else:
                    backoff = self._poll_interval
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Calendar poll cycle crashed; backing off.")
                backoff = min(max(backoff * 2, self._poll_interval), MAX_BACKOFF_SECONDS)
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
#: Substrings in an OAuth ``RefreshError`` that mean the token itself is dead
#: and no amount of retrying will help. Anything else is transport trouble.
_DEAD_TOKEN_MARKERS = (
    "invalid_grant",
    "invalid_client",
    "unauthorized_client",
    "invalid_scope",
)


def local_day_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Return the start and end of ``now``'s *local* day, as aware datetimes.

    The window used to be built from a UTC ``now`` with ``hour=0``, which makes
    "today" run 02:00→02:00 local in Europe/Paris. Two consequences: meetings
    between local midnight and 02:00 were invisible, and ``reset_alerts()``
    fired at 02:00 local — 17:00 local for a US user, i.e. in the middle of the
    workday, wiping the bookkeeping that lets a meeting's end signal fire.
    """
    local_now = now.astimezone()
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _token_is_permanently_dead(exc: BaseException) -> bool:
    """True only when Google's OAuth server has explicitly rejected the token.

    A ``RefreshError`` is *also* raised for connection resets, so the exception
    type alone is not enough — we require an OAuth error code in the message.
    """
    try:
        from google.auth.exceptions import RefreshError
    except Exception:  # pragma: no cover (google libs absent)
        return False
    if not isinstance(exc, RefreshError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _DEAD_TOKEN_MARKERS)


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
                    label=label,
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
