"""The Status submenu: health rows, live meters, and the repair actions.

Split out of ``src/ui/menubar.py`` when the diagnostic fixes of 2026-10-25
pushed it back over the 2,000-line limit audit item 42 set. Same shape as
SettingsMenuMixin: everything here reads or writes app state, so a mixin.

What lives here: rendering the health registry (one row per component, plus
mic/system meters while recording), turning transitions into notifications,
the launch-time BlackHole/routing check, and the two repair actions (Fix
System Audio…, Reconnect Calendar…).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from src.health import (
    MICROPHONE,
    SYSTEM_AUDIO,
    HealthReport,
    HealthState,
)
from src.ui.menubar_state import _GLYPH_FOR, UiState
from src.ui.notifications import NotificationType
from src.ui.rumps_support import SOUND_SETTINGS_URL, _forget_menu_item, _meter_row

if TYPE_CHECKING:
    import queue
    from collections.abc import Callable

    from src.detection.detector import MeetingDetector
    from src.health import HealthRegistry
    from src.ui.menubar_state import UiSnapshot
    from src.ui.notifications import NotificationManager

logger = logging.getLogger("src.ui.menubar")


class HealthPanelMixin:
    """Status panel behaviour for :class:`~src.ui.menubar.MenuBarApp`."""

    if TYPE_CHECKING:
        # Provided by MenuBarApp. Declared so the checker knows what the mixin
        # reads and writes; nothing here is assigned — that is the host's job.
        _app: Any
        _mi: dict[str, Any]
        _health: HealthRegistry
        _notifications: NotificationManager
        _detector: MeetingDetector
        _snapshot: UiSnapshot
        _lock: threading.RLock
        _events: queue.Queue[tuple[str, Any]]
        _recorder: Any
        _routing_check: Callable[[], Any] | None
        _route_at_recording_start: Any
        _last_route: tuple[Any, str] | None
        _user_check_in_progress: bool
        _live_meters: dict[str, str]
        _health_rows_shown: list[str]
        _health_items: list[Any]
        _meter_items: dict[str, Any]

        def _refresh_menu_visibility(self) -> None: ...
        def _drain_main_queue(self, _timer: Any) -> None: ...
        def _open_url(self, url: str) -> None: ...
        def _open_app(self, name: str) -> None: ...

    def _warn_if_blackhole_missing(self) -> None:
        """Check system audio at startup — structurally *and* functionally.

        The old version only asked whether BlackHole and some aggregate device
        existed. That returned ok=True on this Mac for two months while the
        output went to the built-in speakers and BlackHole captured pure
        silence. Existence is not routing, so we now check both.
        """
        try:
            from src.audio.blackhole_check import verify_blackhole_setup

            status = verify_blackhole_setup()
        except Exception:  # pragma: no cover (defensive)
            logger.exception("BlackHole check raised on startup")
            return

        if not status.ok:
            logger.warning(
                "BlackHole / Multi-Output not configured: %s",
                "; ".join(status.issues) or "see check-audio output",
            )
            self._health.report(
                SYSTEM_AUDIO,
                label="System audio",
                state=HealthState.FAIL,
                detail="BlackHole not installed or no Multi-Output device",
                hint="Run scripts/setup_blackhole.sh.",
            )
            return

        # BlackHole exists. Is anything actually being routed into it?
        try:
            self._report_audio_routing(self._check_routing())
        except Exception:  # pragma: no cover (defensive)
            logger.exception("Audio routing check raised on startup")

    def _check_microphone_permission(self, *, at_recording_start: bool = False) -> None:
        """Settle microphone permission now, not at the first recording.

        After a rebuild changes the app's code hash, macOS treats the old grant
        as undecided; left alone, the prompt would appear the first time a
        recording opens the mic — mid-meeting, with zeros captured until it is
        answered. A denial is a failure with its fix, not the "nobody spoke"
        warning a silent mic otherwise gets.
        """
        from src.audio import mic_permission as mp

        status = mp.microphone_authorization()
        if status == mp.AUTHORIZED:
            self._clear_microphone_permission_failure()
        elif status == mp.NOT_DETERMINED and not at_recording_start:
            logger.info("Microphone permission is undecided (a new app signature?); asking now.")
            if not mp.request_microphone_access(self._on_microphone_answer):
                logger.warning("Could not ask for microphone access.")
        elif status in (mp.DENIED, mp.RESTRICTED):
            self._report_microphone_denied(status, force=at_recording_start)

    def _on_microphone_answer(self, granted: bool) -> None:
        """Called by AVFoundation on an arbitrary thread."""
        if granted:
            logger.info("Microphone access granted.")
            self._clear_microphone_permission_failure()
        else:
            from src.audio import mic_permission as mp

            self._report_microphone_denied(mp.DENIED)

    def _refresh_microphone_permission(self) -> None:
        """Clear a permission failure once the user has fixed it in Settings.

        Cheap (a TCC query), so it rides the 60 s schedule tick; without it the
        "access denied" row and the ⚠ badge stayed until a recording heard the
        mic, although the setting had been turned back on.
        """
        rep = self._health.get(MICROPHONE)
        if rep is None or not _is_permission_failure(rep):
            return
        from src.audio import mic_permission as mp

        if mp.microphone_authorization() == mp.AUTHORIZED:
            self._clear_microphone_permission_failure()

    def _clear_microphone_permission_failure(self) -> None:
        rep = self._health.get(MICROPHONE)
        if rep is not None and _is_permission_failure(rep):
            self._health.report(MICROPHONE, label="Microphone", state=HealthState.OK,
                                detail="access granted")

    def _report_microphone_denied(self, status: str, *, force: bool = False) -> None:
        from src.audio import mic_permission as mp

        logger.warning("Microphone access is %s for Otis.", status)
        self._health.report(
            MICROPHONE, label="Microphone", state=HealthState.FAIL,
            detail=f"access {'restricted' if status == mp.RESTRICTED else 'denied'} — "
                   "your side of a call records as silence",
            hint="System Settings → Privacy & Security → Microphone → turn on Otis.",
            force_notify=force,
        )

    def _check_routing(self) -> Any:
        """check_output_routing(), unless a test injected its own."""
        if self._routing_check is not None:
            return self._routing_check()
        from src.audio import blackhole_check

        return blackhole_check.check_output_routing()

    def _refresh_level_meters(self) -> None:
        """Show live mic/system input levels in the Status panel while recording.

        This is the check that would have caught the dead system stream during
        the very first meeting instead of two months later.

        The meters used to be written into each health row's detail with
        touch() — the only place the row's reason lives — so '✕ System audio —
        not routed (…)' turned into '✕ System audio — ▃▃▃·····' and stayed so
        after the meeting. They are their own rows now, one per stream, whether
        or not that stream has a health row yet (the mic usually has none).
        """
        with self._lock:
            recorder = self._recorder
        if recorder is None:
            return
        try:
            meters = recorder.level_meters(width=8)
        except Exception:  # pragma: no cover (defensive)
            return
        self._live_meters = {label: meters[label] for label in ("mic", "system") if label in meters}
        self._refresh_health_menu()

    def _clear_level_meters(self) -> None:
        if self._live_meters:
            self._live_meters = {}
            self._refresh_health_menu()

    def _cb_health_changed(self, report: HealthReport, previous: HealthState) -> None:
        """A component changed state (background thread) — tell the user."""
        logger.info(
            "health change: %s %s → %s", report.key, previous.value, report.state.value
        )
        self._events.put(("health", (report, previous)))

    def _handle_health_change(
        self, report: HealthReport, previous: HealthState
    ) -> None:
        """Main-thread half: repaint the panel and raise one notification."""
        self._refresh_health_menu()
        self._refresh_menu_visibility()
        self._apply_health_badge()

        # A first-ever report is not news. UNSET means "nothing was ever
        # reported for this component", so announcing it would tell the user a
        # perfectly healthy startup had just "recovered".
        if previous is HealthState.UNSET and report.state is HealthState.OK:
            return
        if self._stream_toast_capped(report, previous):
            return

        if report.state in (HealthState.WARN, HealthState.FAIL):
            body = report.detail or "needs attention"
            if report.hint:
                body = f"{body}. {report.hint}"
            self._notifications.notify(
                NotificationType.ERROR,
                f"{report.label} problem",
                body,
                force=True,
            )
        elif report.state is HealthState.OK:
            self._notifications.notify(
                NotificationType.TRANSCRIPTION_COMPLETE,
                f"{report.label} recovered",
                report.detail or "working again",
                force=True,
            )

    def _begin_recording_toasts(self) -> None:
        self._stream_toasts: dict[str, set[str]] = {}

    def _stream_toast_capped(self, report: HealthReport, previous: HealthState) -> bool:
        """At most one "problem" and one "recovered" toast per stream per recording.

        The system stream is flagged after 60 s of digital silence — what a
        muted far side sounds like — and every flip raised a toast with a
        sound: 12 in 21 minutes on 2026-09-24, each time recovering on its own.
        The panel still shows every change. Never capped:

        * a route lost *during* the recording (OK when it started, not routed
          now: earbuds connected mid-call). Not a route that was "not routed"
          all along — this user's default output is the speakers while the
          meeting app plays into the Otis device, so a live check says
          NOT_ROUTED in every meeting, and exempting that capped nothing;
        * a forced report (force_notify: silence since the first block, or
          ten minutes of it) — whatever the row showed before it. Guessing
          "forced" from an unchanged state missed the case where a Fix click
          had set the row OK in the middle of the silence (review round 2);
        * anything raised while the user is running Fix System Audio….
        """
        if report.key not in (SYSTEM_AUDIO, MICROPHONE):
            return False
        if self._snapshot.state not in (UiState.RECORDING, UiState.PAUSED):
            return False
        if self._user_check_in_progress:
            return False
        if report.state in (HealthState.WARN, HealthState.FAIL):
            kind = "problem"
        elif report.state is HealthState.OK:
            kind = "recovered"
        else:
            return False
        shown = self._stream_toasts.setdefault(report.key, set())
        if report.forced:
            # Never capped — but it does spend the allowance, or an ordinary
            # quiet stretch after it would toast again (review round 3).
            shown.add(kind)
            return False
        if kind == "problem" and report.key == SYSTEM_AUDIO and self._route_lost_mid_recording():
            return False
        if kind in shown:
            logger.info("%s %s toast skipped: already shown this recording.",
                        report.label, kind)
            return True
        shown.add(kind)
        return False

    def _route_lost_mid_recording(self) -> bool:
        from src.audio.blackhole_check import RoutingState

        if self._route_at_recording_start is not RoutingState.OK:
            return False
        try:
            return bool(self._check_routing().state is RoutingState.NOT_ROUTED)
        except Exception:
            logger.exception("Routing check for a mid-call silence raised")
            return False

    def _refresh_health_menu(self) -> None:
        """Render the Status submenu: one row per component, then live meters.

        It used to delete and rebuild every item on each call — once a second
        while recording — and rumps keeps every MenuItem it registers, so each
        recording hour leaked ~18,000 entries (8-16 MB). Now rows are rebuilt
        only when their text changes (and the old ones are released), and the
        meter rows are retitled in place.
        """
        root = self._mi.get("health_root")
        if root is None:
            return
        reports = self._health.snapshot()
        rows = [rep.one_line() for rep in reports] or ["(no checks run yet)"]
        meters = dict(self._live_meters)

        if rows != self._health_rows_shown or list(meters) != list(self._meter_items):
            self._rebuild_health_items(root, rows, meters)
        else:
            for label, bar in meters.items():
                text = _meter_row(label, bar)
                item = self._meter_items[label]
                if item.title != text:
                    item.title = text

        overall = self._health.overall()
        root.title = "Status" if overall is HealthState.OK else f"Status  {_GLYPH_FOR[overall]}"

    def _rebuild_health_items(self, root: Any, rows: list[str], meters: dict[str, str]) -> None:
        import rumps

        for item in [*self._health_items, *self._meter_items.values()]:
            _forget_menu_item(item)
        try:
            for key in list(root.keys()):
                del root[key]
        except Exception:  # pragma: no cover (older rumps)
            pass

        self._health_items = []
        for text in rows:
            item = rumps.MenuItem(text)
            item.set_callback(None)  # informational, not clickable
            root.add(item)
            self._health_items.append(item)
        self._meter_items = {}
        for label, bar in meters.items():
            item = rumps.MenuItem(_meter_row(label, bar))
            item.set_callback(None)
            root.add(item)
            self._meter_items[label] = item
        self._health_rows_shown = list(rows)

    def _apply_health_badge(self) -> None:
        """Show ⚠ next to the icon when something needs attention.

        Only while idle — during a recording the title carries the MM:SS timer,
        which matters more in the moment.
        """
        with self._lock:
            state = self._snapshot.state
        if state == UiState.RECORDING:
            return
        try:
            self._app.title = "⚠" if self._health.is_degraded() else ""
        except Exception:  # pragma: no cover
            logger.exception("Could not set health badge")

    def _on_reconnect_calendar(self, _sender: Any) -> None:
        """Run the interactive OAuth flow for every configured account.

        This used to explain itself in a modal alert first, which stopped the
        app's timers until it was dismissed (see _start_timer). A notification
        says the same without blocking anything.
        """
        self._notifications.notify(
            NotificationType.STATUS,
            "Reconnecting Google Calendar",
            "Your browser will open — sign in there, once per account.",
            force=True,
        )
        threading.Thread(
            target=self._run_calendar_reauth, name="otis-calendar-reauth", daemon=True
        ).start()

    def _run_calendar_reauth(self) -> None:
        pollers = list(getattr(self._detector, "calendar_pollers", ()) or ())
        if not pollers:
            self._notifications.notify(
                NotificationType.ERROR, "No calendar configured",
                "Add an account under detection.calendar.accounts first.", force=True,
            )
            return
        for poller in pollers:
            try:
                poller.authenticate(headless=False)
                poller.notify_auth_recovered()
            except Exception as exc:
                logger.exception("Calendar re-auth failed")
                self._notifications.notify(
                    NotificationType.ERROR, "Calendar reconnect failed", str(exc),
                    force=True,
                )
                return
        self._notifications.notify(
            NotificationType.TRANSCRIPTION_COMPLETE, "Calendar reconnected",
            f"{len(pollers)} account(s) authorised.", force=True,
        )

    def _on_fix_audio(self, _sender: Any) -> None:
        """Re-check system-audio routing, show the result, and open the fix.

        This used to explain the result in a modal alert and only update the
        panel after it was dismissed. Twice that alert was never answered and
        the instance stayed frozen (see _start_timer). Now the panel repaints
        at once, one notification carries the explanation, and the settings
        that fix the problem open.
        """
        from src.audio.blackhole_check import RoutingState

        status = self._check_routing()
        before = self._health.get(SYSTEM_AUDIO)
        # A click always gets an answer: nothing raised by this check counts
        # against a recording's toast allowance.
        self._user_check_in_progress = True
        try:
            self._report_audio_routing(status)
            after = self._health.get(SYSTEM_AUDIO)
            # Repaint now rather than on the next 100 ms tick. A state change
            # also raises its own "problem"/"recovered" toast in the drain.
            self._drain_main_queue(None)
        finally:
            self._user_check_in_progress = False

        toasted_by_registry = (
            after is not None
            and (before.state if before else HealthState.UNSET) is not after.state
            and after.state is not HealthState.UNKNOWN
            and not (before is None and after.state is HealthState.OK)
        )
        if not toasted_by_registry:
            self._notifications.notify(
                NotificationType.STATUS, "System audio", status.message(), force=True,
            )

        if status.state is RoutingState.NOT_ROUTED:
            self._open_url(SOUND_SETTINGS_URL)
        elif status.state is RoutingState.INAUDIBLE:
            self._open_app("Audio MIDI Setup")

    def _report_audio_routing(self, status: Any) -> None:
        """Fold a routing check result into the health registry."""
        from src.audio.blackhole_check import RoutingState

        self._last_route = (status.state, status.default_output_name)

        if status.state is RoutingState.OK:
            self._health.report(
                SYSTEM_AUDIO, label="System audio", state=HealthState.OK,
                detail=f"routed via {status.default_output_name}",
            )
        elif status.state is RoutingState.UNKNOWN:
            self._health.report(
                SYSTEM_AUDIO, label="System audio", state=HealthState.UNKNOWN,
                detail="could not check routing", hint=status.message(),
            )
        elif status.state is RoutingState.INAUDIBLE:
            # Otis records fine; it is the user who hears nothing. Labelling
            # this "not routed" told them the opposite of what was wrong.
            self._health.report(
                SYSTEM_AUDIO, label="System audio", state=HealthState.WARN,
                detail=f"recording, but no speakers in {status.default_output_name}",
                hint=status.message(),
            )
        else:
            self._health.report(
                SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                detail=f"not routed (output: {status.default_output_name})",
                hint=status.message(),
            )


def _is_permission_failure(rep: HealthReport) -> bool:
    return rep.state is HealthState.FAIL and rep.detail.startswith("access ")

