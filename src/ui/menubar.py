"""macOS menu-bar UI for Otis.

Architecture
------------
``rumps`` requires the main thread for the menu bar event loop, but our
detection (process monitor + calendar pollers) and recording engine run on
background threads and fire callbacks from those threads. To stay safe we use
a single-direction bridge:

    background thread ── push event ──▶ thread-safe queue
                                           │
                                           ▼
    main thread (rumps.Timer 100ms) ── drains queue ─▶ updates menu/icon

Every menu-bar mutation (icon swap, menu visibility, title text) happens on
the main thread inside the timer or inside a user-clicked callback, both of
which rumps already guarantees are main-thread.

What's wired up
---------------
* IDLE / APPROACHING / DETECTED (blink) / RECORDING / PAUSED / PROCESSING /
  OFF_HOURS — six visual states.
* Menu items toggle visibility based on state (Pause/Resume/Stop only show
  when relevant).
* Recording duration timer updates the menu-bar title text once per second.
* Working-day / working-hour windowing pauses detection outside the schedule.
* Settings (Whisper model, working days, app whitelist) are toggled in-place
  and persisted to ``~/.otis/config.yaml`` — schema changes that need a
  ProcessMonitor rebuild (whitelist edits) take effect on next launch and we
  surface that with a notification.
* The transcription handler is pluggable: pass ``transcription_handler`` to
  the constructor (``main.py`` wires the real mlx-whisper pipeline from
  ``src.pipeline``); the default no-op just logs, which keeps bare test
  instances recording-only.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import single_instance
from src.audio.recorder import DualStreamRecorder, RecorderState
from src.audio.wav_repair import repair_if_needed
from src.config import Config
from src.detection.detector import (
    MeetingContext,
    MeetingDetector,
)
from src.health import (
    CALENDAR,
    SYSTEM_AUDIO,
    HealthRegistry,
    HealthState,
    get_health,
)
from src.schedule import is_within_working_hours, working_hours_from_config
from src.storage.sessions import (
    _format_session_label,
    _resolve_session_audio,
    find_orphan_sessions,
)
from src.storage.transcript_store import TranscriptStore
from src.storage.transcription_queue import TranscriptionQueue
from src.ui.icons import ensure_icons
from src.ui.menubar_health import HealthPanelMixin
from src.ui.menubar_settings import SettingsMenuMixin
from src.ui.menubar_state import (
    BLINK_INTERVAL,
    DURATION_INTERVAL,
    HOUR_END_PRESETS,
    HOUR_START_PRESETS,
    KNOWN_MEETING_APPS,
    MAX_RUNNING_CANDIDATES,
    QUEUE_DRAIN_INTERVAL,
    SCHEDULE_CHECK_INTERVAL,
    SUPPORTED_LANGUAGES,
    WEEKDAY_NAMES,
    WHISPER_MODELS,
    UiSnapshot,
    UiState,
    format_duration_mmss,
    running_app_names,
    whitelist_candidates,
    write_user_config_override,
)
from src.ui.notifications import (
    NotificationManager,
    NotificationType,
    format_process_disappeared,
    format_transcription_complete,
)
from src.ui.rumps_support import (
    _alert,
    _start_timer,
)

logger = logging.getLogger(__name__)

#: How long a terminate signal may take to end the process before the
#: watchdog forces it (see _escalate_unhandled_quit). Stopping a recording
#: can take ~14 s (two writer joins of up to 5 s, plus metadata).
QUIT_GRACE_SECONDS = 20.0

TranscriptionHandler = Callable[[dict[str, Any]], None]

#: Re-exported so ``src.ui.menubar`` stays the one public name for the menu
#: bar after the split (audit item 42). Listed explicitly rather than left as
#: bare imports, which a lint --fix will happily delete as unused.
__all__ = [
    "BLINK_INTERVAL",
    "DURATION_INTERVAL",
    "HOUR_END_PRESETS",
    "HOUR_START_PRESETS",
    "KNOWN_MEETING_APPS",
    "MAX_RUNNING_CANDIDATES",
    "QUEUE_DRAIN_INTERVAL",
    "SCHEDULE_CHECK_INTERVAL",
    "SUPPORTED_LANGUAGES",
    "WEEKDAY_NAMES",
    "WHISPER_MODELS",
    "MenuBarApp",
    "TranscriptionHandler",
    "UiSnapshot",
    "UiState",
    "_format_session_label",
    "_resolve_session_audio",
    "find_orphan_sessions",
    "format_duration_mmss",
    "running_app_names",
    "whitelist_candidates",
    "write_user_config_override",
]


class MenuBarApp(HealthPanelMixin, SettingsMenuMixin):
    """The thing you see in the menu bar.

    Parameters
    ----------
    config:
        Loaded :class:`Config`.
    detector:
        Pre-built :class:`MeetingDetector` (the live app constructs one in
        ``main.py``; tests pass a fake or a barebones one).
    recorder_factory:
        Callable producing a :class:`DualStreamRecorder` per recording session.
    notifications:
        Optional pre-built :class:`NotificationManager` (defaults to a fresh one).
    transcription_handler:
        Called with the recorder's metadata dict after Stop. Phase 4 replaces
        the default no-op with the real transcription pipeline.
    icons_dir:
        Where icons live (defaults to ``~/.otis/icons``).
    user_config_path:
        Where to persist Settings changes (defaults to ``~/.otis/config.yaml``).
    surface_request_path:
        Where a second launch asks this copy to show itself (``~/.otis/surface``
        in real use; None disables the check).
    """

    def __init__(
        self,
        *,
        config: Config,
        detector: MeetingDetector,
        recorder_factory: Callable[[Config], DualStreamRecorder],
        notifications: NotificationManager | None = None,
        transcription_handler: TranscriptionHandler | None = None,
        transcript_store: TranscriptStore | None = None,
        audio_dir: Path | None = None,
        icons_dir: Path | None = None,
        user_config_path: Path | None = None,
        rumps_app_factory: Callable[..., Any] | None = None,
        health: HealthRegistry | None = None,
        surface_request_path: Path | None = None,
        routing_check: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        # A second launch drops a file here instead of starting another copy
        # (src/single_instance.py); the drain timer answers it.
        self._surface_request_path = surface_request_path
        self._routing_check = routing_check
        self._health = health if health is not None else get_health()
        self._detector = detector
        self._recorder_factory = recorder_factory
        self._notifications = notifications or NotificationManager()
        self._transcribe = transcription_handler or _default_transcription_handler
        self._store = transcript_store
        self._audio_dir = (
            audio_dir
            or Path(config.get("storage", "audio_dir", default="~/Otis/audio"))
        ).expanduser()
        self._icons_dir = (icons_dir or Path("~/.otis/icons")).expanduser()
        self._user_config_path = (
            user_config_path or Path("~/.otis/config.yaml")
        ).expanduser()

        # Cross-thread bridge: callbacks push tuples onto this queue, the
        # main-thread drain timer pops and reacts.
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._lock = threading.RLock()

        # Menu contents come from disk, so they are scanned on a worker and
        # rendered from the drain timer. Requests coalesce: a scan already in
        # flight picks up later requests instead of stacking threads.
        self._scan_lock = threading.Lock()
        self._scan_thread: threading.Thread | None = None
        self._scan_pending = False

        self._snapshot = UiSnapshot(
            selected_language_code=config.get("transcription", "language"),
        )
        self._recorder: DualStreamRecorder | None = None
        # Set while the start worker is opening devices, so a second click
        # (or a detector event) cannot launch a competing recorder.
        self._starting = False
        # What to fall back to if starting fails — the user may have been in
        # DETECTED or APPROACHING, not IDLE.
        self._state_before_starting: str = UiState.IDLE
        self._blink_phase = 0  # 0 / 1, used by blink timer

        # Set by "Transcribe Now" to cut a deferral short.
        self._skip_deferral = threading.Event()
        # Survives a quit or a crash mid-transcription; drained by run().
        # Sited next to the config so it lands in ~/.otis/queue in real use and
        # inside the tmp dir under test, without a separate knob.
        self._queue = TranscriptionQueue(
            self._user_config_path.parent / "queue", audio_dir=self._audio_dir
        )
        self._icons = ensure_icons(self._icons_dir)
        self._app = self._build_app(rumps_app_factory)
        self._wire_detector_callbacks()
        for poller in getattr(self._detector, "calendar_pollers", ()) or ():
            poller.is_active = self._calendar_should_poll
        # Health transitions become notifications and a menu refresh. The
        # registry only fires on an actual state change, so a subsystem can
        # report on every poll tick without producing a single extra toast.
        self._health.on_change(self._cb_health_changed)
        # Live input meters while recording, kept apart from the registry so
        # they never overwrite a row's reason (see _refresh_level_meters).
        self._live_meters: dict[str, str] = {}
        self._begin_recording_toasts()
        # The routing verdict when the current recording started (set by the
        # start worker), and whether a user-initiated check is running; both
        # read by the per-recording toast cap.
        self._route_at_recording_start: Any = None
        self._user_check_in_progress = False
        # (state, output name) of the last routing check that was reported.
        self._last_route: tuple[Any, str] | None = None
        self._health_rows_shown: list[str] = []
        self._health_items: list[Any] = []
        self._meter_items: dict[str, Any] = {}
        self._refresh_health_menu()
        # Brand the *application* icon (used in the About box, the force-quit
        # window, Notification Center, etc.) so it isn't the Python framework
        # logo when we run via `python -m src.main`. When we run from Otis.app
        # the bundle's CFBundleIconFile already wins; calling this is a no-op
        # there.
        self._brand_application_icon()

    # =====================================================================
    # Public API
    # =====================================================================
    def run(self) -> None:
        """Start detection + the rumps event loop. Blocks until Quit."""
        n_pollers = len(getattr(self._detector, "calendar_pollers", ()))
        logger.info(
            "Starting MenuBarApp (cwd=%s, pollers=%d, icons=%s).",
            os.getcwd(), n_pollers, self._icons_dir,
        )
        self._warn_if_blackhole_missing()
        self._drain_transcription_queue()
        # First fill of the two disk-backed menus. Deliberately here and not
        # in __init__: it must not run before the drain timer exists to render
        # the result, and constructing the app (tests included) should not
        # start touching disk on its own.
        self._request_menu_rescan()

        # Honour the working-hours window from the very first tick — otherwise
        # we'd run detection for up to ``SCHEDULE_CHECK_INTERVAL`` seconds even
        # though we're outside the window.
        in_window = is_within_working_hours(
            datetime.now(), **working_hours_from_config(self._config)
        )
        if in_window:
            self._detector.start()
        else:
            logger.info("Launched outside working hours; detection paused.")
            self._set_state(UiState.OFF_HOURS)
            # The pollers stay stopped, but a missing token is still news: say
            # so now rather than at the first working-hours poll.
            for poller in getattr(self._detector, "calendar_pollers", ()) or ():
                try:
                    poller.report_credentials_offline()
                except Exception:
                    logger.exception("Offline calendar credential check raised")

        # A menu Quit (and a logout) ends in NSApp.terminate_, which calls
        # exit(): the finally below never runs. applicationWillTerminate_
        # does fire first, and rumps relays it as before_quit.
        import rumps

        rumps.events.before_quit.register(self._on_before_quit)
        self._install_terminate_signals()
        try:
            self._app.run()
        finally:
            self._teardown()

    def _install_terminate_signals(self) -> None:
        """Make `kill` (SIGTERM/SIGHUP) a normal Quit.

        The Python process is not the app LaunchServices tracks (the applet
        is), so at logout or shutdown it can get a bare SIGTERM — which ended
        it instantly: no quit line, no before_quit, a running recording's WAV
        left for wav_repair. Python runs signal handlers on the main thread
        between bytecodes; the 100 ms drain timer guarantees there are some.
        """
        import signal

        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._on_terminate_signal)
            except (ValueError, OSError):  # not the main thread (tests)
                logger.debug("Could not install a handler for %s", sig)
                return
        # The handler above needs the main thread to run Python. Frozen in
        # native code, it never would — and `kill PID`, which used to end the
        # process outright, would do nothing. The C-level handler still writes
        # the signal number to a wakeup fd; a watchdog thread reads it and
        # forces the exit if the normal quit has not finished in time.
        try:
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
        except (ValueError, OSError):
            logger.debug("No wakeup fd; a frozen copy needs kill -9", exc_info=True)
            return
        threading.Thread(
            target=self._escalate_unhandled_quit, args=(read_fd,),
            name="otis-quit-watchdog", daemon=True,
        ).start()

    def _escalate_unhandled_quit(
        self,
        read_fd: int,
        *,
        grace: float = QUIT_GRACE_SECONDS,
        exit_fn: Callable[[int], Any] = os._exit,
    ) -> None:
        """Force the exit if a terminate signal is not acted on within ``grace``.

        A healthy quit ends the process long before (stopping a recording takes
        up to ~14 s, hence the margin), taking this thread with it.
        """
        import faulthandler
        import signal

        wanted = {int(signal.SIGTERM), int(signal.SIGHUP)}
        while True:
            try:
                data = os.read(read_fd, 64)
            except OSError:
                return
            if not data:
                return
            if any(b in wanted for b in data):
                break
        time.sleep(grace)
        logger.error(
            "The main thread did not quit within %.0fs of a terminate signal; "
            "forcing exit. All thread stacks follow on stderr (launch.log).", grace,
        )
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:  # pragma: no cover (closed stderr)
            pass
        exit_fn(143)  # 128 + SIGTERM

    def _on_terminate_signal(self, signum: int, _frame: Any) -> None:
        import signal

        import rumps

        logger.warning("Otis received %s; quitting.", signal.Signals(signum).name)
        rumps.quit_application()

    def notify_transcription_progress(self, pct: int) -> None:
        """Surface transcription progress in the menu-bar title (thread-safe).

        Called by the transcription handler from its worker thread; the
        actual title mutation happens on the main thread via the event queue.
        """
        self._events.put(("progress", int(pct)))

    # ---- read-only views, mostly for tests ----
    @property
    def snapshot(self) -> UiSnapshot:
        return self._snapshot

    @property
    def recorder(self) -> DualStreamRecorder | None:
        return self._recorder

    # =====================================================================
    # rumps construction
    # =====================================================================
    def _build_app(self, rumps_app_factory: Callable[..., Any] | None) -> Any:
        """Lazily import rumps so non-macOS test envs can import this module."""
        if rumps_app_factory is None:
            import rumps
            rumps_app_factory = rumps.App

        app = rumps_app_factory(
            "Otis",
            icon=str(self._icons["idle"]),
            quit_button=None,  # we'll add our own to run a clean shutdown
        )

        # Build menu items — store handles so handlers can mutate them.
        self._mi: dict[str, Any] = {}
        self._build_menu(app)

        # Periodic timers (main-thread).
        import rumps

        self._timer_drain = rumps.Timer(self._drain_main_queue, QUEUE_DRAIN_INTERVAL)
        self._timer_blink = rumps.Timer(self._on_blink_tick, BLINK_INTERVAL)
        self._timer_duration = rumps.Timer(self._on_duration_tick, DURATION_INTERVAL)
        self._timer_schedule = rumps.Timer(self._on_schedule_tick, SCHEDULE_CHECK_INTERVAL)

        _start_timer(self._timer_drain)
        _start_timer(self._timer_schedule)
        # Blink + duration are started on demand when entering DETECTED / RECORDING.

        return app

    def _build_menu(self, app: Any) -> None:
        import rumps

        # Top-level controls
        self._mi["start"] = rumps.MenuItem("Start Recording", callback=self._on_start_clicked)
        self._mi["pause"] = rumps.MenuItem("Pause Recording", callback=self._on_pause_clicked)
        self._mi["resume"] = rumps.MenuItem("Resume Recording", callback=self._on_resume_clicked)
        self._mi["stop"] = rumps.MenuItem("Stop & Transcribe", callback=self._on_stop_clicked)
        # Only shown while DEFERRED: cuts the wait short when the user would
        # rather have the transcript than a cool laptop.
        self._mi["transcribe_now"] = rumps.MenuItem(
            "Transcribe Now", callback=self._on_transcribe_now_clicked
        )

        # Language submenu
        self._mi["language_root"] = rumps.MenuItem("Language: Auto-detect")
        self._lang_items: dict[str, rumps.MenuItem] = {}
        for label, code in SUPPORTED_LANGUAGES:
            it = rumps.MenuItem(label, callback=self._on_language_picked)
            it._otis_lang_code = code
            self._mi["language_root"].add(it)
            self._lang_items[label] = it
        self._update_language_menu(self._snapshot.selected_language_code)

        # Folder / web shortcuts
        self._mi["open_web"] = rumps.MenuItem(
            "Open Transcripts", callback=self._on_open_transcripts_web
        )
        self._mi["open_folder"] = rumps.MenuItem(
            "Open Transcripts Folder", callback=self._on_open_transcripts_folder
        )

        # Settings submenu
        self._mi["settings_root"] = rumps.MenuItem("Settings")
        self._build_settings_submenu(self._mi["settings_root"])

        # Generate Transcript — populated with audio sessions on disk that
        # don't yet have a transcript (e.g. recordings whose Stop & Transcribe
        # crashed mid-flight, leaving orphan WAVs on disk).
        self._mi["generate_root"] = rumps.MenuItem("Generate Transcript")
        self._render_generate_menu(None)

        # Recent transcripts — populated from disk now and refreshed after
        # each transcription completes.
        self._mi["recent_root"] = rumps.MenuItem("Recent Transcripts")
        self._render_recent_menu(None)

        # Status — the panel whose absence let three failures run silently for
        # months. Every subsystem reports here; a degraded component also puts
        # a ⚠ next to the icon so it's visible without opening the menu.
        self._mi["health_root"] = rumps.MenuItem("Status")
        self._mi["reconnect_calendar"] = rumps.MenuItem(
            "Reconnect Calendar…", callback=self._on_reconnect_calendar
        )
        self._mi["fix_audio"] = rumps.MenuItem(
            "Fix System Audio…", callback=self._on_fix_audio
        )

        # About / Quit
        self._mi["about"] = rumps.MenuItem("About Otis", callback=self._on_about)
        self._mi["quit"] = rumps.MenuItem("Quit", callback=self._on_quit_clicked)

        app.menu = [
            self._mi["health_root"],
            self._mi["reconnect_calendar"],
            self._mi["fix_audio"],
            None,
            self._mi["start"],
            self._mi["pause"],
            self._mi["resume"],
            self._mi["stop"],
            self._mi["transcribe_now"],
            None,  # separator
            self._mi["language_root"],
            None,
            self._mi["open_web"],
            self._mi["open_folder"],
            None,
            self._mi["settings_root"],
            None,
            self._mi["generate_root"],
            self._mi["recent_root"],
            None,
            self._mi["about"],
            self._mi["quit"],
        ]
        self._refresh_menu_visibility()

    def _wire_detector_callbacks(self) -> None:
        self._detector.on_meeting_approaching(self._cb_approaching)
        self._detector.on_meeting_detected(self._cb_detected)
        self._detector.on_meeting_ended(self._cb_ended)
        self._detector.on_process_disappeared(self._cb_process_gone)

    def _cb_approaching(self, ctx: MeetingContext) -> None:
        logger.debug("detector → approaching: %s", ctx.title)
        self._events.put(("approaching", ctx))

    def _cb_detected(self, ctx: MeetingContext) -> None:
        logger.debug("detector → detected: app=%s title=%s", ctx.app, ctx.title)
        self._events.put(("detected", ctx))

    def _cb_ended(self, ctx: MeetingContext) -> None:
        logger.debug("detector → ended: %s", ctx.title or ctx.app)
        self._events.put(("ended", ctx))

    def _cb_process_gone(self, app_name: str) -> None:
        logger.debug("detector → process gone: %s", app_name)
        self._events.put(("process_gone", app_name))

    def _cb_device_error(self, stream_label: str, exc: Exception) -> None:
        """Recorder writer thread reports a dead capture stream (bg thread)."""
        logger.error("recorder → device error on %s stream: %s", stream_label, exc)
        self._events.put(("device_error", (stream_label, exc)))

    # =====================================================================
    # Main-thread queue drain (rumps.Timer)
    # =====================================================================
    def _drain_main_queue(self, _timer: Any) -> None:
        self._sync_recorder_state()
        self._answer_surface_request()
        while True:
            try:
                event_type, payload = self._events.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_event(event_type, payload)
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Error handling main-thread event %s", event_type)

    def _answer_surface_request(self) -> None:
        """Someone opened Otis.app again: show them the copy that is running.

        A menu-bar app has no window to bring forward, and the icon may sit
        behind the menu-bar overflow — the user relaunched precisely because
        they could not see it. So: a notification that says so, then the menu
        opened as if the icon had been clicked.
        """
        path = self._surface_request_path
        if path is None or not single_instance.take_surface_request(path):
            return
        logger.info("Otis.app was opened again; showing the running copy.")
        self._notifications.notify(
            NotificationType.ALREADY_RUNNING,
            "Otis is already running",
            "Its icon is in the menu bar — the menu is open now. "
            "If you can't see it, close a few menu-bar apps.",
            force=True,
        )
        try:
            # Scheduled, not called: performClick_ runs menu tracking right
            # here, inside the drain timer's callout, and a timer never
            # re-fires while its callout is on the stack — an alert picked
            # from that menu would freeze the drain again.
            self._app._nsapp.nsstatusitem.button().performSelector_withObject_afterDelay_(
                "performClick:", None, 0.0
            )
        except Exception:  # headless tests, or rumps internals changed
            logger.debug("Could not open the status menu", exc_info=True)

    def _sync_recorder_state(self) -> None:
        """Mirror recorder-initiated pause/resume into the UI.

        The recorder pauses/resumes itself on system sleep/wake (the pause on
        the main thread, the resume on a short-lived worker). Without this
        sync the icon would keep
        showing RECORDING while the recorder is paused. User-initiated
        clicks already update both sides on the main thread, so this only
        fires for the sleep/wake transitions.
        """
        with self._lock:
            recorder = self._recorder
            ui_state = self._snapshot.state
        if recorder is None:
            return
        if recorder.state == RecorderState.PAUSED and ui_state == UiState.RECORDING:
            logger.info("Recorder paused itself (system sleep); updating UI.")
            self._set_state(UiState.PAUSED)
        elif recorder.state == RecorderState.RECORDING and ui_state == UiState.PAUSED:
            logger.info("Recorder resumed itself (system wake); updating UI.")
            self._set_state(UiState.RECORDING)
            _start_timer(self._timer_duration)

    def _handle_event(self, event_type: str, payload: Any) -> None:
        # Background detector signals must never demote the UI out of an
        # active "user task" state. Without this guard, a routine 30s process
        # rescan or a 2-min calendar alert mid-transcription would flip the
        # icon from PROCESSING (blue) to DETECTED (orange ticking) and hide
        # the Stop button — even though the recorder/transcription worker is
        # still running.
        #
        # During RECORDING/PAUSED, we still want the heads-up notification
        # for a NEW back-to-back meeting (so the user knows another call is
        # coming), but we don't change the icon or menu visibility.
        # During PROCESSING, we suppress entirely — a "Meeting detected"
        # toast mid-transcription is noise.
        if event_type in ("approaching", "detected"):
            with self._lock:
                ui_state = self._snapshot.state
            if ui_state == UiState.PROCESSING:
                logger.debug(
                    "Suppressing detector %s event while UI is PROCESSING.",
                    event_type,
                )
                return
            # DEFERRED deliberately falls through: nothing is running, and the
            # whole reason we are waiting is that a call is in progress — so
            # the next meeting is exactly what the user needs to hear about.
            if ui_state in (UiState.RECORDING, UiState.PAUSED):
                # Heads-up only: notify, don't change state.
                ctx = payload
                if event_type == "approaching":
                    self._notifications.notify(
                        NotificationType.MEETING_APPROACHING,
                        "Meeting in 2 min",
                        ctx.title or "(untitled)",
                    )
                else:
                    self._notifications.notify(
                        NotificationType.MEETING_DETECTED,
                        "Meeting detected",
                        ctx.title or ctx.app or "(unknown app)",
                    )
                return

        if event_type == "approaching":
            ctx = payload
            logger.info(
                "UI ← APPROACHING (%s, link=%s)",
                ctx.title or "(untitled)", ctx.meeting_link or "-",
            )
            self._set_state(UiState.APPROACHING, current_meeting=ctx)
            self._notifications.notify(
                NotificationType.MEETING_APPROACHING,
                "Meeting in 2 min",
                ctx.title or "(untitled)",
            )
        elif event_type == "detected":
            ctx = payload
            logger.info(
                "UI ← DETECTED (app=%s, title=%s)",
                ctx.app, ctx.title or "(no calendar)",
            )
            self._set_state(UiState.DETECTED, current_meeting=ctx)
            self._start_blink()
            self._notifications.notify(
                NotificationType.MEETING_DETECTED,
                "Meeting detected",
                ctx.title or ctx.app or "(unknown app)",
            )
        elif event_type == "ended":
            # Only relevant if we never started recording — the detector keeps
            # us in RECORDING when a process exits mid-recording (advisory path).
            # Read under the lock like every other branch: a recorder worker
            # can be writing this field from another thread.
            with self._lock:
                ended_state = self._snapshot.state
            if ended_state in (UiState.APPROACHING, UiState.DETECTED):
                self._set_state(UiState.IDLE)
                self._stop_blink()
        elif event_type == "process_gone":
            app_name: str = payload
            title, body = format_process_disappeared(app_name)
            self._notifications.notify(
                NotificationType.PROCESS_DISAPPEARED, title, body
            )
        elif event_type == "device_error":
            # A capture stream died mid-recording (device unplugged, disk
            # error). Don't auto-stop — the other stream may still be
            # capturing fine; the user decides whether to Stop & Transcribe.
            stream_label, exc = payload
            self._notifications.notify(
                NotificationType.ERROR,
                f"{stream_label.capitalize()} stream failed",
                f"{exc} — the other stream keeps recording. "
                "Stop & Transcribe to save what was captured.",
                force=True,
            )
        elif event_type == "force_state":
            # Used by background threads (e.g. the transcription worker) to
            # request a state change on the main thread.
            self._set_state(payload)
        elif event_type == "progress":
            # Live transcription progress next to the blue PROCESSING icon.
            # Ignore stale events that arrive after the state moved on —
            # _set_state already cleared the title at that transition.
            if self._snapshot.state == UiState.PROCESSING:
                try:
                    self._app.title = f"{payload}%"
                except Exception:  # pragma: no cover
                    logger.exception("Could not set progress title")
        elif event_type == "health":
            self._handle_health_change(*payload)
        elif event_type == "refresh_recent":
            # The transcription worker just landed (or failed); rescan so the
            # new transcript shows up under Recent and the session it came
            # from disappears from Generate Transcript.
            self._request_menu_rescan()
        elif event_type == "recorder_started":
            self._handle_recorder_started(*payload)
        elif event_type == "recorder_start_failed":
            self._handle_recorder_start_failed(payload)
        elif event_type == "recorder_stopped":
            self._handle_recorder_stopped(payload)
        elif event_type == "recorder_stop_failed":
            self._handle_recorder_stop_failed(payload)
        elif event_type == "render_menus":
            recent, orphans = payload
            self._render_recent_menu(recent)
            self._render_generate_menu(orphans)

    # =====================================================================
    # User-clicked menu callbacks (main-thread, courtesy of rumps)
    # =====================================================================
    def _on_start_clicked(self, _sender: Any) -> None:
        self._do_start_recording()

    def _on_pause_clicked(self, _sender: Any) -> None:
        with self._lock:
            if self._recorder is None:
                return
            self._recorder.pause()
        self._set_state(UiState.PAUSED)
        self._notifications.notify(
            NotificationType.RECORDING_PAUSED, "Recording paused", ""
        )

    def _on_resume_clicked(self, _sender: Any) -> None:
        with self._lock:
            if self._recorder is None:
                return
            self._recorder.resume()
        self._set_state(UiState.RECORDING)
        _start_timer(self._timer_duration)

    def _on_stop_clicked(self, _sender: Any) -> None:
        self._do_stop_and_transcribe()

    def _on_open_transcripts_web(self, _sender: Any) -> None:
        host = self._config.get("web", "host", default="127.0.0.1")
        port = self._config.get("web", "port", default=8765)
        self._open_url(f"http://{host}:{port}")

    def _on_open_transcripts_folder(self, _sender: Any) -> None:
        path = Path(self._config.get(
            "storage", "transcript_dir", default="~/Otis/transcripts"
        )).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._open_in_finder(path)

    def _on_about(self, _sender: Any) -> None:
        _alert(
            title="About Otis",
            message="Otis — local macOS meeting transcription.\n\n"
                    "Mic + system audio recording, on-device Whisper "
                    "transcription, MCP integration with Claude.",
        )

    def _on_quit_clicked(self, _sender: Any) -> None:
        import rumps

        logger.info("Quit chosen from the menu.")

        # If a transcription is mid-flight, the worker is a daemon thread —
        # it will be killed the moment we exit, losing the transcript. Warn
        # the user explicitly so they can wait it out.
        with self._lock:
            ui_state = self._snapshot.state
        if ui_state in (UiState.PROCESSING, UiState.DEFERRED):
            # These used to be the same state, so the dialog said a
            # transcription was running during a wait where nothing had
            # started — frightening the user into waiting out a job that had
            # not begun, for up to two hours. The recording is on disk and
            # queued either way; only PROCESSING actually loses work.
            if ui_state == UiState.PROCESSING:
                message = (
                    "A transcription is running. Quitting now will discard it "
                    "and it will restart next launch. Wait for it to finish, "
                    "or quit anyway?"
                )
            else:
                message = (
                    "A recording is waiting for your call to finish before it "
                    "transcribes. Nothing is running — it is saved and will "
                    "resume next launch. Quit?"
                )
            response = _alert(
                title="Transcription pending",
                message=message,
                ok="Quit anyway",
                cancel="Wait",
            )
            # rumps.alert returns 1 for the OK button, 0 for cancel.
            if not response:
                return

        # Stop recording cleanly first; teardown happens in run()'s finally.
        self._do_stop_if_recording()
        rumps.quit_application()

    # =====================================================================
    # Periodic timer callbacks
    # =====================================================================
    def _on_blink_tick(self, _timer: Any) -> None:
        if self._snapshot.state != UiState.DETECTED:
            self._stop_blink()
            return
        self._blink_phase ^= 1
        icon_state = "approaching" if self._blink_phase else "idle"
        self._set_icon(icon_state)

    def _on_duration_tick(self, _timer: Any) -> None:
        if self._snapshot.state != UiState.RECORDING:
            self._timer_duration.stop()
            self._app.title = ""
            self._clear_level_meters()
            self._apply_health_badge()
            return
        self._app.title = self._snapshot.title_text()
        self._refresh_level_meters()

    def _calendar_should_poll(self) -> bool:
        """The pollers' own working-hours gate (called on their threads).

        The schedule tick below is what normally stops them, but it skips four
        states and stops with a stuck main loop. A recording that runs past the
        window keeps its calendar, for the meeting-ended signal.
        """
        if self._snapshot.state in (UiState.RECORDING, UiState.PAUSED):
            return True
        return is_within_working_hours(
            datetime.now(), **working_hours_from_config(self._config)
        )

    def _on_schedule_tick(self, _timer: Any) -> None:
        """Move into / out of OFF_HOURS based on the working-hours window.

        Critical: an active recording must NEVER be interrupted by this. If the
        user starts recording during off-hours (manually) we leave the state
        machine alone so the recorder UI keeps working. The off-hours
        transition only happens from neutral states (IDLE / APPROACHING /
        DETECTED).
        """
        in_window = is_within_working_hours(
            datetime.now(), **working_hours_from_config(self._config)
        )
        with self._lock:
            current_state = self._snapshot.state

        # Don't fiddle with state during active recording / paused / processing —
        # transitioning to OFF_HOURS would clear recorder context and hide the
        # Stop button.
        if current_state in (UiState.RECORDING, UiState.PAUSED,
                             UiState.PROCESSING, UiState.DEFERRED):
            return

        currently_off = current_state == UiState.OFF_HOURS
        if not in_window and not currently_off:
            logger.info("Entering off-hours; pausing detector.")
            try:
                self._detector.stop()
            except Exception:
                logger.exception("Failed to stop detector for off-hours")
            self._set_state(UiState.OFF_HOURS)
        elif in_window and currently_off:
            logger.info("Exiting off-hours; resuming detector.")
            try:
                self._detector.start()
            except Exception:
                logger.exception("Failed to start detector after off-hours")
            self._set_state(UiState.IDLE)

    # =====================================================================
    # State + visual update primitives
    # =====================================================================
    def _set_state(
        self,
        state: str,
        *,
        current_meeting: MeetingContext | None = None,
    ) -> None:
        with self._lock:
            self._snapshot.state = state
            if current_meeting is not None:
                self._snapshot.current_meeting = current_meeting
            if state == UiState.RECORDING:
                if self._snapshot.recording_started_monotonic is None:
                    self._snapshot.recording_started_monotonic = time.monotonic()
                elif self._snapshot.paused_at_monotonic is not None:
                    # Resuming: shift the anchor forward by the pause length
                    # so the MM:SS counter excludes time spent paused.
                    self._snapshot.recording_started_monotonic += (
                        time.monotonic() - self._snapshot.paused_at_monotonic
                    )
                self._snapshot.paused_at_monotonic = None
            elif state == UiState.PAUSED:
                if self._snapshot.paused_at_monotonic is None:
                    self._snapshot.paused_at_monotonic = time.monotonic()
            elif state in (UiState.IDLE, UiState.OFF_HOURS):
                self._snapshot.recording_started_monotonic = None
                self._snapshot.paused_at_monotonic = None
                self._snapshot.current_meeting = None

        self._set_icon(state)
        # Clear the menu-bar title text whenever we leave RECORDING. The
        # duration tick handler also clears it, but it only fires while the
        # timer is running — once we stop the timer (in _do_stop_if_recording)
        # no further ticks happen, so the last MM:SS value would otherwise
        # stay glued next to the icon. Clearing here is the authoritative path.
        if state != UiState.RECORDING:
            try:
                self._app.title = ""
            except Exception:
                logger.exception("Could not clear menu-bar title")
        if state not in (UiState.RECORDING, UiState.PAUSED):
            self._clear_level_meters()

        self._refresh_menu_visibility()

    def _brand_application_icon(self) -> None:
        """Override the global app icon (About dialog, notifications, etc.)
        with the bundled :file:`OtisIcon.png` if it's available.

        Wrapped in try/except — any pyobjc / AppKit failure should not stop
        the menu bar from running.
        """
        try:
            from src.ui.icons import DEFAULT_SOURCE_PATH

            source = DEFAULT_SOURCE_PATH if DEFAULT_SOURCE_PATH.exists() else None
            if source is None:
                return
            from AppKit import NSApplication, NSImage

            image = NSImage.alloc().initWithContentsOfFile_(str(source))
            if image is None:
                logger.debug("Could not load app icon from %s", source)
                return
            NSApplication.sharedApplication().setApplicationIconImage_(image)
            logger.debug("Application icon set to %s", source)
        except Exception:  # pragma: no cover (non-macOS / pyobjc missing)
            logger.exception("Could not set NSApplication icon")

    def _set_icon(self, state: str) -> None:
        path = self._icons.get(state, self._icons["idle"])
        try:
            self._app.icon = str(path)
            logger.debug("icon → %s (%s)", state, path.name)
        except Exception:  # pragma: no cover
            logger.exception("Could not set icon for state %s", state)

    def _start_blink(self) -> None:
        self._blink_phase = 0
        if not self._timer_blink.is_alive():
            _start_timer(self._timer_blink)

    def _stop_blink(self) -> None:
        if self._timer_blink.is_alive():
            self._timer_blink.stop()

    def _refresh_menu_visibility(self) -> None:
        s = self._snapshot.state
        # Each rumps.MenuItem has no real "hidden" flag — we toggle visibility
        # by rebuilding visibility via .set_callback(None) for disabled items.
        # The simpler approach used here: call .hidden = True/False which rumps
        # exposes via the underlying NSMenuItem.
        self._set_visible(
            self._mi["start"],
            s in (UiState.IDLE, UiState.APPROACHING, UiState.DETECTED,
                  UiState.OFF_HOURS, UiState.DEFERRED),
        )
        self._set_visible(self._mi["pause"], s == UiState.RECORDING)
        self._set_visible(self._mi["resume"], s == UiState.PAUSED)
        self._set_visible(self._mi["stop"], s in (UiState.RECORDING, UiState.PAUSED))
        self._set_visible(self._mi["transcribe_now"], s == UiState.DEFERRED)
        # The two repair actions only appear when they'd actually do something.
        # Pollers report per account — "calendar:personal", "calendar:work" —
        # since the verification pass. This gate kept asking for the bare
        # "calendar" key, which nothing reports any more, so the item that
        # runs the re-auth never appeared while the panel above it said
        # "reconnect". Any calendar row failing is reason enough to offer it.
        self._set_visible(
            self._mi["reconnect_calendar"],
            any(
                rep.state is HealthState.FAIL
                and (rep.key == CALENDAR or rep.key.startswith(CALENDAR + ":"))
                for rep in self._health.snapshot()
            ),
        )
        audio = self._health.get(SYSTEM_AUDIO)
        self._set_visible(
            self._mi["fix_audio"],
            audio is not None and audio.state in (HealthState.WARN, HealthState.FAIL),
        )

    def _set_visible(self, menu_item: Any, visible: bool) -> None:
        try:
            menu_item.hidden = not visible  # rumps proxies to NSMenuItem.setHidden_
        except Exception:  # pragma: no cover (older rumps)
            menu_item.set_callback(menu_item._callback if visible else None)

    # =====================================================================
    # Health panel — see src/ui/menubar_health.py
    # =====================================================================
    def _render_generate_menu(self, sessions: list[dict[str, Any]] | None) -> None:
        """Draw the "Generate Transcript" submenu. Main thread, no disk.

        ``None`` means the scan hasn't landed yet — the menu exists from the
        first paint, it just says so.
        """
        import rumps

        root = self._mi.get("generate_root")
        if root is None:
            return
        try:
            for key in list(root.keys()):
                del root[key]
        except Exception:  # pragma: no cover (older rumps)
            pass

        if not sessions:
            placeholder = rumps.MenuItem(
                "(scanning…)" if sessions is None else "(no orphan recordings)"
            )
            placeholder.set_callback(None)  # render disabled
            root.add(placeholder)
            return

        for entry in sessions:
            label = entry["label"]
            item = rumps.MenuItem(label, callback=self._on_generate_clicked)
            item._otis_session_id = entry["session_id"]
            item._otis_metadata_path = str(entry["metadata_path"])
            item._otis_mic_path = str(entry["mic_path"])
            item._otis_system_path = (
                str(entry["system_path"]) if entry["system_path"] else None
            )
            root.add(item)

    # =====================================================================
    # Menu contents (scanned on a worker, rendered on the main thread)
    # =====================================================================
    def _request_menu_rescan(self) -> None:
        """Ask the worker to rescan disk and repaint the two disk-backed menus.

        Audit C5: both menus were built synchronously inside the rumps main
        thread — at startup before ``app.run()``, and again after every
        transcription. That is two rglobs of the audio tree, a JSON parse per
        session, and a store scan, on the thread that also draws the menu:
        ~1.8 s frozen at 1 000 transcripts.

        Requests coalesce rather than stack. A scan already running picks up
        anything asked for while it was working, so a burst of events costs
        one extra pass, not one thread each.
        """
        with self._scan_lock:
            self._scan_pending = True
            if self._scan_thread is not None and self._scan_thread.is_alive():
                return
            self._scan_thread = threading.Thread(
                target=self._scan_loop, name="otis-menu-scan", daemon=True
            )
            self._scan_thread.start()

    def _scan_loop(self) -> None:
        while True:
            with self._scan_lock:
                if not self._scan_pending:
                    # Clear before returning, under the lock, so the next
                    # request starts a thread instead of waiting on this one.
                    self._scan_thread = None
                    return
                self._scan_pending = False
            try:
                payload = self._scan_menus()
            except Exception:
                logger.exception("Menu scan failed")
                continue
            self._events.put(("render_menus", payload))

    def _scan_menus(
        self, *, recent_limit: int = 5, orphan_limit: int = 10,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """All the disk work behind the two menus. Worker thread only."""
        recent: list[dict[str, Any]] = []
        if self._store is not None:
            try:
                recent = self._store.list_transcripts(limit=recent_limit)
            except Exception:
                logger.exception("Could not list transcripts for the Recent menu")
        try:
            orphans = self._find_orphan_sessions(limit=orphan_limit)
        except Exception:
            logger.exception("Could not scan for orphan recordings")
            orphans = []
        return recent, orphans

    def _find_orphan_sessions(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Convenience: pull audio_dir / store / current-session from self."""
        with self._lock:
            current_sid = (
                self._recorder.session_id
                if self._recorder is not None
                else None
            )
        return find_orphan_sessions(
            audio_dir=self._audio_dir,
            store=self._store,
            current_recording_session_id=current_sid,
            limit=limit,
        )

    def _on_generate_clicked(self, sender: Any) -> None:
        """User picked an orphan/untranscribed session → fire transcription."""
        sid = getattr(sender, "_otis_session_id", None)
        meta_str = getattr(sender, "_otis_metadata_path", None)
        mic_str = getattr(sender, "_otis_mic_path", None)
        sys_str = getattr(sender, "_otis_system_path", None)
        if not sid or not mic_str:
            return

        # If the user is in the middle of a real recording / transcription,
        # don't queue a competing one — the WhisperEngine isn't designed to
        # process two streams concurrently and the icon would lie.
        with self._lock:
            current = self._snapshot.state
        # DEFERRED is not busy — nothing is running, and refusing here was
        # part of the same conflation that hid Start for up to two hours.
        if current in (UiState.RECORDING, UiState.PAUSED, UiState.PROCESSING):
            self._notifications.notify(
                NotificationType.ERROR,
                "Busy",
                "Wait for the current recording / transcription to finish.",
                force=True,
            )
            return

        metadata = self._build_recorder_metadata(
            session_id=sid,
            metadata_path=Path(meta_str) if meta_str else None,
            mic_path=Path(mic_str),
            system_path=Path(sys_str) if sys_str else None,
        )
        # The transcription_handler in main.py expects these decorations.
        metadata["_meeting"] = {"title": None, "app": None, "participants": []}
        metadata["_language"] = self._snapshot.selected_language_code

        self._set_state(UiState.PROCESSING)
        self._notifications.notify(
            NotificationType.RECORDING_STARTED,
            "Generating transcript",
            _format_session_label(
                Path(meta_str) if meta_str else None, Path(mic_str)
            ),
            force=True,
        )
        threading.Thread(
            target=self._repair_then_transcribe,
            args=(metadata, Path(mic_str), Path(sys_str) if sys_str else None),
            name="otis-on-demand-transcribe",
            daemon=True,
        ).start()

    def _repair_then_transcribe(
        self, metadata: dict[str, Any], mic_path: Path, system_path: Path | None,
    ) -> None:
        """Fix a crash-truncated header, then transcribe. Worker thread.

        Repair used to happen inside the menu build, against every recording
        on disk, every time the menu was drawn — reading 64 KB per file and
        potentially rewriting headers as a side effect of opening a menu. It
        belongs to whoever is about to read the audio, which is this: two
        files, once, on the session the user actually chose.
        """
        try:
            repair_if_needed(mic_path)
            repair_if_needed(system_path)
        except Exception:
            logger.exception("Could not repair audio for %s", mic_path)
        self._run_transcription(metadata)

    def _build_recorder_metadata(
        self,
        *,
        session_id: str,
        metadata_path: Path | None,
        mic_path: Path,
        system_path: Path | None,
    ) -> dict[str, Any]:
        """Assemble a recorder-style metadata dict for the transcription handler.

        Reads the existing metadata.json if present; otherwise synthesises one
        with reasonable defaults (zero monotonic anchors, mtime as wall clock).
        """
        if metadata_path is not None and metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    meta = dict(meta)
                else:
                    meta = {}
            except Exception:
                meta = {}
        else:
            meta = {}

        meta.setdefault("session_id", session_id)
        # Express paths relative to audio_dir so the processor's session
        # builder resolves them to the right files even if audio_dir moves.
        try:
            meta["mic_wav"] = str(mic_path.relative_to(self._audio_dir))
        except ValueError:
            meta["mic_wav"] = mic_path.name
        if system_path is not None:
            try:
                meta["system_wav"] = str(system_path.relative_to(self._audio_dir))
            except ValueError:
                meta["system_wav"] = system_path.name
        else:
            meta["system_wav"] = None
        meta.setdefault("mic_start_monotonic", 0.0)
        meta.setdefault("system_start_monotonic", 0.0)
        meta.setdefault("sample_rate", int(
            self._config.get("audio", "sample_rate", default=16000)
        ))
        if not meta.get("start_wall_clock"):
            from datetime import datetime

            try:
                ts = datetime.fromtimestamp(mic_path.stat().st_mtime, tz=UTC)
                meta["start_wall_clock"] = ts.isoformat()
            except Exception:
                meta["start_wall_clock"] = datetime.now(UTC).isoformat()
        meta.setdefault("pauses", [])
        return meta

    def _render_recent_menu(self, entries: list[dict[str, Any]] | None) -> None:
        """Draw the Recent Transcripts submenu. Main thread, no disk.

        ``None`` means the scan hasn't landed yet.
        """
        import rumps

        root = self._mi.get("recent_root")
        if root is None:
            return

        # Clear whatever's there. rumps.MenuItem supports dict-style key access.
        try:
            for key in list(root.keys()):
                del root[key]
        except Exception:  # pragma: no cover (older rumps)
            pass

        if not entries:
            root.add(rumps.MenuItem(
                "(scanning…)" if entries is None else "(no transcripts yet)"
            ))
            return

        for fm in entries:
            label = self._format_recent_label(fm)
            item = rumps.MenuItem(label, callback=self._on_recent_picked)
            item._otis_transcript_id = fm.get("id")
            root.add(item)

    @staticmethod
    def _format_recent_label(fm: dict[str, Any]) -> str:
        date = str(fm.get("date") or "")
        time_ = str(fm.get("start_time") or "")
        title = str(fm.get("title") or "(untitled)")
        prefix = f"{date} {time_}".strip() or "—"
        # Cap length so wide menus don't push off the screen.
        max_title = 50
        if len(title) > max_title:
            title = title[: max_title - 1] + "…"
        return f"{prefix}  {title}"

    def _on_recent_picked(self, sender: Any) -> None:
        transcript_id = getattr(sender, "_otis_transcript_id", None)
        if not transcript_id or self._store is None:
            return
        record = self._store.get_transcript(transcript_id)
        if record is None:
            self._notifications.notify(
                NotificationType.ERROR,
                "Transcript not found",
                "The file may have been moved or deleted.",
                force=True,
            )
            return
        self._open_in_finder(record["path"])

    def _drain_transcription_queue(self) -> None:
        """Re-run transcriptions a previous process never finished."""
        try:
            jobs = self._queue.recoverable(self._queue.pending(), store=self._store)
        except Exception:
            logger.exception("Could not read the transcription queue")
            return
        if not jobs:
            return
        logger.info("Resuming %d unfinished transcription(s).", len(jobs))
        self._notifications.notify(
            NotificationType.RECORDING_STARTED,
            "Resuming transcription",
            f"{len(jobs)} recording(s) were interrupted.",
            force=True,
        )
        for job in jobs:
            threading.Thread(
                target=self._run_transcription,
                args=(job.metadata,),
                name="otis-resume-transcribe",
                daemon=True,
            ).start()

    def _on_transcribe_now_clicked(self, _sender: Any) -> None:
        """Stop waiting for the call to end and transcribe immediately."""
        self._skip_deferral.set()
        self._notifications.notify(
            NotificationType.RECORDING_STARTED,
            "Transcribing now",
            "Skipping the wait for your call to finish.",
            force=True,
        )

    def _do_start_recording(self) -> None:
        """Begin a recording. Returns at once; the device work runs on a worker.

        Opening CoreAudio input streams is not fast — 5.3 s measured for the
        first start in a process (cold device enumeration), 200-270 ms after
        that. All of it used to run on the rumps main thread, so the menu bar
        froze from the click until capture was live.
        """
        with self._lock:
            if self._recorder is not None or self._starting:
                return
            self._starting = True
            self._state_before_starting = self._snapshot.state
        self._set_state(UiState.STARTING)
        threading.Thread(
            target=self._start_recorder_worker,
            name="otis-recorder-start",
            daemon=True,
        ).start()

    def _start_recorder_worker(self) -> None:
        """Build and start the recorder off the main thread."""
        # Routing is live state — macOS moves the default output to earbuds
        # the moment they connect — and it used to be checked only at launch.
        # Re-check now, *before* capture, so the stream's first audible block
        # (which reports the row OK) always has the last word.
        self._route_at_recording_start = None
        try:
            status = self._check_routing()
            self._route_at_recording_start = status.state
            # Only a *changed* route is news. With the speakers as default
            # output and the meeting app playing into the Otis device, the
            # check says "not routed" every time while the stream works; the
            # row then carries newer live evidence ("signal ok") that an
            # unchanged verdict must not overwrite with a false alarm.
            if (status.state, status.default_output_name) != self._last_route:
                self._report_audio_routing(status)
        except Exception:
            logger.exception("Routing check at recording start raised")
        try:
            recorder = self._recorder_factory(self._config)
            recorder.on_device_error = self._cb_device_error
            session_id = recorder.start()
        except Exception as exc:
            logger.exception("Failed to start recorder")
            self._events.put(("recorder_start_failed", exc))
            return
        self._events.put(("recorder_started", (recorder, session_id)))

    def _handle_recorder_started(self, recorder: Any, session_id: str) -> None:
        self._begin_recording_toasts()
        with self._lock:
            self._starting = False
            self._recorder = recorder
        try:
            self._detector.user_started_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_started_recording raised")
        self._stop_blink()
        self._set_state(UiState.RECORDING)
        _start_timer(self._timer_duration)
        self._notifications.notify(
            NotificationType.RECORDING_STARTED,
            "Recording started",
            f"Session {session_id[:8]}",
        )

    def _handle_recorder_start_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._starting = False
            previous = self._state_before_starting
        self._notifications.notify(
            NotificationType.ERROR, "Could not start recording", str(exc), force=True
        )
        # Back to whatever we interrupted, not blindly to IDLE — a meeting may
        # still be detected and the user may want to try again.
        self._set_state(previous)

    def _do_stop_and_transcribe(self) -> None:
        """Stop and hand off to transcription, without blocking the UI thread.

        ``recorder.stop()`` drains the capture queues, joins both writer
        threads and writes metadata — ~250 ms measured, and up to ~12 s if a
        writer is wedged. SAVING gives the user something truthful to look at
        meanwhile, and hides Start/Stop so the click cannot be repeated.
        """
        with self._lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is None:
            return
        if recorder.state in (RecorderState.IDLE, RecorderState.STOPPED):
            return
        self._timer_duration.stop()
        self._set_state(UiState.SAVING)
        threading.Thread(
            target=self._stop_recorder_worker,
            args=(recorder,),
            name="otis-recorder-stop",
            daemon=True,
        ).start()

    def _stop_recorder_worker(self, recorder: Any) -> None:
        """Flush the recording off the main thread, then continue the handoff."""
        try:
            metadata = recorder.stop()
        except Exception as exc:
            logger.exception("Recorder stop raised")
            self._events.put(("recorder_stop_failed", exc))
            return
        try:
            self._detector.user_stopped_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_stopped_recording raised")
        self._events.put(("recorder_stopped", metadata))

    def _handle_recorder_stop_failed(self, exc: BaseException) -> None:
        self._notifications.notify(
            NotificationType.ERROR, "Save failed", str(exc), force=True
        )
        try:
            self._detector.user_stopped_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_stopped_recording raised")
        # The recorder is gone and we could not flush it; SAVING must not be
        # where the UI comes to rest.
        self._set_state(UiState.IDLE)

    def _handle_recorder_stopped(self, metadata: dict[str, Any] | None) -> None:
        if not metadata:
            self._set_state(UiState.IDLE)
            return
        # Decorate the recorder metadata with everything the transcription
        # handler needs: the user's chosen Whisper language and the live
        # meeting context (title, app, attendees) from the detector.
        with self._lock:
            metadata["_language"] = self._snapshot.selected_language_code
            current = self._snapshot.current_meeting
        if current is not None:
            metadata["_meeting"] = {
                "title": current.title,
                "app": current.app,
                "participants": list(current.participants),
                "meeting_link": current.meeting_link,
                "calendar_event_id": current.calendar_event_id,
            }
        else:
            metadata["_meeting"] = {"title": None, "app": None, "participants": []}
        self._set_state(UiState.PROCESSING)
        # Run the transcription handler in a background thread so the menu bar
        # stays responsive.
        threading.Thread(
            target=self._run_transcription,
            args=(metadata,),
            name="otis-transcription",
            daemon=True,
        ).start()

    def _do_stop_if_recording(self) -> dict[str, Any] | None:
        with self._lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is None:
            return None
        if recorder.state in (RecorderState.IDLE, RecorderState.STOPPED):
            return None
        try:
            metadata = recorder.stop()
        except Exception as exc:
            logger.exception("Recorder stop raised")
            self._notifications.notify(
                NotificationType.ERROR, "Save failed", str(exc), force=True
            )
            # Recorder is gone and we couldn't flush — the UI must not stay
            # in RECORDING forever. Force back to IDLE so the user can try
            # again (Start becomes visible, the duration timer winds down).
            self._timer_duration.stop()
            try:
                self._detector.user_stopped_recording()
            except Exception:  # pragma: no cover
                logger.exception("Detector user_stopped_recording raised")
            self._events.put(("force_state", UiState.IDLE))
            return None
        try:
            self._detector.user_stopped_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_stopped_recording raised")
        self._timer_duration.stop()
        return metadata

    def _run_transcription(self, metadata: dict[str, Any]) -> None:
        self._queue.enqueue(metadata)
        try:
            self._run_transcription_inner(metadata)
        finally:
            # The handler returned, one way or another. A failure has already
            # written a placeholder the user can see, so this is finished work
            # either way — only a process that died mid-run stays queued.
            self._queue.resolve(str(metadata.get("session_id") or ""))

    def _run_transcription_inner(self, metadata: dict[str, Any]) -> None:
        # Whisper saturates the GPU/CPU for many minutes — never compete
        # with a live call (e.g. back-to-back meetings). Wait until the mic
        # frees up before starting the heavy work; the recording is safe on
        # disk either way.
        #
        # The wait gets its own state. It used to run inside PROCESSING, so
        # for up to DEFER_MAX_SECONDS the app hid Start, dropped detector
        # events, refused Generate Transcript as "Busy" and warned on quit
        # that a transcription would be lost — all while nothing was running.
        # And the trigger for waiting is that another call is live, so Otis
        # was unrecordable during exactly the meeting you would want it for.
        self._events.put(("force_state", UiState.DEFERRED))
        self._defer_while_in_call()
        self._events.put(("force_state", UiState.PROCESSING))

        try:
            self._detector.transcription_started()
        except Exception:  # pragma: no cover
            logger.exception("Detector transcription_started raised")

        succeeded = True
        try:
            self._transcribe(metadata)
        except Exception as exc:
            succeeded = False
            logger.exception("Transcription handler raised")
            # Save a placeholder transcript so the user can SEE the failure
            # in the Recent menu / web UI later, instead of having to grep
            # the log file. Best-effort — if the store is unavailable we
            # just notify and move on.
            self._save_failure_placeholder(metadata, exc)
            self._notifications.notify(
                NotificationType.ERROR, "Transcription failed", str(exc), force=True
            )

        # Only fire the "Transcript ready" toast on actual success — earlier
        # versions sent it even after a failure, which was misleading.
        if succeeded:
            title, body = format_transcription_complete(
                self._meeting_title_for(metadata),
                duration_minutes=self._estimate_duration_min(metadata),
            )
            self._notifications.notify(
                NotificationType.TRANSCRIPTION_COMPLETE, title, body, force=True
            )

        try:
            self._detector.transcription_finished()
        except Exception:  # pragma: no cover
            logger.exception("Detector transcription_finished raised")
        # Back to IDLE on the main thread via the queue (we're in a bg thread).
        self._events.put(("force_state", UiState.IDLE))
        # Whether we succeeded or not, the Recent menu has new content to show.
        self._events.put(("refresh_recent", None))

    def _defer_while_in_call(self) -> None:
        if not self._config.get("transcription", "defer_while_in_call", default=True):
            return
        from src.pipeline import make_call_probe, wait_for_call_to_end

        probe = make_call_probe(self._config, detector=self._detector)

        self._skip_deferral.clear()

        def busy() -> bool:
            if self._skip_deferral.is_set():
                return False
            with self._lock:
                if self._recorder is not None:
                    return True
            return probe()

        def notify_deferred() -> None:
            self._notifications.notify(
                NotificationType.RECORDING_STARTED,
                "Transcription deferred",
                "Waiting for your call to finish so the laptop stays fast.",
                force=True,
            )

        wait_for_call_to_end(is_busy=busy, on_first_wait=notify_deferred)

    def _save_failure_placeholder(
        self, metadata: dict[str, Any], error: BaseException,
    ) -> None:
        if self._store is None:
            return
        meeting = metadata.get("_meeting") or {}
        try:
            self._store.save_failure(
                session_id=str(metadata.get("session_id") or "unknown"),
                error=error,
                title=meeting.get("title"),
                app=meeting.get("app"),
                participants=[
                    _participant_to_str(p) for p in (meeting.get("participants") or [])
                ],
                model=str(self._config.get("transcription", "model", default="small")),
                audio_files={
                    "mic": metadata.get("mic_wav"),
                    "system": metadata.get("system_wav"),
                },
                language=metadata.get("_language"),
            )
        except Exception:
            logger.exception("Could not save failure placeholder transcript")

    def _meeting_title_for(self, metadata: dict[str, Any]) -> str:
        """The best name we have for what was just transcribed.

        This read ``metadata["title"]``, which recorder metadata does not have
        — verified against every metadata.json on disk — so the completion
        toast said "Recording" every single time.

        Order matters: the transcript's own title is the one the user will see
        in Recent and the web UI, and it is the only place an auto-suggested
        title exists. The calendar title is next. A timestamp is a poorer name
        than either, but it is at least specific to this recording.
        """
        session_id = str(metadata.get("session_id") or "")
        if session_id and self._store is not None:
            try:
                entry = self._store.get_transcript(session_id)
            except Exception:
                entry = None
            if entry:
                title = str((entry.get("metadata") or {}).get("title") or "").strip()
                if title:
                    return title

        meeting = metadata.get("_meeting") or {}
        title = str(meeting.get("title") or "").strip()
        if title:
            return title

        started = metadata.get("start_wall_clock")
        if started:
            try:
                from datetime import datetime

                when = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                return f"Recording — {when.astimezone():%Y-%m-%d %H:%M}"
            except Exception:
                pass
        return "Recording"

    @staticmethod
    def _estimate_duration_min(metadata: dict[str, Any]) -> float:
        # mic_bytes / (sample_rate * sample_width) → seconds; sample_width = 2.
        sr = metadata.get("sample_rate") or 16000
        mic_bytes = metadata.get("mic_bytes") or 0
        if not mic_bytes:
            return 0.0
        return mic_bytes / (sr * 2) / 60.0

    # =====================================================================
    # Misc
    # =====================================================================
    @staticmethod
    def _open_url(url: str) -> None:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            logger.exception("Could not open URL %s", url)

    @staticmethod
    def _open_app(name: str) -> None:
        try:
            import subprocess
            subprocess.Popen(["open", "-a", name])
        except Exception:
            logger.exception("Could not open %s", name)

    @staticmethod
    def _open_in_finder(path: Path) -> None:
        try:
            import subprocess
            subprocess.Popen(["open", str(path)])
        except Exception:
            logger.exception("Could not open %s in Finder", path)

    def _on_before_quit(self) -> None:
        """Last words before exit: say so in the log, save any recording.

        Without this a Quit, a crash and a hang all looked the same in
        otis.log. Each step is guarded — rumps prints a raising handler to
        stderr and carries on — and nothing here joins long-running workers,
        because macOS does not wait long at logout.
        """
        logger.info("Otis quitting (pid %d).", os.getpid())
        # A logout or restart reaches here without _on_quit_clicked, with the
        # recording still running; stopping it flushes the WAVs to disk.
        for step in (self._do_stop_if_recording, self._detector.stop):
            try:
                step()
            except Exception:
                logger.exception("Quit step %s raised", getattr(step, "__name__", step))

    def _teardown(self) -> None:
        self._do_stop_if_recording()
        try:
            self._detector.stop()
        except Exception:  # pragma: no cover
            logger.exception("Detector stop raised on teardown")


# ============================================================================
# Defaults
# ============================================================================
def _default_transcription_handler(metadata: dict[str, Any]) -> None:
    """No-op fallback when no handler is injected (bare/test instances)."""
    logger.info(
        "Transcription handler not configured — keeping the recording at %r.",
        metadata.get("mic_wav"),
    )


def _participant_to_str(p: Any) -> str:
    """Format a participant entry (dict or str) into ``"Name <email>"``."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        name = p.get("name") or ""
        email = p.get("email") or ""
        if name and email:
            return f"{name} <{email}>"
        return name or email or "unknown"
    return str(p)


# ============================================================================
# Orphan-session discovery — pure functions so they're testable without rumps
# ============================================================================
