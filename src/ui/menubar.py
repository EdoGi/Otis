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
  the constructor; default is a no-op stub that just logs.

Phase 4 (transcription) replaces the stub with the real pipeline.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any

from src.audio.recorder import DualStreamRecorder, RecorderState
from src.config import Config
from src.detection.detector import (
    MeetingContext,
    MeetingDetector,
    MeetingState,
)
from src.ui.icons import ensure_icons
from src.ui.notifications import (
    NotificationManager,
    NotificationType,
    format_process_disappeared,
    format_transcription_complete,
)

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("French", "fr"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Spanish", "es"),
    ("German", "de"),
]

WHISPER_MODELS: list[str] = ["tiny", "base", "small", "medium", "large-v3"]

WEEKDAY_NAMES: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Polling cadences for main-thread timers (rumps.Timer).
QUEUE_DRAIN_INTERVAL = 0.1   # 100 ms — main-thread event drain
BLINK_INTERVAL = 0.5         # 500 ms — DETECTED state blink
DURATION_INTERVAL = 1.0      # 1 s — recording duration tick
SCHEDULE_CHECK_INTERVAL = 60  # 60 s — off-hours window check


# ============================================================================
# Helpers used both by the live app and by tests.
# ============================================================================
def is_within_working_hours(
    now: datetime,
    *,
    working_days: Iterable[int],
    start_hhmm: str,
    end_hhmm: str,
) -> bool:
    """Return True iff ``now`` falls inside the configured work window.

    ``working_days`` are Python weekday() values (0 = Monday … 6 = Sunday).
    ``start_hhmm`` / ``end_hhmm`` are ``"HH:MM"`` strings in local time.
    """
    if now.weekday() not in set(working_days):
        return False
    try:
        sh, sm = map(int, start_hhmm.split(":"))
        eh, em = map(int, end_hhmm.split(":"))
    except (ValueError, AttributeError):
        return True  # malformed config → don't block
    start = dtime(sh, sm)
    end = dtime(eh, em)
    cur = now.time()
    return start <= cur <= end


def format_duration_mmss(seconds: float) -> str:
    """``93.4 -> "01:33"`` — used in the menu-bar title during recording."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_user_config_override(
    user_config_path: Path,
    overrides: dict[str, Any],
) -> None:
    """Merge ``overrides`` into the user's YAML config and persist it.

    Existing keys in the file are preserved; only the keys we touch are
    written. Used by settings toggles (Whisper model, working days, etc.).
    """
    import yaml

    user_config_path = user_config_path.expanduser()
    user_config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if user_config_path.exists():
        try:
            loaded = yaml.safe_load(user_config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Could not parse %s (%s); rewriting from scratch.",
                user_config_path,
                exc,
            )
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
        elif loaded is not None:
            logger.warning(
                "%s did not contain a mapping (got %s); rewriting from scratch.",
                user_config_path,
                type(loaded).__name__,
            )

    merged = _deep_merge_dicts(existing, overrides)
    user_config_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


# ============================================================================
# State manager — the part of the app that has no rumps dependency, so it's
# unit-testable.
# ============================================================================
class UiState(str):
    """Stringly-typed enum of menu-bar display states (str so .value works)."""

    IDLE = "idle"
    APPROACHING = "approaching"
    DETECTED = "detected"
    RECORDING = "recording"
    PAUSED = "paused"
    PROCESSING = "processing"
    OFF_HOURS = "off_hours"


@dataclass
class UiSnapshot:
    """Pure-data snapshot of what the menu bar wants to show right now."""

    state: str = UiState.IDLE
    recording_started_monotonic: float | None = None
    current_meeting: MeetingContext | None = None
    selected_language_code: str | None = None
    show_off_hours: bool = False

    def title_text(self, *, blink_filled: bool = False) -> str:
        """The text rumps shows next to the icon. Empty for compact icons-only."""
        if self.state == UiState.RECORDING and self.recording_started_monotonic is not None:
            elapsed = time.monotonic() - self.recording_started_monotonic
            return format_duration_mmss(elapsed)
        if self.state == UiState.OFF_HOURS:
            return "off-hours"
        return ""


# ============================================================================
# The MenuBarApp — wraps rumps.App, owns recorder + detector + UI.
# ============================================================================
TranscriptionHandler = Callable[[dict[str, Any]], None]


class MenuBarApp:
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
    """

    def __init__(
        self,
        *,
        config: Config,
        detector: MeetingDetector,
        recorder_factory: Callable[[Config], DualStreamRecorder],
        notifications: NotificationManager | None = None,
        transcription_handler: TranscriptionHandler | None = None,
        icons_dir: Path | None = None,
        user_config_path: Path | None = None,
        rumps_app_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._recorder_factory = recorder_factory
        self._notifications = notifications or NotificationManager()
        self._transcribe = transcription_handler or _default_transcription_handler
        self._icons_dir = (icons_dir or Path("~/.otis/icons")).expanduser()
        self._user_config_path = (
            user_config_path or Path("~/.otis/config.yaml")
        ).expanduser()

        # Cross-thread bridge: callbacks push tuples onto this queue, the
        # main-thread drain timer pops and reacts.
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._lock = threading.RLock()

        self._snapshot = UiSnapshot(
            selected_language_code=config.get("transcription", "language"),
        )
        self._recorder: DualStreamRecorder | None = None
        self._blink_phase = 0  # 0 / 1, used by blink timer

        self._icons = ensure_icons(self._icons_dir)
        self._app = self._build_app(rumps_app_factory)
        self._wire_detector_callbacks()

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

        # Honour the working-hours window from the very first tick — otherwise
        # we'd run detection for up to ``SCHEDULE_CHECK_INTERVAL`` seconds even
        # though we're outside the window.
        in_window = is_within_working_hours(
            datetime.now(),
            working_days=self._config.get("app", "working_days", default=[0, 1, 2, 3, 4]),
            start_hhmm=self._config.get("app", "working_hours", "start", default="08:00"),
            end_hhmm=self._config.get("app", "working_hours", "end", default="20:00"),
        )
        if in_window:
            self._detector.start()
        else:
            logger.info("Launched outside working hours; detection paused.")
            self._set_state(UiState.OFF_HOURS)

        try:
            self._app.run()
        finally:
            self._teardown()

    def _warn_if_blackhole_missing(self) -> None:
        """Surface a one-time warning if BlackHole + Multi-Output isn't set up.

        Fully non-fatal — recording still works mic-only. We just nudge the
        user so they know why the system audio file is empty.
        """
        try:
            from src.audio.blackhole_check import verify_blackhole_setup

            status = verify_blackhole_setup()
        except Exception:  # pragma: no cover (defensive)
            logger.exception("BlackHole check raised on startup")
            return
        if status.ok:
            return
        logger.warning(
            "BlackHole / Multi-Output not configured: %s",
            "; ".join(status.issues) or "see check-audio output",
        )
        self._notifications.notify(
            NotificationType.ERROR,
            "BlackHole not configured",
            "System audio won't be captured. Run scripts/setup_blackhole.sh.",
            force=True,
        )

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
            import rumps  # noqa: WPS433  (lazy by design)
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

        self._timer_drain.start()
        self._timer_schedule.start()
        # Blink + duration are started on demand when entering DETECTED / RECORDING.

        return app

    def _build_menu(self, app: Any) -> None:
        import rumps

        # Top-level controls
        self._mi["start"] = rumps.MenuItem("Start Recording", callback=self._on_start_clicked)
        self._mi["pause"] = rumps.MenuItem("Pause Recording", callback=self._on_pause_clicked)
        self._mi["resume"] = rumps.MenuItem("Resume Recording", callback=self._on_resume_clicked)
        self._mi["stop"] = rumps.MenuItem("Stop & Transcribe", callback=self._on_stop_clicked)

        # Language submenu
        self._mi["language_root"] = rumps.MenuItem("Language: Auto-detect")
        self._lang_items: dict[str, rumps.MenuItem] = {}
        for label, code in SUPPORTED_LANGUAGES:
            it = rumps.MenuItem(label, callback=self._on_language_picked)
            it._otis_lang_code = code  # type: ignore[attr-defined]
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

        # Recent transcripts (populated lazily; placeholder for now)
        self._mi["recent_root"] = rumps.MenuItem("Recent Transcripts")
        self._mi["recent_root"].add(rumps.MenuItem("(no transcripts yet)"))

        # About / Quit
        self._mi["about"] = rumps.MenuItem("About Otis", callback=self._on_about)
        self._mi["quit"] = rumps.MenuItem("Quit", callback=self._on_quit_clicked)

        app.menu = [
            self._mi["start"],
            self._mi["pause"],
            self._mi["resume"],
            self._mi["stop"],
            None,  # separator
            self._mi["language_root"],
            None,
            self._mi["open_web"],
            self._mi["open_folder"],
            None,
            self._mi["settings_root"],
            None,
            self._mi["recent_root"],
            None,
            self._mi["about"],
            self._mi["quit"],
        ]
        self._refresh_menu_visibility()

    def _build_settings_submenu(self, root: Any) -> None:
        import rumps

        # Whisper model
        model_root = rumps.MenuItem(
            f"Whisper Model: {self._config.get('transcription', 'model', default='small')}"
        )
        self._model_items: dict[str, rumps.MenuItem] = {}
        for m in WHISPER_MODELS:
            it = rumps.MenuItem(m, callback=self._on_model_picked)
            self._model_items[m] = it
            model_root.add(it)
        self._mi["settings_model_root"] = model_root
        self._update_model_menu()

        # Working days
        days_root = rumps.MenuItem("Working Days")
        self._day_items: dict[int, rumps.MenuItem] = {}
        active_days = set(self._config.get("app", "working_days", default=[0, 1, 2, 3, 4]))
        for idx, name in enumerate(WEEKDAY_NAMES):
            it = rumps.MenuItem(name, callback=self._on_day_toggled)
            it._otis_day_index = idx  # type: ignore[attr-defined]
            it.state = idx in active_days
            self._day_items[idx] = it
            days_root.add(it)
        self._mi["settings_days_root"] = days_root

        # Working hours (display only — editing is a Phase 4+ polish)
        hours_root = rumps.MenuItem(
            "Working Hours: "
            f"{self._config.get('app', 'working_hours', 'start', default='08:00')} → "
            f"{self._config.get('app', 'working_hours', 'end', default='20:00')}"
        )
        self._mi["settings_hours_root"] = hours_root

        # App whitelist
        wl_root = rumps.MenuItem("App Whitelist")
        self._whitelist_items: dict[str, rumps.MenuItem] = {}
        whitelist = self._config.get(
            "detection", "process_monitor", "whitelisted_apps", default=[]
        )
        for app_name in whitelist:
            it = rumps.MenuItem(app_name, callback=self._on_whitelist_toggled)
            it._otis_app_name = app_name  # type: ignore[attr-defined]
            it.state = True
            self._whitelist_items[app_name] = it
            wl_root.add(it)
        self._mi["settings_whitelist_root"] = wl_root

        root.add(model_root)
        root.add(days_root)
        root.add(hours_root)
        root.add(wl_root)

    # =====================================================================
    # Detector callback wiring (background-thread side)
    # =====================================================================
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

    # =====================================================================
    # Main-thread queue drain (rumps.Timer)
    # =====================================================================
    def _drain_main_queue(self, _timer: Any) -> None:
        while True:
            try:
                event_type, payload = self._events.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_event(event_type, payload)
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Error handling main-thread event %s", event_type)

    def _handle_event(self, event_type: str, payload: Any) -> None:
        if event_type == "approaching":
            ctx: MeetingContext = payload
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
            if self._snapshot.state in (UiState.APPROACHING, UiState.DETECTED):
                self._set_state(UiState.IDLE)
                self._stop_blink()
        elif event_type == "process_gone":
            app_name: str = payload
            title, body = format_process_disappeared(app_name)
            self._notifications.notify(
                NotificationType.PROCESS_DISAPPEARED, title, body
            )
        elif event_type == "force_state":
            # Used by background threads (e.g. the transcription worker) to
            # request a state change on the main thread.
            self._set_state(payload)

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
        self._timer_duration.start()

    def _on_stop_clicked(self, _sender: Any) -> None:
        self._do_stop_and_transcribe()

    def _on_language_picked(self, sender: Any) -> None:
        code = getattr(sender, "_otis_lang_code", None)
        with self._lock:
            self._snapshot.selected_language_code = code
        self._update_language_menu(code)
        # Persist so the choice survives restart and Phase 4's transcription
        # pipeline picks it up via load_user_config().
        write_user_config_override(
            self._user_config_path,
            {"transcription": {"language": code}},
        )

    def _on_model_picked(self, sender: Any) -> None:
        model = sender.title
        if model not in WHISPER_MODELS:
            return
        write_user_config_override(
            self._user_config_path,
            {"transcription": {"model": model}},
        )
        self._mi["settings_model_root"].title = f"Whisper Model: {model}"
        self._update_model_menu(active=model)

    def _on_day_toggled(self, sender: Any) -> None:
        idx = getattr(sender, "_otis_day_index", None)
        if idx is None:
            return
        sender.state = not sender.state
        active = sorted(i for i, item in self._day_items.items() if item.state)
        write_user_config_override(
            self._user_config_path,
            {"app": {"working_days": active}},
        )

    def _on_whitelist_toggled(self, sender: Any) -> None:
        sender.state = not sender.state
        active = [name for name, item in self._whitelist_items.items() if item.state]
        write_user_config_override(
            self._user_config_path,
            {"detection": {"process_monitor": {"whitelisted_apps": active}}},
        )
        self._notifications.notify(
            NotificationType.ERROR,
            "Whitelist saved",
            "Restart Otis for whitelist changes to take effect.",
            force=True,
        )

    def _on_open_transcripts_web(self, _sender: Any) -> None:
        port = self._config.get("web", "port", default=8765)
        self._open_url(f"http://127.0.0.1:{port}")

    def _on_open_transcripts_folder(self, _sender: Any) -> None:
        path = Path(self._config.get(
            "storage", "transcript_dir", default="~/Otis/transcripts"
        )).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._open_in_finder(path)

    def _on_about(self, _sender: Any) -> None:
        import rumps
        rumps.alert(
            title="About Otis",
            message="Otis — local macOS meeting transcription.\n\n"
                    "Mic + system audio recording, on-device Whisper "
                    "transcription, MCP integration with Claude.",
        )

    def _on_quit_clicked(self, _sender: Any) -> None:
        import rumps

        # If a transcription is mid-flight, the worker is a daemon thread —
        # it will be killed the moment we exit, losing the transcript. Warn
        # the user explicitly so they can wait it out.
        with self._lock:
            in_processing = self._snapshot.state == UiState.PROCESSING
        if in_processing:
            response = rumps.alert(
                title="Transcription in progress",
                message=(
                    "A transcription is running. Quitting now will discard it. "
                    "Wait for it to finish, or quit anyway?"
                ),
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
            return
        self._app.title = self._snapshot.title_text()

    def _on_schedule_tick(self, _timer: Any) -> None:
        """Move into / out of OFF_HOURS based on the working-hours window.

        Critical: an active recording must NEVER be interrupted by this. If the
        user starts recording during off-hours (manually) we leave the state
        machine alone so the recorder UI keeps working. The off-hours
        transition only happens from neutral states (IDLE / APPROACHING /
        DETECTED).
        """
        in_window = is_within_working_hours(
            datetime.now(),
            working_days=self._config.get("app", "working_days", default=[0, 1, 2, 3, 4]),
            start_hhmm=self._config.get("app", "working_hours", "start", default="08:00"),
            end_hhmm=self._config.get("app", "working_hours", "end", default="20:00"),
        )
        with self._lock:
            current_state = self._snapshot.state

        # Don't fiddle with state during active recording / paused / processing —
        # transitioning to OFF_HOURS would clear recorder context and hide the
        # Stop button.
        if current_state in (UiState.RECORDING, UiState.PAUSED, UiState.PROCESSING):
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
            elif state in (UiState.IDLE, UiState.OFF_HOURS):
                self._snapshot.recording_started_monotonic = None
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

        self._refresh_menu_visibility()

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
            self._timer_blink.start()

    def _stop_blink(self) -> None:
        if self._timer_blink.is_alive():
            self._timer_blink.stop()

    def _refresh_menu_visibility(self) -> None:
        s = self._snapshot.state
        # Each rumps.MenuItem has no real "hidden" flag — we toggle visibility
        # by rebuilding visibility via .set_callback(None) for disabled items.
        # The simpler approach used here: call .hidden = True/False which rumps
        # exposes via the underlying NSMenuItem.
        self._set_visible(self._mi["start"], s in (UiState.IDLE, UiState.APPROACHING, UiState.DETECTED, UiState.OFF_HOURS))
        self._set_visible(self._mi["pause"], s == UiState.RECORDING)
        self._set_visible(self._mi["resume"], s == UiState.PAUSED)
        self._set_visible(self._mi["stop"], s in (UiState.RECORDING, UiState.PAUSED))

    def _set_visible(self, menu_item: Any, visible: bool) -> None:
        try:
            menu_item.hidden = not visible  # rumps proxies to NSMenuItem.setHidden_
        except Exception:  # pragma: no cover (older rumps)
            menu_item.set_callback(menu_item._callback if visible else None)

    def _update_language_menu(self, code: str | None) -> None:
        title = "Auto-detect"
        for label, c in SUPPORTED_LANGUAGES:
            if c == code:
                title = label
                break
        self._mi["language_root"].title = f"Language: {title}"
        for label, item in self._lang_items.items():
            item.state = (label == title)

    def _update_model_menu(self, active: str | None = None) -> None:
        active = active or self._config.get("transcription", "model", default="small")
        for name, item in self._model_items.items():
            item.state = (name == active)

    # =====================================================================
    # Recording lifecycle (called from main thread)
    # =====================================================================
    def _do_start_recording(self) -> None:
        with self._lock:
            if self._recorder is not None:
                return
        try:
            recorder = self._recorder_factory(self._config)
            session_id = recorder.start()
        except Exception as exc:
            logger.exception("Failed to start recorder")
            self._notifications.notify(
                NotificationType.ERROR, "Could not start recording", str(exc), force=True
            )
            return
        with self._lock:
            self._recorder = recorder
        self._detector.user_started_recording()
        self._stop_blink()
        self._set_state(UiState.RECORDING)
        self._timer_duration.start()
        self._notifications.notify(
            NotificationType.RECORDING_STARTED,
            "Recording started",
            f"Session {session_id[:8]}",
        )

    def _do_stop_and_transcribe(self) -> None:
        metadata = self._do_stop_if_recording()
        if not metadata:
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
            return None
        try:
            self._detector.user_stopped_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_stopped_recording raised")
        self._timer_duration.stop()
        return metadata

    def _run_transcription(self, metadata: dict[str, Any]) -> None:
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

    @staticmethod
    def _meeting_title_for(metadata: dict[str, Any]) -> str:
        return metadata.get("title") or "Recording"

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
    def _open_in_finder(path: Path) -> None:
        try:
            import subprocess
            subprocess.Popen(["open", str(path)])
        except Exception:
            logger.exception("Could not open %s in Finder", path)

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
    """No-op stub used until Phase 4 wires up the real pipeline."""
    logger.info(
        "Transcription handler not configured — keeping the recording at %r. "
        "(Phase 4 will replace this with mlx-whisper.)",
        metadata.get("mic_wav"),
    )
