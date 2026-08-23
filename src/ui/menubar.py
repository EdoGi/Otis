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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.audio.recorder import DualStreamRecorder, RecorderState
from src.audio.wav_repair import repair_if_needed
from src.config import Config
from src.detection.detector import (
    MeetingContext,
    MeetingDetector,
    MeetingState,
)
from src.health import (
    CALENDAR,
    MICROPHONE,
    SYSTEM_AUDIO,
    HealthRegistry,
    HealthReport,
    HealthState,
    get_health,
)
from src.schedule import is_within_working_hours, working_hours_from_config
from src.storage.transcript_store import TranscriptStore
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

# Hour presets offered in Settings → Working Hours.
HOUR_START_PRESETS: list[str] = [f"{h:02d}:00" for h in range(6, 13)]   # 06:00–12:00
HOUR_END_PRESETS: list[str] = [f"{h:02d}:00" for h in range(14, 23)]    # 14:00–22:00

# Polling cadences for main-thread timers (rumps.Timer).
QUEUE_DRAIN_INTERVAL = 0.1   # 100 ms — main-thread event drain
BLINK_INTERVAL = 0.5         # 500 ms — DETECTED state blink
DURATION_INTERVAL = 1.0      # 1 s — recording duration tick
SCHEDULE_CHECK_INTERVAL = 60  # 60 s — off-hours window check

# Suffix shown on the "Status" menu title when something is degraded.
_GLYPH_FOR: dict[HealthState, str] = {
    HealthState.OK: "",
    HealthState.UNKNOWN: "·",
    HealthState.WARN: "!",
    HealthState.FAIL: "✕",
}


# ============================================================================
# Helpers used both by the live app and by tests.
# ============================================================================
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
    #: Opening the capture devices. Transient, but not instant: measured on
    #: this Mac, the first start in a process takes 5.3 s of cold CoreAudio
    #: enumeration and later ones 200-270 ms.
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    #: Draining the capture queues and writing metadata, ~250 ms.
    SAVING = "saving"
    PROCESSING = "processing"
    OFF_HOURS = "off_hours"


@dataclass
class UiSnapshot:
    """Pure-data snapshot of what the menu bar wants to show right now."""

    state: str = UiState.IDLE
    recording_started_monotonic: float | None = None
    # When the current pause began. While paused, elapsed time freezes; on
    # resume, recording_started_monotonic is shifted forward by the pause
    # length so the MM:SS counter only counts recorded time.
    paused_at_monotonic: float | None = None
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
        if self.state == UiState.STARTING:
            return "starting…"
        if self.state == UiState.SAVING:
            return "saving…"
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
        transcript_store: TranscriptStore | None = None,
        audio_dir: Path | None = None,
        icons_dir: Path | None = None,
        user_config_path: Path | None = None,
        rumps_app_factory: Callable[..., Any] | None = None,
        health: HealthRegistry | None = None,
    ) -> None:
        self._config = config
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

        self._icons = ensure_icons(self._icons_dir)
        self._app = self._build_app(rumps_app_factory)
        self._wire_detector_callbacks()
        # Health transitions become notifications and a menu refresh. The
        # registry only fires on an actual state change, so a subsystem can
        # report on every poll tick without producing a single extra toast.
        self._health.on_change(self._cb_health_changed)
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

        try:
            self._app.run()
        finally:
            self._teardown()

    def _warn_if_blackhole_missing(self) -> None:
        """Check system audio at startup — structurally *and* functionally.

        The old version only asked whether BlackHole and some aggregate device
        existed. That returned ok=True on this Mac for two months while the
        output went to the built-in speakers and BlackHole captured pure
        silence. Existence is not routing, so we now check both.
        """
        try:
            from src.audio.blackhole_check import (
                check_output_routing,
                verify_blackhole_setup,
            )

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
            self._report_audio_routing(check_output_routing())
        except Exception:  # pragma: no cover (defensive)
            logger.exception("Audio routing check raised on startup")

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

        # Working hours — Start / End submenus with hourly presets.
        hours_root = rumps.MenuItem(self._working_hours_title())
        self._hour_items: dict[tuple[str, str], rumps.MenuItem] = {}
        for which, presets in (("start", HOUR_START_PRESETS), ("end", HOUR_END_PRESETS)):
            sub = rumps.MenuItem(which.capitalize())
            for hhmm in presets:
                it = rumps.MenuItem(hhmm, callback=self._on_working_hour_picked)
                it._otis_which = which  # type: ignore[attr-defined]
                it._otis_hhmm = hhmm  # type: ignore[attr-defined]
                self._hour_items[(which, hhmm)] = it
                sub.add(it)
            hours_root.add(sub)
        self._mi["settings_hours_root"] = hours_root
        self._update_working_hours_menu()

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

    def _cb_device_error(self, stream_label: str, exc: Exception) -> None:
        """Recorder writer thread reports a dead capture stream (bg thread)."""
        logger.error("recorder → device error on %s stream: %s", stream_label, exc)
        self._events.put(("device_error", (stream_label, exc)))

    # =====================================================================
    # Main-thread queue drain (rumps.Timer)
    # =====================================================================
    def _drain_main_queue(self, _timer: Any) -> None:
        self._sync_recorder_state()
        while True:
            try:
                event_type, payload = self._events.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_event(event_type, payload)
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Error handling main-thread event %s", event_type)

    def _sync_recorder_state(self) -> None:
        """Mirror recorder-initiated pause/resume into the UI.

        The recorder pauses/resumes itself on system sleep/wake (on the
        sleep-wake observer thread). Without this sync the icon would keep
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
            self._timer_duration.start()

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
        self._timer_duration.start()

    def _on_stop_clicked(self, _sender: Any) -> None:
        self._do_stop_and_transcribe()

    def _persist_setting(self, overrides: dict[str, Any]) -> None:
        """Persist a settings change to disk AND the live config.

        Without the in-memory half, anything that re-reads ``self._config``
        (the schedule tick, the next transcription) would keep seeing the
        stale pre-toggle value until restart.
        """
        write_user_config_override(self._user_config_path, overrides)
        self._config.apply_overrides(overrides)

    def _on_language_picked(self, sender: Any) -> None:
        code = getattr(sender, "_otis_lang_code", None)
        with self._lock:
            self._snapshot.selected_language_code = code
        self._update_language_menu(code)
        # Persist so the choice survives restart and the transcription
        # pipeline picks it up via load_user_config().
        self._persist_setting({"transcription": {"language": code}})

    def _on_model_picked(self, sender: Any) -> None:
        model = sender.title
        if model not in WHISPER_MODELS:
            return
        self._persist_setting({"transcription": {"model": model}})
        self._mi["settings_model_root"].title = f"Whisper Model: {model}"
        self._update_model_menu(active=model)

    def _on_day_toggled(self, sender: Any) -> None:
        idx = getattr(sender, "_otis_day_index", None)
        if idx is None:
            return
        sender.state = not sender.state
        active = sorted(i for i, item in self._day_items.items() if item.state)
        self._persist_setting({"app": {"working_days": active}})
        # Re-evaluate the window right away — a day toggle can flip us
        # into/out of OFF_HOURS without waiting up to 60 s for the tick.
        self._on_schedule_tick(None)

    def _on_whitelist_toggled(self, sender: Any) -> None:
        sender.state = not sender.state
        active = [name for name, item in self._whitelist_items.items() if item.state]
        self._persist_setting(
            {"detection": {"process_monitor": {"whitelisted_apps": active}}}
        )
        self._notifications.notify(
            NotificationType.ERROR,
            "Whitelist saved",
            "Restart Otis for whitelist changes to take effect.",
            force=True,
        )

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
            self._apply_health_badge()
            return
        self._app.title = self._snapshot.title_text()
        self._refresh_level_meters()

    def _refresh_level_meters(self) -> None:
        """Show live mic/system input levels in the Status panel while recording.

        This is the check that would have caught the dead system stream during
        the very first meeting instead of two months later.
        """
        with self._lock:
            recorder = self._recorder
        if recorder is None:
            return
        try:
            meters = recorder.level_meters(width=8)
        except Exception:  # pragma: no cover (defensive)
            return
        for label, bar in meters.items():
            key = SYSTEM_AUDIO if label == "system" else MICROPHONE
            if self._health.get(key) is None:
                continue
            # touch() updates the detail line without touching state, so a
            # live meter never fires a notification.
            self._health.touch(key, detail=bar)
        self._refresh_health_menu()

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
        # The two repair actions only appear when they'd actually do something.
        cal = self._health.get(CALENDAR)
        self._set_visible(
            self._mi["reconnect_calendar"],
            cal is not None and cal.state is HealthState.FAIL,
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
    # Health panel
    # =====================================================================
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

    def _refresh_health_menu(self) -> None:
        """Rebuild the Status submenu from the registry snapshot."""
        import rumps

        root = self._mi.get("health_root")
        if root is None:
            return
        try:
            for key in list(root.keys()):
                del root[key]
        except Exception:  # pragma: no cover (older rumps)
            pass

        reports = self._health.snapshot()
        if not reports:
            item = rumps.MenuItem("(no checks run yet)")
            item.set_callback(None)
            root.add(item)
        else:
            for rep in reports:
                item = rumps.MenuItem(rep.one_line())
                item.set_callback(None)  # informational, not clickable
                root.add(item)

        overall = self._health.overall()
        root.title = "Status" if overall is HealthState.OK else f"Status  {_GLYPH_FOR[overall]}"

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
        """Run the interactive OAuth flow for every configured account."""
        import rumps

        rumps.alert(
            title="Reconnect Calendar",
            message=(
                "Otis will open your browser to re-authorise Google Calendar.\n\n"
                "If you have more than one account, run this from a terminal "
                "instead:\n\n    ./scripts/setup_google_cal.sh\n"
                "    ./scripts/setup_google_cal.sh work"
            ),
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
        """Explain exactly what's wrong with system-audio routing right now."""
        import rumps

        from src.audio.blackhole_check import check_output_routing

        status = check_output_routing()
        rumps.alert(title="System audio", message=status.message())
        self._report_audio_routing(status)

    def _report_audio_routing(self, status: Any) -> None:
        """Fold a routing check result into the health registry."""
        from src.audio.blackhole_check import RoutingState

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
        else:
            self._health.report(
                SYSTEM_AUDIO, label="System audio", state=HealthState.FAIL,
                detail=f"not routed (output: {status.default_output_name})",
                hint=status.message(),
            )

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
            item._otis_session_id = entry["session_id"]  # type: ignore[attr-defined]
            item._otis_metadata_path = str(entry["metadata_path"])  # type: ignore[attr-defined]
            item._otis_mic_path = str(entry["mic_path"])  # type: ignore[attr-defined]
            item._otis_system_path = (
                str(entry["system_path"]) if entry["system_path"] else None
            )  # type: ignore[attr-defined]
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
            from datetime import datetime, timezone

            try:
                ts = datetime.fromtimestamp(mic_path.stat().st_mtime, tz=timezone.utc)
                meta["start_wall_clock"] = ts.isoformat()
            except Exception:
                meta["start_wall_clock"] = datetime.now(timezone.utc).isoformat()
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
            item._otis_transcript_id = fm.get("id")  # type: ignore[attr-defined]
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

    def _working_hours_title(self) -> str:
        start = self._config.get("app", "working_hours", "start", default="08:00")
        end = self._config.get("app", "working_hours", "end", default="20:00")
        return f"Working Hours: {start} → {end}"

    def _update_working_hours_menu(self) -> None:
        current = {
            "start": self._config.get("app", "working_hours", "start", default="08:00"),
            "end": self._config.get("app", "working_hours", "end", default="20:00"),
        }
        for (which, hhmm), item in self._hour_items.items():
            item.state = (hhmm == current[which])

    def _on_working_hour_picked(self, sender: Any) -> None:
        which = getattr(sender, "_otis_which", None)
        hhmm = getattr(sender, "_otis_hhmm", None)
        if which not in ("start", "end") or not hhmm:
            return
        self._persist_setting({"app": {"working_hours": {which: hhmm}}})
        self._mi["settings_hours_root"].title = self._working_hours_title()
        self._update_working_hours_menu()
        # Apply immediately instead of waiting up to 60 s for the next tick.
        self._on_schedule_tick(None)

    # =====================================================================
    # Recording lifecycle (called from main thread)
    # =====================================================================
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
        with self._lock:
            self._starting = False
            self._recorder = recorder
        try:
            self._detector.user_started_recording()
        except Exception:  # pragma: no cover
            logger.exception("Detector user_started_recording raised")
        self._stop_blink()
        self._set_state(UiState.RECORDING)
        self._timer_duration.start()
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
        # Whisper saturates the GPU/CPU for many minutes — never compete
        # with a live call (e.g. back-to-back meetings). Wait until the mic
        # frees up before starting the heavy work; the recording is safe on
        # disk either way.
        self._defer_while_in_call()

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

        def busy() -> bool:
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
def find_orphan_sessions(
    *,
    audio_dir: Path,
    store: TranscriptStore | None = None,
    current_recording_session_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return audio sessions on disk without a corresponding transcript.

    Three on-disk layouts are accepted:

    * ``{uuid}_metadata.json`` next to ``{uuid}_*.wav`` — fresh recorder
      output (typically at ``audio_dir`` root).
    * ``YYYY-MM-DD_HHMM_metadata.json`` under ``audio_dir/YYYY/MM/`` —
      already renamed by a successful :class:`TranscriptProcessor` run.
    * Orphan ``{uuid}_*.wav`` at the audio root with **no** metadata.json
      — recorder.stop() crashed mid-write. The caller can still transcribe
      these via a synthesised metadata dict.

    Read-only: this only ever looks. Repairing a crash-truncated header is
    the job of whoever transcribes the session
    (:meth:`MenuBarApp._repair_then_transcribe`), because a listing that
    rewrites files is a listing you cannot run from a menu build.

    ``current_recording_session_id`` excludes the session being actively
    recorded right now — its WAV files are still being written and would
    otherwise show up as a tempting (but corrupted) orphan.
    """
    if not audio_dir.exists():
        return []

    # One store scan up front instead of a full rglob+parse per candidate.
    # Failure placeholders don't count — those sessions must stay retryable.
    completed_ids: set[str] = set()
    if store is not None:
        try:
            completed_ids = store.completed_session_ids()
        except Exception:
            logger.exception("Could not index completed transcripts")

    results: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()
    # We also track resolved WAV paths from Pass 1 so Pass 2 doesn't list
    # the same physical files under a filename-derived "session id" that
    # differs from the metadata's id (e.g. renamed YYYY/MM/HH-MM_mic.wav
    # belongs to a UUID session, but the filename itself looks like a date).
    seen_audio_paths: set[Path] = set()
    if current_recording_session_id:
        seen_session_ids.add(current_recording_session_id)

    # Pass 1 — sessions with explicit metadata.json (preferred).
    for meta_path in sorted(audio_dir.rglob("*_metadata.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        sid = str(meta.get("session_id") or "")
        if not sid or sid in seen_session_ids:
            continue
        mic_relative = meta.get("mic_wav") or f"{sid}_mic.wav"
        sys_relative = meta.get("system_wav")
        mic_path, sys_path = _resolve_session_audio(
            meta_path=meta_path,
            declared_mic=mic_relative,
            declared_sys=sys_relative,
            sid=sid,
        )
        if mic_path is None:
            continue
        seen_audio_paths.add(mic_path)
        if sys_path is not None:
            seen_audio_paths.add(sys_path)
        if sid in completed_ids:
            seen_session_ids.add(sid)
            continue
        seen_session_ids.add(sid)
        results.append({
            "session_id": sid,
            "label": _format_session_label(meta_path, mic_path),
            "metadata_path": meta_path,
            "mic_path": mic_path,
            "system_path": sys_path,
        })
        if len(results) >= limit:
            return results

    # Pass 2 — orphan WAVs anywhere (rglob, in case a partial recording
    # ended up inside the YYYY/MM tree somehow). Skip the actively-recording
    # session, anything already covered by Pass 1, and anything that
    # already has a transcript.
    for mic_path in sorted(audio_dir.rglob("*_mic.wav"), reverse=True):
        if mic_path in seen_audio_paths:
            continue
        sid = mic_path.stem.removesuffix("_mic")
        if sid in seen_session_ids:
            continue
        if sid in completed_ids:
            seen_session_ids.add(sid)
            continue
        seen_session_ids.add(sid)
        sys_path = mic_path.with_name(f"{sid}_system.wav")
        results.append({
            "session_id": sid,
            "label": _format_session_label(mic_path, mic_path) + "  (orphan)",
            "metadata_path": None,
            "mic_path": mic_path,
            "system_path": sys_path if sys_path.exists() else None,
        })
        if len(results) >= limit:
            return results
    return results


def _resolve_session_audio(
    *,
    meta_path: Path,
    declared_mic: str,
    declared_sys: str | None,
    sid: str,
) -> tuple[Path | None, Path | None]:
    """Match the three on-disk layouts retranscribe.py also handles."""
    candidates_mic = [
        meta_path.parent / declared_mic,
        meta_path.with_name(meta_path.name.replace("_metadata.json", "_mic.wav")),
        meta_path.parent / f"{sid}_mic.wav",
    ]
    mic = next((c for c in candidates_mic if c.exists()), None)
    if mic is None:
        return None, None
    sys_path: Path | None = None
    if declared_sys:
        cands = [
            meta_path.parent / declared_sys,
            meta_path.with_name(meta_path.name.replace("_metadata.json", "_system.wav")),
            meta_path.parent / f"{sid}_system.wav",
        ]
        sys_path = next((c for c in cands if c.exists()), None)
    else:
        sibling = meta_path.with_name(
            meta_path.name.replace("_metadata.json", "_system.wav")
        )
        if sibling.exists():
            sys_path = sibling
    return mic, sys_path


def _format_session_label(meta_path: Path | None, mic_path: Path) -> str:
    """Pretty label for the submenu — best-effort timestamp + size."""
    anchor = meta_path or mic_path
    from datetime import datetime
    try:
        mtime = datetime.fromtimestamp(anchor.stat().st_mtime)
        ts = mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts = "unknown"
    try:
        mb = mic_path.stat().st_size / (1024 * 1024)
        size = f"  ({mb:.1f} MB)"
    except Exception:
        size = ""
    return f"{ts}{size}"
