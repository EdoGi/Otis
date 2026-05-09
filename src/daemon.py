"""End-to-end Otis runner: detection → recording → graceful shutdown.

This is the smallest thing that turns the Phase-1+2 building blocks into an app
you can leave running in a terminal. Phase 4 will replace it with a proper
menu-bar UI; until then this is the daily-driver-acceptable testing harness.

What it does
------------
1. Builds a :class:`ProcessMonitor` and one :class:`GoogleCalendarPoller` per
   configured account, then a :class:`MeetingDetector` that orchestrates them.
2. Subscribes to the four detector events and prints them as readable
   notifications on stdout.
3. When a meeting is **detected** (process spotted, or calendar+process
   correlated), it auto-starts a :class:`DualStreamRecorder` writing to the
   configured ``storage.audio_dir``.
4. If the meeting app disappears mid-recording it logs a warning **but keeps
   recording** (per the Phase-2 review decision: process exits are advisory).
5. On Ctrl-C or SIGTERM it stops the recorder, flushes the metadata, stops
   the pollers, and exits cleanly.

Anything richer (menu bar, autostart at login, UI) lands in Phase 4+.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.audio.recorder import DualStreamRecorder
from src.config import Config, load_config
from src.detection.calendar_poller import (
    GoogleCalendarPoller,
    build_pollers_from_config,
)
from src.detection.detector import MeetingContext, MeetingDetector
from src.detection.process_monitor import ProcessMonitor

logger = logging.getLogger(__name__)


class OtisDaemon:
    """Foreground daemon: detection + auto-recording + readable terminal output.

    Parameters
    ----------
    config:
        Loaded :class:`Config`.
    notify:
        Callable used for user-facing messages. Defaults to :func:`print`.
        Override in tests.
    recorder_factory:
        Optional factory ``(config) -> DualStreamRecorder`` for tests. The
        default builds the real recorder from the config's audio settings.
    detector:
        Optional pre-built :class:`MeetingDetector` (tests pass a fake).
    """

    def __init__(
        self,
        config: Config,
        *,
        notify: Callable[[str], None] | None = None,
        recorder_factory: Callable[[Config], DualStreamRecorder] | None = None,
        detector: MeetingDetector | None = None,
    ) -> None:
        self._config = config
        self._notify = notify or print
        self._recorder_factory = recorder_factory or _default_recorder_factory
        self._lock = threading.RLock()
        self._recorder: DualStreamRecorder | None = None
        self._stop_event = threading.Event()

        self._detector = detector or self._build_detector()
        self._wire_callbacks()

    # =====================================================================
    # Public API
    # =====================================================================
    def run(self) -> int:
        """Block until SIGINT/SIGTERM, then shut down cleanly. Returns exit code."""
        self._install_signal_handlers()
        self._notify("Otis is watching for meetings. Press Ctrl-C to stop.")
        self._notify(self._summary_line())
        self._detector.start()
        try:
            self._stop_event.wait()
        finally:
            self._shutdown()
        return 0

    def request_stop(self) -> None:
        """Signal the daemon to exit (called by signal handlers and tests)."""
        self._stop_event.set()

    @property
    def detector(self) -> MeetingDetector:
        return self._detector

    @property
    def recorder(self) -> DualStreamRecorder | None:
        return self._recorder

    # =====================================================================
    # Build helpers
    # =====================================================================
    def _build_detector(self) -> MeetingDetector:
        cfg = self._config

        process_monitor: ProcessMonitor | None = None
        pm_cfg = cfg.get("detection", "process_monitor", default={})
        if pm_cfg.get("enabled", True):
            process_monitor = ProcessMonitor(
                whitelisted_apps=list(pm_cfg.get("whitelisted_apps", [])),
                blacklisted_apps=list(pm_cfg.get("blacklisted_apps", [])),
                poll_interval_seconds=float(pm_cfg.get("poll_interval_seconds", 5)),
            )

        calendar_pollers: list[GoogleCalendarPoller] = []
        cal_cfg = cfg.get("detection", "calendar", default={})
        if cal_cfg.get("enabled", True):
            try:
                calendar_pollers = build_pollers_from_config(cal_cfg)
            except Exception as exc:  # pragma: no cover (rare)
                logger.warning(
                    "Could not build calendar pollers; running without calendar: %s", exc
                )

        return MeetingDetector(
            process_monitor=process_monitor,
            calendar_pollers=calendar_pollers,
        )

    def _wire_callbacks(self) -> None:
        self._detector.on_meeting_approaching(self._on_approaching)
        self._detector.on_meeting_detected(self._on_detected)
        self._detector.on_meeting_ended(self._on_ended)
        self._detector.on_process_disappeared(self._on_process_gone)

    # =====================================================================
    # Detector → user-facing handlers
    # =====================================================================
    def _on_approaching(self, ctx: MeetingContext) -> None:
        when = ctx.start_time.strftime("%H:%M") if ctx.start_time else "soon"
        link = f" → {ctx.meeting_link}" if ctx.meeting_link else ""
        self._notify(f"⏰  Meeting in <2 min at {when}: {ctx.title!r}{link}")

    def _on_detected(self, ctx: MeetingContext) -> None:
        title = ctx.title or "(ad-hoc meeting)"
        self._notify(f"📞  Detected: {title}  via {ctx.app or 'calendar'}")
        self._start_recording()

    def _on_ended(self, ctx: MeetingContext) -> None:
        self._notify(f"✅  Meeting ended: {ctx.title or ctx.app or 'unknown'}")
        self._stop_recording()

    def _on_process_gone(self, app_name: str) -> None:
        self._notify(
            f"⚠️   {app_name} exited but recording continues. "
            f"Press Ctrl-C to stop and save."
        )

    # =====================================================================
    # Recorder lifecycle
    # =====================================================================
    def _start_recording(self) -> None:
        with self._lock:
            if self._recorder is not None:
                return  # idempotent: already recording
            try:
                self._recorder = self._recorder_factory(self._config)
                session_id = self._recorder.start()
            except Exception as exc:
                logger.exception("Could not start recorder: %s", exc)
                self._notify(f"❌  Could not start recording: {exc}")
                self._recorder = None
                return
            self._detector.user_started_recording()
            self._notify(f"🔴  Recording session {session_id}")

    def _stop_recording(self) -> None:
        with self._lock:
            if self._recorder is None:
                return
            try:
                metadata = self._recorder.stop()
            except Exception as exc:
                logger.exception("Recorder stop raised: %s", exc)
                metadata = {}
            self._recorder = None
            self._notify(_format_save_line(metadata))

    # =====================================================================
    # Shutdown
    # =====================================================================
    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, AttributeError):  # pragma: no cover (non-main thread)
            pass

    def _handle_signal(self, signum: int, _frame: Any) -> None:  # pragma: no cover
        self._notify(f"\nReceived signal {signum} — shutting down…")
        self.request_stop()

    def _shutdown(self) -> None:
        self._stop_recording()
        self._detector.stop()
        self._notify("Bye.")

    # =====================================================================
    # Misc
    # =====================================================================
    def _summary_line(self) -> str:
        bits: list[str] = []
        cfg = self._config
        pm_enabled = cfg.get("detection", "process_monitor", "enabled", default=True)
        bits.append("process_monitor=on" if pm_enabled else "process_monitor=off")
        cal_enabled = cfg.get("detection", "calendar", "enabled", default=True)
        if cal_enabled:
            n = len(self._detector.calendar_pollers)
            bits.append(f"calendar={n} account(s)")
        else:
            bits.append("calendar=off")
        bits.append(f"audio_dir={cfg.get('storage', 'audio_dir')}")
        return "  • " + ", ".join(bits)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _default_recorder_factory(config: Config) -> DualStreamRecorder:
    audio_dir = Path(config.get("storage", "audio_dir", default="~/Otis/audio"))
    return DualStreamRecorder(
        audio_dir=audio_dir,
        sample_rate=int(config.get("audio", "sample_rate", default=16000)),
        channels=int(config.get("audio", "channels", default=1)),
        mic_device=config.get("audio", "mic_device"),
        system_device=config.get("audio", "system_audio_device", default="BlackHole 2ch"),
    )


def _format_save_line(metadata: dict[str, Any]) -> str:
    if not metadata:
        return "💾  Recording stopped (nothing was captured)."
    mic = metadata.get("mic_wav") or "(none)"
    sysw = metadata.get("system_wav") or "(no system audio)"
    return f"💾  Saved: {mic} + {sysw}"


# ----------------------------------------------------------------------------
# Entry point used by ``otis run``
# ----------------------------------------------------------------------------
def run_from_config(config_path: str | None = None) -> int:
    """Build a daemon from a YAML config and run it. Returns the exit code."""
    cfg = load_config(config_path)
    daemon = OtisDaemon(cfg)
    return daemon.run()
