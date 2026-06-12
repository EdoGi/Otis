"""End-to-end headless Otis runner: detection → recording → transcription.

``otis run`` for terminals / SSH sessions — same pipeline as the menu-bar
app, no UI:

1. Builds a :class:`ProcessMonitor` and one :class:`GoogleCalendarPoller` per
   configured account, then a :class:`MeetingDetector` that orchestrates them.
2. Honours the configured working days / hours: outside the window the
   detector is stopped (an in-progress recording is never interrupted).
3. When a meeting is **detected**, auto-starts a :class:`DualStreamRecorder`
   writing to the configured ``storage.audio_dir``.
4. When the meeting ends, transcribes the recording on a worker thread via
   the shared :mod:`src.pipeline` (same handler as the menu bar) and saves
   the Markdown transcript; failures leave a retryable placeholder.
5. If the meeting app disappears mid-recording it logs a warning **but keeps
   recording** (process exits are advisory).
6. On Ctrl-C or SIGTERM it stops the recorder, transcribes what was
   captured (bounded wait), stops the pollers, and exits cleanly.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.audio.recorder import DualStreamRecorder
from src.config import Config, load_user_config
from src.detection.calendar_poller import (
    GoogleCalendarPoller,
    build_pollers_from_config,
)
from src.detection.detector import MeetingContext, MeetingDetector
from src.detection.process_monitor import ProcessMonitor
from src.pipeline import (
    TranscriptionPipeline,
    make_recorder_factory,
    make_transcription_handler,
)
from src.schedule import is_within_working_hours, working_hours_from_config

logger = logging.getLogger(__name__)

SCHEDULE_CHECK_SECONDS = 60.0
TRANSCRIBE_SHUTDOWN_JOIN_SECONDS = 600.0


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
        transcription_pipeline: TranscriptionPipeline | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._notify = notify or print
        self._recorder_factory = recorder_factory or make_recorder_factory(config)
        self._clock = clock or datetime.now
        self._lock = threading.RLock()
        self._recorder: DualStreamRecorder | None = None
        self._stop_event = threading.Event()

        # Transcription: same shared pipeline as the menu-bar app. ``None``
        # (tests, or callers that only want recording) degrades to
        # record-only with a logged notice at stop time.
        self._pipeline = transcription_pipeline
        self._transcription_handler = (
            make_transcription_handler(transcription_pipeline)
            if transcription_pipeline is not None
            else None
        )
        self._transcribe_lock = threading.Lock()
        # Every spawned transcription worker, so shutdown can join them ALL —
        # back-to-back meetings queue on _transcribe_lock and a single slot
        # would let the queued one die unrecorded at exit.
        self._transcribe_threads: list[threading.Thread] = []
        self._current_meeting: MeetingContext | None = None

        # Working-hours gating.
        self._detector_started = False
        self._schedule_thread: threading.Thread | None = None

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
        # Honour the working-hours window from tick zero; the schedule
        # thread keeps re-checking every minute.
        self._apply_schedule()
        self._schedule_thread = threading.Thread(
            target=self._schedule_loop, name="otis-daemon-schedule", daemon=True
        )
        self._schedule_thread.start()
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
        mic_cfg = cfg.get("detection", "mic_activation", default={}) or {}
        if pm_cfg.get("enabled", True):
            process_monitor = ProcessMonitor(
                whitelisted_apps=list(pm_cfg.get("whitelisted_apps", [])),
                blacklisted_apps=list(pm_cfg.get("blacklisted_apps", [])),
                poll_interval_seconds=float(pm_cfg.get("poll_interval_seconds", 5)),
                mic_activation_enabled=bool(mic_cfg.get("enabled", True)),
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
        with self._lock:
            self._current_meeting = ctx
        self._start_recording()

    def _on_ended(self, ctx: MeetingContext) -> None:
        self._notify(f"✅  Meeting ended: {ctx.title or ctx.app or 'unknown'}")
        with self._lock:
            if ctx.title or ctx.app:
                self._current_meeting = ctx
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
                self._recorder.on_device_error = self._on_recorder_device_error
                session_id = self._recorder.start()
            except Exception as exc:
                logger.exception("Could not start recorder: %s", exc)
                self._notify(f"❌  Could not start recording: {exc}")
                self._recorder = None
                return
            self._detector.user_started_recording()
            self._notify(f"🔴  Recording session {session_id}")

    def _on_recorder_device_error(self, stream_label: str, exc: Exception) -> None:
        self._notify(
            f"⚠️   {stream_label} stream failed mid-recording: {exc} — "
            f"the other stream keeps recording."
        )

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
            meeting = self._current_meeting
            self._current_meeting = None
            # Spawn + register the transcription worker BEFORE releasing the
            # lock: a concurrent _shutdown() serializes on this lock, so it
            # is guaranteed to see the worker and wait for it. Registering
            # after the unlock left a window where shutdown saw no recorder
            # AND no worker and exited, killing the transcription silently.
            self._maybe_transcribe(metadata, meeting)
        self._notify(_format_save_line(metadata))

    # =====================================================================
    # Transcription (worker thread; same handler as the menu bar)
    # =====================================================================
    def _maybe_transcribe(
        self, metadata: dict[str, Any], meeting: MeetingContext | None,
    ) -> None:
        if not metadata:
            return
        if self._transcription_handler is None:
            logger.info(
                "No transcription pipeline configured — recording kept at %r.",
                metadata.get("mic_wav"),
            )
            return
        metadata["_language"] = self._config.get("transcription", "language")
        if meeting is not None:
            metadata["_meeting"] = {
                "title": meeting.title,
                "app": meeting.app,
                "participants": list(meeting.participants),
                "meeting_link": meeting.meeting_link,
                "calendar_event_id": meeting.calendar_event_id,
            }
        else:
            metadata["_meeting"] = {"title": None, "app": None, "participants": []}

        thread = threading.Thread(
            target=self._run_transcription,
            args=(metadata,),
            name="otis-daemon-transcribe",
            daemon=True,
        )
        with self._lock:
            # Prune finished workers so back-to-back days don't accumulate.
            self._transcribe_threads = [
                t for t in self._transcribe_threads if t.is_alive()
            ]
            self._transcribe_threads.append(thread)
        thread.start()

    def _run_transcription(self, metadata: dict[str, Any]) -> None:
        session_id = str(metadata.get("session_id") or "unknown")
        with self._transcribe_lock:  # back-to-back stops serialize here
            self._defer_while_in_call()
            # Only drive the detector's PROCESSING/IDLE transitions when no
            # NEW recording has started in the meantime — otherwise this
            # worker would demote meeting B's RECORDING to PROCESSING and
            # later reset the detector to IDLE while B is still capturing.
            with self._lock:
                manage_detector_state = self._recorder is None
            if manage_detector_state:
                try:
                    self._detector.transcription_started()
                except Exception:  # pragma: no cover
                    logger.exception("Detector transcription_started raised")
            self._notify(f"📝  Transcribing session {session_id[:8]}…")
            try:
                assert self._transcription_handler is not None
                self._transcription_handler(metadata)
                self._notify(f"📄  Transcript saved for session {session_id[:8]}.")
            except Exception as exc:
                logger.exception("Transcription failed for session %s", session_id)
                self._notify(f"❌  Transcription failed: {exc}")
                self._save_failure_placeholder(metadata, exc)
            finally:
                if manage_detector_state:
                    try:
                        self._detector.transcription_finished()
                    except Exception:  # pragma: no cover
                        logger.exception("Detector transcription_finished raised")

    def _defer_while_in_call(self) -> None:
        """Hold the heavy whisper work while a call / new recording is live."""
        if not self._config.get("transcription", "defer_while_in_call", default=True):
            return
        from src.pipeline import make_call_probe, wait_for_call_to_end

        probe = make_call_probe(self._config)

        def busy() -> bool:
            with self._lock:
                if self._recorder is not None:
                    return True
            return probe()

        waited = wait_for_call_to_end(
            is_busy=busy,
            on_first_wait=lambda: self._notify(
                "⏸  Transcription deferred until your call ends."
            ),
            should_abort=self._stop_event.is_set,
        )
        if waited:
            self._notify("▶️  Call over — starting transcription.")

    def _save_failure_placeholder(
        self, metadata: dict[str, Any], error: BaseException,
    ) -> None:
        if self._pipeline is None:
            return
        meeting = metadata.get("_meeting") or {}
        try:
            self._pipeline.store.save_failure(
                session_id=str(metadata.get("session_id") or "unknown"),
                error=error,
                title=meeting.get("title"),
                app=meeting.get("app"),
                participants=[str(p) for p in (meeting.get("participants") or [])],
                model=str(self._config.get("transcription", "model", default="small")),
                audio_files={
                    "mic": metadata.get("mic_wav"),
                    "system": metadata.get("system_wav"),
                },
                language=metadata.get("_language"),
            )
        except Exception:
            logger.exception("Could not save failure placeholder transcript")

    # =====================================================================
    # Working-hours schedule
    # =====================================================================
    def _schedule_loop(self) -> None:
        while not self._stop_event.wait(SCHEDULE_CHECK_SECONDS):
            try:
                self._apply_schedule()
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Schedule check failed; will retry next tick.")

    def _apply_schedule(self) -> None:
        """Start/stop the detector based on the working-hours window.

        An active recording is never interrupted: if the window closes
        mid-meeting we leave everything running and re-check next tick.
        """
        in_window = is_within_working_hours(
            self._clock(), **working_hours_from_config(self._config)
        )
        stop_needed = False
        with self._lock:
            if self._recorder is not None:
                return
            if in_window and not self._detector_started:
                self._detector.start()
                self._detector_started = True
                self._notify("🌅  Inside working hours — detection active.")
            elif not in_window and self._detector_started:
                self._detector_started = False
                stop_needed = True
        if stop_needed:
            # Stop outside the lock — poller joins can take a moment.
            self._detector.stop()
            self._notify("🌙  Outside working hours — detection paused.")

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
        with self._lock:
            pending = [t for t in self._transcribe_threads if t.is_alive()]
        if pending:
            self._notify(
                f"⏳  Waiting for {len(pending)} transcription(s) to finish "
                f"(up to {int(TRANSCRIBE_SHUTDOWN_JOIN_SECONDS / 60)} min each)…"
            )
        for thread in pending:
            thread.join(timeout=TRANSCRIBE_SHUTDOWN_JOIN_SECONDS)
            if thread.is_alive():  # pragma: no cover (very long audio)
                self._notify(
                    "⚠️   Transcription still running at shutdown — a failure "
                    "placeholder may be missing; retry via scripts/retranscribe.py."
                )
        self._detector.stop()
        if self._pipeline is not None:
            self._pipeline.shutdown()
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
    """Build a full daemon (detection + transcription) and run it."""
    from src.pipeline import build_pipeline

    cfg = load_user_config(config_path)
    daemon = OtisDaemon(cfg, transcription_pipeline=build_pipeline(cfg))
    return daemon.run()
