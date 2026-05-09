"""Dual-stream audio recorder for mic + BlackHole loopback.

Each recording session captures two synchronous WAV files — ``{session_id}_mic.wav``
and ``{session_id}_system.wav`` — plus ``{session_id}_metadata.json`` with the
``time.monotonic()`` anchor for each stream so Phase 4's transcript merge can
align segments precisely.

Design notes
------------
* sounddevice ``InputStream`` callbacks must not block, so each callback only
  pushes raw bytes onto a ``queue.Queue``; a writer thread per stream drains
  the queue into the WAV file.
* The first time the writer thread sees a non-empty buffer, it stamps
  ``time.monotonic()`` into the session metadata. The ``mic_start`` and
  ``system_start`` anchors will differ by a few hundred microseconds — that
  delta is exactly what the merge step in Phase 4 needs.
* macOS sleep/wake is observed via ``NSWorkspace`` notifications running on a
  dedicated CFRunLoop thread. On sleep we pause; on wake we resume. The pause
  is logged in metadata. If pyobjc is unavailable (e.g. CI on Linux), the
  observer becomes a no-op.
* Device errors mid-recording are surfaced via the optional ``on_device_error``
  callback so the UI layer can show a notification and offer to fall back to
  the system default.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.audio.devices import AudioDevice, DeviceManager, DeviceNotFoundError

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class PermissionDeniedError(RuntimeError):
    """Raised when macOS denies microphone access."""

    HINT = (
        "Microphone access is required. Open "
        "System Settings → Privacy & Security → Microphone "
        "and enable Otis (or your terminal/IDE while developing), "
        "then restart the app."
    )

    def __init__(self, message: str = "") -> None:
        super().__init__(f"{message}\n{self.HINT}".strip())


@dataclass
class _StreamState:
    """Per-stream bookkeeping owned by the recorder."""

    label: str  # 'mic' or 'system'
    device: AudioDevice
    wav_path: Path
    queue: queue.Queue[bytes | None] = field(default_factory=queue.Queue)
    start_monotonic: float | None = None
    bytes_written: int = 0
    stream: Any = None  # sounddevice.InputStream
    writer_thread: threading.Thread | None = None
    error: Exception | None = None


class DualStreamRecorder:
    """Records mic + system audio to two WAV files with aligned monotonic anchors.

    Parameters
    ----------
    audio_dir:
        Directory where the WAV files and metadata JSON are written.
    sample_rate:
        Per-stream sample rate. 16 kHz is optimal for Whisper.
    channels:
        Per-stream channel count. 1 (mono) recommended.
    mic_device:
        Name (substring) or device index for the microphone. ``None`` ⇒ default.
    system_device:
        Name (substring) for the BlackHole loopback (default ``"BlackHole 2ch"``).
    device_manager:
        Optional pre-built :class:`DeviceManager` (useful for tests).
    on_device_error:
        Optional callback ``(stream_label, exception) -> None`` invoked when a
        capture stream fails mid-recording.
    observe_sleep_wake:
        If True (default), pause on sleep and resume on wake using
        ``NSWorkspaceWillSleepNotification`` / ``NSWorkspaceDidWakeNotification``.
    """

    SAMPLE_WIDTH_BYTES = 2  # int16 PCM

    def __init__(
        self,
        *,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
        mic_device: str | int | None = None,
        system_device: str | int | None = "BlackHole 2ch",
        device_manager: DeviceManager | None = None,
        on_device_error: Callable[[str, Exception], None] | None = None,
        observe_sleep_wake: bool = True,
    ) -> None:
        self._audio_dir = Path(audio_dir).expanduser()
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._mic_device_spec = mic_device
        self._system_device_spec = system_device
        self._device_manager = device_manager
        self._on_device_error = on_device_error
        self._observe_sleep_wake = observe_sleep_wake

        self._state = RecorderState.IDLE
        self._state_lock = threading.RLock()

        # Pause/resume control: when set ⇒ recording is active; when cleared ⇒
        # paused (callbacks discard incoming frames).
        self._active = threading.Event()
        self._active.set()

        self._session_id: str | None = None
        self._mic_state: _StreamState | None = None
        self._system_state: _StreamState | None = None

        # Pause bookkeeping. Offsets are seconds since ``start()`` returned.
        self._start_monotonic: float | None = None
        self._start_wall_clock: datetime | None = None
        self._pauses: list[dict[str, float]] = []
        self._current_pause_started: float | None = None

        # Sleep/wake observer (pyobjc-based). Lazily started.
        self._sleep_wake_observer: _SleepWakeObserver | None = None

    # =====================================================================
    # Public API
    # =====================================================================
    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def start(self, session_id: str | None = None) -> str:
        """Begin recording. Returns the session id.

        If the configured system-audio device (BlackHole) is not present, the
        recorder logs a warning and continues with **mic-only** capture. The
        ``{session_id}_system.wav`` file is omitted in that case and the
        metadata's system fields are ``None``. This matches the Phase 1 spec's
        graceful-degradation requirement.
        """
        with self._state_lock:
            if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
                raise RuntimeError(f"Recorder already active (state={self._state}).")
            if self._device_manager is None:
                self._device_manager = DeviceManager()
            else:
                self._device_manager.refresh()

            self._session_id = session_id or str(uuid.uuid4())
            self._audio_dir.mkdir(parents=True, exist_ok=True)
            self._pauses = []
            self._current_pause_started = None

            mic_dev = self._resolve_device(self._mic_device_spec, role="mic")
            sys_dev = self._resolve_device(self._system_device_spec, role="system")

            self._mic_state = _StreamState(
                label="mic",
                device=mic_dev,
                wav_path=self._audio_dir / f"{self._session_id}_mic.wav",
            )
            self._system_state = (
                _StreamState(
                    label="system",
                    device=sys_dev,
                    wav_path=self._audio_dir / f"{self._session_id}_system.wav",
                )
                if sys_dev is not None
                else None
            )

            self._active.set()
            self._start_wall_clock = datetime.now(timezone.utc)
            self._start_monotonic = time.monotonic()

            try:
                self._open_stream(self._mic_state)
                if self._system_state is not None:
                    self._open_stream(self._system_state)
            except Exception:
                # Roll back any partially opened resources before re-raising.
                self._teardown_streams()
                self._state = RecorderState.ERROR
                raise

            self._state = RecorderState.RECORDING
            if self._observe_sleep_wake:
                self._start_sleep_wake_observer()

            logger.info(
                "Recording started: session=%s mic=%r system=%r dir=%s",
                self._session_id,
                mic_dev.name,
                self._system_state.device.name if self._system_state else "(none)",
                self._audio_dir,
            )
            return self._session_id

    def pause(self) -> None:
        with self._state_lock:
            if self._state != RecorderState.RECORDING:
                logger.debug("pause() ignored; state=%s", self._state)
                return
            self._active.clear()
            self._current_pause_started = self._elapsed_seconds()
            self._state = RecorderState.PAUSED
            logger.info("Recording paused at %.3fs", self._current_pause_started)

    def resume(self) -> None:
        with self._state_lock:
            if self._state != RecorderState.PAUSED:
                logger.debug("resume() ignored; state=%s", self._state)
                return
            now = self._elapsed_seconds()
            if self._current_pause_started is not None:
                self._pauses.append(
                    {"paused_at": self._current_pause_started, "resumed_at": now}
                )
                self._current_pause_started = None
            self._active.set()
            self._state = RecorderState.RECORDING
            logger.info("Recording resumed at %.3fs", now)

    def stop(self) -> dict[str, Any]:
        """Stop recording, flush WAV files, write metadata. Returns metadata dict.

        Calling :meth:`stop` before :meth:`start` is a no-op that returns an
        empty dict. Calling :meth:`stop` twice returns the metadata that was
        written the first time.
        """
        with self._state_lock:
            if self._state == RecorderState.IDLE:
                logger.debug("stop() called before start(); ignoring.")
                return {}
            if self._state == RecorderState.STOPPED:
                # Idempotent
                return self._read_metadata_file()

            # Close out any open pause window so metadata is internally consistent.
            if self._state == RecorderState.PAUSED and self._current_pause_started is not None:
                self._pauses.append(
                    {
                        "paused_at": self._current_pause_started,
                        "resumed_at": self._elapsed_seconds(),
                    }
                )
                self._current_pause_started = None

            self._active.set()  # release writer threads if still waiting
            self._teardown_streams()
            metadata = self._write_metadata()
            self._stop_sleep_wake_observer()
            self._state = RecorderState.STOPPED
            files = []
            if self._mic_state is not None:
                files.append(str(self._mic_state.wav_path))
            if self._system_state is not None:
                files.append(str(self._system_state.wav_path))
            logger.info("Recording stopped: session=%s files=%s", self._session_id, files)
            return metadata

    # =====================================================================
    # Internals
    # =====================================================================
    def _elapsed_seconds(self) -> float:
        if self._start_monotonic is None:
            return 0.0
        return time.monotonic() - self._start_monotonic

    def _resolve_device(
        self, spec: str | int | None, *, role: str
    ) -> AudioDevice | None:
        """Resolve an input device for ``role`` ('mic' or 'system').

        For ``role='mic'`` we always need a real device — missing means abort.
        For ``role='system'`` we degrade gracefully: if no spec is given, or the
        configured device cannot be found, we log a warning and return ``None``
        so the recorder can capture the mic alone.
        """
        assert self._device_manager is not None  # set in start()

        if role == "mic":
            if spec is None:
                dev = self._device_manager.default_input()
                if dev is None:
                    raise DeviceNotFoundError("No default input device available.")
                return dev
            return self._lookup_input_device(spec)

        # role == "system": graceful degradation
        if spec is None:
            logger.info("No system audio device configured; recording mic only.")
            return None
        try:
            return self._lookup_input_device(spec)
        except DeviceNotFoundError as exc:
            logger.warning(
                "System audio device %r not found (%s). "
                "Recording mic only — install BlackHole and create a Multi-Output "
                "device to capture system audio.",
                spec,
                exc,
            )
            return None

    def _lookup_input_device(self, spec: str | int) -> AudioDevice:
        """Resolve a device spec (substring or index) to an input AudioDevice."""
        assert self._device_manager is not None
        if isinstance(spec, int):
            if 0 <= spec < len(self._device_manager.devices):
                dev = self._device_manager.devices[spec]
                if not dev.is_input:
                    raise DeviceNotFoundError(
                        f"Device #{spec} ({dev.name!r}) has no input channels."
                    )
                return dev
            raise DeviceNotFoundError(f"Device index {spec} out of range.")
        return self._device_manager.require(spec, input_only=True)

    def _open_stream(self, st: _StreamState) -> None:
        """Open a sounddevice InputStream and start the writer thread."""
        import sounddevice as sd

        st.queue = queue.Queue()
        st.bytes_written = 0
        st.start_monotonic = None
        st.error = None

        def callback(indata: "np.ndarray", frames: int, time_info: Any, status: Any) -> None:  # noqa: ARG001
            if status:
                # Non-fatal warnings (input_overflow etc.) — log but keep going.
                logger.warning("[%s] sounddevice status: %s", st.label, status)
            if not self._active.is_set():
                return  # paused — drop frames
            try:
                # ``indata`` is int16 because we requested dtype="int16".
                st.queue.put_nowait(bytes(indata))
            except queue.Full:  # pragma: no cover (queue is unbounded)
                logger.error("[%s] queue full; dropping %d frames", st.label, frames)

        try:
            st.stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=st.device.index,
                callback=callback,
            )
            st.stream.start()
        except Exception as exc:  # pragma: no cover (depends on host)
            msg = str(exc).lower()
            if "permission" in msg or "denied" in msg or "not permitted" in msg:
                raise PermissionDeniedError(str(exc)) from exc
            raise

        st.writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(st,),
            name=f"otis-writer-{st.label}",
            daemon=True,
        )
        st.writer_thread.start()

    def _writer_loop(self, st: _StreamState) -> None:
        """Drain the queue into a WAV file. Sentinel ``None`` ends the loop."""
        try:
            with wave.open(str(st.wav_path), "wb") as wf:
                wf.setnchannels(self._channels)
                wf.setsampwidth(self.SAMPLE_WIDTH_BYTES)
                wf.setframerate(self._sample_rate)
                while True:
                    chunk = st.queue.get()
                    if chunk is None:
                        break
                    if st.start_monotonic is None:
                        # Stamp the first-real-audio anchor exactly here.
                        st.start_monotonic = time.monotonic()
                    wf.writeframes(chunk)
                    st.bytes_written += len(chunk)
        except Exception as exc:  # pragma: no cover (disk failures)
            logger.exception("[%s] writer thread crashed: %s", st.label, exc)
            st.error = exc
            if self._on_device_error is not None:
                try:
                    self._on_device_error(st.label, exc)
                except Exception:
                    logger.exception("on_device_error callback raised")

    def _teardown_streams(self) -> None:
        for st in (self._mic_state, self._system_state):
            if st is None:
                continue
            try:
                if st.stream is not None:
                    st.stream.stop()
                    st.stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("[%s] error closing stream: %s", st.label, exc)
            finally:
                st.stream = None
            # Signal writer to drain remaining queue and exit.
            st.queue.put_nowait(None)
        for st in (self._mic_state, self._system_state):
            if st is None or st.writer_thread is None:
                continue
            st.writer_thread.join(timeout=5.0)
            if st.writer_thread.is_alive():  # pragma: no cover
                logger.error("[%s] writer thread did not exit", st.label)

    def _write_metadata(self) -> dict[str, Any]:
        assert self._mic_state is not None
        assert self._session_id is not None
        sys_st = self._system_state
        meta: dict[str, Any] = {
            "session_id": self._session_id,
            "mic_start_monotonic": self._mic_state.start_monotonic,
            "system_start_monotonic": sys_st.start_monotonic if sys_st else None,
            "start_wall_clock": (
                self._start_wall_clock.isoformat() if self._start_wall_clock else None
            ),
            "pauses": list(self._pauses),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "mic_device": self._mic_state.device.name,
            "system_device": sys_st.device.name if sys_st else None,
            "mic_wav": self._mic_state.wav_path.name,
            "system_wav": sys_st.wav_path.name if sys_st else None,
            "mic_bytes": self._mic_state.bytes_written,
            "system_bytes": sys_st.bytes_written if sys_st else 0,
        }
        meta_path = self._audio_dir / f"{self._session_id}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.debug("Wrote metadata: %s", meta_path)
        return meta

    def _read_metadata_file(self) -> dict[str, Any]:
        if self._session_id is None:
            return {}
        path = self._audio_dir / f"{self._session_id}_metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ sleep / wake
    def _start_sleep_wake_observer(self) -> None:
        if self._sleep_wake_observer is not None:
            return
        observer = _SleepWakeObserver(
            on_sleep=self._handle_system_sleep,
            on_wake=self._handle_system_wake,
        )
        if observer.start():
            self._sleep_wake_observer = observer
        else:
            logger.info("Sleep/wake observation unavailable on this platform.")

    def _stop_sleep_wake_observer(self) -> None:
        if self._sleep_wake_observer is not None:
            self._sleep_wake_observer.stop()
            self._sleep_wake_observer = None

    def _handle_system_sleep(self) -> None:
        logger.info("System sleep detected — pausing recording.")
        try:
            self.pause()
        except Exception:  # pragma: no cover
            logger.exception("Failed to pause on system sleep")

    def _handle_system_wake(self) -> None:
        logger.info("System wake detected — resuming recording.")
        try:
            self.resume()
        except Exception:  # pragma: no cover
            logger.exception("Failed to resume on system wake")


# ============================================================================
# Sleep/wake observer (pyobjc-backed; degrades to no-op if unavailable)
# ============================================================================
class _SleepWakeObserver:
    """Listens for ``NSWorkspaceWillSleepNotification`` / ``DidWakeNotification``.

    Notifications are delivered on a dedicated daemon thread running its own
    ``NSRunLoop``. If pyobjc/AppKit cannot be imported (e.g. CI on Linux), the
    observer reports ``start() -> False`` and does nothing.
    """

    def __init__(
        self,
        on_sleep: Callable[[], None],
        on_wake: Callable[[], None],
    ) -> None:
        self._on_sleep = on_sleep
        self._on_wake = on_wake
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._impl: Any = None  # the Obj-C bridge object

    def start(self) -> bool:
        try:
            # Imported here so non-macOS environments don't blow up at import time.
            import objc  # noqa: F401
            from AppKit import NSWorkspace  # noqa: F401
            from Foundation import NSObject  # noqa: F401
        except Exception as exc:  # pragma: no cover
            logger.debug("pyobjc unavailable; sleep/wake observation disabled: %s", exc)
            return False

        self._thread = threading.Thread(
            target=self._run, name="otis-sleepwake", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:  # pragma: no cover (requires macOS GUI session)
        import objc
        from AppKit import NSWorkspace
        from Foundation import NSDate, NSObject, NSRunLoop

        on_sleep = self._on_sleep
        on_wake = self._on_wake

        class _Bridge(NSObject):
            def workspaceWillSleep_(self, _notification):  # noqa: N802
                try:
                    on_sleep()
                except Exception:
                    logger.exception("on_sleep handler raised")

            def workspaceDidWake_(self, _notification):  # noqa: N802
                try:
                    on_wake()
                except Exception:
                    logger.exception("on_wake handler raised")

        bridge = _Bridge.alloc().init()
        self._impl = bridge

        try:
            nc = NSWorkspace.sharedWorkspace().notificationCenter()
            nc.addObserver_selector_name_object_(
                bridge,
                objc.selector(_Bridge.workspaceWillSleep_, signature=b"v@:@"),
                "NSWorkspaceWillSleepNotification",
                None,
            )
            nc.addObserver_selector_name_object_(
                bridge,
                objc.selector(_Bridge.workspaceDidWake_, signature=b"v@:@"),
                "NSWorkspaceDidWakeNotification",
                None,
            )

            run_loop = NSRunLoop.currentRunLoop()
            while not self._stop_event.is_set():
                run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.5))
        finally:
            try:
                NSWorkspace.sharedWorkspace().notificationCenter().removeObserver_(bridge)
            except Exception:
                pass
