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
* The alignment anchor for each stream is stamped inside the audio callback,
  from PortAudio's ``inputBufferAdcTime`` — the hardware capture time of the
  block. The difference between the two anchors is what the merge step uses,
  so it must not include Python scheduling: stamping it in the writer thread
  put a queue hand-off and a GIL wake inside the measurement.
* macOS sleep/wake is observed via ``NSWorkspace`` notifications running on a
  dedicated CFRunLoop thread. On sleep we pause; on wake we resume. The pause
  is logged in metadata. If pyobjc is unavailable (e.g. CI on Linux), the
  observer becomes a no-op.
* Device errors mid-recording are surfaced via the optional ``on_device_error``
  callback so the UI layer can show a notification and offer to fall back to
  the system default.
"""

from __future__ import annotations

import errno
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
from src.audio.levels import (
    DEFAULT_SILENCE_SECONDS,
    LevelMonitor,
    render_meter,
    rms_from_int16,
)
from src.health import report_stream_health

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


# Capture in ~256 ms blocks instead of PortAudio's default (~9 ms on macOS).
# Measured on an M-series MacBook: the default delivers ~1,100 Python
# callbacks/s across two streams (≈11% CPU in-process + ≈8% in coreaudiod,
# i.e. real fan noise during calls); 4096-frame blocks at 16 kHz cut that to
# ~8 callbacks/s and <1% combined. We record for offline transcription, so
# capture latency is irrelevant — only power matters.
CAPTURE_BLOCKSIZE_FRAMES = 4096
CAPTURE_LATENCY = "high"

#: Bytes in one capture block (mono int16). The unit the two streams drift by.
CAPTURE_BLOCK_BYTES = CAPTURE_BLOCKSIZE_FRAMES * 2

#: Seconds of audio the capture queue may buffer before dropping blocks.
#: Unbounded before: a writer that was merely *slow* (Time Machine, Spotlight,
#: a network volume) grew the queue at 32 KB/s per stream for the whole
#: meeting. The drop-on-full handler already existed but could never fire.
CAPTURE_QUEUE_SECONDS = 30.0

#: How often the watchdog checks that audio is still flowing.
WATCHDOG_INTERVAL_SECONDS = 5.0

#: No new bytes for this long while RECORDING means the capture stream is dead,
#: not quiet. A silent room still delivers blocks of zeros; a disconnected or
#: sleep-stale device delivers nothing at all. That distinction is the whole
#: point — the level monitor deliberately ignores gaps, so it cannot see this.
STREAM_STALL_SECONDS = 15.0


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class DeviceOpenError(RuntimeError):
    """A capture device refused to open, for a reason that isn't permissions.

    Previously any PaError whose text happened to contain "denied" was reported
    as a microphone-permission problem, sending the user to System Settings to
    fix something that wasn't broken. Everything else propagated as a bare
    PaError with no mention of which device, rate or channel count was
    involved.
    """


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
    #: Live signal level + sustained-silence detector. Set in _open_stream.
    meter: Any = None
    #: Capture blocks discarded because the writer fell too far behind.
    dropped_blocks: int = 0
    #: Watchdog bookkeeping: bytes at the last check, and when they last moved.
    last_bytes_seen: int = 0
    last_progress_monotonic: float = 0.0
    #: True once the watchdog has reported this stream as stalled, so it
    #: reports once rather than every tick.
    stalled: bool = False
    #: Rebuilds the capture callback when the input stream has to be recreated.
    make_callback: Any = None


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
        silence_alert_seconds: float = DEFAULT_SILENCE_SECONDS,
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
        # Seconds of continuous digital silence before a stream is reported as
        # broken. Long enough to survive a quiet stretch in a meeting.
        self._silence_alert_seconds = float(silence_alert_seconds)

        # Pause bookkeeping. Offsets are seconds since ``start()`` returned.
        self._start_monotonic: float | None = None
        self._start_wall_clock: datetime | None = None
        self._pauses: list[dict[str, float]] = []
        self._current_pause_started: float | None = None
        # Who initiated the current pause: "user" or "sleep". A wake event
        # only auto-resumes a sleep-initiated pause — never one the user
        # made deliberately before closing the lid.
        self._pause_source: str | None = None

        # Sleep/wake observer (pyobjc-based). Lazily started.
        self._sleep_wake_observer: _SleepWakeObserver | None = None
        # Capture watchdog: notices a stream that stops delivering audio.
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop: threading.Event | None = None

    # =====================================================================
    # Public API
    # =====================================================================
    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def on_device_error(self) -> Callable[[str, Exception], None] | None:
        return self._on_device_error

    @on_device_error.setter
    def on_device_error(self, callback: Callable[[str, Exception], None] | None) -> None:
        """Late-bind the device-error callback.

        The UI builds recorders through a config-only factory, so it can't
        pass the callback at construction time — it sets it here right
        before ``start()``.
        """
        self._on_device_error = callback

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
            except Exception:
                # Roll back any partially opened resources before re-raising.
                self._teardown_streams()
                self._state = RecorderState.ERROR
                raise

            # The system stream opens separately and is allowed to fail. It
            # used to share the mic's try-block, so any PaError from BlackHole
            # — device busy, sample-rate refusal, driver wedged after a sleep —
            # tore down the mic recording too. That defeated the whole point of
            # the graceful-degradation path below: mic-only is a usable
            # recording, no recording is not.
            if self._system_state is not None:
                try:
                    self._open_stream(self._system_state)
                except Exception as exc:
                    logger.warning(
                        "System audio stream failed to open (%s); recording "
                        "mic-only. System audio will be missing from this "
                        "transcript.", exc,
                    )
                    self._system_state = None
                    try:
                        report_stream_health(
                            "system", silent=True,
                            detail=f"capture device would not open: {exc}",
                        )
                    except Exception:  # pragma: no cover (defensive)
                        logger.exception("Could not report system-stream failure")
                    if self._on_device_error is not None:
                        try:
                            self._on_device_error("system", exc)
                        except Exception:  # pragma: no cover
                            logger.exception("on_device_error callback raised")

            self._state = RecorderState.RECORDING
            self._start_watchdog()
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

    def pause(self, *, source: str = "user") -> None:
        with self._state_lock:
            if self._state != RecorderState.RECORDING:
                logger.debug("pause() ignored; state=%s", self._state)
                return
            self._active.clear()
            self._current_pause_started = self._elapsed_seconds()
            self._pause_source = source
            self._state = RecorderState.PAUSED
            logger.info(
                "Recording paused at %.3fs (source=%s)",
                self._current_pause_started, source,
            )

    def resume(self, *, source: str = "user") -> None:
        with self._state_lock:
            if self._state != RecorderState.PAUSED:
                logger.debug("resume() ignored; state=%s", self._state)
                return
            if source == "wake" and self._pause_source != "sleep":
                logger.info(
                    "Wake resume skipped — pause was user-initiated; "
                    "recording stays paused."
                )
                return
            now = self._elapsed_seconds()
            if self._current_pause_started is not None:
                self._pauses.append(
                    {"paused_at": self._current_pause_started, "resumed_at": now}
                )
                self._current_pause_started = None
            self._pause_source = None
            self._active.set()
            self._state = RecorderState.RECORDING
            # Don't let the watchdog count the sleep as a stall.
            for st in (self._mic_state, self._system_state):
                if st is not None:
                    st.last_progress_monotonic = time.monotonic()
            logger.info("Recording resumed at %.3fs (source=%s)", now, source)

        if source == "wake":
            # CoreAudio routinely re-enumerates across sleep — Bluetooth always
            # does. resume() used to only set a flag, so a stream whose device
            # went away came back dead: no error, no bytes, and the rest of the
            # meeting silently lost.
            self._revive_dead_streams()

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
            self._stop_watchdog()
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

    def _queue_maxsize(self) -> int:
        """Blocks of headroom before the callback starts dropping audio.

        Sized in seconds so it doesn't silently change with the block size.
        Dropping a block is bad; growing without bound until the machine
        swaps, during a meeting, is worse.
        """
        blocks_per_second = self._sample_rate / CAPTURE_BLOCKSIZE_FRAMES
        return max(8, int(CAPTURE_QUEUE_SECONDS * blocks_per_second))

    # ------------------------------------------------------------- metering
    def _make_meter(self, label: str) -> LevelMonitor:
        """Build the level monitor for one stream, wired to the health panel."""

        def on_silence(stream_label: str, seconds: float) -> None:
            logger.warning(
                "[%s] no signal for %.0fs — the stream is capturing silence.",
                stream_label, seconds,
            )
            report_stream_health(
                stream_label,
                silent=True,
                detail=f"no signal for {int(seconds)}s",
            )

        def on_signal(stream_label: str) -> None:
            logger.info("[%s] signal returned.", stream_label)
            report_stream_health(stream_label, silent=False, detail="signal ok")

        return LevelMonitor(
            label,
            silence_seconds=self._silence_alert_seconds,
            on_silence=on_silence,
            on_signal=on_signal,
        )

    @property
    def levels(self) -> dict[str, float]:
        """Current smoothed input level per stream, 0.0-1.0 (for the menu)."""
        out: dict[str, float] = {}
        for st in (self._mic_state, self._system_state):
            if st is not None and st.meter is not None:
                out[st.label] = st.meter.level
        return out

    def level_meters(self, width: int = 8) -> dict[str, str]:
        """Text meters per stream, ready to drop into a menu item title."""
        return {
            label: render_meter(level, width=width)
            for label, level in self.levels.items()
        }

    def silent_streams(self) -> list[str]:
        """Streams that have been silent long enough to look broken."""
        out: list[str] = []
        for st in (self._mic_state, self._system_state):
            if st is not None and st.meter is not None and st.meter.is_silent:
                out.append(st.label)
        return out

    def _open_stream(self, st: _StreamState) -> None:
        """Open a sounddevice InputStream and start the writer thread."""
        import sounddevice as sd

        st.queue = queue.Queue(maxsize=self._queue_maxsize())
        st.bytes_written = 0
        st.start_monotonic = None
        st.error = None
        st.meter = self._make_meter(st.label)

        block_seconds = CAPTURE_BLOCKSIZE_FRAMES / float(self._sample_rate)

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                # Non-fatal warnings (input_overflow etc.) — log but keep going.
                logger.warning("[%s] sounddevice status: %s", st.label, status)
            if st.error is not None:
                return  # writer thread is dead; queueing would grow unbounded
            if not self._active.is_set():
                return  # paused — drop frames
            # Stamp the alignment anchor here, from the hardware capture time
            # of this block's first sample. See _capture_timestamp.
            if st.start_monotonic is None:
                st.start_monotonic = _capture_timestamp(time_info)
            # Measure before queueing: this is the only place the samples are
            # seen live, and it's how a dead stream gets caught during the call
            # instead of after transcription two months later.
            try:
                st.meter.observe(rms_from_int16(indata), block_seconds)
            except Exception:  # pragma: no cover (never break capture)
                pass
            try:
                # ``indata`` is int16 because we requested dtype="int16".
                st.queue.put_nowait(bytes(indata))
            except queue.Full:
                # The queue is bounded now, so this path is live: the writer
                # has fallen more than CAPTURE_QUEUE_SECONDS behind. Drop the
                # block rather than growing memory for the rest of the meeting,
                # and say so once instead of per block.
                st.dropped_blocks += 1
                if st.dropped_blocks == 1:
                    logger.error(
                        "[%s] capture queue full after %.0fs of buffering; "
                        "dropping audio. The disk can't keep up.",
                        st.label, CAPTURE_QUEUE_SECONDS,
                    )

        # Kept so the watchdog can rebuild the input stream after a device
        # disconnect or a sleep that re-enumerated CoreAudio, without
        # disturbing the queue, the writer thread or the open WAV file.
        st.make_callback = lambda: callback
        self._start_input_stream(st, sd)

        st.last_bytes_seen = 0
        st.last_progress_monotonic = time.monotonic()
        st.stalled = False

        st.writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(st,),
            name=f"otis-writer-{st.label}",
            daemon=True,
        )
        st.writer_thread.start()

    def _revive_dead_streams(self) -> None:
        """Rebuild any stream whose device did not survive sleep."""
        for st in (self._mic_state, self._system_state):
            if st is None or st.stream is None:
                continue
            active = getattr(st.stream, "active", None)
            if active is None:
                continue  # host doesn't report it; the watchdog will catch it
            if not active:
                self._revive_stream(st)

    # ------------------------------------------------------------ watchdog
    def _start_watchdog(self) -> None:
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(self._watchdog_stop,),
            name="otis-capture-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        if self._watchdog_stop is not None:
            self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._watchdog_thread = None

    def _watchdog_loop(self, stop_event: threading.Event) -> None:
        """Notice when a capture stream stops delivering audio.

        Until now the only way a device failure reached the user was if the
        *writer* raised — i.e. disk errors only. A device that was unplugged,
        or that re-enumerated while the Mac slept, simply stopped calling the
        callback: no exception, no error, ``stop()`` reporting success, and the
        rest of the meeting written as nothing at all.

        Silence is not the signal here — a quiet room still delivers blocks of
        zeros. The signal is *no blocks*, which is why this watches bytes
        rather than level.
        """
        while not stop_event.wait(WATCHDOG_INTERVAL_SECONDS):
            try:
                self._check_streams_alive()
            except Exception:  # pragma: no cover (defensive)
                logger.exception("Capture watchdog tick failed; continuing.")

    def _check_streams_alive(self) -> None:
        with self._state_lock:
            if self._state is not RecorderState.RECORDING:
                return  # paused or stopped: no data expected
            streams = [s for s in (self._mic_state, self._system_state) if s is not None]

        now = time.monotonic()
        for st in streams:
            if st.error is not None:
                continue  # already reported by the writer
            if st.bytes_written != st.last_bytes_seen:
                st.last_bytes_seen = st.bytes_written
                st.last_progress_monotonic = now
                if st.stalled:
                    st.stalled = False
                    logger.info("[%s] capture recovered.", st.label)
                    report_stream_health(st.label, silent=False, detail="signal ok")
                continue
            stalled_for = now - st.last_progress_monotonic
            if stalled_for < STREAM_STALL_SECONDS or st.stalled:
                continue
            st.stalled = True
            logger.error(
                "[%s] no audio for %.0fs — the capture device has stopped "
                "delivering.", st.label, stalled_for,
            )
            report_stream_health(
                st.label, silent=True,
                detail=f"capture stopped {int(stalled_for)}s ago",
            )
            if self._on_device_error is not None:
                try:
                    self._on_device_error(
                        st.label,
                        RuntimeError(f"no audio for {int(stalled_for)}s"),
                    )
                except Exception:  # pragma: no cover
                    logger.exception("on_device_error callback raised")
            self._revive_stream(st)

    def _start_input_stream(self, st: _StreamState, sd: Any = None) -> None:
        """Create and start the InputStream for ``st``. Raises on failure."""
        if sd is None:
            import sounddevice as sd
        try:
            st.stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=st.device.index,
                callback=st.make_callback(),
                blocksize=CAPTURE_BLOCKSIZE_FRAMES,
                latency=CAPTURE_LATENCY,
            )
            st.stream.start()
        except Exception as exc:  # pragma: no cover (depends on host)
            if _is_permission_error(exc):
                raise PermissionDeniedError(str(exc)) from exc
            raise DeviceOpenError(
                f"Could not open {st.device.name!r} at "
                f"{self._sample_rate} Hz / {self._channels}ch: {exc}"
            ) from exc

    def _revive_stream(self, st: _StreamState) -> bool:
        """Rebuild a dead input stream in place. Returns True on success.

        The WAV file, the queue and the writer thread are untouched, so the
        recording continues into the same file — the gap is simply missing
        audio, which is a far better outcome than the rest of the meeting
        being silently lost.
        """
        logger.warning("[%s] capture stream looks dead; rebuilding it.", st.label)
        try:
            if st.stream is not None:
                try:
                    st.stream.stop()
                except Exception:  # pragma: no cover (already dead)
                    pass
                try:
                    st.stream.close()
                except Exception:  # pragma: no cover (already dead)
                    pass
            st.stream = None
            self._start_input_stream(st)
        except Exception as exc:
            logger.error("[%s] could not rebuild the capture stream: %s", st.label, exc)
            return False
        st.last_progress_monotonic = time.monotonic()
        logger.info("[%s] capture stream rebuilt.", st.label)
        return True

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
                    wf.writeframes(chunk)
                    st.bytes_written += len(chunk)
        except Exception as exc:
            logger.exception("[%s] writer thread crashed: %s", st.label, exc)
            st.error = exc
            # The recorder used to stay in RECORDING after this: the UI kept
            # showing a live session, the callback silently dropped every
            # subsequent frame, and stop() wrote metadata with bytes_written
            # frozen at whatever got through. You could record a full meeting
            # and find three minutes of audio. Mark the failure in the state
            # machine so the truth is visible while it's still actionable.
            self._note_writer_failure(st, exc)
            if self._on_device_error is not None:
                try:
                    self._on_device_error(st.label, exc)
                except Exception:
                    logger.exception("on_device_error callback raised")

    def _note_writer_failure(self, st: _StreamState, exc: BaseException) -> None:
        """Report a dead writer to the health panel; degrade state if total."""
        detail = f"{st.label} write failed: {exc}"
        if _looks_like_disk_full(exc):
            detail = f"{st.label} stopped — the disk is full"
        try:
            report_stream_health(st.label, silent=True, detail=detail)
        except Exception:  # pragma: no cover (defensive)
            logger.exception("Could not report writer failure to health")

        # If every stream's writer is dead there is nothing left to capture.
        others = [
            s for s in (self._mic_state, self._system_state)
            if s is not None and s is not st
        ]
        if any(s.error is None for s in others):
            return
        with self._state_lock:
            if self._state == RecorderState.RECORDING:
                logger.error(
                    "All capture writers have failed; leaving RECORDING so the "
                    "UI stops claiming a recording is in progress."
                )
                self._state = RecorderState.ERROR

    def _teardown_streams(self) -> None:
        """Stop both streams together, then close, then drain.

        This used to run stop()+close() fully for the mic before touching the
        system stream — so the system stream kept capturing for the whole
        duration of the mic's teardown, and the two files ended up different
        lengths. Every real session on this machine showed the delta as an
        exact multiple of one capture block, and three of four were desynced:

            2026-07-17  +1 block      2026-07-20  +1 block
            2026-08-07  +2 blocks     2026-07-22   0

        Commit 973f440 raised the block size from ~9 ms to 256 ms for the CPU
        win, which multiplied the error ~28x without adding a barrier. Stopping
        both streams first collapses the window to the gap between two stop()
        calls instead of a whole stop-and-close cycle.
        """
        streams = [st for st in (self._mic_state, self._system_state) if st is not None]

        # Phase 1 — halt capture on every stream before doing slower work on
        # any of them. This is the barrier that keeps the two files aligned.
        for st in streams:
            try:
                if st.stream is not None:
                    st.stream.stop()
            except Exception as exc:  # pragma: no cover
                logger.warning("[%s] error stopping stream: %s", st.label, exc)

        # Phase 2 — now the slow part, with no capture still running.
        for st in streams:
            try:
                if st.stream is not None:
                    st.stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("[%s] error closing stream: %s", st.label, exc)
            finally:
                st.stream = None
            # Signal writer to drain remaining queue and exit.
            st.queue.put_nowait(None)

        for st in streams:
            if st.writer_thread is None:
                continue
            st.writer_thread.join(timeout=5.0)
            if st.writer_thread.is_alive():  # pragma: no cover
                logger.error("[%s] writer thread did not exit", st.label)

        self._warn_if_streams_desynced()

    def _warn_if_streams_desynced(self) -> None:
        """Surface a cross-stream length mismatch instead of shipping it.

        The alignment step applies a single constant offset, so it cannot
        correct a drift that appears at teardown. Nothing checked this before,
        which is why the regression went unnoticed through a release.
        """
        mic, sys_st = self._mic_state, self._system_state
        if mic is None or sys_st is None:
            return
        if not mic.bytes_written or not sys_st.bytes_written:
            return
        delta = abs(mic.bytes_written - sys_st.bytes_written)
        if delta <= CAPTURE_BLOCK_BYTES:
            return
        blocks = delta / CAPTURE_BLOCK_BYTES
        seconds = delta / (self._sample_rate * self.SAMPLE_WIDTH_BYTES)
        logger.warning(
            "Streams ended %d bytes apart (%.1f blocks, %.0f ms): mic=%d system=%d. "
            "Transcript alignment may drift.",
            delta, blocks, seconds * 1000, mic.bytes_written, sys_st.bytes_written,
        )

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
            self.pause(source="sleep")
        except Exception:  # pragma: no cover
            logger.exception("Failed to pause on system sleep")

    def _handle_system_wake(self) -> None:
        logger.info("System wake detected — resuming recording.")
        try:
            self.resume(source="wake")
        except Exception:  # pragma: no cover
            logger.exception("Failed to resume on system wake")


# ============================================================================
# Sleep/wake observer (pyobjc-backed; degrades to no-op if unavailable)
# ============================================================================
#: PortAudio error text that genuinely means "macOS denied microphone access".
#: Matching on the bare word "denied" also caught unrelated failures and
#: reported them as a permissions problem, sending the user to System Settings
#: to fix something that wasn't broken.
_PERMISSION_MARKERS = (
    "permission denied",
    "not permitted",
    "microphone access",
    "access denied",
)


def _is_permission_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _PERMISSION_MARKERS)


def _warn_if_device_cannot_do(device: Any, sample_rate: int, channels: int) -> None:
    """Log when a device's declared capabilities don't match what we request.

    CoreAudio usually resamples 48 kHz down to 16 kHz transparently, and this
    is deliberately a warning rather than a refusal — the capability table is
    advisory and being wrong about it should never stop a recording. But when
    an open *does* fail, having said this first turns an opaque PaError into
    something diagnosable.

    Channel count is the one that quietly costs data: BlackHole 2ch reports two
    input channels and we open it as mono, and whether CoreAudio downmixes or
    just takes channel 0 is host-dependent. If it takes channel 0, a source
    panned right is lost with no error anywhere.
    """
    max_channels = getattr(device, "max_input_channels", None)
    if isinstance(max_channels, int) and max_channels and channels > max_channels:
        logger.warning(
            "%r reports %d input channel(s) but %d were requested; "
            "the open will probably fail.",
            device.name, max_channels, channels,
        )
    declared = getattr(device, "default_sample_rate", None)
    if isinstance(declared, (int, float)) and declared and abs(declared - sample_rate) > 1:
        logger.debug(
            "%r runs natively at %.0f Hz; requesting %d Hz (CoreAudio resamples).",
            device.name, declared, sample_rate,
        )


def _capture_timestamp(time_info: Any) -> float:
    """When the first sample of this block was actually captured, in seconds.

    The two streams are aligned by the difference between their anchors, so
    what matters is that both are measured on the same clock at the same point
    in the pipeline.

    PortAudio hands the callback ``inputBufferAdcTime`` — the hardware capture
    time of the buffer. Measured on this Mac it shares ``time.monotonic``'s
    base (a constant ~0.28 s input-latency offset, and ``currentTime`` tracking
    ``monotonic`` to 0.4 ms), so the two streams' values are directly
    comparable. Falls back to ``time.monotonic()`` when a host doesn't supply
    it; both streams degrade together, so the difference stays meaningful.

    Measured notes, because the obvious story here is wrong
    ------------------------------------------------------
    The audit attributed the observed 5-99 ms mic-vs-system spread to stamping
    this in the *writer* thread, behind a queue hand-off and the GIL. Measuring
    all three sources side by side over six runs, that is not what dominates:

        writer thread        skew spread 173.28 ms, stdev 59.09
        callback monotonic   skew spread 173.21 ms, stdev 59.12
        callback ADC         skew spread 173.05 ms, stdev 59.02

    The hand-off contributes well under a millisecond. The spread is the two
    devices genuinely starting at different times, which an accurate anchor is
    supposed to record, not erase. So this change buys accuracy (~1 ms of
    measurement error removed, and the input latency excluded properly), not
    the order-of-magnitude the audit implied.

    It also rules out the tempting fix. Starting both streams back to back —
    the startup analogue of the teardown barrier — makes things *worse*,
    because the two devices then contend during CoreAudio spin-up:

        start() 50 ms apart     skew spread   6.89 ms
        start() back to back    skew spread 504.97 ms

    So the streams are still opened one at a time, deliberately.
    """
    adc = getattr(time_info, "inputBufferAdcTime", None)
    if isinstance(adc, (int, float)) and adc > 0:
        return float(adc)
    return time.monotonic()


def _looks_like_disk_full(exc: BaseException) -> bool:
    """True for ENOSPC, however the OS phrased it."""
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    text = str(exc).lower()
    return "no space left" in text or "enospc" in text


_BRIDGE_CLASS: Any = None
_BRIDGE_CLASS_LOCK = threading.Lock()


def sleep_wake_bridge_class() -> Any:
    """Define the sleep/wake Objective-C bridge exactly once per process.

    This class used to be declared inside ``_SleepWakeObserver._run``. ObjC
    registers class names process-wide, so the first recorder defined it and
    every recorder after that died on::

        objc.error: _Bridge is overriding existing Objective-C class

    inside its observer thread, where nothing was watching. The menu bar builds
    a fresh recorder for every meeting, so sleep/wake handling worked for the
    first recording of a session and silently stopped for all the rest — and
    with it the wake-time stream revival added in P2, whose entire purpose is
    that a lid closed mid-meeting does not produce a dead stream.
    """
    global _BRIDGE_CLASS
    with _BRIDGE_CLASS_LOCK:
        if _BRIDGE_CLASS is not None:
            return _BRIDGE_CLASS

        from Foundation import NSObject

        class _OtisSleepWakeBridge(NSObject):
            def workspaceWillSleep_(self, _notification):
                handler = getattr(self, "otis_on_sleep", None)
                if handler is None:
                    return
                try:
                    handler()
                except Exception:
                    logger.exception("on_sleep handler raised")

            def workspaceDidWake_(self, _notification):
                handler = getattr(self, "otis_on_wake", None)
                if handler is None:
                    return
                try:
                    handler()
                except Exception:
                    logger.exception("on_wake handler raised")

        _BRIDGE_CLASS = _OtisSleepWakeBridge
        return _BRIDGE_CLASS


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
            # Imported here so non-macOS environments don't blow up at import
            # time. These are availability probes — the real imports happen on
            # the observer thread — so they are deliberately unused.
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
        from Foundation import NSDate, NSRunLoop

        cls = sleep_wake_bridge_class()
        bridge = cls.alloc().init()
        # Handlers live on the instance, not in a closure, because the class
        # is now shared by every recorder in the process.
        bridge.otis_on_sleep = self._on_sleep
        bridge.otis_on_wake = self._on_wake
        self._impl = bridge

        try:
            nc = NSWorkspace.sharedWorkspace().notificationCenter()
            nc.addObserver_selector_name_object_(
                bridge,
                objc.selector(cls.workspaceWillSleep_, signature=b"v@:@"),
                "NSWorkspaceWillSleepNotification",
                None,
            )
            nc.addObserver_selector_name_object_(
                bridge,
                objc.selector(cls.workspaceDidWake_, signature=b"v@:@"),
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
