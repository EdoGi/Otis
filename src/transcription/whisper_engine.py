"""Wrapper around ``mlx-whisper`` for local-only transcription.

Why mlx-whisper?
    The MLX framework runs natively on Apple Silicon (Metal-backed). On an M1
    Pro a 1-hour meeting transcribes with the ``small`` model in ~3-8 minutes
    — significantly faster than ``openai-whisper`` (which only uses CPU on
    macOS unless you wrestle with PyTorch + MPS).

What this module does
---------------------
* Lazy-loads the model on the first ``transcribe()`` call (saves ~600 MB of
  RAM while idle).
* Drops the model reference after ``idle_timeout`` seconds of inactivity so
  the OS can reclaim the memory.
* Reports approximate progress to a caller-supplied callback by sampling
  wall-clock time vs the audio's intrinsic duration (mlx-whisper itself
  doesn't expose a per-step hook).
* Translates the most common failure modes into actionable errors:
  - model download failed → ``ModelDownloadError`` (mentions the network)
  - out of memory → ``OutOfMemoryError`` (suggests a smaller model)
  - corrupt / empty audio → returns an empty ``TranscriptionResult``
    rather than crashing the daemon.

The actual call to mlx-whisper goes through a pluggable ``transcribe_fn``
parameter so unit tests can run without the framework installed.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model name → Hugging Face repo (mlx-community/whisper-* mirrors)
# ---------------------------------------------------------------------------
_MODEL_REPO_MAP: dict[str, str] = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large":    "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0

# RMS below this counts as "essentially silent" and skips Whisper entirely.
# Empirically, real (even quiet) speech via the built-in MacBook mic sits at
# 0.01-0.05 RMS; ambient room noise / muted streams are 0.0001-0.001. We pick
# 0.001 (~-60 dBFS) — comfortably below quiet speech, comfortably above the
# noise floor of a healthy capture chain. The other anti-hallucination defences
# (``condition_on_previous_text=False`` + the post-filter) are what catch the
# really pathological cases.
SILENCE_RMS_THRESHOLD = 0.001

#: Ceiling on an afconvert decode. A 2-hour FLAC decodes in a couple of
#: seconds; anything near this means something is wrong.
DECODE_TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """One Whisper segment: a contiguous chunk of timed text."""

    start: float            # seconds from the start of the audio file
    end: float              # seconds from the start of the audio file
    text: str               # already-stripped human-readable text

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Segment:
        return cls(
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=str(raw.get("text", "")).strip(),
        )


@dataclass
class TranscriptionResult:
    """Output of one ``transcribe()`` call."""

    segments: list[Segment] = field(default_factory=list)
    detected_language: str | None = None
    duration_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    #: True when the pre-check found the audio to be effectively silent and
    #: Whisper was skipped. The caller records this in the transcript so a
    #: half-lost meeting is visible in the file, not only in the log.
    was_silent: bool = False
    #: Peak-window RMS measured by the pre-check (``None`` if unmeasurable).
    measured_rms: float | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class WhisperError(RuntimeError):
    """Base for transcription failures we know how to handle."""


class ModelDownloadError(WhisperError):
    """mlx-whisper couldn't pull the model weights (usually no network)."""


class OutOfMemoryError(WhisperError):
    """Model + audio didn't fit in RAM/VRAM. Suggest a smaller model."""


# ---------------------------------------------------------------------------
# WhisperEngine
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[float], None]
TranscribeFn = Callable[..., dict[str, Any]]


class WhisperEngine:
    """Thread-safe lazy wrapper around mlx-whisper.

    Parameters
    ----------
    model_name:
        ``tiny`` | ``base`` | ``small`` | ``medium`` | ``large-v3``. Maps to a
        ``mlx-community/whisper-*-mlx`` Hugging Face repo.
    idle_timeout_seconds:
        Seconds of inactivity before we drop our cached model reference. The
        OS won't free GPU memory while another process holds it, but Python's
        GC will release whatever the engine itself was holding.
    transcribe_fn:
        Override the underlying transcription call. Defaults to
        :func:`mlx_whisper.transcribe`. Tests pass a deterministic fake.
    """

    def __init__(
        self,
        *,
        model_name: str = "small",
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        transcribe_fn: TranscribeFn | None = None,
        rtf_state_path: str | Path | None = None,
        health: Any | None = None,
        report_health: bool = True,
    ) -> None:
        if model_name not in _MODEL_REPO_MAP:
            raise ValueError(
                f"Unknown model {model_name!r}; choose from {sorted(_MODEL_REPO_MAP)}"
            )
        self._model_name = model_name
        self._model_repo = _MODEL_REPO_MAP[model_name]
        #: Local snapshot directory, resolved on first use. See
        #: :func:`resolve_model_path` for why we do not just pass the repo id.
        self._resolved_model_path: str | None = None
        self._shutdown = False
        self._idle_timeout = float(idle_timeout_seconds)
        self._transcribe_fn = transcribe_fn or _default_mlx_transcribe
        self._health = health
        self._report_health = bool(report_health)

        self._lock = threading.RLock()
        # Serialises the actual decode. The old code bumped an ``_inflight``
        # counter under ``_lock`` and then called _call_transcribe OUTSIDE it,
        # so two workers could decode simultaneously on one shared engine —
        # contending for the GPU and thrashing mlx's ModelHolder. Counting
        # concurrency is not the same as preventing it.
        self._run_lock = threading.Lock()
        # mlx-whisper caches the loaded model in its own module state
        # (ModelHolder) — 0.5-2 GB of unified memory depending on the model.
        # We track warmth + an idle timer, and on expiry we actually CLEAR
        # that cache so the memory goes back to the OS instead of being held
        # until the app quits.
        self._warm = False
        self._idle_timer: threading.Timer | None = None
        self._inflight = 0  # transcriptions currently running
        # Measured real-time factor (audio seconds per wall-clock second),
        # EMA-updated after each successful run so progress estimates adapt
        # to this machine + model instead of assuming the M1-Pro/small 8x.
        # Optionally persisted per-model so restarts keep honest estimates.
        self._rtf_state_path = Path(rtf_state_path).expanduser() if rtf_state_path else None
        self._measured_rtf: float | None = self._load_persisted_rtf()

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_warm(self) -> bool:
        return self._warm

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def transcribe(
        self,
        wav_path: str | Path,
        *,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
        stream: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe ``wav_path`` and return :class:`TranscriptionResult`.

        ``language=None`` lets Whisper auto-detect; pass ``"en"`` / ``"fr"``
        / etc. to force a specific language and skip the detection pass.

        Empty / unreadable audio returns an empty result (no segments) rather
        than raising — this keeps a single corrupt file from killing the
        daemon when it tries to transcribe.
        """
        path = Path(wav_path).expanduser()
        if not path.exists() or path.stat().st_size == 0:
            logger.warning("Transcribe skipped — missing or empty: %s", path)
            return TranscriptionResult(detected_language=language)

        # Cheap silence pre-check: if the WAV is essentially silent, skip
        # Whisper entirely. The threshold is module-level (SILENCE_RMS_THRESHOLD)
        # so it can be tuned without re-reading argument plumbing. The post-
        # transcription hallucination filter handles the rest.
        # Peak window RMS across the whole file — see _wav_rms docstring.
        rms = _wav_rms(path)
        label = stream or _infer_stream_label(path)
        if rms is not None and rms < SILENCE_RMS_THRESHOLD:
            logger.info(
                "Transcribe skipped — %s is essentially silent "
                "(peak-window RMS=%.4f, threshold=%.4f).",
                path.name, rms, SILENCE_RMS_THRESHOLD,
            )
            # This exact line was logged at every transcription for two months
            # while the app told the user "Transcript ready ✅". The signal now
            # goes somewhere a human will actually see it.
            self._report_stream_health(label, silent=True)
            return TranscriptionResult(
                detected_language=language,
                duration_seconds=_safe_wav_duration_seconds(path),
                was_silent=True,
                measured_rms=rms,
            )
        if rms is None:
            # The pre-check could not measure this file (24-bit, resampled,
            # partly corrupt). Saying "signal ok" about audio we never read is
            # the same failure the whole health surface exists to prevent, so
            # leave the component untouched rather than turn the panel green.
            logger.info("Signal level unmeasurable for %s; not reporting health.",
                        path.name)
        else:
            self._report_stream_health(label, silent=False, rms=rms)

        duration = _safe_wav_duration_seconds(path)
        progress = (
            _ProgressEstimator(duration, on_progress, rtf=self._current_rtf())
            if on_progress
            else None
        )
        # One decode at a time on this engine. We block here rather than inside
        # the try so the progress estimator measures the run, not the queue —
        # and so a queued job can't report 40% while it hasn't started.
        with self._run_lock:
            if progress is not None:
                progress.start()

            started = time.monotonic()
            try:
                with self._lock:
                    self._warm = True
                    self._inflight += 1
                    self._reset_idle_timer()

                raw = self._call_transcribe(path, language)
            except _MlxImportFailure as exc:
                raise ModelDownloadError(
                    "mlx-whisper isn't installed; run `pip install mlx-whisper`."
                ) from exc
            except Exception as exc:
                msg = str(exc).lower()
                if "memory" in msg or "alloc" in msg or "oom" in msg:
                    raise OutOfMemoryError(
                        f"Out of memory while running model={self._model_name!r}. "
                        "Try a smaller model in Settings → Whisper Model."
                    ) from exc
                if any(tok in msg for tok in ("download", "network", "connection", "resolve")):
                    raise ModelDownloadError(
                        "Could not download the Whisper model. "
                        "Check your internet connection and try again."
                    ) from exc
                raise
            finally:
                if progress is not None:
                    progress.stop()
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    # Long meetings transcribe for far more than the idle
                    # timeout — re-arm at the END too, so the cooldown counts
                    # from when we actually went quiet.
                    self._reset_idle_timer()

        result = self._parse_result(raw, language)
        self._record_rtf(duration, time.monotonic() - started)

        if on_progress is not None:
            try:
                on_progress(100.0)
            except Exception:
                logger.exception("on_progress(100) raised")
        return result

    def shutdown(self) -> None:
        """Cancel the idle timer, drop the warm flag, release model memory."""
        # The release stays INSIDE the lock. Previously we read _inflight,
        # dropped the lock, then cleared mlx's ModelHolder — so a transcription
        # starting in that window could get a half-cleared holder back from
        # ModelHolder.get_model() and fail deep inside decoding.
        with self._lock:
            self._shutdown = True
            self._warm = False
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self._inflight == 0:
                _release_mlx_model_cache()

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------
    def _registry(self) -> Any | None:
        if not self._report_health:
            return None
        if self._health is not None:
            return self._health
        from src.health import get_health

        return get_health()

    def _report_stream_health(
        self, label: str | None, *, silent: bool, rms: float | None = None,
    ) -> None:
        """Turn the silence pre-check into a user-visible health state.

        A silent *system* stream means the far side of every meeting is being
        lost — that is a hard failure with a known fix. A silent *mic* usually
        just means nobody spoke on this side, so it is only a warning.
        """
        registry = self._registry()
        if registry is None or label not in ("mic", "system"):
            return
        from src.health import report_stream_health

        if silent:
            detail = (
                "captured silence — the other side of the call was not recorded"
                if label == "system"
                else "captured silence"
            )
        else:
            detail = f"signal ok (rms {rms:.3f})" if rms is not None else "signal ok"
        report_stream_health(label, silent=silent, detail=detail, registry=registry)

    def _call_transcribe(self, path: Path, language: str | None) -> dict[str, Any]:
        # Tuning notes:
        # * ``condition_on_previous_text=False`` is critical. With the default
        #   (True) Whisper feeds each segment's tokens into the context for the
        #   next one, so on near-silent audio it locks into a hallucinated loop
        #   ("Joyeux Joyeux Joyeux…", or YouTube subtitle leftovers).
        # * ``no_speech_threshold=0.6`` is the mlx-whisper default; we keep it.
        # * ``compression_ratio_threshold=2.4`` is also the default; segments
        #   that compress better than this are dropped (another hallucination
        #   guard).
        return self._transcribe_fn(
            str(path),
            path_or_hf_repo=self.model_path,
            language=language,
            verbose=False,
            condition_on_previous_text=False,
        )

    @staticmethod
    def _parse_result(raw: dict[str, Any], language_override: str | None) -> TranscriptionResult:
        if not isinstance(raw, dict):
            return TranscriptionResult()
        segments = [Segment.from_raw(s) for s in raw.get("segments") or []]
        # Some return shapes only use ``text`` with no segments. Synthesise one
        # so downstream code always sees a list.
        if not segments and raw.get("text"):
            segments = [
                Segment(start=0.0, end=float(raw.get("duration", 0.0)),
                        text=str(raw["text"]).strip())
            ]
        segments = filter_hallucinations(segments)
        return TranscriptionResult(
            segments=segments,
            detected_language=str(raw.get("language") or language_override or "")
                              or None,
            duration_seconds=float(raw["duration"]) if raw.get("duration") else None,
            raw=raw,
        )

    def set_model(self, model_name: str) -> bool:
        """Switch models. Blocks until any running transcription finishes.

        The picker used to write config and stop there, so the checkmark said
        `medium` while the engine kept running `small` — and the transcript's
        frontmatter recorded `small` while a *failure* placeholder for the same
        session read config directly and recorded `medium`. Two files
        disagreeing about what produced them.

        Never call this from the UI thread: ``_run_lock`` is held for the whole
        of a transcription, which is minutes.
        """
        if model_name not in _MODEL_REPO_MAP:
            raise ValueError(
                f"Unknown model {model_name!r}; choose from {sorted(_MODEL_REPO_MAP)}"
            )
        with self._run_lock:
            with self._lock:
                if model_name == self._model_name:
                    return False
                previous = self._model_name
                self._model_name = model_name
                self._model_repo = _MODEL_REPO_MAP[model_name]
                self._resolved_model_path = None
                self._warm = False
                if self._idle_timer is not None:
                    self._idle_timer.cancel()
                    self._idle_timer = None
            # Outside the state lock, still inside _run_lock: nothing can be
            # transcribing, so dropping mlx's held model is safe here.
            _release_mlx_model_cache()
            logger.info("Whisper model switched: %s -> %s", previous, model_name)
            return True

    @property
    def model_path(self) -> str:
        """Local directory for the model, resolved once and remembered.

        Falls back to the repo id if it cannot be resolved, so a first run on
        a machine with no cached model still downloads normally.
        """
        with self._lock:
            if self._resolved_model_path is not None:
                return self._resolved_model_path
        resolved = resolve_model_path(self._model_repo)
        with self._lock:
            self._resolved_model_path = resolved
        return resolved

    def _current_rtf(self) -> float:
        with self._lock:
            return self._measured_rtf or _ProgressEstimator.REAL_TIME_FACTOR

    def _record_rtf(self, audio_seconds: float, elapsed_seconds: float) -> None:
        """EMA-update the measured real-time factor after a successful run.

        Short clips and sub-half-second runs are skipped — they're dominated
        by model load / fixed overhead and would skew the estimate.
        """
        if audio_seconds <= 5.0 or elapsed_seconds <= 0.5:
            return
        measured = audio_seconds / elapsed_seconds
        with self._lock:
            if self._measured_rtf is None:
                self._measured_rtf = measured
            else:
                self._measured_rtf = 0.5 * self._measured_rtf + 0.5 * measured
            smoothed = self._measured_rtf
        self._persist_rtf(smoothed)
        logger.debug("Measured RTF %.1fx (smoothed %.1fx)", measured, smoothed)

    def _reset_idle_timer(self) -> None:
        if self._shutdown:
            # shutdown() cancelled the timer, but transcribe()'s finally block
            # re-armed it on the way out — leaving a timer alive on a shut-down
            # engine to fire minutes later and clear mlx's cache under whoever
            # owns the process next.
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self._idle_timeout, self._on_idle_expired)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle_expired(self) -> None:
        with self._lock:
            if self._inflight > 0:
                # A long transcription is still running — not actually idle.
                # Re-arm and check again later.
                self._reset_idle_timer()
                return
            self._warm = False
            self._idle_timer = None
            # Held under the lock so a transcription cannot start between the
            # _inflight check above and the cache clear below. See shutdown().
            _release_mlx_model_cache()
        logger.info(
            "WhisperEngine cooled down after %.0fs of inactivity — "
            "model memory released.",
            self._idle_timeout,
        )

    # ----------------------------------------------------------- RTF state
    def _load_persisted_rtf(self) -> float | None:
        if self._rtf_state_path is None or not self._rtf_state_path.exists():
            return None
        try:
            import json

            data = json.loads(self._rtf_state_path.read_text(encoding="utf-8"))
            value = data.get(self._model_name)
            return float(value) if value else None
        except Exception:
            return None

    def _persist_rtf(self, value: float) -> None:
        if self._rtf_state_path is None:
            return
        try:
            import json

            data: dict[str, Any] = {}
            if self._rtf_state_path.exists():
                loaded = json.loads(self._rtf_state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            data[self._model_name] = round(value, 2)
            self._rtf_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._rtf_state_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            logger.debug("Could not persist RTF state", exc_info=True)


# ---------------------------------------------------------------------------
# Progress estimator — runs in its own daemon thread
# ---------------------------------------------------------------------------
class _ProgressEstimator:
    """Approximate progress (0-99 %) by wall-clock vs estimated total time.

    Whisper's runtime depends on the model, audio length, and Mac model. For
    the small model on M1 Pro, real-time factor is roughly 8x — i.e. a 5-min
    audio file takes ~40 s. We estimate total = ``duration / RTF`` and tick
    every 0.5 s.
    """

    REAL_TIME_FACTOR = 8.0  # default until the engine has measured this machine

    def __init__(
        self,
        duration_seconds: float,
        callback: ProgressCallback,
        *,
        rtf: float | None = None,
    ) -> None:
        self._duration = max(1.0, duration_seconds)
        self._callback = callback
        self._rtf = float(rtf) if rtf else self.REAL_TIME_FACTOR
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        start = time.monotonic()
        estimated_total = self._duration / self._rtf
        while not self._stop.is_set():
            elapsed = time.monotonic() - start
            pct = min(99.0, 100.0 * elapsed / estimated_total)
            try:
                self._callback(pct)
            except Exception:  # pragma: no cover (defensive)
                logger.exception("progress callback raised")
            if self._stop.wait(0.5):
                break


def _infer_stream_label(path: Path) -> str | None:
    """``…_mic.wav`` → ``"mic"``, ``…_system.flac`` → ``"system"``, else None.

    The processor passes ``stream=`` explicitly, but retranscribe.py and the
    orphan-recovery path don't — and the filename already carries the answer.
    """
    stem = path.stem.lower()
    if stem.endswith("_mic"):
        return "mic"
    if stem.endswith("_system"):
        return "system"
    return None


def _release_mlx_model_cache() -> None:
    """Drop mlx-whisper's module-level model cache and flush Metal buffers.

    mlx-whisper keeps the last-used model in ``transcribe.ModelHolder`` —
    0.5-2 GB of unified memory depending on the model — until the process
    exits. Clearing the holder + ``mx.clear_cache()`` actually returns that
    memory to the OS. No-op when mlx isn't installed (tests, CI).

    Only call while no transcription is in flight: an active run keeps its
    own reference to the model (safe), but the next run would reload it.

    If mlx-whisper was never imported, no model was ever loaded and there is
    nothing to release. Importing it anyway (it pulls in scipy and numba) cost
    2.6 s on every shutdown of a session that transcribed nothing.
    """
    if not _mlx_whisper_loaded():
        return
    try:
        holder = _import_model_holder()
    except Exception:
        # A bare `except: return` used to hide this completely — if mlx-whisper
        # renamed ModelHolder, memory release silently became a no-op AND the
        # buffer flush below was skipped, leaking 1.4 GB per model with no
        # signal at all. Log it, then still flush what we can.
        logger.warning(
            "Could not reach mlx-whisper's ModelHolder — model memory will not "
            "be released. mlx-whisper internals may have changed.",
            exc_info=True,
        )
    else:
        holder.model = None
        holder.model_path = None

    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # pragma: no cover (older mlx / mlx absent)
        logger.debug("mx.clear_cache unavailable", exc_info=True)


def _mlx_whisper_loaded() -> bool:
    return "mlx_whisper.transcribe" in sys.modules


def _import_model_holder() -> Any:
    """Return mlx-whisper's ``ModelHolder`` class.

    NB: ``from mlx_whisper import transcribe`` yields the FUNCTION (the package
    ``__init__`` shadows the submodule); ``import_module`` gets the actual
    module that owns ``ModelHolder``. Split out so tests can simulate the
    upstream rename this used to swallow.
    """
    import importlib

    return importlib.import_module("mlx_whisper.transcribe").ModelHolder


# ---------------------------------------------------------------------------
# mlx-whisper invocation, isolated so tests don't need the framework
# ---------------------------------------------------------------------------
class _MlxImportFailure(RuntimeError):
    """Raised when mlx-whisper isn't installed."""


#: Transcribe long recordings in windows of this many seconds.
#:
#: mlx-whisper builds the mel spectrogram for the WHOLE file before decoding a
#: single token — ``log_mel_spectrogram`` runs one STFT across the entire array
#: — so MLX's allocator grows with meeting length. Measured on this Mac with
#: ``medium``, via ``mx.get_peak_memory()``:
#:
#:      5 min  2.31 GB      20 min  2.78 GB
#:     10 min  2.50 GB      45 min  3.53 GB
#:
#: about 31 MB per minute of audio over a ~2.2 GB floor, which puts a 97-minute
#: meeting near 5.2 GB of unified memory. None of it shows up in RSS (1.05 GB
#: for that 45-minute run) because MLX allocates through Metal, which is why
#: this went unnoticed. Windowing caps the audio-dependent part.
CHUNK_SECONDS = 600.0

#: Never advance less than this between windows. Only a pathological window
#: (Whisper returning almost nothing for ten minutes of audio) can hit it; it
#: exists so the loop always terminates.
MIN_ADVANCE_SECONDS = 30.0


def resolve_model_path(repo: str) -> str:
    """Return a local directory holding ``repo``, contacting the network only
    if the model is not on disk yet.

    mlx-whisper passes whatever it is given straight to ``snapshot_download``,
    which calls huggingface.co on *every* load even when all four files are
    already cached — measured 265.9 ms against 0.1 ms for the same resolution
    done locally. The engine drops the model after ``idle_timeout_seconds``,
    so a normal day of meetings repeats that request once per meeting.

    Two reasons to stop. The README promises "No telemetry. No analytics. No
    phone-home", and this was the one thing contradicting it — the timing of
    every meeting you transcribe, leaked to a third party. And a laptop with
    no connection should still be able to transcribe what it just recorded.

    Resolution is: cached copy, else download it once, else hand back the repo
    id and let mlx-whisper raise the error that actually explains the problem.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:  # pragma: no cover (hub always ships with mlx-whisper)
        return repo

    try:
        return str(snapshot_download(repo_id=repo, local_files_only=True))
    except Exception:
        logger.info("Model %s is not cached yet; downloading once.", repo)

    try:
        return str(snapshot_download(repo_id=repo))
    except Exception:
        logger.warning(
            "Could not resolve %s locally or fetch it; letting mlx-whisper try.",
            repo, exc_info=True,
        )
        return repo


def _default_mlx_transcribe(audio: str, **kwargs: Any) -> dict[str, Any]:
    """Hand audio to mlx-whisper as a pre-decoded numpy array.

    Why we don't pass the file path: mlx-whisper's ``load_audio`` shells out
    to ``ffmpeg`` for *any* input — even plain 16 kHz mono 16-bit WAV which
    we already produce. Decoding ourselves avoids the ffmpeg system
    dependency and one extra subprocess per call.
    """
    try:
        import mlx_whisper
    except Exception as exc:  # pragma: no cover (no mlx-whisper in tests)
        raise _MlxImportFailure(str(exc)) from exc

    samples = decode_to_float32_mono16k(audio)
    if len(samples) <= int(CHUNK_SECONDS * TARGET_SAMPLE_RATE):
        return mlx_whisper.transcribe(samples, **kwargs)
    return _transcribe_windowed(mlx_whisper, samples, **kwargs)


def _transcribe_windowed(
    mlx_whisper: Any, samples: Any, **kwargs: Any,
) -> dict[str, Any]:
    """Transcribe a long recording window by window, stitched back together.

    Each window resumes exactly where the previous window's last *complete*
    segment ended, so there is no gap and nothing is transcribed twice.

    The subtlety is the final segment of a window: its audio was cut off by
    the window edge, so it is the one Whisper had least context for. It is
    dropped and re-read as part of the next window, which sees the whole
    sentence. Advancing by a fixed step with a fixed overlap instead — the
    obvious design — lost about 15 words at every seam, measured as 91.6%
    word agreement against a single pass on a real 45-minute meeting, where
    Whisper agrees with *itself* 100.00%.

    Language is decided once. mlx-whisper detects it from the first 30 s of
    whatever it is handed, so left on auto every window votes again and a
    meeting could change language halfway through.
    """
    window = int(CHUNK_SECONDS * TARGET_SAMPLE_RATE)
    floor = int(MIN_ADVANCE_SECONDS * TARGET_SAMPLE_RATE)
    total = len(samples)
    language = kwargs.get("language")
    segments: list[dict[str, Any]] = []
    pos = 0
    windows = 0

    while pos < total:
        chunk = samples[pos:pos + window]
        offset = pos / TARGET_SAMPLE_RATE
        raw = mlx_whisper.transcribe(chunk, **{**kwargs, "language": language})
        windows += 1
        if language is None:
            language = raw.get("language") or None

        kept = list(raw.get("segments") or [])
        is_last = pos + window >= total
        if not is_last and len(kept) > 1:
            kept = kept[:-1]  # cut by the window edge; the next window owns it

        for seg in kept:
            shifted = dict(seg)
            shifted["start"] = float(seg.get("start", 0.0)) + offset
            shifted["end"] = float(seg.get("end", 0.0)) + offset
            segments.append(shifted)

        _flush_mlx_buffers()
        if is_last:
            break
        # Resume where the kept text ended. A window that recognised nothing
        # advances by its whole length — there is nothing there to re-read.
        if kept:
            advance = int(float(kept[-1].get("end", 0.0)) * TARGET_SAMPLE_RATE)
        else:
            advance = window
        pos += max(advance, floor)

    logger.info(
        "Transcribed %.1f min in %d windows of %.0f s.",
        total / TARGET_SAMPLE_RATE / 60.0, windows, CHUNK_SECONDS,
    )
    return {
        "text": " ".join(str(s.get("text", "")).strip() for s in segments).strip(),
        "segments": segments,
        "language": language,
        "duration": total / TARGET_SAMPLE_RATE,
    }


def _flush_mlx_buffers() -> None:
    """Return MLX's cached buffers to the OS, keeping the loaded model.

    Distinct from :func:`_release_mlx_model_cache`, which also drops the model
    and so forces a multi-second reload on the next run.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # pragma: no cover (older mlx / mlx absent)
        logger.debug("mx.clear_cache unavailable", exc_info=True)


#: Formats we can decode without ffmpeg, via macOS's own AudioToolbox.
#: `which ffmpeg` returns nothing on a stock Mac — and mlx-whisper's loader
#: shells out to it for *any* input — so this is what makes non-WAV audio
#: usable at all. afconvert ships with macOS; no new dependency.
DECODABLE_SUFFIXES = (".wav", ".flac", ".caf", ".m4a", ".aiff", ".aif", ".mp3")

TARGET_SAMPLE_RATE = 16000


def decode_to_float32_mono16k(path: str | Path) -> Any:
    """Decode any supported audio file to 1-D float32 mono at 16 kHz.

    Whisper wants exactly that shape. Our recorder writes it directly, so a
    conforming WAV is one ``frombuffer`` and an in-place scale. Anything else —
    FLAC, or a WAV at some other rate or width — is transcoded through
    ``afconvert``, which is part of macOS.

    This ordering matters for the FLAC migration: the decoder has to exist and
    work before anything starts *writing* FLAC, or retranscribe and orphan
    recovery break silently on a machine with no ffmpeg.
    """
    path = Path(path)
    arr = _read_conforming_wav(path)
    if arr is not None:
        return arr
    arr = _afconvert_decode(path)
    if arr is not None:
        return arr
    # Last resort: mlx-whisper's own loader. Needs ffmpeg, which is usually
    # absent — but if the user has it, better to use it than to fail.
    logger.info("Falling back to mlx-whisper's loader for %s", path.name)
    import mlx_whisper.audio

    return mlx_whisper.audio.load_audio(str(path))


def _read_conforming_wav(path: Path) -> Any | None:
    """Fast path: a WAV already at 16 kHz / mono / 16-bit. ``None`` otherwise."""
    import numpy as np

    try:
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getframerate() != TARGET_SAMPLE_RATE
                or wf.getnchannels() != 1
                or wf.getsampwidth() != 2
            ):
                return None
            raw = wf.readframes(wf.getnframes())
    except Exception:
        return None
    # astype() then /= in place. Writing `.astype(...) / 32768.0` allocates a
    # second full copy of the array — 372 MB for a 97-minute recording — for a
    # scale by a constant.
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def _afconvert_decode(path: Path) -> Any | None:
    """Transcode via macOS ``afconvert`` into a temp WAV, then read it."""
    import shutil
    import subprocess
    import tempfile

    tool = shutil.which("afconvert")
    if tool is None:  # pragma: no cover (afconvert ships with macOS)
        return None

    with tempfile.TemporaryDirectory(prefix="otis-decode-") as tmp:
        out = Path(tmp) / "decoded.wav"
        try:
            subprocess.run(
                [tool, "-f", "WAVE", "-d", f"LEI16@{TARGET_SAMPLE_RATE}",
                 "-c", "1", str(path), str(out)],
                check=True, capture_output=True, timeout=DECODE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("afconvert could not decode %s: %s", path.name, exc)
            return None
        return _read_conforming_wav(out)


def _safe_wav_duration_seconds(path: Path) -> float:
    """Duration in seconds, for any format we can read. 0.0 on failure.

    A WAV header answers this for free. For FLAC and friends we ask ``afinfo``
    rather than decoding the whole file, so a 2-hour recording still costs
    milliseconds.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / (wf.getframerate() or 1)
    except Exception:
        pass
    return _afinfo_duration_seconds(path)


def _afinfo_duration_seconds(path: Path) -> float:
    """Duration via macOS ``afinfo``. Returns 0.0 when it can't be determined."""
    import re as _re
    import shutil
    import subprocess

    tool = shutil.which("afinfo")
    if tool is None:  # pragma: no cover (afinfo ships with macOS)
        return 0.0
    try:
        out = subprocess.run(
            [tool, str(path)], check=True, capture_output=True,
            text=True, timeout=30.0,
        ).stdout
    except Exception:  # pragma: no cover (corrupt / missing file)
        return 0.0
    match = _re.search(r"estimated duration:\s*([0-9.]+)", out)
    return float(match.group(1)) if match else 0.0


def _rms_from_decoded(path: Path, *, window_seconds: float = 5.0) -> float | None:
    """Peak-window RMS for a format the ``wave`` module can't stream."""
    try:
        import numpy as np

        samples = decode_to_float32_mono16k(path)
        if samples is None or len(samples) == 0:
            return None
        win = max(1, int(window_seconds * TARGET_SAMPLE_RATE))
        peak = 0.0
        for start in range(0, len(samples), win):
            chunk = samples[start:start + win]
            if chunk.size:
                peak = max(peak, float(np.sqrt(np.mean(chunk * chunk))))
        return peak
    except Exception:
        return None


def _wav_rms(path: Path, *, window_seconds: float = 5.0) -> float | None:
    """Return the **peak** RMS across non-overlapping windows of the WAV file.

    Earlier versions only sampled the first 5 seconds. That was enough for
    "completely silent" detection but mis-fired on real recordings where the
    audio stream had a quiet head — e.g. the user hit Start before YouTube
    started playing, so the first 5 s of system audio is dead but the rest
    is normal speech. We now scan the whole file in chunks, keep the loudest
    window, and decide based on that. O(1) memory, O(file_size) time.

    Returns ``None`` on any failure so the caller falls through to normal
    transcription instead of skipping a recording we couldn't measure.
    """
    if path.suffix.lower() != ".wav":
        # Not streamable with the `wave` module. Decoding is affordable because
        # this only runs once per stream and the result is the same array the
        # transcription would have produced anyway.
        return _rms_from_decoded(path)
    try:
        with wave.open(str(path), "rb") as wf:
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate() or 16000
            if sampwidth != 2:
                return None  # only int16 expected from our recorder
            chunk = max(1, int(window_seconds * rate))

            import numpy as np

            best = 0.0
            while True:
                raw = wf.readframes(chunk)
                if not raw:
                    break
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if arr.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(arr * arr)))
                if rms > best:
                    best = rms
            return best
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hallucination filter — runs over every transcribe() result
# ---------------------------------------------------------------------------
# Phrases Whisper memorised from YouTube subtitle dumps and regurgitates on
# silent / low-info audio. Lower-cased before comparison.
_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "merci d'avoir regardé",
    "merci d'avoir regardé cette vidéo",
    "abonnez-vous",
    "rendez-vous sur patreon",
    "le lien est dans la description",
    "sous-titres réalisés par la communauté d'amara",
    "sous-titrage st 501",
    "thanks for watching",
    "subscribe to my channel",
    "see you in the next video",
    "buise sexuellement",  # disturbing common Whisper hallucination
)

# Max consecutive identical-text segments to keep before we treat the rest
# as a hallucination loop.
_MAX_CONSECUTIVE_DUPLICATES = 2


def filter_hallucinations(segments: list[Segment]) -> list[Segment]:
    """Drop common Whisper hallucination patterns.

    Three rules:
    1. Drop segments whose normalised text matches a known training-leak phrase.
    2. Collapse runs of >2 identical-text consecutive segments (the classic
       "Joyeux Joyeux Joyeux..." failure mode).
    3. Drop very short segments (<0.2 s) that contain no real word.
    """
    if not segments:
        return segments
    out: list[Segment] = []
    last_text: str | None = None
    duplicate_run = 0

    for seg in segments:
        text_lower = seg.text.lower().strip()

        if any(phrase in text_lower for phrase in _HALLUCINATION_PHRASES):
            logger.debug("Dropping hallucinated segment: %r", seg.text)
            continue

        if (seg.end - seg.start) < 0.2 and not _has_real_word(text_lower):
            continue

        if text_lower == last_text:
            duplicate_run += 1
            if duplicate_run >= _MAX_CONSECUTIVE_DUPLICATES:
                continue
        else:
            duplicate_run = 0
            last_text = text_lower
        out.append(seg)

    return out


def _has_real_word(text: str) -> bool:
    """True if there's at least one alphabetic word ≥3 chars long."""
    import re as _re

    for token in _re.findall(r"[a-zà-ÿ]+", text, flags=_re.IGNORECASE):
        if len(token) >= 3:
            return True
    return False


__all__ = [
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
    "ModelDownloadError",
    "OutOfMemoryError",
    "Segment",
    "TranscriptionResult",
    "WhisperEngine",
    "WhisperError",
]
