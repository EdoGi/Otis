"""Wrapper around ``mlx-whisper`` for local-only transcription.

Why mlx-whisper?
    The MLX framework runs natively on Apple Silicon (Metal-backed). On an M1
    Pro a 1-hour meeting transcribes with the ``small`` model in ~3–8 minutes
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
# 0.01–0.05 RMS; ambient room noise / muted streams are 0.0001–0.001. We pick
# 0.001 (~−60 dBFS) — comfortably below quiet speech, comfortably above the
# noise floor of a healthy capture chain. The other anti-hallucination defences
# (``condition_on_previous_text=False`` + the post-filter) are what catch the
# really pathological cases.
SILENCE_RMS_THRESHOLD = 0.001


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
    def from_raw(cls, raw: dict[str, Any]) -> "Segment":
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
    ) -> None:
        if model_name not in _MODEL_REPO_MAP:
            raise ValueError(
                f"Unknown model {model_name!r}; choose from {sorted(_MODEL_REPO_MAP)}"
            )
        self._model_name = model_name
        self._model_repo = _MODEL_REPO_MAP[model_name]
        self._idle_timeout = float(idle_timeout_seconds)
        self._transcribe_fn = transcribe_fn or _default_mlx_transcribe

        self._lock = threading.RLock()
        # mlx-whisper caches the loaded model in its own module state
        # (ModelHolder) — 0.5–2 GB of unified memory depending on the model.
        # We track warmth + an idle timer, and on expiry we actually CLEAR
        # that cache so the memory goes back to the OS instead of being held
        # until the app quits.
        self._warm = False
        self._idle_timer: threading.Timer | None = None
        self._inflight = 0  # transcriptions currently running
        # Measured real-time factor (audio seconds per wall-clock second),
        # EMA-updated after each successful run so progress estimates adapt
        # to this machine + model instead of assuming the M1-Pro/small 8×.
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
        if rms is not None and rms < SILENCE_RMS_THRESHOLD:
            logger.info(
                "Transcribe skipped — %s is essentially silent "
                "(peak-window RMS=%.4f, threshold=%.4f).",
                path.name, rms, SILENCE_RMS_THRESHOLD,
            )
            return TranscriptionResult(
                detected_language=language, duration_seconds=_safe_wav_duration_seconds(path),
            )

        duration = _safe_wav_duration_seconds(path)
        progress = (
            _ProgressEstimator(duration, on_progress, rtf=self._current_rtf())
            if on_progress
            else None
        )
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
        with self._lock:
            self._warm = False
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            release = self._inflight == 0
        if release:
            _release_mlx_model_cache()

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------
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
            path_or_hf_repo=self._model_repo,
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
    the small model on M1 Pro, real-time factor is roughly 8× — i.e. a 5-min
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


def _release_mlx_model_cache() -> None:
    """Drop mlx-whisper's module-level model cache and flush Metal buffers.

    mlx-whisper keeps the last-used model in ``transcribe.ModelHolder`` —
    0.5–2 GB of unified memory depending on the model — until the process
    exits. Clearing the holder + ``mx.clear_cache()`` actually returns that
    memory to the OS. No-op when mlx isn't installed (tests, CI).

    Only call while no transcription is in flight: an active run keeps its
    own reference to the model (safe), but the next run would reload it.
    """
    try:
        # NB: ``from mlx_whisper import transcribe`` yields the FUNCTION
        # (the package __init__ shadows the submodule); import_module gets
        # the actual module that owns ModelHolder.
        import importlib

        mlx_transcribe_mod = importlib.import_module("mlx_whisper.transcribe")
        mlx_transcribe_mod.ModelHolder.model = None
        mlx_transcribe_mod.ModelHolder.model_path = None
    except Exception:
        return
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # pragma: no cover (older mlx)
        logger.debug("mx.clear_cache unavailable", exc_info=True)


# ---------------------------------------------------------------------------
# mlx-whisper invocation, isolated so tests don't need the framework
# ---------------------------------------------------------------------------
class _MlxImportFailure(RuntimeError):
    """Raised when mlx-whisper isn't installed."""


def _default_mlx_transcribe(audio: str, **kwargs: Any) -> dict[str, Any]:
    """Hand audio to mlx-whisper as a pre-decoded numpy array.

    Why we don't pass the file path: mlx-whisper's ``load_audio`` shells out
    to ``ffmpeg`` for *any* input — even plain 16 kHz mono 16-bit WAV which
    we already produce. Decoding ourselves avoids the ffmpeg system
    dependency and one extra subprocess per call.
    """
    try:
        import mlx_whisper  # type: ignore[import-not-found]
        import numpy as np  # noqa: WPS433
    except Exception as exc:  # pragma: no cover (no mlx-whisper in tests)
        raise _MlxImportFailure(str(exc)) from exc

    samples = _wav_to_float32_mono16k(audio)
    return mlx_whisper.transcribe(samples, **kwargs)  # type: ignore[no-any-return]


def _wav_to_float32_mono16k(path: str) -> "Any":
    """Decode a WAV file to a 1-D float32 numpy array at 16 kHz mono.

    Whisper expects exactly that shape: 16 kHz, mono, normalised float in
    ``[-1, 1]``. Our recorder always writes that format so the work here
    is one frombuffer + a divide. If the file is at a different rate or
    width we still bail to mlx-whisper's path-based ffmpeg loader as a
    safety net (raises the same actionable error if ffmpeg is missing).
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if rate == 16000 and channels == 1 and sampwidth == 2:
            raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Format we don't natively support — let mlx-whisper deal (will need ffmpeg).
    logger.info(
        "WAV at %s is %dHz/%dch/%d-bit; falling back to mlx-whisper's loader.",
        path, rate, channels, sampwidth * 8,
    )
    import mlx_whisper.audio  # type: ignore[import-not-found]

    return mlx_whisper.audio.load_audio(str(path))


def _safe_wav_duration_seconds(path: Path) -> float:
    """Read a WAV header to compute duration. Returns 0 on failure."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / rate
    except Exception:  # pragma: no cover (corrupt files)
        return 0.0


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
