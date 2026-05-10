"""Tests for src/transcription/whisper_engine.py.

We never actually call mlx-whisper — every test injects a fake
``transcribe_fn`` so the engine's wiring (lazy state, language pass-through,
result parsing, error mapping) is exercised deterministically.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from src.transcription.whisper_engine import (
    ModelDownloadError,
    OutOfMemoryError,
    Segment,
    TranscriptionResult,
    WhisperEngine,
    filter_hallucinations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _write_audible_wav(path: Path, *, seconds: float = 0.5, rate: int = 16000) -> None:
    """A 440 Hz sine at ~50 % amplitude — non-silent, so the RMS pre-check
    in :class:`WhisperEngine` doesn't short-circuit the test."""
    import math
    import struct

    n = int(seconds * rate)
    amp = 16000  # ~half of int16 range
    frames = bytearray()
    for i in range(n):
        sample = int(amp * math.sin(2 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", sample)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    p = tmp_path / "sample.wav"
    _write_audible_wav(p, seconds=0.5)
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_transcribe_returns_segments(sample_wav: Path) -> None:
    fake_raw = {
        "text": "Hello there",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello "},
            {"start": 1.0, "end": 2.0, "text": " there "},
        ],
    }
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: fake_raw)
    result = engine.transcribe(sample_wav)

    assert isinstance(result, TranscriptionResult)
    assert [s.text for s in result.segments] == ["Hello", "there"]
    assert result.detected_language == "en"


def test_language_override_passed_through(sample_wav: Path) -> None:
    seen: dict[str, Any] = {}

    def fake(audio: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"text": "bonjour", "language": "fr", "segments": []}

    engine = WhisperEngine(transcribe_fn=fake)
    engine.transcribe(sample_wav, language="fr")
    assert seen["language"] == "fr"


def test_progress_callback_fires_at_least_once(sample_wav: Path) -> None:
    fake_raw = {"text": "hi", "segments": [], "language": "en"}
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: fake_raw)

    seen: list[float] = []
    engine.transcribe(sample_wav, on_progress=seen.append)
    # The estimator may or may not emit pre-100 ticks (transcribe is fast in
    # tests), but it MUST emit the final 100 once we're done.
    assert seen
    assert seen[-1] == 100.0


def test_warm_state_after_first_call(sample_wav: Path) -> None:
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: {"text": "", "segments": []})
    assert engine.is_warm is False
    engine.transcribe(sample_wav)
    assert engine.is_warm is True


def test_idle_timer_cools_engine_down(sample_wav: Path) -> None:
    """Use a 0.05 s idle timeout to avoid sleeping in tests."""
    engine = WhisperEngine(
        idle_timeout_seconds=0.05,
        transcribe_fn=lambda *a, **kw: {"text": "", "segments": []},
    )
    engine.transcribe(sample_wav)
    assert engine.is_warm is True
    # Wait a hair longer than the timer.
    import time
    time.sleep(0.2)
    assert engine.is_warm is False


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
def test_oom_error_is_translated(sample_wav: Path) -> None:
    def boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("mlx_metal_oom: not enough memory for buffer")

    engine = WhisperEngine(transcribe_fn=boom)
    with pytest.raises(OutOfMemoryError) as exc:
        engine.transcribe(sample_wav)
    assert "smaller model" in str(exc.value).lower()


def test_download_error_is_translated(sample_wav: Path) -> None:
    def boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise ConnectionError("Failed to resolve huggingface.co")

    engine = WhisperEngine(transcribe_fn=boom)
    with pytest.raises(ModelDownloadError):
        engine.transcribe(sample_wav)


def test_missing_audio_returns_empty_result(tmp_path: Path) -> None:
    """Don't crash on missing files — return empty so the daemon survives."""
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: {"text": "x", "segments": []})
    result = engine.transcribe(tmp_path / "doesnotexist.wav")
    assert result.segments == []


def test_empty_audio_returns_empty_result(tmp_path: Path) -> None:
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: {"text": "x", "segments": []})
    result = engine.transcribe(p)
    assert result.segments == []


def test_silent_wav_short_circuits_before_calling_whisper(tmp_path: Path) -> None:
    """A silent WAV must skip Whisper entirely — otherwise hallucinations."""
    p = tmp_path / "silent.wav"
    rate = 16000
    silent_frames = bytes(rate * 2)  # 1 second of zero samples
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(silent_frames)

    called = {"hit": False}

    def fake(*_a, **_kw):
        called["hit"] = True
        return {"text": "Joyeux Joyeux Joyeux", "segments": []}

    engine = WhisperEngine(transcribe_fn=fake)
    result = engine.transcribe(p)
    assert called["hit"] is False, "WhisperEngine should not call Whisper on silent audio"
    assert result.segments == []


def test_quiet_but_audible_wav_is_NOT_skipped(tmp_path: Path) -> None:
    """RMS ~0.003 (real-world quiet speech) must still be transcribed.

    The first iteration of the silence pre-check used a 0.005 threshold and
    incorrectly skipped real recordings whose mic was just a bit far / quiet.
    """
    import math
    import struct

    p = tmp_path / "quiet.wav"
    rate = 16000
    n = rate  # 1 second
    target_rms = 0.003
    amp = int(target_rms * 32768.0 * math.sqrt(2))  # peak for sine RMS
    frames = bytearray()
    for i in range(n):
        sample = int(amp * math.sin(2 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", sample)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))

    called = {"hit": False}

    def fake(*_a, **_kw):
        called["hit"] = True
        return {"text": "hello", "segments": [{"start": 0, "end": 1, "text": "hello"}]}

    engine = WhisperEngine(transcribe_fn=fake)
    engine.transcribe(p)
    assert called["hit"] is True, "0.003 RMS is real audio; pre-check must NOT skip it"


# ---------------------------------------------------------------------------
# Unknown model rejection
# ---------------------------------------------------------------------------
def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        WhisperEngine(model_name="nonexistent-v9000")


# ---------------------------------------------------------------------------
# Result parsing details
# ---------------------------------------------------------------------------
def test_result_text_property_concatenates_segments() -> None:
    r = TranscriptionResult(
        segments=[
            Segment(0.0, 1.0, "hello"),
            Segment(1.0, 2.0, "world"),
        ],
        detected_language="en",
    )
    assert r.text == "hello world"


def test_segment_from_raw_strips_text() -> None:
    s = Segment.from_raw({"start": 0.5, "end": 1.5, "text": "  spaced  "})
    assert s.text == "spaced"
    assert s.start == 0.5
    assert s.end == 1.5


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------
def test_filter_drops_known_youtube_intro_phrases() -> None:
    segs = [
        Segment(0.0, 2.0, "Hello there"),
        Segment(2.0, 5.0, "Merci d'avoir regardé cette vidéo"),
        Segment(5.0, 8.0, "Rendez-vous sur Patreon, le lien est dans la description"),
        Segment(8.0, 10.0, "Subscribe to my channel"),
        Segment(10.0, 12.0, "Real content here"),
    ]
    out = filter_hallucinations(segs)
    assert [s.text for s in out] == ["Hello there", "Real content here"]


def test_filter_collapses_runaway_duplicate_runs() -> None:
    """The classic 'Joyeux Joyeux Joyeux...' hallucination must collapse."""
    segs = [Segment(i * 0.5, (i + 1) * 0.5, "Joyeux") for i in range(10)]
    out = filter_hallucinations(segs)
    # 2 duplicates allowed (so the user knows it was repeated), then truncated.
    assert len(out) == 2


def test_filter_keeps_normal_repeated_words_separately() -> None:
    """Real conversation can have the same short answer twice — keep them."""
    segs = [
        Segment(0.0, 2.0, "Yes."),
        Segment(5.0, 7.0, "Different question?"),
        Segment(8.0, 10.0, "Yes."),
    ]
    out = filter_hallucinations(segs)
    assert len(out) == 3


def test_filter_drops_subsecond_noise_segments() -> None:
    """Segments shorter than 0.2 s and without a real word are noise."""
    segs = [
        Segment(0.0, 0.05, "."),
        Segment(0.1, 0.15, "uh"),  # <0.2 s, no real word
        Segment(0.5, 1.5, "good morning everyone"),
    ]
    out = filter_hallucinations(segs)
    assert [s.text for s in out] == ["good morning everyone"]
