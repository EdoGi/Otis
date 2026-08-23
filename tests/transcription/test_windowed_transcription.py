"""Cover for audit finding C3 — peak memory grew with meeting length.

mlx-whisper builds the mel spectrogram for the *whole* file before decoding a
single token, so MLX's allocator scales with duration. Measured on this Mac
with ``medium`` via ``mx.get_peak_memory()``: 2.31 GB at 5 min, 2.50 GB at
10 min, 2.78 GB at 20 min, 3.37 GB at 45 min — roughly 31 MB per minute over a
2.2 GB floor, which puts a 97-minute meeting near 5.2 GB. On an 8 GB machine
that is the difference between running and swapping.

None of it appears in RSS (1.05 GB for the 45-minute run) because MLX
allocates through Metal, which is why the audit could only estimate it.

The stitching is what these tests are really about. Whisper is deterministic —
two single passes over the same 45-minute file agreed 100.00%, segment for
segment — so *any* disagreement a windowed run introduces is ours, not noise.
A fixed step with a fixed overlap scored 91.6%, losing about 15 words per
seam. Advancing to the end of the last complete segment is what fixes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.transcription import whisper_engine as we
from src.transcription.whisper_engine import (
    CHUNK_SECONDS,
    MIN_ADVANCE_SECONDS,
    TARGET_SAMPLE_RATE,
    _transcribe_windowed,
)

SR = TARGET_SAMPLE_RATE


class FakeMlx:
    """Records each window it was handed and replies with scripted segments."""

    def __init__(self, reply=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply = reply or (lambda i, n: {
            "segments": [
                {"start": 10.0, "end": 20.0, "text": f"body {i}"},
                {"start": CHUNK_SECONDS - 20.0, "end": CHUNK_SECONDS - 5.0,
                 "text": f"tail {i}"},
            ],
            "language": "en",
        })

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), "kwargs": kwargs})
        return self._reply(len(self.calls) - 1, len(audio))


def audio(seconds: float):
    return np.zeros(int(seconds * SR), dtype=np.float32)


@pytest.fixture(autouse=True)
def _flush_calls(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(we, "_flush_mlx_buffers", lambda: calls.append(1))
    return calls


# ---------------------------------------------------------------------------
# When windowing kicks in
# ---------------------------------------------------------------------------
def test_short_audio_is_one_pass(monkeypatch) -> None:
    """mlx's own result dict is handed back untouched.

    The windowed path rebuilds the dict from the fields it knows about, so
    routing a short recording through it would quietly drop anything else
    mlx-whisper returned.
    """
    reply = {"text": "hello", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
             "language": "en", "some_future_field": 42}
    fake = FakeMlx(reply=lambda i, n: reply)
    monkeypatch.setattr(we, "decode_to_float32_mono16k", lambda _p: audio(60))
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", fake)

    out = we._default_mlx_transcribe("x.wav", language="en")

    assert len(fake.calls) == 1
    assert out is reply, "a short recording must not be reshaped"


def test_a_long_recording_is_split(monkeypatch) -> None:
    fake = FakeMlx()
    monkeypatch.setattr(we, "decode_to_float32_mono16k",
                        lambda _p: audio(CHUNK_SECONDS * 3))
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", fake)
    we._default_mlx_transcribe("x.wav", language="en")
    assert len(fake.calls) > 1


# ---------------------------------------------------------------------------
# Stitching: no gap, no duplicate, no truncated segment
# ---------------------------------------------------------------------------
def test_each_window_resumes_where_the_last_kept_segment_ended() -> None:
    fake = FakeMlx()
    _transcribe_windowed(fake, audio(CHUNK_SECONDS * 2.5), language="en")
    # First window keeps "body" (ends 20 s) and drops "tail"; so the second
    # window must start at 20 s, not at CHUNK_SECONDS.
    assert fake.calls[1]["samples"] == int(min(CHUNK_SECONDS,
                                               CHUNK_SECONDS * 2.5 - 20.0) * SR)


def test_the_segment_cut_by_the_window_edge_is_reread_not_kept() -> None:
    """It is the segment Whisper had least context for."""
    out = _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 2), language="en")
    texts = [s["text"] for s in out["segments"]]
    assert "tail 0" not in texts, "a window-edge segment must not be trusted"
    assert "body 0" in texts


def test_the_final_window_keeps_everything() -> None:
    """Nothing follows it, so its last segment is not cut by anything."""
    out = _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 2), language="en")
    assert any(t.startswith("tail") for t in [s["text"] for s in out["segments"]])


def test_no_text_is_emitted_twice() -> None:
    out = _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 3), language="en")
    texts = [s["text"] for s in out["segments"]]
    assert len(texts) == len(set(texts))


def test_timestamps_are_offset_and_monotonic() -> None:
    out = _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 3), language="en")
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts)
    assert starts[0] == 10.0
    assert starts[-1] > CHUNK_SECONDS, "later windows must be shifted forward"


def test_a_single_segment_window_still_advances() -> None:
    """Dropping the only segment would emit nothing and never move on.

    It also re-reads from where that segment ended rather than from the window
    edge — Whisper returned nothing for the rest of the window, so that audio
    has not been transcribed yet and skipping it would lose it.
    """
    fake = FakeMlx(reply=lambda i, n: {
        "segments": [{"start": 0.0, "end": 300.0, "text": f"only {i}"}],
        "language": "en"})
    out = _transcribe_windowed(fake, audio(CHUNK_SECONDS * 2), language="en")

    assert [s["text"] for s in out["segments"]] == ["only 0", "only 1", "only 2"]
    starts = [s["start"] for s in out["segments"]]
    assert starts == [0.0, 300.0, 600.0], "each window resumes at the last end"


def test_a_window_that_recognises_nothing_skips_its_whole_length() -> None:
    calls: list[int] = []

    def reply(i, n):
        calls.append(n)
        if i == 0:
            return {"segments": [], "language": "en"}
        return {"segments": [{"start": 1.0, "end": 2.0, "text": "speech"}],
                "language": "en"}

    out = _transcribe_windowed(FakeMlx(reply), audio(CHUNK_SECONDS * 2),
                               language="en")
    assert [s["text"] for s in out["segments"]] == ["speech"]
    assert out["segments"][0]["start"] == pytest.approx(CHUNK_SECONDS + 1.0)


def test_a_window_that_barely_advances_is_floored() -> None:
    """Whisper returning half a second of text for ten minutes of audio.

    Advancing by that half second would re-read the same window thousands of
    times. Guarding only against zero is not enough — the floor is what bounds
    the work.
    """
    fake = FakeMlx(reply=lambda i, n: {
        "segments": [{"start": 0.0, "end": 0.5, "text": f"x{i}"}],
        "language": "en"})
    total = CHUNK_SECONDS * 3

    out = _transcribe_windowed(fake, audio(total), language="en")

    ceiling = int(total / MIN_ADVANCE_SECONDS) + 2
    assert len(fake.calls) <= ceiling, f"{len(fake.calls)} windows for {total}s"
    assert out["duration"] == pytest.approx(total)


def test_a_window_whose_text_ends_at_zero_still_moves_on() -> None:
    fake = FakeMlx(reply=lambda i, n: {
        "segments": [{"start": 0.0, "end": 0.0, "text": f"x{i}"}],
        "language": "en"})
    out = _transcribe_windowed(fake, audio(MIN_ADVANCE_SECONDS * 4), language="en")
    assert len(fake.calls) <= 5
    assert out["duration"] == pytest.approx(MIN_ADVANCE_SECONDS * 4)


def test_the_result_shape_matches_a_single_pass() -> None:
    out = _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 2), language="fr")
    assert set(out) == {"text", "segments", "language", "duration"}
    assert out["language"] == "fr"
    assert out["duration"] == pytest.approx(CHUNK_SECONDS * 2)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
def test_auto_detected_language_is_pinned_after_the_first_window() -> None:
    """Left on auto, every window votes again and a meeting can switch language."""
    langs = iter(["en", "fr", "de", "es", "it"])
    fake = FakeMlx(reply=lambda i, n: {
        "segments": [{"start": 0.0, "end": 300.0, "text": f"w{i}"}],
        "language": next(langs)})

    out = _transcribe_windowed(fake, audio(CHUNK_SECONDS * 3))

    assert out["language"] == "en"
    passed = [c["kwargs"]["language"] for c in fake.calls]
    assert passed[0] is None
    assert set(passed[1:]) == {"en"}


def test_an_explicit_language_is_never_overridden() -> None:
    fake = FakeMlx(reply=lambda i, n: {
        "segments": [{"start": 0.0, "end": 300.0, "text": "x"}], "language": "zh"})
    out = _transcribe_windowed(fake, audio(CHUNK_SECONDS * 2), language="fr")
    assert out["language"] == "fr"
    assert {c["kwargs"]["language"] for c in fake.calls} == {"fr"}


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------
def test_buffers_are_flushed_between_windows(_flush_calls) -> None:
    _transcribe_windowed(FakeMlx(), audio(CHUNK_SECONDS * 3), language="en")
    assert len(_flush_calls) >= 2, "windowing without flushing caps nothing"


# ---------------------------------------------------------------------------
# The model's own residency is the other half of the memory story
# ---------------------------------------------------------------------------
def test_idle_timeout_is_configurable(tmp_path) -> None:
    """`medium` is 1.5 GB resident; on a small machine that is worth tuning.

    The engine had the parameter since P0 and nothing ever passed it, so the
    300 s default was the only value reachable.
    """
    from src.config import Config
    from src.pipeline import build_pipeline

    cfg = Config({
        "transcription": {"model": "small", "idle_timeout_seconds": 45},
        "storage": {"audio_dir": str(tmp_path / "a"),
                    "transcript_dir": str(tmp_path / "t")},
    })
    pipeline = build_pipeline(cfg)
    try:
        assert pipeline.processor._engine._idle_timeout == 45.0
    finally:
        pipeline.shutdown()


def test_idle_timeout_defaults_when_unset(tmp_path) -> None:
    from src.config import Config
    from src.pipeline import build_pipeline
    from src.transcription.whisper_engine import DEFAULT_IDLE_TIMEOUT_SECONDS

    cfg = Config({
        "transcription": {"model": "small"},
        "storage": {"audio_dir": str(tmp_path / "a"),
                    "transcript_dir": str(tmp_path / "t")},
    })
    pipeline = build_pipeline(cfg)
    try:
        assert pipeline.processor._engine._idle_timeout == DEFAULT_IDLE_TIMEOUT_SECONDS
    finally:
        pipeline.shutdown()
