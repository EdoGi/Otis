"""P3 item 23 — decode FLAC without ffmpeg.

This has to land and be proven BEFORE anything writes FLAC. `which ffmpeg`
returns nothing on a stock Mac, and mlx-whisper's loader shells out to it for
any input — so encoding first would silently break retranscribe.py and orphan
recovery, with the only symptom being transcripts that never appear.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from src.transcription.whisper_engine import (
    TARGET_SAMPLE_RATE,
    WhisperEngine,
    _safe_wav_duration_seconds,
    _wav_rms,
    decode_to_float32_mono16k,
)

pytestmark = pytest.mark.skipif(
    shutil.which("afconvert") is None, reason="afconvert is macOS-only"
)


def _tone_wav(path: Path, *, seconds: float = 2.0, amp: int = 12000,
              rate: int = TARGET_SAMPLE_RATE, channels: int = 1,
              width: int = 2) -> None:
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * 440 * i / rate))
        for _ in range(channels):
            frames += struct.pack("<h", v) if width == 2 else struct.pack("<i", v << 16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        w.writeframes(bytes(frames))


def _to_flac(src: Path, dst: Path) -> None:
    subprocess.run(["afconvert", "-f", "flac", "-d", "flac", str(src), str(dst)],
                   check=True, capture_output=True)


@pytest.fixture()
def wav_and_flac(tmp_path: Path) -> tuple[Path, Path]:
    wav = tmp_path / "clip.wav"
    flac = tmp_path / "clip.flac"
    _tone_wav(wav)
    _to_flac(wav, flac)
    return wav, flac


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def test_flac_decodes_to_the_same_samples_as_the_wav(wav_and_flac) -> None:
    np = pytest.importorskip("numpy")
    wav, flac = wav_and_flac
    from_wav = decode_to_float32_mono16k(wav)
    from_flac = decode_to_float32_mono16k(flac)

    assert from_flac.dtype == np.float32
    assert from_flac.ndim == 1
    assert len(from_flac) == len(from_wav)
    assert np.allclose(from_flac, from_wav, atol=1e-4), "FLAC must be lossless"


def test_flac_is_actually_smaller(wav_and_flac) -> None:
    wav, flac = wav_and_flac
    assert flac.stat().st_size < wav.stat().st_size


def test_a_nonconforming_wav_is_transcoded_not_rejected(tmp_path: Path) -> None:
    """44.1 kHz stereo used to fall through to mlx-whisper's ffmpeg loader,
    which on this machine simply fails."""
    np = pytest.importorskip("numpy")
    odd = tmp_path / "odd.wav"
    _tone_wav(odd, rate=44100, channels=2)

    out = decode_to_float32_mono16k(odd)

    assert out.ndim == 1, "must be downmixed to mono"
    assert out.dtype == np.float32
    # 2 s at 16 kHz, allowing for resampler edge effects.
    assert abs(len(out) - 2 * TARGET_SAMPLE_RATE) < TARGET_SAMPLE_RATE * 0.05


def test_decoding_does_not_need_ffmpeg(wav_and_flac, monkeypatch) -> None:
    """The whole point. Make the ffmpeg path explode and decode anyway."""
    _wav, flac = wav_and_flac

    import mlx_whisper.audio as mlx_audio

    def explode(*_a, **_k):
        raise AssertionError("fell back to ffmpeg — the decoder did not work")

    monkeypatch.setattr(mlx_audio, "load_audio", explode)
    assert len(decode_to_float32_mono16k(flac)) > 0


# ---------------------------------------------------------------------------
# The helpers that gate transcription must work on FLAC too
# ---------------------------------------------------------------------------
def test_duration_is_read_without_decoding(wav_and_flac) -> None:
    wav, flac = wav_and_flac
    assert _safe_wav_duration_seconds(wav) == pytest.approx(2.0, abs=0.05)
    assert _safe_wav_duration_seconds(flac) == pytest.approx(2.0, abs=0.05)


def test_rms_works_on_flac(wav_and_flac) -> None:
    """Without this the silence pre-check returns None for every FLAC, and the
    health surface goes quiet again — the exact bug P0 set out to kill."""
    wav, flac = wav_and_flac
    assert _wav_rms(wav) == pytest.approx(_wav_rms(flac), abs=0.01)


def test_a_silent_flac_is_still_detected_as_silent(tmp_path: Path) -> None:
    src = tmp_path / "quiet.wav"
    _tone_wav(src, amp=0)
    flac = tmp_path / "quiet.flac"
    _to_flac(src, flac)
    assert (_wav_rms(flac) or 0.0) < 0.001


# ---------------------------------------------------------------------------
# End to end through the engine
# ---------------------------------------------------------------------------
def test_the_engine_transcribes_a_flac_file(wav_and_flac) -> None:
    _wav, flac = wav_and_flac
    seen: dict = {}

    def fake(audio, **kw):
        seen["type"] = type(audio).__name__
        seen["len"] = len(audio)
        return {"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                "language": "en"}

    # Exercise the real _default_mlx_transcribe decode path by patching only
    # the mlx call itself.
    import mlx_whisper

    import src.transcription.whisper_engine as we

    orig = mlx_whisper.transcribe
    mlx_whisper.transcribe = fake
    try:
        engine = we.WhisperEngine(model_name="small", report_health=False)
        result = engine.transcribe(flac, stream="system")
    finally:
        mlx_whisper.transcribe = orig

    assert seen["type"] == "ndarray", "mlx must receive decoded samples"
    assert seen["len"] > 0
    assert result.segments and result.segments[0].text == "hi"


def test_stream_label_is_inferred_from_a_flac_filename(tmp_path: Path) -> None:
    from src.transcription.whisper_engine import _infer_stream_label

    assert _infer_stream_label(tmp_path / "2026-08-21_1749_system.flac") == "system"
    assert _infer_stream_label(tmp_path / "abc_mic.flac") == "mic"


def test_engine_skips_a_missing_flac_gracefully(tmp_path: Path) -> None:
    engine = WhisperEngine(model_name="small", report_health=False,
                           transcribe_fn=lambda *a, **k: {})
    result = engine.transcribe(tmp_path / "nope.flac")
    assert result.segments == []
