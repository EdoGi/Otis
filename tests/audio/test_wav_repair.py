"""Tests for src/audio/wav_repair.py — crash-truncated WAV header recovery."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from src.audio.wav_repair import repair_if_needed, repair_wav_header, wav_needs_repair


def _write_wav(path: Path, *, seconds: float = 1.0, rate: int = 16000) -> bytes:
    """Valid 16 kHz mono 16-bit WAV; returns the PCM payload."""
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return frames


def _zero_size_fields(path: Path) -> None:
    """Simulate a crash: wave only patches sizes on close, so a killed
    process leaves the placeholder values. Standard wave output puts the
    data chunk at offset 36."""
    with path.open("r+b") as fh:
        fh.seek(4)
        fh.write(struct.pack("<I", 36))  # RIFF size as-at-open (header only)
        fh.seek(40)
        fh.write(struct.pack("<I", 0))   # data size 0


def test_crashed_wav_detected_and_repaired(tmp_path: Path) -> None:
    path = tmp_path / "rec_mic.wav"
    payload = _write_wav(path)
    _zero_size_fields(path)

    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 0  # confirmed broken

    assert wav_needs_repair(path) is True
    assert repair_wav_header(path) is True

    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == len(payload) // 2
        assert wf.readframes(wf.getnframes()) == payload  # bit-exact recovery
    assert wav_needs_repair(path) is False


def test_healthy_wav_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "ok.wav"
    _write_wav(path)
    before = path.read_bytes()
    assert wav_needs_repair(path) is False
    assert repair_wav_header(path) is False
    assert path.read_bytes() == before


def test_header_only_wav_not_repaired(tmp_path: Path) -> None:
    """Crash before the first frame: data chunk is genuinely empty."""
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
    assert wav_needs_repair(path) is False
    assert repair_wav_header(path) is False


def test_garbage_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"\x00\x01\x02 this is not a wav file at all" * 10)
    assert wav_needs_repair(path) is False
    assert repair_wav_header(path) is False


def test_truncation_lands_on_frame_boundary(tmp_path: Path) -> None:
    """A crash mid-sample leaves an odd byte; repair truncates to block_align."""
    path = tmp_path / "odd.wav"
    payload = _write_wav(path)
    _zero_size_fields(path)
    with path.open("ab") as fh:
        fh.write(b"\x7f")  # half a sample

    assert repair_wav_header(path) is True
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == len(payload) // 2  # odd byte dropped


def test_repair_if_needed_handles_none_and_missing(tmp_path: Path) -> None:
    assert repair_if_needed(None) is False
    assert repair_if_needed(tmp_path / "nope.wav") is False
