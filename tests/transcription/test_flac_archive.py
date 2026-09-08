"""P3 item 24 — archive recordings compressed, and never at the cost of one.

16-bit PCM at 16 kHz costs 230 MB per meeting-hour across the two streams.
FLAC is lossless, roughly halves speech, and collapses a stream that captured
nothing to almost nothing (721x measured on a real 85 MB silent stream).

The constraint that shapes this: compression must never be able to lose a
recording. Every failure path keeps the audio.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import (
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
    encode_audio,
)
from src.transcription.whisper_engine import WhisperEngine, decode_to_float32_mono16k

pytestmark = pytest.mark.skipif(
    shutil.which("afconvert") is None, reason="afconvert is macOS-only"
)


def _tone(path: Path, *, seconds: float = 1.0, amp: int = 9000) -> None:
    n, rate = int(seconds * 16000), 16000
    frames = bytearray()
    for i in range(n):
        frames += struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _session(tmp_path: Path, sid: str = "s1") -> RecordingSession:
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    mic, sysw = audio / f"{sid}_mic.wav", audio / f"{sid}_system.wav"
    _tone(mic)
    _tone(sysw, amp=6000)
    meta = audio / f"{sid}_metadata.json"
    meta.write_text(json.dumps({
        "session_id": sid, "mic_wav": mic.name, "system_wav": sysw.name,
        "mic_start_monotonic": 0.0, "system_start_monotonic": 0.0,
        "start_wall_clock": "2026-05-09T14:00:00+00:00", "sample_rate": 16000,
    }))
    return RecordingSession(
        session_id=sid, audio_dir=audio, mic_wav=mic, system_wav=sysw,
        metadata_path=meta, mic_start_monotonic=0.0, system_start_monotonic=0.0,
        start_wall_clock=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc),
        sample_rate=16000,
    )


def _engine() -> WhisperEngine:
    return WhisperEngine(
        model_name="small", report_health=False,
        transcribe_fn=lambda *a, **k: {
            "language": "en", "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
    )


def _process(tmp_path: Path, **kw):
    session = _session(tmp_path)
    proc = TranscriptProcessor(
        engine=_engine(), store=TranscriptStore(tmp_path / "t"),
        audio_dir=session.audio_dir, **kw,
    )
    return proc, session, proc.process(session, meeting=MeetingSnapshot())


# ---------------------------------------------------------------------------
# encode_audio
# ---------------------------------------------------------------------------
def test_encoding_is_lossless(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    src, dst = tmp_path / "a.wav", tmp_path / "a.flac"
    _tone(src, seconds=2.0)
    assert encode_audio(src, dst) is True
    assert np.allclose(
        decode_to_float32_mono16k(src), decode_to_float32_mono16k(dst), atol=1e-4
    )


def test_encoding_actually_shrinks_the_file(tmp_path: Path) -> None:
    src, dst = tmp_path / "a.wav", tmp_path / "a.flac"
    _tone(src, seconds=2.0)
    encode_audio(src, dst)
    assert dst.stat().st_size < src.stat().st_size


def test_a_silent_stream_costs_almost_nothing(tmp_path: Path) -> None:
    """The audit's 85 MB of digital silence — the single biggest win."""
    src, dst = tmp_path / "q.wav", tmp_path / "q.flac"
    _tone(src, seconds=5.0, amp=0)
    encode_audio(src, dst)
    assert dst.stat().st_size < src.stat().st_size / 50


def test_an_unknown_format_is_declined_not_guessed(tmp_path: Path) -> None:
    src, dst = tmp_path / "a.wav", tmp_path / "a.mp3"
    _tone(src)
    assert encode_audio(src, dst) is False


def test_a_failed_encode_leaves_no_stub_behind(tmp_path: Path) -> None:
    missing, dst = tmp_path / "nope.wav", tmp_path / "nope.flac"
    assert encode_audio(missing, dst) is False
    assert not dst.exists()


# ---------------------------------------------------------------------------
# The archive step
# ---------------------------------------------------------------------------
def test_recordings_are_archived_as_flac_by_default(tmp_path: Path) -> None:
    _proc, _s, result = _process(tmp_path)
    mic = result.metadata["audio_files"]["mic"]
    assert mic.endswith(".flac"), mic
    assert (tmp_path / "audio" / mic).exists()
    assert result.metadata["audio_available"] is True


def test_the_original_wav_is_removed_after_a_successful_encode(tmp_path: Path) -> None:
    _proc, session, _r = _process(tmp_path)
    assert not session.mic_wav.exists()
    assert not session.system_wav.exists()


def test_wav_can_still_be_chosen(tmp_path: Path) -> None:
    _proc, _s, result = _process(tmp_path, archive_format="wav")
    assert result.metadata["audio_files"]["mic"].endswith(".wav")


def test_an_unknown_format_falls_back_to_wav(tmp_path: Path) -> None:
    _proc, _s, result = _process(tmp_path, archive_format="ogg")
    assert result.metadata["audio_files"]["mic"].endswith(".wav")


def test_metadata_json_follows_the_audio(tmp_path: Path) -> None:
    """retranscribe and orphan discovery both read this file."""
    _proc, _s, result = _process(tmp_path)
    mic_rel = result.metadata["audio_files"]["mic"]
    meta_path = (tmp_path / "audio" / mic_rel)
    meta_path = meta_path.with_name(meta_path.stem.removesuffix("_mic") + "_metadata.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["mic_wav"] == mic_rel
    assert (tmp_path / "audio" / meta["mic_wav"]).exists()


def test_an_archived_flac_can_be_transcribed_again(tmp_path: Path) -> None:
    """The whole reason the decoder had to land first."""
    _proc, _s, result = _process(tmp_path)
    flac = tmp_path / "audio" / result.metadata["audio_files"]["mic"]
    samples = decode_to_float32_mono16k(flac)
    assert len(samples) > 0


# ---------------------------------------------------------------------------
# Compression must never cost a recording
# ---------------------------------------------------------------------------
def test_a_broken_encoder_falls_back_to_keeping_the_audio(
    tmp_path: Path, monkeypatch
) -> None:
    import src.transcription.processor as proc_mod

    monkeypatch.setattr(proc_mod, "encode_audio", lambda src, dst: False)
    _proc, _session, result = _process(tmp_path)

    mic = result.metadata["audio_files"]["mic"]
    assert mic.endswith(".wav"), "must keep the raw capture rather than lose it"
    assert (tmp_path / "audio" / mic).exists()
    assert result.metadata["audio_available"] is True


def test_a_failed_move_marks_the_transcript_honestly(
    tmp_path: Path, monkeypatch
) -> None:
    """Metadata used to be rewritten to the new paths BEFORE the moves, so a
    failure left it pointing at files that never arrived."""
    import src.transcription.processor as proc_mod

    monkeypatch.setattr(proc_mod, "encode_audio", lambda src, dst: False)
    monkeypatch.setattr(
        proc_mod.shutil, "move",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")),
    )
    _proc, _s, _result = _process(tmp_path)

    store = TranscriptStore(tmp_path / "t")
    fm = store.list_transcripts(limit=5)[0]
    assert fm["audio_available"] is False
    assert fm["audio_files"]["mic"] is None
