"""Mic and system streams drift apart over a meeting; alignment corrects it.

Both streams start within ~0.1 s and stop together (the teardown barrier),
yet the files ended up to 2.1 s different in length on real meetings
(2026-10-21: +2.09 s over 75.7 min; 09-03: -1.18 s; 08-31: -1.79 s). The
processor shifted system timestamps by the start offset only, so late in a
long meeting the two tracks were off by more than the 1 s echo-dedup window:
an echo survived as a duplicate "Me:" line, and the interleave could put a
reply before the question.

The system stream should be (mic length - start offset) long; the processor
now scales system timestamps onto the mic timeline by that ratio — but only
for a mismatch above one capture block (0.25 s) and below 2000 ppm, beyond
which it is not drift and a linear stretch would misplace audio.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import (
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
    apply_offset,
    compute_drift_scale,
)
from src.transcription.whisper_engine import Segment, WhisperEngine

BPS = 16000 * 2  # 16 kHz, 16-bit mono


def test_no_mismatch_means_no_scaling() -> None:
    assert compute_drift_scale(4520 * BPS, int((4520 - 0.04) * BPS), 0.04, BPS) == 1.0


def test_a_mismatch_inside_one_capture_block_is_left_alone() -> None:
    assert compute_drift_scale(4520 * BPS, int((4520 - 0.04 - 0.2) * BPS), 0.04, BPS) == 1.0


def test_two_seconds_over_75_minutes_is_corrected() -> None:
    scale = compute_drift_scale(4520 * BPS, 4518 * BPS, 0.04, BPS)
    assert scale == pytest.approx((4520 - 0.04) / 4518)
    assert 400 < (scale - 1) * 1e6 < 500  # ~434 ppm, within the measured range


@pytest.mark.parametrize("mic, sys", [(None, 4518 * BPS), (4520 * BPS, None), (0, 0)])
def test_missing_byte_counts_mean_no_scaling(mic, sys) -> None:
    """Older recordings, or a mic-only one, carry no usable counts."""
    assert compute_drift_scale(mic, sys, 0.04, BPS) == 1.0


def test_an_implausible_mismatch_is_not_treated_as_drift() -> None:
    """A minute missing from one stream is a gap we cannot place, not drift."""
    assert compute_drift_scale(4520 * BPS, 4460 * BPS, 0.04, BPS) == 1.0


def test_apply_offset_scales_before_shifting() -> None:
    out = apply_offset([Segment(1000.0, 1001.0, "x")], 0.5, scale=1.001)
    assert out[0].start == pytest.approx(0.5 + 1001.0)
    assert out[0].end == pytest.approx(0.5 + 1002.001)


# ---------------------------------------------------------------------------
# End to end through the processor
# ---------------------------------------------------------------------------
def _make_wav(path: Path) -> None:
    rate, n = 16000, 8000
    frames = b"".join(struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / rate)))
                      for i in range(n))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)


def _session(tmp_path: Path, *, mic_seconds: float, sys_seconds: float) -> RecordingSession:
    sid = "d41f0000-0000-0000-0000-000000000000"
    audio = tmp_path / "audio"
    audio.mkdir()
    mic, sysw, meta = audio / f"{sid}_mic.wav", audio / f"{sid}_system.wav", audio / f"{sid}_metadata.json"
    _make_wav(mic)
    _make_wav(sysw)
    meta.write_text(json.dumps({
        "session_id": sid, "mic_start_monotonic": 1000.0, "system_start_monotonic": 1000.04,
        "start_wall_clock": "2026-10-21T08:09:56+00:00", "pauses": [], "sample_rate": 16000,
        "channels": 1, "mic_wav": mic.name, "system_wav": sysw.name,
        "mic_bytes": int(mic_seconds * BPS), "system_bytes": int(sys_seconds * BPS),
    }))
    return RecordingSession(
        session_id=sid, audio_dir=audio, mic_wav=mic, system_wav=sysw, metadata_path=meta,
        mic_start_monotonic=1000.0, system_start_monotonic=1000.04,
        start_wall_clock=datetime(2026, 10, 21, 8, 9, 56, tzinfo=UTC), sample_rate=16000,
    )


def _engine(mic: list[tuple[float, float, str]], system: list[tuple[float, float, str]]) -> WhisperEngine:
    calls = {"n": 0}

    def fake(audio: str, **_kw: Any) -> dict[str, Any]:
        calls["n"] += 1
        segs = mic if calls["n"] == 1 else system
        return {"language": "fr", "segments": [{"start": s, "end": e, "text": t} for s, e, t in segs]}

    return WhisperEngine(transcribe_fn=fake)


def test_an_echo_late_in_a_drifting_meeting_is_still_deduplicated(tmp_path: Path) -> None:
    # The far side said this 75 minutes in; the mic heard it off the speakers.
    # In the system file it sits 2 s earlier, because that file is 2 s short.
    words = "on se retrouve jeudi pour valider le planning"
    session = _session(tmp_path, mic_seconds=4520.0, sys_seconds=4518.0)
    processor = TranscriptProcessor(
        engine=_engine(mic=[(4501.95, 4504.0, words)], system=[(4500.0, 4502.0, words)]),
        store=TranscriptStore(tmp_path / "transcripts"), audio_dir=session.audio_dir,
        dedup_echoes=True,
    )
    result = processor.process(session, meeting=MeetingSnapshot(), language="fr")
    assert result.echo_dropped == 1, "uncorrected, the copies sat 1.9 s apart"
    assert f"Me: {words}" not in result.body
