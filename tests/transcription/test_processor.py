"""Tests for src/transcription/processor.py.

The merge logic is the heart of Phase 4. We test each pure helper directly,
then run the end-to-end ``process`` with a mocked WhisperEngine and real
files on disk so the rename + frontmatter paths are exercised.
"""

from __future__ import annotations

import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.storage.transcript_store import TranscriptStore, split_frontmatter
from src.transcription.processor import (
    ECHO_TIME_WINDOW_SECONDS,
    ECHO_WORD_OVERLAP_THRESHOLD,
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
    apply_offset,
    deduplicate_echo,
    interleave_with_overlaps,
    render_markdown_body,
)
from src.transcription.processor import _LabelledSegment as LabelledSegment
from src.transcription.whisper_engine import (
    Segment,
    TranscriptionResult,
    WhisperEngine,
)


# ============================================================================
# Pure helpers
# ============================================================================
def test_apply_offset_shifts_all_timestamps() -> None:
    segments = [Segment(0.0, 1.0, "a"), Segment(2.0, 3.0, "b")]
    shifted = apply_offset(segments, 0.2)
    assert [s.start for s in shifted] == [0.2, 2.2]
    assert [s.end for s in shifted] == [1.2, 3.2]
    # Original list is untouched.
    assert segments[0].start == 0.0


def test_apply_offset_zero_returns_copy() -> None:
    segments = [Segment(0.0, 1.0, "a")]
    out = apply_offset(segments, 0.0)
    assert out == segments
    assert out is not segments  # defensive copy


def test_deduplicate_echo_drops_overlapping_speech() -> None:
    """Mic and system both pick up 'good morning everyone' within 1 s."""
    mic = [Segment(10.0, 12.0, "good morning everyone")]
    system = [
        Segment(10.05, 12.05, "good morning everyone"),  # echo of mic
        Segment(20.0, 22.0, "thanks for joining"),       # genuine speaker
    ]
    kept, dropped = deduplicate_echo(system, mic)
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].text == "thanks for joining"


def test_deduplicate_echo_keeps_distinct_text() -> None:
    """Same time, different content → not an echo, both keep."""
    mic = [Segment(10.0, 12.0, "what about the deadline")]
    system = [Segment(10.1, 12.1, "we should ship friday")]
    kept, dropped = deduplicate_echo(system, mic)
    assert dropped == 0
    assert kept == system


def test_deduplicate_echo_respects_time_window() -> None:
    """Same words but >1 s apart must not be treated as echo."""
    mic = [Segment(10.0, 12.0, "thanks everyone")]
    system = [Segment(15.0, 17.0, "thanks everyone")]
    kept, dropped = deduplicate_echo(system, mic)
    assert dropped == 0
    assert kept == system


def test_deduplicate_echo_threshold_matches_spec() -> None:
    """80 % overlap is the cutoff per the Phase 4 spec."""
    assert ECHO_TIME_WINDOW_SECONDS == 1.0
    assert ECHO_WORD_OVERLAP_THRESHOLD == pytest.approx(0.80)


def test_interleave_marks_overlap_between_speakers() -> None:
    a = LabelledSegment(Segment(0.0, 5.0, "speaking…"), "Me")
    b = LabelledSegment(Segment(2.0, 6.0, "interrupting"), "Participant")
    out = interleave_with_overlaps([a, b])
    assert [ls.speaker for ls, _ in out] == ["Me", "Participant"]
    assert [ovl for _, ovl in out] == [False, True]


def test_interleave_does_not_mark_same_speaker_overlap() -> None:
    """Two consecutive 'Me' segments that overlap should NOT be flagged."""
    a = LabelledSegment(Segment(0.0, 5.0, "first"), "Me")
    b = LabelledSegment(Segment(3.0, 8.0, "second"), "Me")
    out = interleave_with_overlaps([a, b])
    assert all(not ovl for _, ovl in out)


def test_render_markdown_body_includes_overlap_marker() -> None:
    a = LabelledSegment(Segment(0.0, 5.0, "I think we should"), "Me")
    b = LabelledSegment(Segment(2.0, 6.0, "yes go ahead"), "Participant")
    body = render_markdown_body(
        title="Sprint Planning",
        date_str="2026-05-09",
        segments=interleave_with_overlaps([a, b]),
    )
    assert "[overlap]" in body
    assert "**[00:00]** Me: I think we should" in body
    assert "**[00:02]** Participant: yes go ahead" in body
    assert "## Transcript" in body


def test_render_markdown_body_handles_empty_segments() -> None:
    body = render_markdown_body(
        title="Empty Recording", date_str="2026-05-09", segments=[]
    )
    assert "(no speech detected)" in body


# ============================================================================
# End-to-end process()
# ============================================================================
def _make_wav(path: Path, *, seconds: float = 0.5) -> None:
    """Audible (sine-wave) fixture — silent fills are now skipped by the
    silence pre-check in :class:`WhisperEngine`, which would otherwise return
    no segments and break our merge tests."""
    import math
    import struct

    rate = 16000
    n = int(seconds * rate)
    amp = 16000
    frames = bytearray()
    for i in range(n):
        sample = int(amp * math.sin(2 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", sample)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))


def _make_recording_session(
    tmp_path: Path,
    *,
    session_id: str = "abc12345-0000-0000-0000-000000000000",
    mic_anchor: float = 1000.0,
    sys_anchor: float = 1000.2,
    with_system: bool = True,
) -> RecordingSession:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    mic_wav = audio_dir / f"{session_id}_mic.wav"
    sys_wav = audio_dir / f"{session_id}_system.wav"
    metadata_path = audio_dir / f"{session_id}_metadata.json"
    _make_wav(mic_wav)
    if with_system:
        _make_wav(sys_wav)
    metadata_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "mic_start_monotonic": mic_anchor,
                "system_start_monotonic": sys_anchor,
                "start_wall_clock": "2026-05-09T14:00:02+00:00",
                "pauses": [],
                "sample_rate": 16000,
                "mic_wav": mic_wav.name,
                "system_wav": sys_wav.name if with_system else None,
            }
        )
    )
    return RecordingSession(
        session_id=session_id,
        audio_dir=audio_dir,
        mic_wav=mic_wav,
        system_wav=sys_wav if with_system else None,
        metadata_path=metadata_path,
        mic_start_monotonic=mic_anchor,
        system_start_monotonic=sys_anchor,
        start_wall_clock=datetime(2026, 5, 9, 14, 0, 2, tzinfo=timezone.utc),
        sample_rate=16000,
    )


def _engine_returning(
    mic_segments: list[tuple[float, float, str]],
    system_segments: list[tuple[float, float, str]],
    *,
    language: str = "fr",
) -> WhisperEngine:
    """Build a WhisperEngine with a fake transcribe_fn that returns canned data per file."""
    state = {"call": 0}

    def fake(audio: str, **_kw: Any) -> dict[str, Any]:
        state["call"] += 1
        segs = mic_segments if state["call"] == 1 else system_segments
        return {
            "language": language,
            "duration": (segs[-1][1] if segs else 0.0),
            "segments": [{"start": s, "end": e, "text": t} for s, e, t in segs],
        }

    return WhisperEngine(transcribe_fn=fake)


def test_process_writes_transcript_with_frontmatter(tmp_path: Path) -> None:
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 2.0, "Bonjour tout le monde")],
        system_segments=[(0.5, 2.5, "Salut")],
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir, model_name="small"
    )
    meeting = MeetingSnapshot(
        title="Sprint Planning",
        app="zoom.us",
        participants=[{"name": "Alice", "email": "alice@example.com"}],
    )

    result = processor.process(session, meeting=meeting, language="fr")

    assert result.transcript_path.exists()
    content = result.transcript_path.read_text()
    fm, body = split_frontmatter(content)
    assert fm["id"] == session.session_id
    assert fm["title"] == "Sprint Planning"
    assert fm["language"] == "fr"
    assert fm["model"] == "small"
    assert fm["app"] == "zoom.us"
    assert fm["participants"] == ["Alice <alice@example.com>"]
    assert fm["audio_available"] is True
    assert "Me: Bonjour tout le monde" in body
    assert "Participant: Salut" in body


def test_process_applies_monotonic_offset_to_system_segments(tmp_path: Path) -> None:
    """A 200 ms offset between anchors must shift system segments by 0.2 s."""
    session = _make_recording_session(
        tmp_path, mic_anchor=1000.0, sys_anchor=1000.2
    )
    engine = _engine_returning(
        mic_segments=[(10.0, 11.0, "hi")],
        system_segments=[(10.0, 11.0, "hello")],
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir
    )

    result = processor.process(session, meeting=MeetingSnapshot(), language=None)
    body = result.body
    # Offset means the participant segment now starts at 10.2 → still 00:10 in MM:SS,
    # but should appear AFTER the mic segment in chronological order.
    me_idx = body.index("Me: hi")
    them_idx = body.index("Participant: hello")
    assert me_idx < them_idx


def test_process_dedupes_echo_when_opted_in(tmp_path: Path) -> None:
    """When ``dedup_echoes=True``, identical text from both tracks → only mic stays.

    Default (False) preserves both streams; this test asserts the optional
    dedup path still works for callers that explicitly want it.
    """
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 2.0, "good morning everyone")],
        system_segments=[(0.05, 2.05, "good morning everyone")],
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir,
        dedup_echoes=True,
    )

    result = processor.process(session, meeting=MeetingSnapshot())
    assert result.echo_dropped == 1
    assert "Participant: good morning everyone" not in result.body
    assert "Me: good morning everyone" in result.body


def test_process_keeps_both_streams_by_default(tmp_path: Path) -> None:
    """Default behaviour: same content on both tracks → both kept (no dedup)."""
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 2.0, "good morning everyone")],
        system_segments=[(0.05, 2.05, "good morning everyone")],
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir,
    )

    result = processor.process(session, meeting=MeetingSnapshot())
    assert result.echo_dropped == 0
    assert "Me: good morning everyone" in result.body
    assert "Participant: good morning everyone" in result.body


def test_process_handles_missing_system_audio(tmp_path: Path) -> None:
    """If BlackHole wasn't configured, system_wav is None — keep going."""
    session = _make_recording_session(tmp_path, with_system=False)
    engine = _engine_returning(
        mic_segments=[(0.0, 1.0, "hello")],
        system_segments=[],
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir
    )

    result = processor.process(session, meeting=MeetingSnapshot())
    assert "Me: hello" in result.body
    assert result.system_segments == 0
    assert result.metadata["audio_files"]["system"] is None


def test_process_relocates_audio_into_year_month_tree(tmp_path: Path) -> None:
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 1.0, "x")], system_segments=[(0.0, 1.0, "y")]
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=session.audio_dir
    )

    result = processor.process(session, meeting=MeetingSnapshot())

    # Expected path is computed from the LOCAL-tz view of start_wall_clock —
    # the processor uses ``.astimezone()``, so we mirror that here so the test
    # passes regardless of which timezone CI / a developer's box runs in.
    local_start = session.start_wall_clock.astimezone()
    expected_dir = (
        session.audio_dir / local_start.strftime("%Y") / local_start.strftime("%m")
    )
    expected_mic = expected_dir / local_start.strftime("%Y-%m-%d_%H%M_mic.wav")
    expected_sys = expected_dir / local_start.strftime("%Y-%m-%d_%H%M_system.wav")
    assert expected_mic.exists()
    assert expected_sys.exists()
    # Original UUID-named files no longer exist.
    assert not session.mic_wav.exists()
    assert not session.system_wav.exists()
    # Frontmatter audio_files paths are relative to audio_dir.
    expected_relative = (
        f"{local_start.strftime('%Y')}/{local_start.strftime('%m')}/"
        f"{local_start.strftime('%Y-%m-%d_%H%M_mic.wav')}"
    )
    assert result.metadata["audio_files"]["mic"] == expected_relative


def test_process_progress_reaches_100(tmp_path: Path) -> None:
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 1.0, "x")], system_segments=[(0.0, 1.0, "y")]
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(engine=engine, store=store, audio_dir=session.audio_dir)

    seen: list[float] = []
    processor.process(session, meeting=MeetingSnapshot(), on_progress=seen.append)
    assert seen, "progress callback never fired"
    assert seen[-1] == 100.0
    # Should pass the 45 % and 90 % checkpoints.
    assert any(p >= 45.0 for p in seen)
    assert any(p >= 90.0 for p in seen)


def test_process_async_runs_in_background(tmp_path: Path) -> None:
    session = _make_recording_session(tmp_path)
    engine = _engine_returning(
        mic_segments=[(0.0, 1.0, "x")], system_segments=[]
    )
    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(engine=engine, store=store, audio_dir=session.audio_dir)

    received: list[Path] = []
    t = processor.process_async(
        session,
        meeting=MeetingSnapshot(),
        on_complete=lambda r: received.append(r.transcript_path),
    )
    t.join(timeout=5.0)
    assert received and received[0].exists()


def test_recording_session_from_recorder_metadata(tmp_path: Path) -> None:
    """The factory must read the recorder's metadata dict correctly."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    metadata = {
        "session_id": "deadbeef-0000-0000-0000-000000000000",
        "mic_wav": "deadbeef-0000-0000-0000-000000000000_mic.wav",
        "system_wav": "deadbeef-0000-0000-0000-000000000000_system.wav",
        "mic_start_monotonic": 100.5,
        "system_start_monotonic": 100.7,
        "start_wall_clock": "2026-05-09T14:00:00+00:00",
        "sample_rate": 16000,
    }
    session = RecordingSession.from_recorder_metadata(metadata, audio_dir=audio_dir)
    assert session.session_id == metadata["session_id"]
    assert session.mic_wav.name == metadata["mic_wav"]
    assert session.system_wav is not None
    assert session.mic_start_monotonic == 100.5
    assert session.start_wall_clock is not None
