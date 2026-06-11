"""Tests for the Generate-Transcript orphan-discovery helpers in menubar.py.

These tests don't touch rumps — they exercise the module-level
``find_orphan_sessions`` / ``_resolve_session_audio`` functions directly
against a real filesystem layout in ``tmp_path``.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from src.storage.transcript_store import TranscriptStore
from src.ui.menubar import (
    _format_session_label,
    _resolve_session_audio,
    find_orphan_sessions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_wav(path: Path, *, size_bytes: int = 1024) -> None:
    """Minimal valid WAV file with a chosen body size."""
    rate = 16000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00" * size_bytes)


def _write_metadata(path: Path, *, session_id: str, mic_name: str | None = None,
                    sys_name: str | None = None) -> None:
    payload = {
        "session_id": session_id,
        "mic_wav": mic_name or f"{session_id}_mic.wav",
        "system_wav": sys_name,
        "mic_start_monotonic": 0.0,
        "system_start_monotonic": 0.0,
        "start_wall_clock": "2026-05-10T14:00:00+00:00",
        "sample_rate": 16000,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _basic_transcript(session_id: str) -> dict:
    return {
        "id": session_id,
        "title": "Existing",
        "date": "2026-05-09",
        "start_time": "10:00",
        "end_time": "10:30",
        "duration_minutes": 30,
        "language": "en",
        "app": None,
        "participants": [],
        "tags": [],
        "audio_files": {"mic": None, "system": None},
        "audio_available": True,
        "model": "small",
    }


# ---------------------------------------------------------------------------
# Empty / missing audio dir
# ---------------------------------------------------------------------------
def test_returns_empty_when_audio_dir_missing(tmp_path: Path) -> None:
    out = find_orphan_sessions(audio_dir=tmp_path / "does-not-exist")
    assert out == []


def test_returns_empty_when_audio_dir_empty(tmp_path: Path) -> None:
    out = find_orphan_sessions(audio_dir=tmp_path)
    assert out == []


# ---------------------------------------------------------------------------
# Pass 1 — sessions with metadata.json
# ---------------------------------------------------------------------------
def test_finds_uuid_session_with_metadata(tmp_path: Path) -> None:
    sid = "abc12345-0000-0000-0000-000000000000"
    _make_wav(tmp_path / f"{sid}_mic.wav")
    _make_wav(tmp_path / f"{sid}_system.wav")
    _write_metadata(
        tmp_path / f"{sid}_metadata.json",
        session_id=sid,
        sys_name=f"{sid}_system.wav",
    )
    out = find_orphan_sessions(audio_dir=tmp_path)
    assert len(out) == 1
    entry = out[0]
    assert entry["session_id"] == sid
    assert entry["mic_path"].name == f"{sid}_mic.wav"
    assert entry["system_path"].name == f"{sid}_system.wav"
    assert entry["metadata_path"] is not None


def test_finds_renamed_yyyymm_session(tmp_path: Path) -> None:
    """Layout produced by a successful TranscriptProcessor run."""
    sid = "renamed-1"
    sub = tmp_path / "2026" / "05"
    sub.mkdir(parents=True)
    prefix = sub / "2026-05-10_1400"
    _make_wav(prefix.with_name(f"{prefix.name}_mic.wav"))
    _make_wav(prefix.with_name(f"{prefix.name}_system.wav"))
    _write_metadata(
        prefix.with_name(f"{prefix.name}_metadata.json"),
        session_id=sid,
        mic_name="2026/05/2026-05-10_1400_mic.wav",
        sys_name="2026/05/2026-05-10_1400_system.wav",
    )
    out = find_orphan_sessions(audio_dir=tmp_path)
    assert len(out) == 1
    assert out[0]["session_id"] == sid


def test_skips_sessions_already_transcribed(tmp_path: Path) -> None:
    sid = "transcribed-1"
    audio = tmp_path / "audio"
    transcripts = tmp_path / "transcripts"
    audio.mkdir(); transcripts.mkdir()
    _make_wav(audio / f"{sid}_mic.wav")
    _write_metadata(audio / f"{sid}_metadata.json", session_id=sid)
    store = TranscriptStore(transcripts)
    store.save(_basic_transcript(sid), "## Transcript\n\nbody")

    out = find_orphan_sessions(audio_dir=audio, store=store)
    assert out == []


def test_failed_placeholder_does_not_hide_session(tmp_path: Path) -> None:
    """A session whose transcription failed must STILL be offered for retry.

    save_failure writes a status:failed transcript with the session id; that
    placeholder used to make the session look transcribed and vanish from the
    Generate Transcript menu.
    """
    sid = "failed-1"
    audio = tmp_path / "audio"
    transcripts = tmp_path / "transcripts"
    audio.mkdir(); transcripts.mkdir()
    _make_wav(audio / f"{sid}_mic.wav")
    _write_metadata(audio / f"{sid}_metadata.json", session_id=sid)
    store = TranscriptStore(transcripts)
    store.save_failure(session_id=sid, error=RuntimeError("oom"), title="Big Call")

    out = find_orphan_sessions(audio_dir=audio, store=store)
    assert [e["session_id"] for e in out] == [sid]

    # Once a real transcript lands, the session disappears from the list.
    store.save(_basic_transcript(sid), "## Transcript\n\nbody")
    assert find_orphan_sessions(audio_dir=audio, store=store) == []


# ---------------------------------------------------------------------------
# Pass 2 — orphan WAVs without metadata.json
# ---------------------------------------------------------------------------
def test_finds_orphan_uuid_wavs_without_metadata(tmp_path: Path) -> None:
    sid = "deadbeef-cafe-1234-5678-aaaaaaaaaaaa"
    _make_wav(tmp_path / f"{sid}_mic.wav")
    _make_wav(tmp_path / f"{sid}_system.wav")

    out = find_orphan_sessions(audio_dir=tmp_path)
    assert len(out) == 1
    assert out[0]["session_id"] == sid
    assert out[0]["metadata_path"] is None
    assert "(orphan)" in out[0]["label"]


def test_orphan_without_system_track_still_works(tmp_path: Path) -> None:
    sid = "mic-only-orphan"
    _make_wav(tmp_path / f"{sid}_mic.wav")
    out = find_orphan_sessions(audio_dir=tmp_path)
    assert len(out) == 1
    assert out[0]["system_path"] is None


# ---------------------------------------------------------------------------
# In-progress recording exclusion (the bug I caught during the audit)
# ---------------------------------------------------------------------------
def test_excludes_actively_recording_session_from_orphan_list(tmp_path: Path) -> None:
    """A WAV being written right now must NOT appear as an orphan."""
    active_sid = "active-recording-uuid"
    other_sid = "other-orphan-uuid"
    _make_wav(tmp_path / f"{active_sid}_mic.wav")
    _make_wav(tmp_path / f"{active_sid}_system.wav")
    _make_wav(tmp_path / f"{other_sid}_mic.wav")

    out = find_orphan_sessions(
        audio_dir=tmp_path,
        current_recording_session_id=active_sid,
    )
    assert len(out) == 1
    assert out[0]["session_id"] == other_sid


# ---------------------------------------------------------------------------
# Limit + ordering
# ---------------------------------------------------------------------------
def test_respects_limit_argument(tmp_path: Path) -> None:
    for i in range(5):
        sid = f"sid-{i:02d}"
        _make_wav(tmp_path / f"{sid}_mic.wav")
        _write_metadata(tmp_path / f"{sid}_metadata.json", session_id=sid)

    out = find_orphan_sessions(audio_dir=tmp_path, limit=2)
    assert len(out) == 2


def test_corrupt_metadata_files_are_skipped(tmp_path: Path) -> None:
    """A bad metadata.json must not stop us from listing other sessions."""
    good_sid = "good-1"
    _make_wav(tmp_path / f"{good_sid}_mic.wav")
    _write_metadata(tmp_path / f"{good_sid}_metadata.json", session_id=good_sid)
    # Plant a bad metadata file.
    (tmp_path / "broken_metadata.json").write_text(":::not yaml:::")

    out = find_orphan_sessions(audio_dir=tmp_path)
    assert {e["session_id"] for e in out} == {good_sid}


def test_metadata_pointing_at_missing_audio_is_skipped(tmp_path: Path) -> None:
    """Don't list sessions whose metadata.json references a deleted .wav."""
    sid = "ghost"
    _write_metadata(tmp_path / f"{sid}_metadata.json", session_id=sid)
    # NO mic.wav written — the file's gone.
    out = find_orphan_sessions(audio_dir=tmp_path)
    assert out == []


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------
def test_resolve_session_audio_uses_sibling_when_declared_path_missing(tmp_path: Path) -> None:
    """When metadata's declared path doesn't exist, fall back to siblings."""
    sub = tmp_path / "2026" / "05"
    sub.mkdir(parents=True)
    sid = "x-1"
    meta_path = sub / "2026-05-10_1400_metadata.json"
    meta_path.write_text("{}")
    _make_wav(sub / "2026-05-10_1400_mic.wav")
    _make_wav(sub / "2026-05-10_1400_system.wav")

    mic, sys_path = _resolve_session_audio(
        meta_path=meta_path,
        declared_mic="x-1_mic.wav",   # this path doesn't exist
        declared_sys="x-1_system.wav",
        sid=sid,
    )
    assert mic is not None and mic.name == "2026-05-10_1400_mic.wav"
    assert sys_path is not None and sys_path.name == "2026-05-10_1400_system.wav"


def test_resolve_session_audio_returns_none_when_nothing_resolves(tmp_path: Path) -> None:
    meta = tmp_path / "x_metadata.json"
    meta.write_text("{}")
    mic, sys_path = _resolve_session_audio(
        meta_path=meta, declared_mic="x_mic.wav", declared_sys=None, sid="x",
    )
    assert mic is None and sys_path is None


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------
def test_format_session_label_includes_mb(tmp_path: Path) -> None:
    p = tmp_path / "test_mic.wav"
    _make_wav(p, size_bytes=2 * 1024 * 1024)
    label = _format_session_label(None, p)
    assert "MB" in label


def test_format_session_label_handles_missing_file_gracefully() -> None:
    label = _format_session_label(None, Path("/does/not/exist.wav"))
    # Doesn't crash — just may have unknown values.
    assert isinstance(label, str)
