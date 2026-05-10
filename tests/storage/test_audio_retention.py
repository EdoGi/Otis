"""Tests for src/storage/audio_retention.py."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.storage.audio_retention import AudioRetentionManager
from src.storage.transcript_store import TranscriptStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_audio_session(
    audio_dir: Path,
    *,
    session_id: str,
    age_days: float,
    with_system: bool = True,
    use_uuid_layout: bool = True,
) -> list[Path]:
    """Create a fake recorded session with mtime ``age_days`` in the past."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    mtime = time.time() - age_days * 86400.0

    if use_uuid_layout:
        prefix = audio_dir / session_id
        mic = Path(f"{prefix}_mic.wav")
        sys_ = Path(f"{prefix}_system.wav") if with_system else None
        meta = Path(f"{prefix}_metadata.json")
    else:
        # Simulate the post-rename layout under YYYY/MM/.
        sub = audio_dir / "2026" / "05"
        sub.mkdir(parents=True, exist_ok=True)
        prefix = sub / "2026-05-09_1400"
        mic = Path(f"{prefix}_mic.wav")
        sys_ = Path(f"{prefix}_system.wav") if with_system else None
        meta = Path(f"{prefix}_metadata.json")

    mic.write_bytes(b"\x00" * 1024)
    if sys_ is not None:
        sys_.write_bytes(b"\x00" * 1024)
    meta.write_text(json.dumps({"session_id": session_id}))

    for p in (mic, sys_, meta):
        if p is not None:
            os.utime(p, (mtime, mtime))
    return [p for p in (mic, sys_, meta) if p is not None]


def _make_transcript(store: TranscriptStore, session_id: str) -> Path:
    fm = {
        "id": session_id,
        "title": "Old Meeting",
        "date": "2026-05-09",
        "start_time": "14:00",
        "end_time": "14:30",
        "duration_minutes": 30,
        "language": "fr",
        "app": "zoom.us",
        "participants": [],
        "tags": [],
        "audio_files": {"mic": "x", "system": "y"},
        "audio_available": True,
        "model": "small",
    }
    return store.save(fm, "## Transcript")


# ---------------------------------------------------------------------------
# cleanup_now
# ---------------------------------------------------------------------------
def test_old_files_are_deleted(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    files = _make_audio_session(audio_dir, session_id="old-session", age_days=45)
    rm = AudioRetentionManager(audio_dir=audio_dir, retention_days=30)
    rm.cleanup_now()
    for p in files:
        assert not p.exists()


def test_recent_files_are_kept(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    files = _make_audio_session(audio_dir, session_id="recent", age_days=5)
    rm = AudioRetentionManager(audio_dir=audio_dir, retention_days=30)
    rm.cleanup_now()
    for p in files:
        assert p.exists(), f"expected {p} to survive a 30-day retention sweep"


def test_cleanup_marks_transcript_audio_unavailable(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    store = TranscriptStore(transcript_dir)
    sid = "33333333-3333-3333-3333-333333333333"
    _make_audio_session(audio_dir, session_id=sid, age_days=45)
    _make_transcript(store, sid)

    rm = AudioRetentionManager(
        audio_dir=audio_dir, transcript_store=store, retention_days=30
    )
    report = rm.cleanup_now()
    assert sid in report.transcripts_marked
    fm = store.get_transcript(sid)["metadata"]
    assert fm["audio_available"] is False


def test_cleanup_finds_renamed_layout(tmp_path: Path) -> None:
    """Files under YYYY/MM/ — the renamed layout — must also be swept."""
    audio_dir = tmp_path / "audio"
    sid = "44444444-4444-4444-4444-444444444444"
    files = _make_audio_session(
        audio_dir, session_id=sid, age_days=60, use_uuid_layout=False
    )
    rm = AudioRetentionManager(audio_dir=audio_dir, retention_days=30)
    report = rm.cleanup_now()
    for p in files:
        assert not p.exists()
    assert sid in report.deleted_session_ids


def test_cleanup_handles_nonexistent_audio_dir(tmp_path: Path) -> None:
    """Don't crash when audio_dir hasn't been created yet (fresh install)."""
    rm = AudioRetentionManager(
        audio_dir=tmp_path / "never-existed", retention_days=30
    )
    report = rm.cleanup_now()
    assert report.deleted_files == []


# ---------------------------------------------------------------------------
# manual delete_audio
# ---------------------------------------------------------------------------
def test_delete_audio_removes_only_targeted_session(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    sid_a = "55555555-5555-5555-5555-555555555555"
    sid_b = "66666666-6666-6666-6666-666666666666"
    files_a = _make_audio_session(audio_dir, session_id=sid_a, age_days=1)
    files_b = _make_audio_session(audio_dir, session_id=sid_b, age_days=1)

    rm = AudioRetentionManager(audio_dir=audio_dir, retention_days=30)
    report = rm.delete_audio(sid_a)
    for p in files_a:
        assert not p.exists()
    for p in files_b:
        assert p.exists()
    assert report.deleted_session_ids == [sid_a]


def test_delete_audio_marks_transcript_unavailable(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    store = TranscriptStore(transcript_dir)
    sid = "77777777-7777-7777-7777-777777777777"
    _make_audio_session(audio_dir, session_id=sid, age_days=0.1)
    _make_transcript(store, sid)
    rm = AudioRetentionManager(audio_dir=audio_dir, transcript_store=store, retention_days=30)
    rm.delete_audio(sid)
    assert store.get_transcript(sid)["metadata"]["audio_available"] is False


# ---------------------------------------------------------------------------
# Periodic timer
# ---------------------------------------------------------------------------
def test_start_periodic_runs_initial_sweep(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    files = _make_audio_session(audio_dir, session_id="x", age_days=45)
    rm = AudioRetentionManager(
        audio_dir=audio_dir, retention_days=30, interval_seconds=86400
    )
    rm.start_periodic()
    try:
        for p in files:
            assert not p.exists(), "initial sweep should have run synchronously"
    finally:
        rm.stop()


def test_start_periodic_double_call_does_not_leak_timers(tmp_path: Path) -> None:
    """Calling start_periodic twice must replace the old timer, not stack them."""
    audio_dir = tmp_path / "audio"
    rm = AudioRetentionManager(
        audio_dir=audio_dir, retention_days=30, interval_seconds=86400
    )
    rm.start_periodic()
    first_timer = rm._timer
    rm.start_periodic()
    second_timer = rm._timer
    try:
        assert first_timer is not second_timer, "second start should create a fresh timer"
        # The first timer must be cancelled (not is_alive) so it doesn't leak.
        assert not first_timer.is_alive(), "first timer should have been cancelled"
        assert second_timer.is_alive()
    finally:
        rm.stop()
