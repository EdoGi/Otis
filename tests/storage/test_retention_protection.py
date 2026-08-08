"""Regression cover for audit finding B2 — retention deleted unrecoverable audio.

``cleanup_now()`` selected purely on mtime and never consulted the transcript
store, even though ``completed_session_ids()`` exists for exactly this purpose
and is already used by the orphan-discovery menu. Two ways that lost data:

* a failed transcription writes a placeholder whose body says "run
  retranscribe.py <id>" — at day 30 the only input to that retry was deleted;
* a session never transcribed at all (crash, or the user never opened
  "Generate Transcript") was deleted at day 30 with no transcript ever made.

Audio is only deletable once a *successful* transcript exists.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import pytest

from src.storage.audio_retention import AudioRetentionManager
from src.storage.transcript_store import TranscriptStore


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def _age(path: Path, days: float) -> None:
    old = time.time() - days * 86400
    import os

    os.utime(path, (old, old))


def make_session(
    audio_dir: Path, sid: str, *, days_old: float, relocated: bool = False,
) -> dict[str, Path]:
    """Write a session's WAVs + metadata, aged ``days_old``."""
    if relocated:
        base = audio_dir / "2026" / "07"
        stem = "2026-07-01_1030"
    else:
        base = audio_dir
        stem = sid
    mic = base / f"{stem}_mic.wav"
    system = base / f"{stem}_system.wav"
    meta = base / f"{stem}_metadata.json"
    _wav(mic)
    _wav(system)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"session_id": sid}), encoding="utf-8")
    for p in (mic, system, meta):
        _age(p, days_old)
    return {"mic": mic, "system": system, "meta": meta}


def frontmatter(sid: str, *, status: str | None = None) -> dict:
    fm = {
        "id": sid,
        "title": "Meeting",
        "date": "2026-07-01",
        "start_time": "10:30",
        "end_time": "11:00",
        "duration_minutes": 30,
        "language": "en",
        "app": None,
        "participants": [],
        "tags": [],
        "audio_files": {"mic": None, "system": None},
        "audio_available": True,
        "model": "small",
    }
    if status:
        fm["status"] = status
        fm["tags"] = ["failed"]
    return fm


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, TranscriptStore]:
    audio = tmp_path / "audio"
    audio.mkdir()
    store = TranscriptStore(tmp_path / "transcripts")
    return audio, store


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------
def test_never_transcribed_audio_survives_retention(dirs) -> None:
    audio, store = dirs
    files = make_session(audio, "never-done", days_old=45)

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert files["mic"].exists(), "audio with no transcript is the only copy of that meeting"
    assert files["system"].exists()
    assert report.deleted_files == []
    assert "never-done" in report.protected_session_ids


def test_failed_transcription_audio_survives_retention(dirs) -> None:
    """The placeholder literally tells the user to run retranscribe.py."""
    audio, store = dirs
    files = make_session(audio, "failed-one", days_old=45)
    store.save(frontmatter("failed-one", status="failed"), "# failed\n\nrun retranscribe.py")

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert files["mic"].exists()
    assert "failed-one" in report.protected_session_ids


def test_successfully_transcribed_audio_is_deleted(dirs) -> None:
    audio, store = dirs
    files = make_session(audio, "all-good", days_old=45)
    store.save(frontmatter("all-good"), "# ok\n\nbody")

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert not files["mic"].exists()
    assert not files["system"].exists()
    assert len(report.deleted_files) == 3
    assert "all-good" in report.deleted_session_ids


def test_protection_works_for_relocated_date_named_audio(dirs) -> None:
    """Successful sessions are renamed to YYYY-MM-DD_HHMM_*.wav and the UUID
    disappears from the filename — the session id survives only in the JSON."""
    audio, store = dirs
    files = make_session(audio, "uuid-style-id", days_old=45, relocated=True)

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert files["mic"].exists(), "date-named audio must be protected too"
    assert "uuid-style-id" in report.protected_session_ids


def test_recent_audio_is_untouched_regardless(dirs) -> None:
    audio, store = dirs
    files = make_session(audio, "fresh", days_old=2)
    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()
    assert files["mic"].exists()
    assert report.deleted_files == []
    assert report.protected_session_ids == []


def test_backstop_deletes_very_old_protected_audio(dirs) -> None:
    """Protection cannot be unbounded or a crashed session pins gigabytes
    forever. After grace_multiplier x retention, it goes."""
    audio, store = dirs
    files = make_session(audio, "ancient", days_old=200)

    report = AudioRetentionManager(
        audio_dir=audio,
        transcript_store=store,
        retention_days=30,
        protected_grace_multiplier=3.0,  # backstop at 90 days
    ).cleanup_now()

    assert not files["mic"].exists()
    assert "ancient" in report.deleted_session_ids


def test_without_a_store_behaviour_is_unchanged(dirs) -> None:
    """No store wired (tests, minimal daemon) → we cannot know, so don't
    change the old behaviour."""
    audio, _store = dirs
    files = make_session(audio, "no-store", days_old=45)

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=None, retention_days=30
    ).cleanup_now()

    assert not files["mic"].exists()
    assert len(report.deleted_files) == 3


def test_audio_with_no_discoverable_session_id_is_protected(dirs) -> None:
    """A stray WAV with no metadata and no UUID name can't be matched to a
    transcript; deleting it is a guess we shouldn't make."""
    audio, store = dirs
    stray = audio / "recording.wav"
    _wav(stray)
    _age(stray, 45)

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert stray.exists()
    assert report.deleted_files == []
