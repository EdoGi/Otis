"""Cover for audit finding E8 — a quit mid-transcription lost the work.

Every transcription worker is a ``daemon=True`` thread, so anything running
when the process goes away is gone. The menu bar could half-recover through
"Generate Transcript" if the user happened to look; ``otis run`` had no scan
at all, so a headless reboot lost every pending transcription silently until
retention deleted the audio 30 days later.

The queue records *in-flight* work only. A job is dropped once its handler
returns — success, or a failure placeholder the user can see are both
outcomes. Only a process that died mid-run leaves an entry behind.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from src.storage.transcript_store import TranscriptStore
from src.storage.transcription_queue import QueuedJob, TranscriptionQueue


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return path


@pytest.fixture()
def queue(tmp_path: Path) -> TranscriptionQueue:
    return TranscriptionQueue(tmp_path / "queue")


def meta(sid: str, mic: Path | None = None) -> dict:
    return {"session_id": sid, "mic_wav": str(mic) if mic else None,
            "system_wav": None, "_language": "en"}


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------
def test_a_queued_job_survives_a_new_process(queue, tmp_path) -> None:
    mic = _wav(tmp_path / "a_mic.wav")
    queue.enqueue(meta("sess-a", mic))

    fresh = TranscriptionQueue(tmp_path / "queue")
    jobs = fresh.pending()

    assert [j.session_id for j in jobs] == ["sess-a"]
    assert jobs[0].metadata["_language"] == "en"
    assert jobs[0].mic_path == mic


def test_resolving_removes_the_job(queue, tmp_path) -> None:
    queue.enqueue(meta("sess-a", _wav(tmp_path / "a_mic.wav")))
    assert queue.resolve("sess-a") is True
    assert queue.pending() == []
    assert queue.resolve("sess-a") is False


def test_jobs_come_back_oldest_first(queue, tmp_path) -> None:
    """Filenames sort c/a/b; queue order must follow when they were queued."""
    for sid in ("c", "a", "b"):
        queue.enqueue(meta(f"sess-{sid}", _wav(tmp_path / f"{sid}_mic.wav")))

    jobs = queue.pending()

    assert [j.session_id for j in jobs] == ["sess-c", "sess-a", "sess-b"]
    assert [j.queued_at for j in jobs] == sorted(j.queued_at for j in jobs)


def test_requeueing_the_same_session_does_not_duplicate(queue, tmp_path) -> None:
    mic = _wav(tmp_path / "a_mic.wav")
    queue.enqueue(meta("sess-a", mic))
    queue.enqueue(meta("sess-a", mic))
    assert len(queue.pending()) == 1


def test_a_job_with_no_session_id_is_not_queued(queue) -> None:
    assert queue.enqueue({"mic_wav": "x.wav"}) is None
    assert queue.pending() == []


# ---------------------------------------------------------------------------
# What is worth replaying
# ---------------------------------------------------------------------------
def test_a_job_whose_audio_is_gone_is_dropped(queue, tmp_path) -> None:
    mic = _wav(tmp_path / "a_mic.wav")
    queue.enqueue(meta("sess-a", mic))
    mic.unlink()  # retention got there first

    assert queue.recoverable(queue.pending()) == []
    assert queue.pending() == [], "and it is not left to be re-checked forever"


def test_a_job_that_already_has_a_transcript_is_dropped(queue, tmp_path) -> None:
    """The process finished the work, then died before it could say so.

    Replaying it would write a duplicate transcript on every single launch.
    """
    store = TranscriptStore(tmp_path / "transcripts")
    store.save({"id": "sess-a", "title": "Done", "date": "2026-08-01",
                "start_time": "10:00"}, "# Done\n\nbody")
    queue.enqueue(meta("sess-a", _wav(tmp_path / "a_mic.wav")))

    assert queue.recoverable(queue.pending(), store=store) == []


def test_a_failure_placeholder_does_not_count_as_done(queue, tmp_path) -> None:
    """It tells the user to retry, so the audio still has work left in it."""
    store = TranscriptStore(tmp_path / "transcripts")
    store.save({"id": "sess-a", "title": "Failed", "date": "2026-08-01",
                "start_time": "10:00", "status": "failed"}, "# failed\n\nretry")
    queue.enqueue(meta("sess-a", _wav(tmp_path / "a_mic.wav")))

    assert [j.session_id for j in
            queue.recoverable(queue.pending(), store=store)] == ["sess-a"]


def test_a_recoverable_job_is_left_queued(queue, tmp_path) -> None:
    queue.enqueue(meta("sess-a", _wav(tmp_path / "a_mic.wav")))
    assert len(queue.recoverable(queue.pending())) == 1
    assert len(queue.pending()) == 1, "still pending until its handler returns"


# ---------------------------------------------------------------------------
# Corruption must not wedge startup
# ---------------------------------------------------------------------------
def test_an_unreadable_entry_is_dropped_not_fatal(queue, tmp_path) -> None:
    good = _wav(tmp_path / "a_mic.wav")
    queue.enqueue(meta("sess-a", good))
    (queue.directory / "garbage.json").write_text("{not json", encoding="utf-8")

    jobs = queue.pending()

    assert [j.session_id for j in jobs] == ["sess-a"]
    assert not (queue.directory / "garbage.json").exists()


def test_an_entry_missing_its_metadata_is_dropped(queue) -> None:
    queue.directory.mkdir(parents=True, exist_ok=True)
    (queue.directory / "half.json").write_text(
        json.dumps({"session_id": "sess-x"}), encoding="utf-8")
    assert queue.pending() == []


def test_a_missing_queue_directory_is_empty_not_an_error(tmp_path) -> None:
    assert TranscriptionQueue(tmp_path / "never-made").pending() == []


def test_a_session_id_cannot_escape_the_queue_directory(queue, tmp_path) -> None:
    queue.enqueue({"session_id": "../../etc/passwd", "mic_wav": "x.wav"})
    written = list(queue.directory.glob("*.json"))
    assert len(written) == 1
    assert written[0].parent == queue.directory


def test_metadata_holding_paths_is_still_writable(queue, tmp_path) -> None:
    """Callers decorate metadata before it reaches us; a Path must not lose it."""
    mic = _wav(tmp_path / "a_mic.wav")
    assert queue.enqueue({"session_id": "sess-a", "mic_wav": mic,
                          "_meeting": {"title": None}}) is not None
    assert queue.pending()[0].metadata["mic_wav"] == str(mic)


def test_queued_job_with_no_mic_has_no_path() -> None:
    job = QueuedJob(session_id="s", metadata={}, path=Path("x"), queued_at="")
    assert job.mic_path is None
