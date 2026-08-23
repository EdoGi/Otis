"""Both front ends must resume work a previous process never finished.

Audit E8. The menu bar leaves every transcription on a daemon thread, so a
quit discards it; ``otis run`` additionally had no orphan scan, so a crash
between ``recorder.stop()`` and the handler starting left WAVs that nothing
would ever look at again.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from src.config import Config
from src.daemon import OtisDaemon


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return path


class FakePipeline:
    def __init__(self, tmp_path: Path) -> None:
        from src.storage.transcript_store import TranscriptStore

        self.audio_dir = tmp_path / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.store = TranscriptStore(tmp_path / "transcripts")
        self.processor = None

    def shutdown(self) -> None:
        pass


def _config(tmp_path: Path) -> Config:
    return Config({
        "app": {"working_days": [0, 1, 2, 3, 4, 5, 6],
                "working_hours": {"start": "00:00", "end": "23:59"}},
        "transcription": {"model": "small", "defer_while_in_call": False},
        "storage": {"audio_dir": str(tmp_path / "audio"),
                    "transcript_dir": str(tmp_path / "transcripts"),
                    "queue_dir": str(tmp_path / "queue")},
        "detection": {"process_monitor": {"enabled": False},
                      "calendar": {"enabled": False}},
    })


def _daemon(tmp_path: Path, handler_calls: list, *, fail: bool = False):
    pipeline = FakePipeline(tmp_path)
    daemon = OtisDaemon(
        _config(tmp_path),
        notify=lambda _m: None,
        recorder_factory=lambda _c: None,
        transcription_pipeline=pipeline,  # type: ignore[arg-type]
    )

    def handler(metadata):
        handler_calls.append(metadata)
        if fail:
            raise RuntimeError("whisper exploded")

    daemon._transcription_handler = handler
    return daemon


def _join_workers(daemon, timeout: float = 5.0) -> None:
    for t in list(daemon._transcribe_threads):
        t.join(timeout)


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------
def test_the_daemon_resumes_a_queued_job(tmp_path: Path) -> None:
    mic = _wav(tmp_path / "audio" / "sess-a_mic.wav")
    calls: list = []
    daemon = _daemon(tmp_path, calls)
    daemon._queue.enqueue({"session_id": "sess-a", "mic_wav": str(mic)})

    daemon._recover_unfinished_work()
    _join_workers(daemon)

    assert [c["session_id"] for c in calls] == ["sess-a"]
    assert daemon._queue.pending() == [], "resolved once the handler returned"


def test_the_daemon_finds_an_orphan_that_never_reached_the_handler(
    tmp_path: Path
) -> None:
    """A crash between recorder.stop() and transcription starting."""
    audio = tmp_path / "audio"
    _wav(audio / "sess-orphan_mic.wav")
    _wav(audio / "sess-orphan_system.wav")
    (audio / "sess-orphan_metadata.json").write_text(
        json.dumps({"session_id": "sess-orphan", "mic_wav": "sess-orphan_mic.wav"}),
        encoding="utf-8",
    )
    calls: list = []
    daemon = _daemon(tmp_path, calls)

    daemon._recover_unfinished_work()
    _join_workers(daemon)

    assert [c["session_id"] for c in calls] == ["sess-orphan"]


def test_an_orphan_already_covered_by_the_queue_runs_once(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    mic = _wav(audio / "sess-a_mic.wav")
    (audio / "sess-a_metadata.json").write_text(
        json.dumps({"session_id": "sess-a", "mic_wav": "sess-a_mic.wav"}),
        encoding="utf-8",
    )
    calls: list = []
    daemon = _daemon(tmp_path, calls)
    daemon._queue.enqueue({"session_id": "sess-a", "mic_wav": str(mic)})

    daemon._recover_unfinished_work()
    _join_workers(daemon)

    assert len(calls) == 1, f"transcribed {len(calls)} times"


def test_a_transcribed_session_is_not_replayed(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    mic = _wav(audio / "sess-a_mic.wav")
    calls: list = []
    daemon = _daemon(tmp_path, calls)
    daemon._pipeline.store.save(
        {"id": "sess-a", "title": "Done", "date": "2026-08-01",
         "start_time": "10:00"}, "# Done\n\nbody")
    daemon._queue.enqueue({"session_id": "sess-a", "mic_wav": str(mic)})

    daemon._recover_unfinished_work()
    _join_workers(daemon)

    assert calls == [], "a duplicate transcript on every launch"


def test_a_normal_transcription_leaves_nothing_queued(tmp_path: Path) -> None:
    mic = _wav(tmp_path / "audio" / "sess-a_mic.wav")
    calls: list = []
    daemon = _daemon(tmp_path, calls)

    daemon._maybe_transcribe({"session_id": "sess-a", "mic_wav": str(mic)}, None)
    _join_workers(daemon)

    assert len(calls) == 1
    assert daemon._queue.pending() == []


def test_a_failed_transcription_also_leaves_nothing_queued(tmp_path: Path) -> None:
    """It wrote a placeholder the user can see; that is an outcome, not a loss."""
    mic = _wav(tmp_path / "audio" / "sess-a_mic.wav")
    calls: list = []
    daemon = _daemon(tmp_path, calls, fail=True)

    daemon._maybe_transcribe({"session_id": "sess-a", "mic_wav": str(mic)}, None)
    _join_workers(daemon)

    assert len(calls) == 1
    assert daemon._queue.pending() == []


def test_recovery_is_a_no_op_without_a_transcription_handler(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, [])
    daemon._transcription_handler = None
    daemon._recover_unfinished_work()  # must not raise


# ---------------------------------------------------------------------------
# The menu bar
# ---------------------------------------------------------------------------
def test_the_menu_bar_queues_and_resolves(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("rumps")
    from src.detection.detector import MeetingDetector
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    seen_while_running: list = []

    cfg = Config({
        "app": {"working_days": [0], "working_hours": {"start": "00:00",
                                                       "end": "23:59"}},
        "transcription": {"model": "small", "defer_while_in_call": False},
        "storage": {"audio_dir": str(tmp_path / "audio"),
                    "transcript_dir": str(tmp_path / "transcripts")},
        "detection": {"process_monitor": {"whitelisted_apps": []}},
    })
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )

    def handler(_metadata):
        # The whole point: visible on disk while the work is in flight.
        seen_while_running.extend(j.session_id for j in bar._queue.pending())

    bar._transcribe = handler
    bar._run_transcription({"session_id": "sess-a", "mic_wav": "x.wav",
                            "_meeting": {"title": None}, "_language": None})

    assert seen_while_running == ["sess-a"]
    assert bar._queue.pending() == []


def test_the_menu_bar_queue_lives_next_to_the_config(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.detection.detector import MeetingDetector
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    cfg = Config({"storage": {"audio_dir": str(tmp_path / "a"),
                              "transcript_dir": str(tmp_path / "t")}})
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "cfg" / "config.yaml",
    )
    assert bar._queue.directory == tmp_path / "cfg" / "queue"


def test_run_itself_recovers_before_it_starts_watching(tmp_path: Path) -> None:
    """Wired into run(), not merely implemented.

    Calling _recover_unfinished_work() directly in a test passes just as
    happily when run() never calls it.
    """
    mic = _wav(tmp_path / "audio" / "sess-a_mic.wav")
    calls: list = []
    daemon = _daemon(tmp_path, calls)
    daemon._queue.enqueue({"session_id": "sess-a", "mic_wav": str(mic)})

    daemon.request_stop()  # run() exits as soon as it reaches the wait
    assert daemon.run() == 0
    _join_workers(daemon)

    assert [c["session_id"] for c in calls] == ["sess-a"]
