"""Regression: crash recovery dropped every interrupted menu-bar transcription.

The recorder writes ``mic_wav`` as a bare file name (relative to the audio
dir), but ``recoverable()`` checked ``Path(mic_wav).exists()`` — against the
process's working directory, which is the repo when run.sh starts Otis. So
every resumed job looked audio-less and was deleted. Real case: the 84-minute
recording 029ff6ac from 2026-09-03 was interrupted at 48% of transcription;
the next launch logged "has no audio left; dropping." while both WAVs (161 MB
each) were still in ~/Otis/audio. The existing tests only ever enqueued
absolute paths, the one shape production never writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.storage.transcription_queue import TranscriptionQueue

SID = "029ff6ac-c227-43f9-b096-92e2284e08ae"


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audio"
    d.mkdir()
    (d / f"{SID}_mic.wav").write_bytes(b"RIFF")
    return d


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "repo"
    d.mkdir()
    monkeypatch.chdir(d)  # run.sh cds into the repo
    return d


def test_a_recorder_shaped_entry_is_recoverable(tmp_path, audio_dir, elsewhere) -> None:
    queue = TranscriptionQueue(tmp_path / "queue", audio_dir=audio_dir)
    queue.enqueue({"session_id": SID, "mic_wav": f"{SID}_mic.wav"})

    jobs = queue.recoverable(queue.pending())

    assert [j.session_id for j in jobs] == [SID]
    assert (tmp_path / "queue" / f"{SID}.json").exists(), "entry must survive"


def test_absolute_entries_still_work(tmp_path, audio_dir, elsewhere) -> None:
    queue = TranscriptionQueue(tmp_path / "queue", audio_dir=audio_dir)
    queue.enqueue({"session_id": SID, "mic_wav": str(audio_dir / f"{SID}_mic.wav")})
    assert [j.session_id for j in queue.recoverable(queue.pending())] == [SID]


def test_audio_that_is_really_gone_is_still_dropped(tmp_path, audio_dir, elsewhere) -> None:
    queue = TranscriptionQueue(tmp_path / "queue", audio_dir=audio_dir)
    queue.enqueue({"session_id": "gone", "mic_wav": "gone_mic.wav"})
    assert queue.recoverable(queue.pending()) == []
    assert not (tmp_path / "queue" / "gone.json").exists()


def test_the_menu_bar_queue_resolves_against_its_audio_dir(tmp_path, audio_dir, elsewhere) -> None:
    pytest.importorskip("rumps")
    from src.detection.detector import MeetingDetector
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    app = MenuBarApp(
        config=Config({"storage": {"audio_dir": str(audio_dir)}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=HealthRegistry(),
    )
    app._queue.enqueue({"session_id": SID, "mic_wav": f"{SID}_mic.wav"})
    assert [j.session_id for j in app._queue.recoverable(app._queue.pending())] == [SID]


def test_the_daemon_queue_resolves_against_the_configured_audio_dir(
    tmp_path, audio_dir, elsewhere
) -> None:
    from src.daemon import OtisDaemon
    from src.detection.detector import MeetingDetector

    cfg = Config({
        "storage": {"audio_dir": str(audio_dir), "queue_dir": str(tmp_path / "queue"),
                    "transcript_dir": str(tmp_path / "transcripts")},
        "detection": {"process_monitor": {"enabled": False}, "calendar": {"enabled": False}},
        "transcription": {"defer_while_in_call": False},
    })
    daemon = OtisDaemon(
        config=cfg, notify=lambda _m: None, recorder_factory=lambda _c: None,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
    )
    daemon._queue.enqueue({"session_id": SID, "mic_wav": f"{SID}_mic.wav"})
    assert [j.session_id for j in daemon._queue.recoverable(daemon._queue.pending())] == [SID]
