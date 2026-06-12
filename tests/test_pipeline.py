"""Tests for src/pipeline.py — the shared menu-bar/daemon transcription wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline import (
    TranscriptionPipeline,
    build_pipeline,
    make_recorder_factory,
    make_transcription_handler,
)
from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import TranscriptProcessor
from src.transcription.whisper_engine import WhisperEngine


def _cfg(tmp_path: Path) -> Config:
    return Config(
        {
            "storage": {
                "audio_dir": str(tmp_path / "audio"),
                "transcript_dir": str(tmp_path / "transcripts"),
            },
            "audio": {"sample_rate": 16000, "channels": 1},
            "transcription": {"model": "tiny", "language": None},
        }
    )


def test_build_pipeline_wires_components(tmp_path: Path) -> None:
    pipeline = build_pipeline(_cfg(tmp_path))
    assert isinstance(pipeline.store, TranscriptStore)
    assert isinstance(pipeline.engine, WhisperEngine)
    assert isinstance(pipeline.processor, TranscriptProcessor)
    assert pipeline.engine.model_name == "tiny"
    assert pipeline.audio_dir == tmp_path / "audio"
    assert pipeline.store.root == tmp_path / "transcripts"
    pipeline.shutdown()  # must not raise


def test_make_recorder_factory_reads_audio_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    factory = make_recorder_factory(cfg)
    recorder = factory(cfg)
    # The recorder is built but never started — no audio stack touched.
    assert recorder._audio_dir == tmp_path / "audio"
    assert recorder._sample_rate == 16000
    assert recorder._channels == 1


class _FakeProcessor:
    """Records the process() call so we can assert handler plumbing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process(self, session, *, meeting=None, language=None, on_progress=None):
        self.calls.append(
            {"session": session, "meeting": meeting, "language": language}
        )
        if on_progress is not None:
            for pct in (0.0, 0.4, 1.7, 1.9, 50.0, 100.0):
                on_progress(pct)


def _fake_pipeline(tmp_path: Path) -> tuple[TranscriptionPipeline, _FakeProcessor]:
    fake = _FakeProcessor()
    store = TranscriptStore(tmp_path / "transcripts")
    engine = WhisperEngine(model_name="tiny", transcribe_fn=lambda *_a, **_k: {})
    pipeline = TranscriptionPipeline(
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        store=store,
        engine=engine,
        processor=fake,  # type: ignore[arg-type]
    )
    return pipeline, fake


def test_handler_builds_session_and_meeting_from_metadata(tmp_path: Path) -> None:
    pipeline, fake = _fake_pipeline(tmp_path)
    handler = make_transcription_handler(pipeline)

    handler(
        {
            "session_id": "sess-1",
            "mic_wav": "sess-1_mic.wav",
            "system_wav": None,
            "_language": "fr",
            "_meeting": {
                "title": "Standup",
                "app": "zoom.us",
                "participants": [{"name": "Alice", "email": "a@x.com"}],
                "meeting_link": "https://meet.example",
                "calendar_event_id": "evt-9",
            },
        }
    )

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["session"].session_id == "sess-1"
    assert call["session"].mic_wav == tmp_path / "audio" / "sess-1_mic.wav"
    assert call["meeting"].title == "Standup"
    assert call["meeting"].calendar_event_id == "evt-9"
    assert call["language"] == "fr"


def test_handler_forwards_integer_percent_once_each(tmp_path: Path) -> None:
    pipeline, _ = _fake_pipeline(tmp_path)
    seen: list[int] = []
    handler = make_transcription_handler(pipeline, on_progress_pct=seen.append)

    handler({"session_id": "s", "mic_wav": "s_mic.wav", "system_wav": None})

    # 0.4 collapses into 0; 1.7 and 1.9 collapse into a single 1.
    assert seen == [0, 1, 50, 100]


def test_handler_survives_progress_sink_exception(tmp_path: Path) -> None:
    pipeline, fake = _fake_pipeline(tmp_path)

    def bad_sink(_pct: int) -> None:
        raise RuntimeError("sink exploded")

    handler = make_transcription_handler(pipeline, on_progress_pct=bad_sink)
    handler({"session_id": "s", "mic_wav": "s_mic.wav", "system_wav": None})
    assert len(fake.calls) == 1  # processing completed despite sink errors


# ---------------------------------------------------------------------------
# Defer-while-in-call
# ---------------------------------------------------------------------------
def test_wait_returns_immediately_when_not_busy() -> None:
    from src.pipeline import wait_for_call_to_end

    waited = wait_for_call_to_end(is_busy=lambda: False, sleep=lambda _s: None)
    assert waited == 0.0


def test_wait_polls_until_call_ends_and_notifies_once() -> None:
    from src.pipeline import wait_for_call_to_end

    busy_answers = [True, True, True, False]
    notifications: list[str] = []
    waited = wait_for_call_to_end(
        is_busy=lambda: busy_answers.pop(0),
        on_first_wait=lambda: notifications.append("deferred"),
        poll_seconds=30.0,
        sleep=lambda _s: None,
    )
    assert waited == 90.0  # three busy polls
    assert notifications == ["deferred"]  # only one toast


def test_wait_gives_up_after_max_and_proceeds() -> None:
    from src.pipeline import wait_for_call_to_end

    waited = wait_for_call_to_end(
        is_busy=lambda: True,
        poll_seconds=30.0,
        max_seconds=120.0,
        sleep=lambda _s: None,
    )
    assert waited == 120.0  # bounded — transcription still happens


def test_wait_aborts_on_shutdown_signal() -> None:
    from src.pipeline import wait_for_call_to_end

    calls = {"n": 0}

    def should_abort() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    waited = wait_for_call_to_end(
        is_busy=lambda: True,
        should_abort=should_abort,
        poll_seconds=30.0,
        sleep=lambda _s: None,
    )
    assert waited == 60.0


def test_call_probe_disabled_when_mic_signal_invalid(tmp_path: Path) -> None:
    """mic_activation disabled (dictation app holds the mic) ⇒ never defer
    on the mic signal."""
    from src.pipeline import make_call_probe

    cfg = Config({"detection": {"mic_activation": {"enabled": False}}})
    probe = make_call_probe(cfg)
    assert probe() is False
