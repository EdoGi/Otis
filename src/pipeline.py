"""Shared transcription pipeline wiring.

Both entry points — the menu-bar app (``otis ui``) and the headless daemon
(``otis run``) — need the same plumbing: a TranscriptStore, a WhisperEngine,
a TranscriptProcessor, and a "transcription handler" that turns a recorder
metadata dict into a saved transcript. This module owns that wiring so the
two front ends stay in sync.

It deliberately lives outside ``src.main`` (which imports ``src.daemon`` at
module level) so the daemon can import it without a circular import.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.audio.recorder import DualStreamRecorder
from src.config import Config
from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import (
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
)
from src.transcription.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

TranscriptionHandler = Callable[[dict[str, Any]], None]


@dataclass
class TranscriptionPipeline:
    """The full post-recording stack, built once per process."""

    audio_dir: Path
    transcript_dir: Path
    store: TranscriptStore
    engine: WhisperEngine
    processor: TranscriptProcessor

    def shutdown(self) -> None:
        self.engine.shutdown()


def build_pipeline(cfg: Config) -> TranscriptionPipeline:
    """Construct store + engine + processor from config."""
    audio_dir = Path(cfg.get("storage", "audio_dir", default="~/Otis/audio")).expanduser()
    transcript_dir = Path(
        cfg.get("storage", "transcript_dir", default="~/Otis/transcripts")
    ).expanduser()
    store = TranscriptStore(transcript_dir)
    engine = WhisperEngine(
        model_name=str(cfg.get("transcription", "model", default="small")),
    )
    processor = TranscriptProcessor(
        engine=engine,
        store=store,
        audio_dir=audio_dir,
        model_name=engine.model_name,
        suggest_titles=bool(cfg.get("transcription", "suggest_titles", default=True)),
    )
    return TranscriptionPipeline(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        store=store,
        engine=engine,
        processor=processor,
    )


def make_recorder_factory(cfg: Config) -> Callable[[Config], DualStreamRecorder]:
    """Factory-of-factories: one recorder per recording session."""
    audio_dir = Path(cfg.get("storage", "audio_dir", default="~/Otis/audio")).expanduser()

    def recorder_factory(_cfg: Config) -> DualStreamRecorder:
        return DualStreamRecorder(
            audio_dir=audio_dir,
            sample_rate=int(_cfg.get("audio", "sample_rate", default=16000)),
            channels=int(_cfg.get("audio", "channels", default=1)),
            mic_device=_cfg.get("audio", "mic_device"),
            system_device=_cfg.get("audio", "system_audio_device", default="BlackHole 2ch"),
        )

    return recorder_factory


def make_transcription_handler(
    pipeline: TranscriptionPipeline,
    *,
    on_progress_pct: Callable[[int], None] | None = None,
) -> TranscriptionHandler:
    """Build the handler that the UI/daemon call with recorder metadata.

    The handler is synchronous — callers run it on their own worker thread
    so their PROCESSING state stays coherent. Progress is logged at every
    integer-percent change (the estimator ticks every 0.5 s, so this keeps
    the log readable) and forwarded to ``on_progress_pct`` when provided.
    """

    def transcription_handler(metadata: dict[str, Any]) -> None:
        session = RecordingSession.from_recorder_metadata(
            metadata, audio_dir=pipeline.audio_dir
        )
        meeting_dict = metadata.get("_meeting") or {}
        meeting = MeetingSnapshot(
            title=meeting_dict.get("title"),
            app=meeting_dict.get("app"),
            participants=list(meeting_dict.get("participants") or []),
            meeting_link=meeting_dict.get("meeting_link"),
            calendar_event_id=meeting_dict.get("calendar_event_id"),
        )
        language = metadata.get("_language")

        last_logged_pct = [-1]
        sid = metadata.get("session_id") or "unknown"

        def on_progress(pct: float) -> None:
            current = int(pct)
            if current != last_logged_pct[0]:
                last_logged_pct[0] = current
                logger.info("Transcription progress: %d%% (session=%s)", current, sid)
                if on_progress_pct is not None:
                    try:
                        on_progress_pct(current)
                    except Exception:
                        logger.exception("on_progress_pct sink raised")

        pipeline.processor.process(
            session, meeting=meeting, language=language, on_progress=on_progress,
        )

    return transcription_handler


__all__ = [
    "TranscriptionHandler",
    "TranscriptionPipeline",
    "build_pipeline",
    "make_recorder_factory",
    "make_transcription_handler",
]
