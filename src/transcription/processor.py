"""Post-recording transcription pipeline.

Takes a :class:`DualStreamRecorder` session (two WAV files + metadata JSON)
and produces a single Markdown transcript with both speakers interleaved on
one timeline. Runs in a background thread so the menu-bar UI stays responsive.

Pipeline
--------
1. **Read metadata** ``{session_id}_metadata.json`` for the
   ``mic_start_monotonic`` / ``system_start_monotonic`` anchors.
2. **Transcribe mic** stream (0–45 % progress).
3. **Transcribe system** stream (45–90 % progress) if it exists; fallback to
   mic-only if BlackHole was unconfigured.
4. **Align**: shift system segments by
   ``offset = system_start - mic_start`` so both tracks live on the mic
   timeline.
5. **Echo dedup**: drop system segments that mirror a mic segment within
   ±1 s and ≥80 % word-set overlap (mic picked up speaker audio).
6. **Interleave** chronologically and mark concurrent segments with
   ``[overlap]``.
7. **Render** Markdown with a YAML frontmatter block.
8. **Save** via :class:`TranscriptStore` and rename audio files into
   ``{audio_dir}/YYYY/MM/YYYY-MM-DD_HHMM_*.wav``.

The merge is implemented as small pure helpers (``apply_offset``,
``deduplicate_echo``, ``interleave_with_overlaps``, ``render_markdown_body``)
so each can be unit-tested without touching mlx-whisper.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.storage.transcript_store import TranscriptStore, slugify
from src.transcription.titling import suggest_title
from src.transcription.whisper_engine import (
    Segment,
    TranscriptionResult,
    WhisperEngine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (exposed at module level so tests can patch them)
# ---------------------------------------------------------------------------
ECHO_TIME_WINDOW_SECONDS = 1.0
ECHO_WORD_OVERLAP_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class RecordingSession:
    """Everything we know about a finished recording, from the recorder.

    The recorder dict (output of ``DualStreamRecorder.stop()``) can be turned
    into one of these via :meth:`from_recorder_metadata`.
    """

    session_id: str
    audio_dir: Path
    mic_wav: Path
    system_wav: Path | None
    metadata_path: Path
    mic_start_monotonic: float | None = None
    system_start_monotonic: float | None = None
    start_wall_clock: datetime | None = None
    sample_rate: int = 16000
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_recorder_metadata(
        cls,
        recorder_metadata: dict[str, Any],
        *,
        audio_dir: str | Path,
    ) -> "RecordingSession":
        audio_dir = Path(audio_dir).expanduser()
        session_id = str(recorder_metadata["session_id"])
        mic_name = recorder_metadata.get("mic_wav") or f"{session_id}_mic.wav"
        sys_name = recorder_metadata.get("system_wav")
        meta_path = audio_dir / f"{session_id}_metadata.json"
        return cls(
            session_id=session_id,
            audio_dir=audio_dir,
            mic_wav=audio_dir / mic_name,
            system_wav=(audio_dir / sys_name) if sys_name else None,
            metadata_path=meta_path,
            mic_start_monotonic=recorder_metadata.get("mic_start_monotonic"),
            system_start_monotonic=recorder_metadata.get("system_start_monotonic"),
            start_wall_clock=_parse_wall_clock(recorder_metadata.get("start_wall_clock")),
            sample_rate=int(recorder_metadata.get("sample_rate") or 16000),
            raw_metadata=recorder_metadata,
        )


@dataclass
class MeetingSnapshot:
    """Calendar / detection context the menu bar passes to us at Stop time."""

    title: str | None = None
    app: str | None = None
    participants: list[dict[str, str]] = field(default_factory=list)
    meeting_link: str | None = None
    calendar_event_id: str | None = None


@dataclass
class ProcessingResult:
    transcript_path: Path
    metadata: dict[str, Any]
    detected_language: str | None
    body: str
    mic_segments: int
    system_segments: int
    echo_dropped: int


ProgressCallback = Callable[[float], None]
CompleteCallback = Callable[[ProcessingResult], None]


# ============================================================================
# TranscriptProcessor
# ============================================================================
class TranscriptProcessor:
    """Owns the full post-recording pipeline.

    Parameters
    ----------
    engine:
        :class:`WhisperEngine` (or anything with a compatible ``transcribe``).
    store:
        :class:`TranscriptStore`. The processor calls ``store.save`` once at
        the end.
    audio_dir:
        Where the recorder writes WAV files. The processor renames them into
        ``audio_dir/YYYY/MM/`` after a successful transcription.
    model_name:
        Stamped into the transcript frontmatter so the user can tell which
        model produced it. Defaults to ``engine.model_name``.
    """

    def __init__(
        self,
        *,
        engine: WhisperEngine,
        store: TranscriptStore,
        audio_dir: str | Path,
        model_name: str | None = None,
        dedup_echoes: bool = False,
        suggest_titles: bool = True,
    ) -> None:
        self._engine = engine
        self._store = store
        self._audio_dir = Path(audio_dir).expanduser()
        self._model_name = model_name or engine.model_name
        # When a recording has no calendar title, mine the transcript for a
        # descriptive one ("Onboarding with Acme") instead of saving every
        # ad-hoc session as "Ad-hoc Recording". Local heuristic only.
        self._suggest_titles = suggest_titles
        # Echo dedup off by default. Real meetings often have both streams
        # carrying the other person's voice (mic picks up speakers / earphone
        # leak), and our heuristic was preferring the mic copy and dropping
        # the system one — labeling the other speaker as "Me:". Keeping both
        # tracks unfiltered is the safer default; ``[overlap]`` markers in
        # the rendered Markdown still show when concurrent segments happen.
        self._dedup_echoes = dedup_echoes

    # ------------------------------------------------------------------ sync
    def process(
        self,
        session: RecordingSession,
        *,
        meeting: MeetingSnapshot | None = None,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Run the full pipeline synchronously. Returns the final result."""
        meeting = meeting or MeetingSnapshot()
        progress = on_progress or (lambda _p: None)

        # Phase 1 — read metadata file.
        try:
            file_meta = json.loads(self._read_text(session.metadata_path))
        except Exception:
            logger.warning(
                "Could not read metadata file %s; using in-memory only.",
                session.metadata_path,
            )
            file_meta = {}
        merged_meta = {**file_meta, **session.raw_metadata}
        mic_anchor = (
            merged_meta.get("mic_start_monotonic")
            if isinstance(merged_meta.get("mic_start_monotonic"), (int, float))
            else session.mic_start_monotonic
        )
        sys_anchor = (
            merged_meta.get("system_start_monotonic")
            if isinstance(merged_meta.get("system_start_monotonic"), (int, float))
            else session.system_start_monotonic
        )

        # Phase 2 — mic transcription (0-45 %).
        progress(1.0)
        mic_result = self._engine.transcribe(
            session.mic_wav,
            language=language,
            on_progress=lambda p, base=0.0, span=45.0: progress(base + (p / 100.0) * span),
        )
        progress(45.0)

        # Phase 3 — system transcription (45-90 %).
        sys_result: TranscriptionResult | None = None
        if session.system_wav is not None and session.system_wav.exists() and session.system_wav.stat().st_size > 0:
            sys_result = self._engine.transcribe(
                session.system_wav,
                language=language,
                on_progress=lambda p, base=45.0, span=45.0: progress(base + (p / 100.0) * span),
            )
        progress(90.0)

        detected_language = (
            mic_result.detected_language or (sys_result.detected_language if sys_result else None)
        )

        # Phase 4 — align + dedup + interleave.
        sys_segments = sys_result.segments if sys_result is not None else []
        offset = _compute_offset(mic_anchor, sys_anchor)
        sys_segments = apply_offset(sys_segments, offset)
        if self._dedup_echoes:
            deduped_sys, dropped = deduplicate_echo(sys_segments, mic_result.segments)
        else:
            deduped_sys, dropped = sys_segments, 0

        labelled_mic = [_LabelledSegment(s, "Me") for s in mic_result.segments]
        labelled_sys = [_LabelledSegment(s, "Participant") for s in deduped_sys]
        merged = interleave_with_overlaps(labelled_mic + labelled_sys)

        # Phase 5 — frontmatter + body.
        wall_start = session.start_wall_clock or datetime.now(timezone.utc)
        local_start = wall_start.astimezone()
        # End time = start + the larger of the two transcription extents.
        last_t = max((s.segment.end for s in labelled_mic), default=0.0)
        last_t_sys = max((s.segment.end for s in labelled_sys), default=0.0)
        duration_seconds = max(last_t, last_t_sys)
        local_end = local_start + timedelta(seconds=duration_seconds)

        # Where the audio files will live after rename — relative to audio_dir.
        relocated_relative = self._relocated_audio_paths(session, local_start)

        # Ad-hoc recordings (no calendar title) get a transcript-derived
        # title; the filename slug — and so the saved name — follows it,
        # keeping the date+time prefix: 2026-06-12_1430_onboarding-with-acme.md
        suggested_title: str | None = None
        if self._suggest_titles and not (meeting.title or "").strip():
            transcript_text = " ".join(ls.segment.text for ls, _ in merged)
            suggested_title = suggest_title(
                transcript_text, participants=meeting.participants
            )
            if suggested_title:
                logger.info("Suggested title for ad-hoc session: %r", suggested_title)

        frontmatter = self._build_frontmatter(
            session=session,
            meeting=meeting,
            local_start=local_start,
            local_end=local_end,
            duration_seconds=duration_seconds,
            language=detected_language or language,
            audio_relative_paths=relocated_relative,
            suggested_title=suggested_title,
        )

        body = render_markdown_body(
            title=frontmatter["title"],
            date_str=frontmatter["date"],
            segments=merged,
        )

        # Phase 6 — write transcript, rename audio, mark progress 100 %.
        transcript_path = self._store.save(frontmatter, body)
        self._relocate_audio(session, relocated_relative, frontmatter)
        progress(100.0)

        return ProcessingResult(
            transcript_path=transcript_path,
            metadata=frontmatter,
            detected_language=detected_language,
            body=body,
            mic_segments=len(labelled_mic),
            system_segments=len(labelled_sys),
            echo_dropped=dropped,
        )

    # ----------------------------------------------------------- async glue
    def process_async(
        self,
        session: RecordingSession,
        *,
        meeting: MeetingSnapshot | None = None,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
        on_complete: CompleteCallback | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> threading.Thread:
        """Same as :meth:`process` but runs on a daemon thread."""

        def _runner() -> None:
            try:
                result = self.process(
                    session, meeting=meeting, language=language, on_progress=on_progress
                )
            except Exception as exc:  # noqa: BLE001 — surface to UI
                logger.exception("Transcription failed for session %s", session.session_id)
                if on_error is not None:
                    on_error(exc)
                return
            if on_complete is not None:
                try:
                    on_complete(result)
                except Exception:
                    logger.exception("on_complete callback raised")

        t = threading.Thread(
            target=_runner,
            name=f"otis-transcribe-{session.session_id[:8]}",
            daemon=True,
        )
        t.start()
        return t

    # =====================================================================
    # Internals
    # =====================================================================
    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _build_frontmatter(
        self,
        *,
        session: RecordingSession,
        meeting: MeetingSnapshot,
        local_start: datetime,
        local_end: datetime,
        duration_seconds: float,
        language: str | None,
        audio_relative_paths: dict[str, str | None],
        suggested_title: str | None = None,
    ) -> dict[str, Any]:
        title = (meeting.title or "").strip() or suggested_title or "Ad-hoc Recording"
        return {
            "id": session.session_id,
            "title": title,
            "date": local_start.strftime("%Y-%m-%d"),
            "start_time": local_start.strftime("%H:%M"),
            "end_time": local_end.strftime("%H:%M"),
            "duration_minutes": int(round(duration_seconds / 60.0)),
            "language": language,
            "app": meeting.app,
            "participants": [_format_participant(p) for p in meeting.participants],
            "tags": [],
            "audio_files": {
                "mic": audio_relative_paths.get("mic"),
                "system": audio_relative_paths.get("system"),
            },
            "audio_available": True,
            "model": self._model_name,
        }

    def _relocated_audio_paths(
        self, session: RecordingSession, local_start: datetime,
    ) -> dict[str, str | None]:
        """Compute the post-rename relative paths under ``audio_dir``.

        Two recordings stopped within the same minute resolve to the same
        ``YYYY-MM-DD_HHMM`` prefix, and ``shutil.move`` would silently
        overwrite the first session's WAVs. Bump a numeric suffix until the
        whole destination set is free. A destination that already exists but
        *is* the session's own file (re-transcription of already-relocated
        audio) doesn't count as a collision — the move becomes a no-op.
        """
        date_str = local_start.strftime("%Y-%m-%d")
        time_compact = local_start.strftime("%H%M")
        rel_dir = f"{local_start.strftime('%Y')}/{local_start.strftime('%m')}"
        attempt = 1
        while True:
            base = f"{date_str}_{time_compact}" if attempt == 1 else (
                f"{date_str}_{time_compact}_{attempt}"
            )
            prefix = f"{rel_dir}/{base}"
            paths: dict[str, str | None] = {
                "mic": f"{prefix}_mic.wav",
                "system": f"{prefix}_system.wav" if session.system_wav else None,
            }
            if not self._relocation_collides(session, paths):
                return paths
            attempt += 1

    def _relocation_collides(
        self, session: RecordingSession, relative_paths: dict[str, str | None],
    ) -> bool:
        """True if any destination is taken by a file other than the source."""
        mic_rel = relative_paths["mic"]
        assert mic_rel is not None
        pairs: list[tuple[Path, str]] = [
            (session.mic_wav, mic_rel),
            (session.metadata_path, mic_rel.replace("_mic.wav", "_metadata.json")),
        ]
        if session.system_wav is not None and relative_paths.get("system"):
            pairs.append((session.system_wav, relative_paths["system"]))
        for src, rel in pairs:
            dst = self._audio_dir / rel
            if dst.exists() and dst.resolve() != src.resolve():
                return True
        return False

    def _relocate_audio(
        self,
        session: RecordingSession,
        relative_paths: dict[str, str | None],
        frontmatter: dict[str, Any],
    ) -> None:
        """Move the recorder UUID-named WAVs into the ``YYYY/MM/`` tree.

        Importantly: rewrite ``metadata.json`` so its ``mic_wav`` /
        ``system_wav`` / ``metadata_path`` fields point at the *new*
        locations. Earlier versions only renamed the file but left the
        contents pointing at the old UUID names, breaking retranscribe.
        """
        # Compute destinations up front so we can update metadata in-place.
        new_mic = self._audio_dir / relative_paths["mic"] if relative_paths.get("mic") else None
        new_sys = self._audio_dir / relative_paths["system"] if relative_paths.get("system") else None
        new_meta = (
            self._audio_dir / relative_paths["mic"].replace("_mic.wav", "_metadata.json")
            if relative_paths.get("mic")
            else None
        )

        # Rewrite the metadata file BEFORE moving it so the on-disk content is
        # already correct in its new home.
        if session.metadata_path.exists():
            try:
                meta = json.loads(session.metadata_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    if new_mic is not None:
                        meta["mic_wav"] = relative_paths["mic"]
                    meta["system_wav"] = relative_paths.get("system")
                    session.metadata_path.write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
            except Exception as exc:
                logger.warning(
                    "Could not rewrite metadata at %s: %s",
                    session.metadata_path,
                    exc,
                )

        moves: list[tuple[Path, Path]] = []
        if session.mic_wav.exists() and new_mic is not None:
            moves.append((session.mic_wav, new_mic))
        if session.system_wav is not None and session.system_wav.exists() and new_sys is not None:
            moves.append((session.system_wav, new_sys))
        if session.metadata_path.exists() and new_meta is not None:
            moves.append((session.metadata_path, new_meta))

        for src, dst in moves:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                logger.debug("Moved %s → %s", src, dst)
            except Exception as exc:
                logger.warning("Could not move %s → %s: %s", src, dst, exc)
                # If audio relocation fails, mark the transcript so the user
                # knows the WAV path in the frontmatter is wrong.
                self._store.update_frontmatter(
                    session.session_id, {"audio_available": False}
                )


# ============================================================================
# Pure helpers — testable without touching mlx-whisper or filesystem
# ============================================================================
@dataclass
class _LabelledSegment:
    segment: Segment
    speaker: str


def _compute_offset(mic_anchor: float | None, sys_anchor: float | None) -> float:
    """Return the seconds to add to system segments so they align with mic."""
    if mic_anchor is None or sys_anchor is None:
        return 0.0
    return float(sys_anchor) - float(mic_anchor)


def apply_offset(segments: list[Segment], offset: float) -> list[Segment]:
    """Return new segments with ``offset`` seconds added to start/end."""
    if not offset:
        return list(segments)
    return [
        Segment(start=s.start + offset, end=s.end + offset, text=s.text)
        for s in segments
    ]


def deduplicate_echo(
    system_segments: list[Segment],
    mic_segments: list[Segment],
    *,
    time_window: float = ECHO_TIME_WINDOW_SECONDS,
    overlap_threshold: float = ECHO_WORD_OVERLAP_THRESHOLD,
) -> tuple[list[Segment], int]:
    """Drop system segments that look like an echo of a mic segment.

    Returns ``(kept_system_segments, dropped_count)``.

    Heuristic: same logical speech if there's a mic segment whose start time
    is within ``time_window`` seconds AND whose word set overlaps the system
    segment's by ≥``overlap_threshold`` (intersection / max-set-size).
    """
    if not system_segments or not mic_segments:
        return list(system_segments), 0

    kept: list[Segment] = []
    dropped = 0
    mic_words_cache = [(s, _normalise_words(s.text)) for s in mic_segments]

    for sys_seg in system_segments:
        sys_words = _normalise_words(sys_seg.text)
        if not sys_words:
            kept.append(sys_seg)
            continue
        is_echo = False
        for mic_seg, mic_words in mic_words_cache:
            if abs(mic_seg.start - sys_seg.start) > time_window:
                continue
            if not mic_words:
                continue
            shared = len(sys_words & mic_words)
            ratio = shared / max(len(sys_words), len(mic_words))
            if ratio >= overlap_threshold:
                is_echo = True
                break
        if is_echo:
            dropped += 1
        else:
            kept.append(sys_seg)
    return kept, dropped


def interleave_with_overlaps(
    labelled: list[_LabelledSegment],
) -> list[tuple[_LabelledSegment, bool]]:
    """Sort labelled segments by start time and flag overlap edges.

    The flag is True for any segment whose start falls inside the previous
    segment's [start, end] window AND that previous segment is from a
    different speaker. The renderer inserts ``[overlap]`` before such items.
    """
    sorted_segs = sorted(labelled, key=lambda ls: ls.segment.start)
    out: list[tuple[_LabelledSegment, bool]] = []
    for i, ls in enumerate(sorted_segs):
        overlap = False
        if i > 0:
            prev = sorted_segs[i - 1]
            if (
                prev.speaker != ls.speaker
                and ls.segment.start < prev.segment.end
            ):
                overlap = True
        out.append((ls, overlap))
    return out


def render_markdown_body(
    *,
    title: str,
    date_str: str,
    segments: list[tuple[_LabelledSegment, bool]],
) -> str:
    """Compose the body of the transcript .md file."""
    lines: list[str] = [f"# {title} — {date_str}", "", "## Transcript", ""]
    if not segments:
        lines.append("_(no speech detected)_")
        return "\n".join(lines)
    for ls, overlap in segments:
        ts = _format_mmss(ls.segment.start)
        if overlap:
            lines.append("[overlap]")
        text = ls.segment.text.strip()
        if not text:
            continue
        lines.append(f"**[{ts}]** {ls.speaker}: {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# Small utilities
# ============================================================================
_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", flags=re.IGNORECASE)


def _normalise_words(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _format_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _parse_wall_clock(raw: Any) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        # ``datetime.isoformat`` accepts trailing-Z indirectly via fromisoformat
        # in Python 3.11+. Be conservative: try, fall back to None.
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _format_participant(p: dict[str, Any]) -> str:
    name = p.get("name") or ""
    email = p.get("email") or ""
    if name and email:
        return f"{name} <{email}>"
    return name or email or "unknown"


__all__ = [
    "ECHO_TIME_WINDOW_SECONDS",
    "ECHO_WORD_OVERLAP_THRESHOLD",
    "MeetingSnapshot",
    "ProcessingResult",
    "RecordingSession",
    "TranscriptProcessor",
    "apply_offset",
    "deduplicate_echo",
    "interleave_with_overlaps",
    "render_markdown_body",
    "slugify",
]
