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
5. **Echo dedup**: drop *mic* segments that mirror a *system* segment within
   ±1 s and >=80 % word-set overlap. BlackHole captures system output, which
   cannot contain your microphone — so a match means the mic heard the
   speakers, and the system copy is the direct capture worth keeping.
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
from dataclasses import dataclass, field, replace
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

#: Ceiling on one afconvert compression. A 2-hour stream encodes in ~1 s.
ENCODE_TIMEOUT_SECONDS = 600.0


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
        dedup_echoes: bool = True,
        suggest_titles: bool = True,
        archive_format: str = "flac",
    ) -> None:
        self._engine = engine
        self._store = store
        self._audio_dir = Path(audio_dir).expanduser()
        # None means "whatever the engine is running now" — the model can be
        # changed from the menu mid-session, and frontmatter must record what
        # actually produced the transcript.
        self._model_name_override = model_name
        # When a recording has no calendar title, mine the transcript for a
        # descriptive one ("Onboarding with Acme") instead of saving every
        # ad-hoc session as "Ad-hoc Recording". Local heuristic only.
        self._suggest_titles = suggest_titles
        # Echo dedup, on by default now that it runs in the right direction.
        # Without it, a call held on speakers transcribes every sentence twice
        # — once from the far end via BlackHole and once from the mic hearing
        # the speakers — which makes the transcript unreadable. Verified on a
        # real recording: 25 of 66 lines were duplicated.
        self._dedup_echoes = dedup_echoes
        # Recordings are archived compressed. 16-bit PCM at 16 kHz costs
        # 230 MB per meeting-hour across the two streams; FLAC is lossless and
        # roughly halves speech, while a stream that captured nothing costs
        # essentially zero (measured 721x on a real 85 MB silent stream).
        # Set to "wav" to keep the raw capture.
        fmt = str(archive_format or "flac").lower().lstrip(".")
        if fmt not in ("flac", "wav"):
            logger.warning("Unknown archive format %r; keeping WAV.", archive_format)
            fmt = "wav"
        self._archive_format = fmt
        self._archive_suffix = f".{fmt}"

    # ------------------------------------------------------------------ sync
    @property
    def _model_name(self) -> str:
        """The model that will actually run, unless a caller pinned one."""
        return self._model_name_override or self._engine.model_name

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
        #
        # Two full Whisper passes per meeting is the honest cost, not an
        # oversight. Audit item 31 proposed halving it; both routes were
        # measured on real recordings and both failed.
        #
        # Merging the streams into one mono track saves 64% of the time and
        # loses half the meeting. Measured on two genuinely independent
        # speakers overlapping in time: the merged pass recovered 36% of the
        # near-end speaker's words and 56% of the far end's, 263 words against
        # 491 transcribed separately. Whisper cannot separate two voices from
        # one channel. (An earlier check looked fine only because that session
        # was a video on speakers, so the streams were 94.5% identical and
        # there was no second speaker to lose.)
        #
        # Gating out silence keeps the streams separate, so attribution is
        # safe, but the saving is not there. On a real 10-minute recording,
        # dropping 9% of the audio saved 2.4% of wall time and changed 9.3% of
        # the words; dropping 23% ran 8.4% *slower*. Whisper's decoder runs
        # until it emits an end-of-segment, so compressing silence makes each
        # remaining 30 s window denser and costlier — fewer windows, more
        # tokens per window, and the two cancel.
        #
        # A stream that is genuinely silent already costs nothing: the engine
        # skips it on the RMS pre-check (SILENCE_RMS_THRESHOLD).
        progress(1.0)
        mic_result = self._engine.transcribe(
            session.mic_wav,
            language=language,
            stream="mic",
            on_progress=_progress_window(progress, 0.0, 45.0),
        )
        progress(45.0)

        # Phase 3 — system transcription (45-90 %).
        sys_result: TranscriptionResult | None = None
        if session.system_wav is not None and session.system_wav.exists() and session.system_wav.stat().st_size > 0:
            sys_result = self._engine.transcribe(
                session.system_wav,
                language=language,
                stream="system",
                on_progress=_progress_window(progress, 45.0, 45.0),
            )
        progress(90.0)

        detected_language = (
            mic_result.detected_language or (sys_result.detected_language if sys_result else None)
        )

        # Phase 4 — align + dedup + interleave.
        sys_segments = sys_result.segments if sys_result is not None else []
        offset = _compute_offset(mic_anchor, sys_anchor)
        sys_segments = apply_offset(sys_segments, offset)
        # Which copy is the echo? BlackHole captures system *output*, which
        # cannot contain your microphone — so when the same words appear in
        # both streams, the mic picked them up off the speakers. The system
        # copy is the direct digital capture and the one to keep.
        #
        # The original code deduped the other way and dropped the system copy,
        # which labelled the other person as "Me:". That is why it shipped
        # disabled rather than fixed.
        mic_segments = mic_result.segments
        dropped = 0
        if self._dedup_echoes:
            mic_segments, dropped = deduplicate_echo(mic_segments, sys_segments)
        deduped_sys = sys_segments

        # Re-stamp both streams onto the wall clock. Pause drops frames rather
        # than writing silence, so up to this point every timestamp is an
        # offset into a file that is shorter than the meeting was. Dedup and
        # alignment above both compare stream-to-stream, so they run first on
        # the raw audio timeline; the shift is the last thing that touches
        # timestamps before they are rendered.
        pause_timeline = PauseTimeline(merged_meta.get("pauses"))
        if pause_timeline:
            logger.info(
                "Session %s paused for %.0fs total; shifting transcript timestamps.",
                session.session_id, pause_timeline.total_pause_seconds,
            )
        labelled_mic = [
            _LabelledSegment(s, "Me") for s in pause_timeline.shift_segments(mic_segments)
        ]
        labelled_sys = [
            _LabelledSegment(s, "Participant") for s in pause_timeline.shift_segments(deduped_sys)
        ]
        merged = interleave_with_overlaps(labelled_mic + labelled_sys)

        # Phase 5 — frontmatter + body.
        wall_start = session.start_wall_clock or datetime.now(timezone.utc)
        local_start = wall_start.astimezone()
        # End time = start + the larger of the two transcription extents.
        last_t = max((s.segment.end for s in labelled_mic), default=0.0)
        last_t_sys = max((s.segment.end for s in labelled_sys), default=0.0)
        duration_seconds = max(last_t, last_t_sys)
        # A meeting where Whisper found nothing — everyone on mute, a failed
        # capture — has no segments, and taking the last segment's end then
        # reported a 45-minute recording as 0 minutes, in the frontmatter, the
        # web UI and any time accounting. The audio knows how long it was.
        if duration_seconds <= 0.0:
            duration_seconds = _recorded_seconds(merged_meta)
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
            system_audio=_system_audio_state(sys_result),
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
        system_audio: str = "ok",
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
            # "ok" | "silent" | "missing". A meeting recorded with a silent
            # system stream is only half a transcript; saying so in the file
            # means the loss is visible where the transcript is read, not only
            # in a log line nobody opens.
            "system_audio": system_audio,
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
            ext = self._archive_suffix
            paths: dict[str, str | None] = {
                "mic": f"{prefix}_mic{ext}",
                "system": f"{prefix}_system{ext}" if session.system_wav else None,
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
            # Derive the metadata name from the stem, not by patching a
            # literal ".wav" — the archive extension is configurable now, so a
            # substring replace silently stopped matching and made every
            # re-transcription look like a collision, bumping to _2.
            (session.metadata_path, _metadata_name_for(mic_rel)),
        ]
        sys_rel = relative_paths.get("system")
        if session.system_wav is not None and sys_rel:
            pairs.append((session.system_wav, sys_rel))
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
        """Archive the recorder's WAVs into ``YYYY/MM/``, compressing en route.

        Order matters. This used to rewrite ``metadata.json`` to the new paths
        *before* attempting the moves, so a failed move left the metadata
        pointing at files that never arrived — and retranscribe and orphan
        discovery both read that metadata. Now nothing is rewritten until the
        files are actually in place, and the metadata records where they really
        ended up.

        A transcode failure is never fatal: the original WAV is moved instead,
        so the recording survives even if compression doesn't.
        """
        mic_rel_path = relative_paths.get("mic")
        sys_rel_path = relative_paths.get("system")
        new_mic = self._audio_dir / mic_rel_path if mic_rel_path else None
        new_sys = self._audio_dir / sys_rel_path if sys_rel_path else None
        new_meta = (
            self._audio_dir / _metadata_name_for(mic_rel_path) if mic_rel_path else None
        )

        landed: dict[str, str | None] = {"mic": None, "system": None}
        failures = 0
        for key, src, dst in (
            ("mic", session.mic_wav, new_mic),
            ("system", session.system_wav, new_sys),
        ):
            if src is None or dst is None or not src.exists():
                continue
            rel = self._archive_one(src, dst)
            if rel is None:
                failures += 1
            else:
                landed[key] = rel

        # Only now is it safe to say where things are.
        if session.metadata_path.exists() and new_meta is not None:
            try:
                meta = json.loads(session.metadata_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    meta["mic_wav"] = landed["mic"]
                    meta["system_wav"] = landed["system"]
                    session.metadata_path.write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
                new_meta.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(session.metadata_path), str(new_meta))
            except Exception as exc:
                logger.warning(
                    "Could not finalise metadata at %s: %s", session.metadata_path, exc
                )

        updates: dict[str, Any] = {}
        if landed != {k: relative_paths.get(k) for k in ("mic", "system")}:
            updates["audio_files"] = {"mic": landed["mic"], "system": landed["system"]}
        if failures:
            # The transcript must not advertise audio that isn't there.
            updates["audio_available"] = bool(landed["mic"] or landed["system"])
        if updates:
            # Keep the caller's copy in step with what actually landed, so the
            # returned result and the saved transcript can't disagree.
            frontmatter.update(updates)
            try:
                self._store.update_frontmatter(session.session_id, updates)
            except Exception:
                logger.exception("Could not correct transcript audio paths")

    def _archive_one(self, src: Path, dst: Path) -> str | None:
        """Put one capture file at ``dst``. Returns its path relative to audio_dir.

        Compresses when the archive format asks for it and the source isn't
        already in that format. Falls back to a plain move on any failure —
        losing compression is a rounding error, losing the recording is not.
        """
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.suffix.lower() != src.suffix.lower():
                if encode_audio(src, dst):
                    src.unlink(missing_ok=True)
                    logger.info(
                        "Archived %s -> %s (%.1f MB -> %.1f MB)",
                        src.name, dst.name,
                        _size_mb(src), _size_mb(dst),
                    )
                    return str(dst.relative_to(self._audio_dir))
                # Compression failed: keep the audio as-is under its own name.
                dst = dst.with_suffix(src.suffix)
                logger.warning("Falling back to an uncompressed archive for %s", src.name)
            shutil.move(str(src), str(dst))
            return str(dst.relative_to(self._audio_dir))
        except Exception as exc:
            logger.warning("Could not archive %s -> %s: %s", src, dst, exc)
            return None


# ============================================================================
# Pure helpers — testable without touching mlx-whisper or filesystem
# ============================================================================
@dataclass
class _LabelledSegment:
    segment: Segment
    speaker: str


def _metadata_name_for(mic_relative: str) -> str:
    """``2026/08/2026-08-21_1749_mic.flac`` -> ``..._metadata.json``."""
    stem = Path(mic_relative)
    return str(stem.with_name(stem.stem.removesuffix("_mic") + "_metadata.json"))


def _size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1e6
    except Exception:
        return 0.0


def encode_audio(src: Path, dst: Path) -> bool:
    """Losslessly compress ``src`` into ``dst`` using macOS ``afconvert``.

    Returns False (without raising) if it can't, so callers fall back to
    keeping the original. There is no ffmpeg on a stock Mac; afconvert ships
    with the OS, and its FLAC round-trip is bit-exact — verified against real
    recordings before any of this was allowed to write.
    """
    import shutil as _shutil
    import subprocess

    fmt = dst.suffix.lower().lstrip(".")
    if fmt != "flac":
        return False
    tool = _shutil.which("afconvert")
    if tool is None:  # pragma: no cover (afconvert ships with macOS)
        return False
    try:
        subprocess.run(
            [tool, "-f", "flac", "-d", "flac", str(src), str(dst)],
            check=True, capture_output=True, timeout=ENCODE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Could not compress %s: %s", src.name, exc)
        dst.unlink(missing_ok=True)
        return False
    if not dst.exists() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        return False
    return True


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
    candidate_segments: list[Segment],
    reference_segments: list[Segment],
    *,
    time_window: float = ECHO_TIME_WINDOW_SECONDS,
    overlap_threshold: float = ECHO_WORD_OVERLAP_THRESHOLD,
) -> tuple[list[Segment], int]:
    """Drop candidate segments that look like an echo of a reference segment.

    Returns ``(kept_candidates, dropped_count)``.

    Heuristic: same logical speech if there's a reference segment whose start
    time is within ``time_window`` seconds AND whose word set overlaps the
    candidate's by >= ``overlap_threshold`` (intersection / max-set-size).

    Direction is the caller's choice and it matters — see the call site in
    :meth:`TranscriptProcessor.process`.
    """
    if not candidate_segments or not reference_segments:
        return list(candidate_segments), 0

    kept: list[Segment] = []
    dropped = 0
    ref_words_cache = [(s, _normalise_words(s.text)) for s in reference_segments]

    for sys_seg in candidate_segments:
        sys_words = _normalise_words(sys_seg.text)
        if not sys_words:
            kept.append(sys_seg)
            continue
        is_echo = False
        for mic_seg, mic_words in ref_words_cache:
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


class PauseTimeline:
    """Maps audio-file time onto wall-clock time across recording pauses.

    ``DualStreamRecorder.pause()`` *drops* frames instead of writing silence,
    so a 25-minute pause makes the WAV 25 minutes shorter than the meeting. The
    recorder has always written a ``pauses`` list into the session metadata —
    nothing read it, so every transcript with a pause carried timestamps that
    drifted by the pause length and an ``end_time`` that was simply wrong.

    ``paused_at`` / ``resumed_at`` are elapsed *wall* seconds since the start of
    the recording, so the audio position at which pause *i* begins is its
    ``paused_at`` minus every earlier pause's duration.
    """

    def __init__(self, pauses: "list[Any] | None") -> None:
        self._gaps: list[tuple[float, float]] = []  # (audio_position, duration)
        cumulative = 0.0
        for entry in _clean_pauses(pauses):
            paused_at, resumed_at = entry
            duration = resumed_at - paused_at
            audio_position = paused_at - cumulative
            self._gaps.append((audio_position, duration))
            cumulative += duration
        self._total = cumulative

    @property
    def total_pause_seconds(self) -> float:
        return self._total

    def __bool__(self) -> bool:
        return bool(self._gaps)

    def to_wall(self, audio_seconds: float) -> float:
        """Convert a timestamp in the audio file to elapsed wall-clock time."""
        wall = float(audio_seconds)
        for audio_position, duration in self._gaps:
            if audio_seconds >= audio_position:
                wall += duration
        return wall

    def shift_segments(self, segments: "list[Segment]") -> "list[Segment]":
        """Return segments with start/end re-stamped onto the wall clock."""
        if not self._gaps:
            return list(segments)
        return [
            replace(s, start=self.to_wall(s.start), end=self.to_wall(s.end))
            for s in segments
        ]


def _clean_pauses(pauses: "list[Any] | None") -> list[tuple[float, float]]:
    """Extract usable ``(paused_at, resumed_at)`` pairs, sorted, junk dropped.

    A crash mid-pause leaves an entry with no ``resumed_at``; a malformed file
    can contain anything at all. Neither may take down a transcription.
    """
    out: list[tuple[float, float]] = []
    for entry in pauses or []:
        if not isinstance(entry, dict):
            continue
        start = entry.get("paused_at")
        end = entry.get("resumed_at")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end <= start:
            continue
        out.append((float(start), float(end)))
    out.sort()
    return out


def _progress_window(
    progress: Callable[[float], None], base: float, span: float,
) -> Callable[[float], None]:
    """Map one stream's 0-100 progress onto its slice of the whole job."""

    def scaled(pct: float) -> None:
        progress(base + (pct / 100.0) * span)

    return scaled


def _recorded_seconds(metadata: dict[str, Any]) -> float:
    """How long the recorder actually captured, from the byte counts."""
    rate = metadata.get("sample_rate") or 16000
    width = 2  # the recorder writes 16-bit PCM
    best = 0.0
    for key in ("mic_bytes", "system_bytes"):
        raw = metadata.get(key)
        if isinstance(raw, (int, float)) and raw > 0:
            best = max(best, float(raw) / (float(rate) * width))
    return best


def _system_audio_state(sys_result: "TranscriptionResult | None") -> str:
    """Classify the system stream for the transcript frontmatter."""
    if sys_result is None:
        return "missing"
    if sys_result.was_silent:
        return "silent"
    return "ok"


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
