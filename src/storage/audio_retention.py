"""Audio retention sweep.

Recordings can be tens of MB per minute. We don't keep the WAV files forever
— after ``audio_retention_days`` we delete them but keep the transcript
``.md`` (the user mostly reads the text anyway). The deletion is reflected in
the transcript's frontmatter: ``audio_available: false``.

Runs:

* once at app startup,
* and then every 24 h on a daemon ``threading.Timer``.

The sweep is filesystem-stat based so it works regardless of how the audio
files were named (UUID-style from the recorder, or ``YYYY-MM-DD_HHMM_*.wav``
after :class:`TranscriptProcessor` renamed them).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DAILY_INTERVAL_SECONDS = 24 * 60 * 60
DEFAULT_RETENTION_DAYS = 30


@dataclass
class CleanupReport:
    deleted_files: list[Path]
    deleted_session_ids: list[str]
    transcripts_marked: list[str]


class AudioRetentionManager:
    """Sweeps ``audio_dir`` for files past the retention horizon.

    Parameters
    ----------
    audio_dir:
        Root of the audio tree. We scan recursively so files in
        ``audio_dir/YYYY/MM/`` are picked up alongside any UUID-named files
        the recorder dropped at the root.
    transcript_store:
        Used to update transcript frontmatter when audio is deleted. Pass
        ``None`` to skip that step (e.g. in tests).
    retention_days:
        Files older than this (by mtime) are deleted on each sweep.
    clock:
        Override ``time.time()`` for tests.
    """

    def __init__(
        self,
        *,
        audio_dir: str | os.PathLike[str],
        transcript_store: Any | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        clock: Callable[[], float] = time.time,
        interval_seconds: float = DAILY_INTERVAL_SECONDS,
    ) -> None:
        self._audio_dir = Path(audio_dir).expanduser()
        self._store = transcript_store
        self._retention_days = int(retention_days)
        self._clock = clock
        self._interval = float(interval_seconds)

        self._timer: threading.Timer | None = None
        self._lock = threading.RLock()

    # =====================================================================
    # Public API
    # =====================================================================
    def start_periodic(self) -> None:
        """Run a sweep now and schedule one every 24 h.

        Idempotent: if a previous timer is already running (e.g. a second
        ``start_periodic()`` call after a config reload), it's cancelled
        first so we don't accumulate daemon timers.
        """
        with self._lock:
            if self._timer is not None:
                logger.debug("start_periodic called while a timer was active; replacing it.")
                self._timer.cancel()
                self._timer = None
        self.cleanup_now()
        self._schedule_next()

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def cleanup_now(self) -> CleanupReport:
        """Delete WAV/JSON files older than the retention horizon. Returns a report."""
        if not self._audio_dir.exists():
            return CleanupReport([], [], [])

        cutoff = self._clock() - self._retention_days * 86400.0
        candidates = list(self._iter_audio_files())
        old_files = [p for p in candidates if _safe_mtime(p) < cutoff]

        if not old_files:
            logger.info(
                "Audio retention: nothing to delete in %s (retention=%d days).",
                self._audio_dir, self._retention_days,
            )
            return CleanupReport([], [], [])

        # Determine the session ids touched, before we delete metadata.json.
        session_ids = self._collect_session_ids(old_files)

        deleted: list[Path] = []
        for path in old_files:
            try:
                path.unlink()
                deleted.append(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not delete %s: %s", path, exc)

        transcripts_marked: list[str] = []
        if self._store is not None and session_ids:
            for sid in session_ids:
                try:
                    if self._store.mark_audio_unavailable(sid) is not None:
                        transcripts_marked.append(sid)
                except Exception as exc:
                    logger.warning(
                        "Could not mark transcript %s audio_available=False: %s",
                        sid,
                        exc,
                    )

        # Best-effort: prune empty YYYY/MM directories left behind.
        self._prune_empty_dirs()

        logger.info(
            "Audio retention: deleted %d file(s), updated %d transcript(s).",
            len(deleted), len(transcripts_marked),
        )
        return CleanupReport(
            deleted_files=deleted,
            deleted_session_ids=list(session_ids),
            transcripts_marked=transcripts_marked,
        )

    def delete_audio(self, session_id: str) -> CleanupReport:
        """Manual delete: drop every audio artefact for ``session_id``."""
        candidates = [
            p for p in self._iter_audio_files() if _matches_session(p, session_id)
        ]
        deleted: list[Path] = []
        for p in candidates:
            try:
                p.unlink()
                deleted.append(p)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not delete %s: %s", p, exc)

        transcripts_marked: list[str] = []
        if self._store is not None:
            try:
                if self._store.mark_audio_unavailable(session_id) is not None:
                    transcripts_marked.append(session_id)
            except Exception as exc:
                logger.warning(
                    "Could not mark transcript %s audio_available=False: %s",
                    session_id,
                    exc,
                )

        self._prune_empty_dirs()
        return CleanupReport(
            deleted_files=deleted,
            deleted_session_ids=[session_id] if deleted else [],
            transcripts_marked=transcripts_marked,
        )

    # =====================================================================
    # Internals
    # =====================================================================
    def _schedule_next(self) -> None:
        with self._lock:
            self._timer = threading.Timer(self._interval, self._on_tick)
            self._timer.daemon = True
            self._timer.start()

    def _on_tick(self) -> None:
        try:
            self.cleanup_now()
        except Exception:  # pragma: no cover (defensive)
            logger.exception("Audio retention sweep crashed; rescheduling anyway.")
        self._schedule_next()

    def _iter_audio_files(self) -> Iterable[Path]:
        if not self._audio_dir.exists():
            return
        for pattern in ("*.wav", "*_metadata.json"):
            yield from self._audio_dir.rglob(pattern)

    def _collect_session_ids(self, paths: Iterable[Path]) -> list[str]:
        """Inspect metadata.json files (where session_id is authoritative)."""
        ids: set[str] = set()
        # First pass — walk metadata.json files included in the deletion set.
        for path in paths:
            if path.name.endswith("_metadata.json"):
                try:
                    import json

                    data = json.loads(path.read_text(encoding="utf-8"))
                    sid = data.get("session_id")
                    if sid:
                        ids.add(str(sid))
                except Exception:
                    pass
        # Second pass — guess from filename for any audio whose metadata
        # was already gone before this sweep.
        for path in paths:
            if path.suffix == ".wav":
                guess = _session_id_from_filename(path)
                if guess:
                    ids.add(guess)
        return sorted(ids)

    def _prune_empty_dirs(self) -> None:
        # Walk bottom-up; rmdir empty dirs but never the root.
        if not self._audio_dir.exists():
            return
        for dirpath, _dirs, _files in os.walk(self._audio_dir, topdown=False):
            d = Path(dirpath)
            if d == self._audio_dir:
                continue
            try:
                d.rmdir()  # only succeeds if empty
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_UUID_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(mic|system)\.wav$",
    flags=re.IGNORECASE,
)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return float("inf")


def _session_id_from_filename(path: Path) -> str | None:
    """Pull a UUID out of a recorder filename like ``{uuid}_mic.wav``."""
    m = _UUID_RE.match(path.name)
    return m.group(1) if m else None


def _matches_session(path: Path, session_id: str) -> bool:
    """True if a path's filename references this session id."""
    name = path.name
    return name.startswith(f"{session_id}_") or session_id in name


__all__ = [
    "AudioRetentionManager",
    "CleanupReport",
    "DAILY_INTERVAL_SECONDS",
    "DEFAULT_RETENTION_DAYS",
]
