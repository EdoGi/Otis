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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DAILY_INTERVAL_SECONDS = 24 * 60 * 60
DEFAULT_RETENTION_DAYS = 30


#: Protected audio can't be kept forever or one crashed session pins gigabytes
#: indefinitely. After ``retention_days * this`` it is deleted anyway.
DEFAULT_PROTECTED_GRACE_MULTIPLIER = 3.0


@dataclass
class CleanupReport:
    deleted_files: list[Path]
    deleted_session_ids: list[str]
    transcripts_marked: list[str]
    #: Sessions past the horizon whose audio we deliberately kept because no
    #: successful transcript exists — deleting it would destroy the only copy.
    protected_session_ids: list[str] = field(default_factory=list)


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
        protected_grace_multiplier: float = DEFAULT_PROTECTED_GRACE_MULTIPLIER,
    ) -> None:
        self._audio_dir = Path(audio_dir).expanduser()
        self._store = transcript_store
        self._retention_days = int(retention_days)
        self._clock = clock
        self._interval = float(interval_seconds)
        self._grace_multiplier = max(1.0, float(protected_grace_multiplier))

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
            return CleanupReport([], [], self.reconcile_audio_availability())

        # Audio is only expendable once a *successful* transcript exists. Until
        # then it is the sole copy of that meeting, and the failure placeholder
        # explicitly tells the user to re-run retranscribe.py against it.
        completed = self._completed_session_ids()
        path_to_session = self._map_paths_to_sessions()
        hard_cutoff = self._clock() - self._retention_days * self._grace_multiplier * 86400.0

        deletable: list[Path] = []
        protected: set[str] = set()
        unresolved: list[Path] = []
        for path in old_files:
            sid = path_to_session.get(path)
            if completed is None:
                deletable.append(path)  # no store wired — old behaviour
                continue
            if sid is not None and sid in completed:
                deletable.append(path)
                continue
            # Unknown or un-transcribed. Keep it, unless it is so old that the
            # backstop kicks in — protection can't be unbounded.
            if _safe_mtime(path) < hard_cutoff:
                logger.warning(
                    "Deleting %s past the %.0f-day backstop even though session %s "
                    "has no successful transcript.",
                    path.name, self._retention_days * self._grace_multiplier, sid or "?",
                )
                deletable.append(path)
                continue
            if sid is not None:
                protected.add(sid)
            else:
                # No metadata and no UUID in the filename — we cannot tell which
                # meeting this belongs to, so we keep it. Count it, because
                # silently hoarding files until the backstop is how you end up
                # wondering where the disk went.
                unresolved.append(path)

        # Determine the session ids touched, before we delete metadata.json.
        # `path_to_session` already resolved every file — including relocated,
        # date-named ones whose filename no longer carries the UUID. Recomputing
        # this from _collect_session_ids alone silently missed them, so their
        # audio was deleted while the transcript went on advertising
        # `audio_available: true` forever.
        session_ids = sorted(
            {sid for p in deletable if (sid := path_to_session.get(p))}
            | set(self._collect_session_ids(deletable))
        )

        deleted: list[Path] = []
        for path in deletable:
            try:
                path.unlink()
                deleted.append(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not delete %s: %s", path, exc)

        if unresolved:
            logger.warning(
                "Audio retention: kept %d file(s) past the horizon that could not "
                "be matched to any session (%s%s).",
                len(unresolved), ", ".join(p.name for p in unresolved[:3]),
                ", …" if len(unresolved) > 3 else "",
            )
        if protected:
            logger.info(
                "Audio retention: kept %d session(s) past the horizon with no "
                "successful transcript (%s).",
                len(protected), ", ".join(sorted(protected)),
            )

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
        # Catch anything that slipped through earlier sweeps too.
        for sid in self.reconcile_audio_availability():
            if sid not in transcripts_marked:
                transcripts_marked.append(sid)

        return CleanupReport(
            deleted_files=deleted,
            deleted_session_ids=list(session_ids),
            transcripts_marked=transcripts_marked,
            protected_session_ids=sorted(protected),
        )

    def reconcile_audio_availability(self) -> list[str]:
        """Fix transcripts that advertise audio which is no longer on disk.

        ``audio_available`` is written at transcription time and flipped when
        retention deletes the files — so any gap between those two events
        leaves the transcript lying. On this machine nine transcripts claimed
        audio that had been swept months earlier, and the web UI and MCP server
        both believe that flag.

        Cheap and idempotent: it only rewrites a transcript whose stored paths
        genuinely resolve to nothing. Returns the session ids corrected.
        """
        if self._store is None:
            return []
        corrected: list[str] = []
        try:
            entries = self._store.list_transcripts(limit=10_000)
        except Exception:
            logger.exception("Could not list transcripts to reconcile audio state")
            return []

        for fm in entries:
            if not fm.get("audio_available"):
                continue
            sid = fm.get("id")
            if not sid:
                continue
            paths = [v for v in (fm.get("audio_files") or {}).values() if v]
            # No paths recorded at all means we can't prove anything; leave it.
            if not paths:
                continue
            if any((self._audio_dir / rel).exists() for rel in paths):
                continue
            try:
                if self._store.mark_audio_unavailable(str(sid)) is not None:
                    corrected.append(str(sid))
            except Exception as exc:
                logger.warning("Could not reconcile transcript %s: %s", sid, exc)

        if corrected:
            logger.info(
                "Audio retention: corrected %d transcript(s) that still claimed "
                "audio was available.", len(corrected),
            )
        return corrected

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

    def _completed_session_ids(self) -> set[str] | None:
        """Sessions with a real (non-``status: failed``) transcript.

        ``None`` means "we can't tell" — no store wired — in which case the
        caller keeps the pre-existing mtime-only behaviour rather than
        protecting everything forever.
        """
        if self._store is None:
            return None
        try:
            return set(self._store.completed_session_ids())
        except Exception:
            logger.exception(
                "Could not index completed transcripts; keeping audio this sweep."
            )
            return set()  # empty = nothing is provably safe to delete

    def _map_paths_to_sessions(self) -> dict[Path, str]:
        """Map every audio file to its session id.

        Successful sessions are renamed to ``YYYY-MM-DD_HHMM_*.wav``, so the
        UUID is gone from the filename and only the sibling metadata JSON knows
        the id. We read every metadata file (not just expired ones) and claim
        its siblings by stem, then fall back to UUID-style filenames.
        """
        mapping: dict[Path, str] = {}
        for meta_path in self._audio_dir.rglob("*_metadata.json"):
            try:
                import json

                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            sid = data.get("session_id")
            if not sid:
                continue
            sid = str(sid)
            mapping[meta_path] = sid
            stem = meta_path.name[: -len("_metadata.json")]
            for suffix in ("mic", "system"):
                for ext in (".wav", ".flac"):
                    sibling = meta_path.with_name(f"{stem}_{suffix}{ext}")
                    if sibling.exists():
                        mapping[sibling] = sid

        for path in self._iter_audio_files():
            if path in mapping:
                continue
            guess = _session_id_from_filename(path)
            if guess:
                mapping[path] = guess
        return mapping

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
