"""Pending transcriptions that survive a quit, a crash, or a power loss.

Every transcription worker is a ``daemon=True`` thread, so anything still
running when the process goes away is simply gone. What was left behind was
the audio on disk and nothing that said "this still needs transcribing".

The menu bar could half-recover: its "Generate Transcript" menu rediscovers
audio that has no transcript, so a user who happened to look would find it.
``otis run`` had no such scan at all — a headless reboot lost every pending
transcription silently, and retention deleted the audio 30 days later.

This is deliberately a record of *in-flight* work, not a retry log. A job is
removed once its handler returns, whether it succeeded or wrote a failure
placeholder, because both of those are outcomes the user can see. Only a
process that died mid-run leaves an entry behind.

One JSON file per session in ``~/.otis/queue/`` — no database, no index, and a
file that cannot be parsed is dropped rather than allowed to wedge startup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DIR = Path("~/.otis/queue")


@dataclass(frozen=True)
class QueuedJob:
    """One transcription that was started but never observed to finish."""

    session_id: str
    metadata: dict[str, Any]
    path: Path
    queued_at: str

    @property
    def mic_path(self) -> Path | None:
        raw = self.metadata.get("mic_wav")
        return Path(str(raw)) if raw else None


class TranscriptionQueue:
    """Crash-visible record of transcriptions that have not finished."""

    def __init__(self, queue_dir: str | os.PathLike[str] | None = None) -> None:
        self._dir = Path(queue_dir or DEFAULT_QUEUE_DIR).expanduser()

    @property
    def directory(self) -> Path:
        return self._dir

    # -----------------------------------------------------------------
    def enqueue(self, metadata: dict[str, Any]) -> Path | None:
        """Record ``metadata`` as pending. Returns the file, or None."""
        session_id = str(metadata.get("session_id") or "").strip()
        if not session_id:
            logger.debug("Not queueing a job with no session_id.")
            return None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._path_for(session_id)
            payload = {
                "session_id": session_id,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
            }
            tmp = path.with_suffix(".json.tmp")
            # default=str: metadata carries Paths once a caller decorates it,
            # and a queue write must never be the thing that loses a recording.
            tmp.write_text(json.dumps(payload, indent=2, default=str),
                           encoding="utf-8")
            tmp.replace(path)
            return path
        except Exception:
            logger.exception("Could not queue session %s", session_id)
            return None

    def resolve(self, session_id: str) -> bool:
        """Drop a job whose handler returned. True if something was removed."""
        try:
            path = self._path_for(str(session_id))
            if not path.exists():
                return False
            path.unlink()
            return True
        except Exception:
            logger.exception("Could not dequeue session %s", session_id)
            return False

    def pending(self) -> list[QueuedJob]:
        """Every job left behind, oldest first."""
        if not self._dir.exists():
            return []
        jobs: list[QueuedJob] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Dropping unreadable queue entry %s", path.name)
                self._discard(path)
                continue
            if not isinstance(payload, dict):
                self._discard(path)
                continue
            metadata = payload.get("metadata")
            session_id = str(payload.get("session_id") or "")
            if not session_id or not isinstance(metadata, dict):
                logger.warning("Dropping malformed queue entry %s", path.name)
                self._discard(path)
                continue
            jobs.append(QueuedJob(
                session_id=session_id,
                metadata=metadata,
                path=path,
                queued_at=str(payload.get("queued_at") or ""),
            ))
        jobs.sort(key=lambda j: j.queued_at)
        return jobs

    def recoverable(self, jobs: list[QueuedJob], *, store: Any | None = None,
                    ) -> list[QueuedJob]:
        """Jobs still worth running: audio present, transcript not already made.

        A job whose audio is gone can never be run again, and one that already
        has a transcript was finished by a process that died before it could
        say so. Both are dropped rather than replayed — replaying the second
        would write a duplicate transcript on every launch.
        """
        keep: list[QueuedJob] = []
        for job in jobs:
            mic = job.mic_path
            if mic is None or not mic.exists():
                logger.info(
                    "Queued session %s has no audio left; dropping.", job.session_id
                )
                self._discard(job.path)
                continue
            if store is not None:
                try:
                    if store.has_completed_transcript(job.session_id):
                        logger.info(
                            "Queued session %s already has a transcript; dropping.",
                            job.session_id,
                        )
                        self._discard(job.path)
                        continue
                except Exception:  # pragma: no cover (defensive)
                    logger.exception("Could not check transcript for %s",
                                     job.session_id)
            keep.append(job)
        return keep

    # -----------------------------------------------------------------
    def _path_for(self, session_id: str) -> Path:
        # Session ids are UUIDs, but never let one escape its directory.
        safe = session_id.replace("/", "_").replace("..", "_")
        return self._dir / f"{safe}.json"

    def _discard(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover (defensive)
            logger.exception("Could not remove queue entry %s", path)


__all__ = ["DEFAULT_QUEUE_DIR", "QueuedJob", "TranscriptionQueue"]
