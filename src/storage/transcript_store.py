"""On-disk transcript storage (Markdown + YAML frontmatter).

Layout::

    ~/Otis/transcripts/
    ├── 2026/
    │   ├── 05/
    │   │   ├── 2026-05-09_1400_sprint-planning.md
    │   │   └── 2026-05-09_1530_ad-hoc-recording.md
    │   └── 06/
    │       └── 2026-06-01_0930_team-sync.md
    └── .trash/
        └── 2026-04-21_1100_old-call.md

Files are simple Markdown with a YAML frontmatter block:

    ---
    id: "uuid"
    title: "..."
    date: "YYYY-MM-DD"
    ...
    ---

    # Title
    <body>

We deliberately avoid a database; ``rglob`` over a year-tree is plenty fast
for years' worth of meetings, and the user can read / grep / version-control
the files directly.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

FRONTMATTER_DELIM = "---"
SLUG_MAX_LEN = 50


# ---------------------------------------------------------------------------
# Pure-function helpers (testable without a TranscriptStore instance)
# ---------------------------------------------------------------------------
def slugify(title: str, *, max_len: int = SLUG_MAX_LEN) -> str:
    """``"Sprint Planning!" → "sprint-planning"`` — safe for filenames."""
    if not title:
        return "untitled"
    text = title.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    text = text[:max_len].rstrip("-")
    return text or "untitled"


def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body_text). Empty dict if missing."""
    if not content.startswith(FRONTMATTER_DELIM):
        return {}, content
    # Find the closing --- on its own line.
    rest = content[len(FRONTMATTER_DELIM):].lstrip("\n")
    end_marker = f"\n{FRONTMATTER_DELIM}"
    end_idx = rest.find(end_marker)
    if end_idx == -1:
        return {}, content
    raw_fm = rest[:end_idx]
    body = rest[end_idx + len(end_marker):].lstrip("\n")
    try:
        fm = yaml.safe_load(raw_fm) or {}
    except Exception as exc:
        logger.warning("Bad frontmatter: %s", exc)
        return {}, content
    if not isinstance(fm, dict):
        return {}, content
    return fm, body


def render_with_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Compose the full ``.md`` content from a frontmatter dict + body."""
    fm_yaml = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip()
    body = body.rstrip() + "\n"
    return f"{FRONTMATTER_DELIM}\n{fm_yaml}\n{FRONTMATTER_DELIM}\n\n{body}"


# ---------------------------------------------------------------------------
# TranscriptStore
# ---------------------------------------------------------------------------
class TranscriptStore:
    """Filesystem-backed CRUD for transcript ``.md`` files.

    Parameters
    ----------
    transcript_dir:
        Root directory containing the ``YYYY/MM/`` tree. Created if missing.
    """

    def __init__(self, transcript_dir: str | os.PathLike[str]) -> None:
        self._root = Path(transcript_dir).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def trash_dir(self) -> Path:
        return self._root / ".trash"

    # =====================================================================
    # Save / Update
    # =====================================================================
    def save(self, frontmatter: dict[str, Any], body: str) -> Path:
        """Write a transcript and return its path.

        Path is computed from the frontmatter:
        ``YYYY/MM/YYYY-MM-DD_HHMM_{slug}.md``.

        Collisions are resolved **atomically** so concurrent saves never
        clobber each other:

        * If the target already exists with the **same** ``id`` as the new
          frontmatter, this is a re-save (tag toggle, retranscribe, etc.) —
          we overwrite in place.
        * Otherwise we try to claim ``foo.md``, ``foo_2.md``, ``foo_3.md`` …
          using ``O_CREAT | O_EXCL`` so two threads never resolve to the same
          target.
        """
        path = self._compute_path(frontmatter)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_id = frontmatter.get("id")

        # Re-save: same id at the same logical slot → overwrite in place.
        if path.exists() and new_id:
            existing_id = self._read_frontmatter_fast(path).get("id")
            if existing_id == new_id:
                self._atomic_write(path, render_with_frontmatter(frontmatter, body))
                logger.info("Saved transcript (re-save): %s", path)
                return path

        # Otherwise atomically claim a free filename.
        target = self._claim_free_path(path, owner_id=new_id)
        self._atomic_write(target, render_with_frontmatter(frontmatter, body))
        if target != path:
            logger.info(
                "Save collision avoided: %s already taken, used %s instead.",
                path.name, target.name,
            )
        else:
            logger.info("Saved transcript: %s", target)
        return target

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write ``content`` to ``path`` via a tmp neighbour + rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _claim_free_path(self, path: Path, *, owner_id: Any) -> Path:
        """Atomically reserve ``path`` (or a numeric variant) for ``owner_id``.

        Uses ``O_CREAT | O_EXCL`` so two callers racing into the same logical
        slot end up on different filenames. The reserved file is created
        empty; the caller follows up with ``_atomic_write`` to fill it.
        """
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        attempt = 1
        while True:
            candidate = path if attempt == 1 else parent / f"{stem}_{attempt}{suffix}"
            try:
                # O_EXCL: fail if the file exists. This is what makes the
                # claim atomic across threads / processes.
                fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                # Already reserved. If it's an in-place re-save of the same
                # owner, accept the slot; otherwise advance to the next suffix.
                if owner_id is not None and candidate.exists():
                    existing_id = self._read_frontmatter_fast(candidate).get("id")
                    if existing_id == owner_id:
                        return candidate
                attempt += 1
                continue
            os.close(fd)
            return candidate

    def update_frontmatter(self, transcript_id: str, updates: dict[str, Any]) -> Path | None:
        """Merge ``updates`` into an existing transcript's frontmatter."""
        path = self.path_for(transcript_id)
        if path is None:
            return None
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        fm.update(updates)
        path.write_text(render_with_frontmatter(fm, body), encoding="utf-8")
        return path

    # =====================================================================
    # Read
    # =====================================================================
    def get_transcript(self, transcript_id: str) -> dict[str, Any] | None:
        """Return ``{"metadata": dict, "body": str, "path": Path}`` or None."""
        path = self.path_for(transcript_id)
        if path is None:
            return None
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        return {"metadata": fm, "body": body, "path": path}

    def path_for(self, transcript_id: str) -> Path | None:
        """Locate the .md file whose frontmatter ``id`` matches."""
        for path in self._iter_transcripts():
            try:
                fm = self._read_frontmatter_fast(path)
            except Exception:
                continue
            if fm.get("id") == transcript_id:
                return path
        return None

    def list_transcripts(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        participant: str | None = None,
        tag: str | None = None,
        language: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return frontmatter dicts (sorted by date desc), filtered."""
        results: list[tuple[str, dict[str, Any]]] = []
        needle = query.lower() if query else None

        for path in self._iter_transcripts():
            try:
                fm = self._read_frontmatter_fast(path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            # Empty / unparseable frontmatter ⇒ corrupt or in-progress write.
            # Don't surface these in listings.
            if not fm or not fm.get("id"):
                logger.debug("Skipping %s — frontmatter missing or has no id.", path)
                continue

            date_str = str(fm.get("date") or "")
            if date_from and date_str < date_from:
                continue
            if date_to and date_str > date_to:
                continue
            if language and fm.get("language") != language:
                continue
            if tag and tag not in (fm.get("tags") or []):
                continue
            if participant and not _has_participant(fm, participant):
                continue
            if needle and needle not in str(fm.get("title", "")).lower():
                # Cheap title pass; full-text uses .search().
                continue

            results.append((self._sort_key(fm, path), fm))

        results.sort(key=lambda kv: kv[0], reverse=True)
        return [fm for _, fm in results[:limit]]

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Grep through every transcript body. Returns metadata + snippets."""
        if not query:
            return []
        needle = query.lower()
        results: list[dict[str, Any]] = []
        for path in self._iter_transcripts():
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, body = split_frontmatter(content)
            if needle not in body.lower():
                continue
            snippets = _extract_snippets(body, needle)
            if not snippets:
                continue
            results.append({"metadata": fm, "snippets": snippets, "path": path})
            if len(results) >= limit:
                break
        return results

    # =====================================================================
    # Tags
    # =====================================================================
    def add_tag(self, transcript_id: str, tag: str) -> bool:
        path = self.path_for(transcript_id)
        if path is None:
            return False
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        tags = list(fm.get("tags") or [])
        if tag in tags:
            return True
        tags.append(tag)
        fm["tags"] = tags
        path.write_text(render_with_frontmatter(fm, body), encoding="utf-8")
        return True

    def remove_tag(self, transcript_id: str, tag: str) -> bool:
        path = self.path_for(transcript_id)
        if path is None:
            return False
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        tags = list(fm.get("tags") or [])
        if tag not in tags:
            return True
        tags.remove(tag)
        fm["tags"] = tags
        path.write_text(render_with_frontmatter(fm, body), encoding="utf-8")
        return True

    # =====================================================================
    # Lifecycle
    # =====================================================================
    def delete_transcript(self, transcript_id: str) -> Path | None:
        """Move the transcript to ``.trash/`` and return the new path."""
        path = self.path_for(transcript_id)
        if path is None:
            return None
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        target = self.trash_dir / path.name
        # Disambiguate if a file with the same name is already there.
        i = 1
        while target.exists():
            target = self.trash_dir / f"{path.stem}.{i}{path.suffix}"
            i += 1
        shutil.move(str(path), str(target))
        logger.info("Moved transcript to trash: %s → %s", path, target)
        return target

    def mark_audio_unavailable(self, session_id: str) -> Path | None:
        """Update a transcript's ``audio_available`` flag to ``False``.

        Called by :class:`AudioRetentionManager` after deleting WAV files.
        """
        return self.update_frontmatter(session_id, {"audio_available": False})

    # =====================================================================
    # Failure placeholders
    # =====================================================================
    def save_failure(
        self,
        *,
        session_id: str,
        error: BaseException,
        title: str | None = None,
        app: str | None = None,
        participants: list[str] | None = None,
        model: str | None = None,
        audio_files: dict[str, str | None] | None = None,
        start_wall_clock: datetime | None = None,
        language: str | None = None,
    ) -> Path:
        """Save a placeholder transcript for a failed transcription.

        The placeholder lives in the same ``YYYY/MM/`` tree as successful
        transcripts so it shows up in listings — but with ``status: failed``
        and a tag of ``failed`` so the user can find / filter / retry. The
        body explains the error and tells them how to retry.
        """
        when = (start_wall_clock or datetime.now(timezone.utc)).astimezone()
        clean_title = (title or "").strip() or "Recording"
        fm: dict[str, Any] = {
            "id": session_id,
            "title": f"{clean_title} — failed",
            "date": when.strftime("%Y-%m-%d"),
            "start_time": when.strftime("%H:%M"),
            "end_time": when.strftime("%H:%M"),
            "duration_minutes": 0,
            "language": language,
            "app": app,
            "participants": list(participants or []),
            "tags": ["failed"],
            "audio_files": dict(audio_files or {"mic": None, "system": None}),
            "audio_available": True,
            "model": model,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        body = self._render_failure_body(
            session_id=session_id,
            title=clean_title,
            error=error,
            audio_files=fm["audio_files"],
            when=when,
        )
        return self.save(fm, body)

    @staticmethod
    def _render_failure_body(
        *,
        session_id: str,
        title: str,
        error: BaseException,
        audio_files: dict[str, str | None],
        when: datetime,
    ) -> str:
        timestamp = when.strftime("%Y-%m-%d %H:%M")
        lines: list[str] = [
            f"# {title} — failed ({timestamp})",
            "",
            f"**`{type(error).__name__}`**: {error}",
            "",
            "## Recording",
            "",
        ]
        mic = audio_files.get("mic")
        sysw = audio_files.get("system")
        if mic:
            lines.append(f"- mic: `{mic}`")
        if sysw:
            lines.append(f"- system: `{sysw}`")
        if not mic and not sysw:
            lines.append("_Audio file path unknown — check `~/Otis/audio/`._")
        lines += [
            "",
            "## Retry",
            "",
            "```sh",
            f"python scripts/retranscribe.py {session_id[:8]}",
            "```",
            "",
        ]
        return "\n".join(lines)

    # =====================================================================
    # Internals
    # =====================================================================
    def _iter_transcripts(self) -> Iterable[Path]:
        """Yield every ``.md`` under root, skipping ``.trash/``."""
        if not self._root.exists():
            return
        for path in self._root.rglob("*.md"):
            if ".trash" in path.parts:
                continue
            yield path

    @staticmethod
    def _read_frontmatter_fast(path: Path) -> dict[str, Any]:
        """Read just enough of the file to parse the frontmatter."""
        with path.open("r", encoding="utf-8") as fh:
            first = fh.readline()
            if first.rstrip() != FRONTMATTER_DELIM:
                return {}
            lines: list[str] = []
            for line in fh:
                if line.rstrip() == FRONTMATTER_DELIM:
                    break
                lines.append(line)
            try:
                fm = yaml.safe_load("".join(lines)) or {}
            except Exception:
                return {}
            return fm if isinstance(fm, dict) else {}

    @staticmethod
    def _sort_key(fm: dict[str, Any], path: Path) -> str:
        """Sort key: date + start_time, falling back to filename."""
        date = str(fm.get("date") or "")
        start = str(fm.get("start_time") or "")
        return f"{date} {start}".strip() or path.name

    @staticmethod
    def _compute_path_parts(frontmatter: dict[str, Any]) -> tuple[str, str, str]:
        """Resolve ``(year, month, filename)`` from a frontmatter dict."""
        date_str = str(frontmatter.get("date") or "")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
            today = datetime.now()
            date_str = today.strftime("%Y-%m-%d")
            frontmatter["date"] = date_str
        year, month, _day = date_str.split("-")
        start_time = str(frontmatter.get("start_time") or "")
        time_compact = re.sub(r"[^\d]", "", start_time)[:4] or "0000"
        slug = slugify(str(frontmatter.get("title") or "untitled"))
        filename = f"{date_str}_{time_compact}_{slug}.md"
        return year, month, filename

    def _compute_path(self, frontmatter: dict[str, Any]) -> Path:
        year, month, filename = self._compute_path_parts(frontmatter)
        return self._root / year / month / filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_participant(fm: dict[str, Any], needle: str) -> bool:
    needle_l = needle.lower()
    for entry in fm.get("participants") or []:
        if needle_l in str(entry).lower():
            return True
    return False


def _extract_snippets(body: str, needle: str, *, max_per_file: int = 3,
                     context: int = 40) -> list[str]:
    """Return up to ``max_per_file`` 80-char-ish context windows around hits."""
    snippets: list[str] = []
    for line in body.splitlines():
        idx = line.lower().find(needle)
        if idx == -1:
            continue
        start = max(0, idx - context)
        end = min(len(line), idx + len(needle) + context)
        snippet = line[start:end].strip()
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= max_per_file:
            break
    return snippets


__all__ = [
    "FRONTMATTER_DELIM",
    "TranscriptStore",
    "render_with_frontmatter",
    "slugify",
    "split_frontmatter",
]
