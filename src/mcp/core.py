"""Tool implementations behind the Otis MCP server.

Kept free of any ``mcp``-runtime imports so the logic is unit-testable and
reusable: the server module registers thin shims over these functions.

Everything is read-only — the MCP surface can search and fetch transcripts
but never modify or delete them.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config import load_user_config
from src.storage.transcript_store import TranscriptStore


def jsonify(value: Any) -> Any:
    """Recursively coerce a value into JSON-serialisable primitives."""
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def store_from_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    user_config_path: str | os.PathLike[str] | None = None,
) -> TranscriptStore:
    """Build a TranscriptStore from the same config stack the app uses."""
    cfg = load_user_config(config_path, user_config_path=user_config_path)
    transcript_dir = cfg.get("storage", "transcript_dir", default="~/Otis/transcripts")
    return TranscriptStore(Path(transcript_dir).expanduser())


def list_transcripts_core(
    store: TranscriptStore,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    tag: str | None = None,
    language: str | None = None,
    query: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Frontmatter metadata for matching transcripts, newest first."""
    entries = store.list_transcripts(
        date_from=date_from,
        date_to=date_to,
        participant=participant,
        tag=tag,
        language=language,
        query=query,
        limit=max(1, min(int(limit), 200)),
    )
    return [jsonify(fm) for fm in entries]


def search_transcripts_core(
    store: TranscriptStore, query: str, *, limit: int = 10,
) -> list[dict[str, Any]]:
    """Full-text search over transcript bodies; metadata + snippets per hit."""
    hits = store.search(query, limit=max(1, min(int(limit), 50)))
    return [
        {
            "metadata": jsonify(hit["metadata"]),
            "snippets": list(hit["snippets"]),
            "path": str(hit["path"]),
        }
        for hit in hits
    ]


def get_transcript_core(store: TranscriptStore, transcript_id: str) -> dict[str, Any]:
    """One transcript by id: metadata + full Markdown body.

    Unknown ids return an ``{"error": ...}`` payload rather than raising —
    a model-friendly tool result the caller can react to.
    """
    record = store.get_transcript(transcript_id)
    if record is None:
        return {"error": f"No transcript with id {transcript_id!r}."}
    return {
        "metadata": jsonify(record["metadata"]),
        "body": record["body"],
        "path": str(record["path"]),
    }


__all__ = [
    "get_transcript_core",
    "jsonify",
    "list_transcripts_core",
    "search_transcripts_core",
    "store_from_config",
]
