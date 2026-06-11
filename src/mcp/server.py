"""MCP server exposing the local transcript store to Claude.

Runs over **stdio** — the MCP client (Claude Desktop, Claude Code, …)
launches this process itself, so there's nothing to start from the Otis app
and no port to configure:

    claude mcp add otis -- /path/to/Otis/.venv/bin/python -m src.mcp.server

All tools are read-only; nothing leaves the machine. Tool logic lives in
:mod:`src.mcp.core` so it stays unit-testable without the MCP runtime.

IMPORTANT: stdout belongs to the MCP protocol. Any logging here goes to
stderr — a stray print() would corrupt the JSON-RPC stream.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp import core
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "otis",
    instructions=(
        "Read-only access to local Otis meeting transcripts (Markdown with "
        "YAML frontmatter). Use list_transcripts to browse by date/participant/"
        "tag, search_transcripts for full-text lookup, and get_transcript to "
        "read one meeting in full."
    ),
)

_store: TranscriptStore | None = None


def _get_store() -> TranscriptStore:
    global _store
    if _store is None:
        _store = core.store_from_config()
    return _store


@mcp.tool()
def list_transcripts(
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    tag: str | None = None,
    language: str | None = None,
    query: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List meeting transcripts (metadata only), newest first.

    Filters: date_from/date_to are YYYY-MM-DD (inclusive); participant
    matches name or email substring; tag/language match exactly; query
    matches the meeting title. All filters are optional and combine.
    """
    return core.list_transcripts_core(
        _get_store(),
        date_from=date_from,
        date_to=date_to,
        participant=participant,
        tag=tag,
        language=language,
        query=query,
        limit=limit,
    )


@mcp.tool()
def search_transcripts(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Full-text search across all transcript bodies.

    Returns metadata plus the matching snippets for each hit — use
    get_transcript with the result's metadata.id to read the full meeting.
    """
    return core.search_transcripts_core(_get_store(), query, limit=limit)


@mcp.tool()
def get_transcript(transcript_id: str) -> dict[str, Any]:
    """Fetch one transcript by id: frontmatter metadata + full Markdown body."""
    return core.get_transcript_core(_get_store(), transcript_id)


def main() -> None:
    # stderr only — stdout carries the stdio JSON-RPC stream.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    logger.info("Otis MCP server starting (stdio).")
    mcp.run()


# Back-compat with the old stub's entry point name.
run = main

if __name__ == "__main__":
    main()
