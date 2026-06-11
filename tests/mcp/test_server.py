"""Smoke tests for src/mcp/server.py — tool registration over FastMCP.

The mcp package is a project dependency, so importing the real server here
is safe. We don't spin up the stdio transport — just verify the surface.
"""

from __future__ import annotations

import asyncio


def test_three_readonly_tools_registered() -> None:
    from src.mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"list_transcripts", "search_transcripts", "get_transcript"}
    for tool in tools:
        assert tool.description, f"{tool.name} is missing a description"


def test_tools_delegate_to_core(tmp_path, monkeypatch) -> None:
    from src.mcp import server
    from src.storage.transcript_store import TranscriptStore

    store = TranscriptStore(tmp_path)
    store.save(
        {"id": "abc", "title": "Demo", "date": "2026-06-01", "start_time": "09:00",
         "tags": [], "participants": [], "audio_files": {}, "language": "en"},
        "## Transcript\n\nhello world",
    )
    monkeypatch.setattr(server, "_store", store)

    assert server.list_transcripts()[0]["id"] == "abc"
    assert server.search_transcripts("hello")[0]["metadata"]["id"] == "abc"
    assert "hello world" in server.get_transcript("abc")["body"]
    assert "error" in server.get_transcript("missing")
