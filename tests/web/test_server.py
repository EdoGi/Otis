"""Tests for src/web/server.py — the local transcript browser."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from src.storage.transcript_store import TranscriptStore
from src.web.server import create_app, render_markdown_min, serve_in_background


# ---------------------------------------------------------------------------
# render_markdown_min — pure function
# ---------------------------------------------------------------------------
def test_render_headings_bold_and_paragraphs() -> None:
    body = (
        "# Standup — 2026-06-01\n\n"
        "## Transcript\n\n"
        "**[00:01]** Me: hello there\n"
        "**[00:03]** Participant: hi\n\n"
        "next paragraph"
    )
    out = render_markdown_min(body)
    assert "<h1>Standup — 2026-06-01</h1>" in out
    assert "<h2>Transcript</h2>" in out
    assert "<strong>[00:01]</strong> Me: hello there<br>" in out
    assert out.count("<p>") == 2


def test_render_escapes_html_before_markup() -> None:
    out = render_markdown_min("# <script>alert(1)</script>\n\n**<b>bold</b>**")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<strong>&lt;b&gt;bold&lt;/b&gt;</strong>" in out


def test_render_empty_body() -> None:
    assert render_markdown_min("") == ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _fm(sid: str, *, title: str, date: str = "2026-06-01", status: str | None = None) -> dict:
    fm = {
        "id": sid,
        "title": title,
        "date": date,
        "start_time": "10:00",
        "end_time": "10:30",
        "duration_minutes": 30,
        "language": "en",
        "app": "zoom.us",
        "participants": ["Alice <alice@example.com>"],
        "tags": ["weekly"],
        "audio_files": {"mic": None, "system": None},
        "audio_available": False,
        "model": "small",
    }
    if status:
        fm["status"] = status
        fm["tags"] = ["failed"]
    return fm


@pytest.fixture
def client(tmp_path: Path):
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("id-1", title="Sprint Planning"),
               "## Transcript\n\n**[00:01]** Me: the roadmap is ready\n")
    store.save(_fm("id-2", title="Retro", date="2026-06-05"),
               "## Transcript\n\n**[00:02]** Participant: shipping went well\n")
    store.save(_fm("id-3", title="Broken call", status="failed"),
               "# Broken call — failed\n\n**`RuntimeError`**: oom\n")
    return create_app(store).test_client()


def test_index_lists_transcripts_newest_first(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "Sprint Planning" in text and "Retro" in text
    assert text.index("Retro") < text.index("Sprint Planning")  # newest first
    assert '<span class="badge">failed</span>' in text  # failed badge shown


def test_index_empty_store(tmp_path: Path) -> None:
    app = create_app(TranscriptStore(tmp_path / "empty"))
    resp = app.test_client().get("/")
    assert resp.status_code == 200
    assert "No transcripts yet" in resp.get_data(as_text=True)


def test_index_title_filter(client) -> None:
    text = client.get("/?q=retro").get_data(as_text=True)
    assert "Retro" in text and "Sprint Planning" not in text


def test_transcript_view_renders_body_and_meta(client) -> None:
    text = client.get("/transcript/id-1").get_data(as_text=True)
    assert "<h1>Sprint Planning</h1>" in text
    assert "<strong>[00:01]</strong> Me: the roadmap is ready" in text
    assert "Alice" in text


def test_transcript_unknown_id_404(client) -> None:
    resp = client.get("/transcript/nope")
    assert resp.status_code == 404
    assert "Transcript not found" in resp.get_data(as_text=True)


def test_transcript_escapes_malicious_title(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("evil", title="<script>alert(1)</script>"),
               "## Transcript\n\n<img src=x onerror=alert(1)>\n")
    text = create_app(store).test_client().get("/transcript/evil").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert "<img src=x" not in text


def test_search_returns_snippets(client) -> None:
    text = client.get("/search?q=roadmap").get_data(as_text=True)
    assert "Sprint Planning" in text
    assert "roadmap" in text
    assert "Retro" not in text


def test_search_empty_query_shows_form(client) -> None:
    resp = client.get("/search")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# serve_in_background — bind semantics
# ---------------------------------------------------------------------------
def test_serve_in_background_raises_synchronously_on_busy_port(tmp_path: Path) -> None:
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(OSError):
            serve_in_background(
                TranscriptStore(tmp_path / "t"), host="127.0.0.1", port=port
            )
    finally:
        blocker.close()


def test_serve_in_background_serves_real_requests(tmp_path: Path) -> None:
    import urllib.request

    store = TranscriptStore(tmp_path / "t")
    store.save(_fm("live-1", title="Live Check"), "## Transcript\n\nhello\n")
    # Pick a free port up front (make_server doesn't expose port=0 cleanly).
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    thread = serve_in_background(store, host="127.0.0.1", port=port)
    assert thread.daemon
    body = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/", timeout=5
    ).read().decode()
    assert "Live Check" in body


def test_page_title_is_escaped_exactly_once(tmp_path: Path) -> None:
    """'Q&A — <sync>' must render in <title> as 'Q&amp;A — &lt;sync&gt;',
    not the double-escaped 'Q&amp;amp;A'."""
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("amp-1", title="Q&A — <sync>"), "## Transcript\n\nbody\n")
    text = create_app(store).test_client().get("/transcript/amp-1").get_data(as_text=True)
    assert "<title>Q&amp;A — &lt;sync&gt; — Otis</title>" in text
    assert "&amp;amp;" not in text
