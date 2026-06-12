"""Local web UI for browsing transcripts.

A deliberately tiny Flask app — list, full-text search, and a single
transcript view. No JavaScript, no static assets, no database: every page is
rendered from the Markdown files via :class:`TranscriptStore`.

Binds to ``web.host``/``web.port`` from the config (127.0.0.1:8765 by
default) and is started as a daemon thread by ``otis ui``; it can also run
standalone::

    .venv/bin/python -m src.web.server

Security model: localhost-only, read-only. All transcript content is
HTML-escaped before any markup transformation, so a meeting title like
``<script>`` renders as text.
"""

from __future__ import annotations

import html
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

from src.storage.transcript_store import TranscriptStore

if TYPE_CHECKING:  # pragma: no cover
    from flask import Flask

logger = logging.getLogger(__name__)

_PAGE_STYLE = """
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 56rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; }
  a { color: #0a66c2; text-decoration: none; } a:hover { text-decoration: underline; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #8884; }
  tr:hover td { background: #8881; }
  form.search { margin: 1rem 0; display: flex; gap: 0.5rem; }
  input[type=text] { flex: 1; padding: 0.4rem 0.6rem; font-size: 1rem; }
  .muted { opacity: 0.65; font-size: 0.9rem; }
  .badge { background: #c33; color: #fff; border-radius: 4px; padding: 0 0.4rem;
           font-size: 0.78rem; margin-left: 0.4rem; }
  .tag { background: #8882; border-radius: 4px; padding: 0 0.4rem;
         font-size: 0.78rem; margin-right: 0.25rem; }
  .snippet { margin: 0.2rem 0 0.8rem; padding-left: 0.8rem; border-left: 3px solid #8884; }
  .meta-card { background: #8881; border-radius: 8px; padding: 0.8rem 1rem; margin: 1rem 0; }
  .meta-card dt { font-weight: 600; } .meta-card dd { margin: 0 0 0.4rem 0; }
"""

_BASE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} — Otis</title>
  <style>""" + _PAGE_STYLE + """</style>
</head>
<body>
  <p class="muted"><a href="/">Otis transcripts</a></p>
  {{ body|safe }}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Minimal, escape-first Markdown rendering (no external dependency)
# ---------------------------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def render_markdown_min(body: str) -> str:
    """Render the subset of Markdown our transcripts use, XSS-safely.

    Everything is HTML-escaped FIRST; only then do we add tags for headings,
    ``**bold**`` (the speaker/timestamp labels), and paragraphs. Covers
    exactly what ``TranscriptProcessor.render_markdown_body`` and the
    failure-placeholder bodies emit.
    """
    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(paragraph) + "</p>")
            paragraph.clear()

    for raw_line in body.splitlines():
        line = html.escape(raw_line.rstrip())
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        heading = None
        for level, prefix in ((3, "### "), (2, "## "), (1, "# ")):
            if stripped.startswith(prefix):
                heading = (level, stripped[len(prefix):])
                break
        if heading is not None:
            flush()
            level, text = heading
            out.append(f"<h{level}>{_BOLD_RE.sub(r'<strong>\1</strong>', text)}</h{level}>")
            continue
        paragraph.append(_BOLD_RE.sub(r"<strong>\1</strong>", stripped))
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------
def create_app(store: TranscriptStore) -> "Flask":
    from flask import Flask, abort, render_template_string, request

    app = Flask("otis-web")

    def page(title: str, body_html: str) -> str:
        return render_template_string(_BASE, title=title, body=body_html)

    def fm_row(fm: dict[str, Any]) -> str:
        tid = html.escape(str(fm.get("id") or ""))
        title = html.escape(str(fm.get("title") or "(untitled)"))
        date = html.escape(str(fm.get("date") or ""))
        start = html.escape(str(fm.get("start_time") or ""))
        minutes = fm.get("duration_minutes")
        duration = f"{minutes} min" if isinstance(minutes, (int, float)) else "—"
        badge = '<span class="badge">failed</span>' if fm.get("status") == "failed" else ""
        tags = "".join(
            f'<span class="tag">{html.escape(str(t))}</span>' for t in (fm.get("tags") or [])
            if t != "failed"
        )
        return (
            f"<tr><td>{date} {start}</td>"
            f'<td><a href="/transcript/{tid}">{title}</a>{badge}</td>'
            f"<td>{duration}</td><td>{tags}</td></tr>"
        )

    def search_form(value: str = "", *, action: str = "/search",
                    placeholder: str = "Search transcript text…") -> str:
        return (
            f'<form class="search" action="{action}" method="get">'
            f'<input type="text" name="q" value="{html.escape(value)}" '
            f'placeholder="{placeholder}">'
            f"<button>Search</button></form>"
        )

    @app.get("/")
    def index() -> str:
        q = (request.args.get("q") or "").strip()
        tag = (request.args.get("tag") or "").strip() or None
        participant = (request.args.get("participant") or "").strip() or None
        date_from = (request.args.get("from") or "").strip() or None
        date_to = (request.args.get("to") or "").strip() or None
        entries = store.list_transcripts(
            query=q or None, tag=tag, participant=participant,
            date_from=date_from, date_to=date_to, limit=200,
        )
        parts = ["<h1>Meeting transcripts</h1>", search_form()]
        if q or tag or participant or date_from or date_to:
            parts.append(
                f'<p class="muted">Filtered title list — <a href="/">clear</a></p>'
            )
        if not entries:
            parts.append("<p>No transcripts yet. Record a meeting and it will show up here.</p>")
        else:
            rows = "".join(fm_row(fm) for fm in entries)
            parts.append(
                "<table><tr><th>When</th><th>Title</th><th>Duration</th><th>Tags</th></tr>"
                + rows + "</table>"
            )
        return page("Transcripts", "\n".join(parts))

    @app.get("/search")
    def search() -> str:
        q = (request.args.get("q") or "").strip()
        parts = ["<h1>Search</h1>", search_form(q)]
        if q:
            hits = store.search(q, limit=25)
            parts.append(f'<p class="muted">{len(hits)} transcript(s) matched.</p>')
            for hit in hits:
                fm = hit["metadata"]
                tid = html.escape(str(fm.get("id") or ""))
                title = html.escape(str(fm.get("title") or "(untitled)"))
                date = html.escape(str(fm.get("date") or ""))
                parts.append(f'<h2><a href="/transcript/{tid}">{title}</a> '
                             f'<span class="muted">{date}</span></h2>')
                for snippet in hit["snippets"]:
                    parts.append(f'<p class="snippet">…{html.escape(snippet)}…</p>')
        return page("Search", "\n".join(parts))

    @app.get("/transcript/<transcript_id>")
    def transcript(transcript_id: str) -> str:
        record = store.get_transcript(transcript_id)
        if record is None:
            abort(404)
        fm = record["metadata"]
        raw_title = str(fm.get("title") or "(untitled)")
        title = html.escape(raw_title)
        meta_bits: list[str] = []
        for label, key in (("Date", "date"), ("Start", "start_time"),
                           ("End", "end_time"), ("Language", "language"),
                           ("App", "app"), ("Model", "model")):
            value = fm.get(key)
            if value:
                meta_bits.append(f"<dt>{label}</dt><dd>{html.escape(str(value))}</dd>")
        participants = fm.get("participants") or []
        if participants:
            joined = ", ".join(html.escape(str(p)) for p in participants)
            meta_bits.append(f"<dt>Participants</dt><dd>{joined}</dd>")
        body_html = (
            f"<h1>{title}</h1>"
            f'<div class="meta-card"><dl>{"".join(meta_bits)}</dl></div>'
            + render_markdown_min(record["body"])
        )
        # Raw title here: the <title> placeholder is Jinja-autoescaped, so
        # passing the pre-escaped form would double-escape ("Q&amp;A").
        return page(raw_title, body_html)

    @app.errorhandler(404)
    def not_found(_e: Any) -> tuple[str, int]:
        return page("Not found", "<h1>Transcript not found</h1>"
                    '<p>It may have been deleted or re-saved. <a href="/">Back to the list.</a></p>'), 404

    return app


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
def serve_in_background(
    store: TranscriptStore, *, host: str = "127.0.0.1", port: int = 8765,
) -> threading.Thread:
    """Start the web UI on a daemon thread. Raises OSError if the port is busy.

    Uses ``werkzeug.serving.make_server`` (not ``app.run``) so the bind
    happens synchronously on the caller's thread — a busy port surfaces as
    an exception the caller can contain instead of killing a background
    thread silently.
    """
    from werkzeug.serving import make_server

    try:
        # threaded=True is essential: the default single-threaded server
        # services one connection at a time, so a browser's idle preconnect
        # socket (no request bytes) would block every real request behind it.
        server = make_server(host, port, create_app(store), threaded=True)
    except SystemExit as exc:
        # werkzeug calls sys.exit(1) when the port is taken — re-raise as a
        # normal error so a busy port can't take down the menu-bar app.
        raise OSError(f"could not bind {host}:{port} (address in use?)") from exc
    thread = threading.Thread(target=server.serve_forever, name="otis-web", daemon=True)
    thread.start()
    logger.info("Web UI serving on http://%s:%d", host, port)
    return thread


def main() -> None:
    """Standalone entry point: ``python -m src.web.server``."""
    import logging as _logging
    from pathlib import Path

    from src.config import load_user_config

    _logging.basicConfig(level=_logging.INFO)
    cfg = load_user_config()
    store = TranscriptStore(
        Path(cfg.get("storage", "transcript_dir", default="~/Otis/transcripts")).expanduser()
    )
    host = str(cfg.get("web", "host", default="127.0.0.1"))
    port = int(cfg.get("web", "port", default=8765))
    app = create_app(store)
    print(f"Otis web UI: http://{host}:{port}  (Ctrl-C to stop)")
    app.run(host=host, port=port, use_reloader=False)


# Back-compat with the old stub's entry point name.
run = main

if __name__ == "__main__":
    main()
