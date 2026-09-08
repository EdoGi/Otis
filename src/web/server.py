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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.storage.transcript_store import TranscriptStore

if TYPE_CHECKING:  # pragma: no cover
    from flask import Flask

logger = logging.getLogger(__name__)

#: Colours are declared once as tokens and re-declared for dark. The old
#: sheet set `color-scheme: light dark` — which asks the browser to flip the
#: page background — and then hardcoded #0a66c2 links and a #c33 badge, so in
#: dark mode both sat on near-black at painful contrast.
_PAGE_STYLE = """
  :root {
    color-scheme: light dark;
    --fg: #1a1a1a;  --bg: #ffffff;  --muted: #5c5c5c;
    --link: #0a66c2;  --rule: #d8d8d8;  --card: #f4f4f5;
    --danger-bg: #c62828;  --danger-fg: #ffffff;  --chip: #e6e6e8;
    --accent: #0a66c2;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fg: #e8e8ea;  --bg: #16171a;  --muted: #a0a0a8;
      --link: #79b8ff;  --rule: #33343a;  --card: #202126;
      --danger-bg: #ff6b6b;  --danger-fg: #1a1a1a;  --chip: #2c2d33;
      --accent: #79b8ff;
    }
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 56rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
         color: var(--fg); background: var(--bg); }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; }
  a { color: var(--link); text-decoration: none; } a:hover { text-decoration: underline; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--rule); }
  tr:hover td { background: var(--card); }
  form.search { margin: 1rem 0; display: flex; gap: 0.5rem; }
  input[type=text] { flex: 1; padding: 0.4rem 0.6rem; font-size: 1rem;
                     color: var(--fg); background: var(--bg);
                     border: 1px solid var(--rule); border-radius: 6px; }
  button { padding: 0.4rem 0.8rem; font-size: 0.95rem; cursor: pointer;
           color: var(--fg); background: var(--card);
           border: 1px solid var(--rule); border-radius: 6px; }
  button:hover { border-color: var(--accent); }
  .muted { color: var(--muted); font-size: 0.9rem; }
  .badge { background: var(--danger-bg); color: var(--danger-fg); border-radius: 4px;
           padding: 0 0.4rem; font-size: 0.78rem; margin-left: 0.4rem; }
  .tag { background: var(--chip); border-radius: 4px; padding: 0 0.4rem;
         font-size: 0.78rem; margin-right: 0.25rem; }
  .snippet { margin: 0.2rem 0 0.8rem; padding-left: 0.8rem;
             border-left: 3px solid var(--rule); }
  .meta-card { background: var(--card); border-radius: 8px; padding: 0.8rem 1rem;
               margin: 1rem 0; }
  .meta-card dt { font-weight: 600; } .meta-card dd { margin: 0 0 0.4rem 0; }
  .player { margin: 1rem 0; display: grid; gap: 0.5rem; }
  .player audio { width: 100%; }
  .player label { font-size: 0.85rem; color: var(--muted); }
  .toolbar { display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap; }
  .pager { display: flex; gap: 0.75rem; align-items: center; margin: 1rem 0; }
  a.ts { font-variant-numeric: tabular-nums; }
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
  <script>
  // Timestamps seek every player on the page: a line can come from either
  // stream, and seeking both keeps them lined up.
  document.addEventListener("click", function (ev) {
    var el = ev.target.closest("a.ts");
    if (!el) return;
    ev.preventDefault();
    var at = parseFloat(el.dataset.t || "0");
    document.querySelectorAll("audio").forEach(function (a) {
      a.currentTime = at;
      if (a.paused) { a.play().catch(function () {}); }
    });
  });
  document.addEventListener("click", function (ev) {
    if (!ev.target.matches("[data-copy]")) return;
    var src = document.getElementById(ev.target.dataset.copy);
    if (!src) return;
    navigator.clipboard.writeText(src.innerText).then(function () {
      var was = ev.target.textContent;
      ev.target.textContent = "Copied";
      setTimeout(function () { ev.target.textContent = was; }, 1200);
    }).catch(function () {});
  });
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Minimal, escape-first Markdown rendering (no external dependency)
# ---------------------------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
#: `**[04:31]**` and `**[1:04:31]**` — already HTML-escaped when this runs.
_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def _linkify_timestamps(text: str) -> str:
    """Turn ``[MM:SS]`` into something that seeks the audio player."""

    def repl(m: re.Match[str]) -> str:
        a, b, c = m.group(1), m.group(2), m.group(3)
        seconds = (
            int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
        )
        return (f'<a class="ts" href="#" data-t="{seconds}" '
                f'title="Play from here">{m.group(0)}</a>')

    return _TIMESTAMP_RE.sub(repl, text)


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
        paragraph.append(
            _linkify_timestamps(_BOLD_RE.sub(r"<strong>\1</strong>", stripped))
        )
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------
#: Rows per page. The list used to be a hard limit=200 with nothing after it,
#: so transcript 201 was simply unreachable from the web UI.
PAGE_SIZE = 50

#: What a browser will actually play. Everything else is offered as a download
#: rather than an <audio> element that silently shows a broken player.
_PLAYABLE_SUFFIXES = {".wav": "audio/wav", ".flac": "audio/flac",
                      ".m4a": "audio/mp4", ".mp3": "audio/mpeg"}


def create_app(
    store: TranscriptStore, audio_dir: str | Path | None = None,
) -> Flask:
    from flask import Flask, abort, render_template_string, request, send_file

    app = Flask("otis-web")
    root = Path(audio_dir).expanduser().resolve() if audio_dir else None

    def audio_path(fm: dict[str, Any], stream: str) -> Path | None:
        """Resolve a transcript's audio, refusing anything outside audio_dir."""
        if root is None or not fm.get("audio_available"):
            return None
        rel = (fm.get("audio_files") or {}).get(stream)
        if not rel:
            return None
        try:
            candidate = (root / str(rel)).resolve()
        except OSError:
            return None
        # The frontmatter is a file on disk and a file on disk can say
        # anything, including ../../../etc/passwd.
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

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

    def search_form(value: str = "", *, action: str = "/",
                    placeholder: str = "Search titles and transcript text…"
                    ) -> str:
        return (
            f'<form class="search" action="{action}" method="get">'
            f'<input type="text" name="q" value="{html.escape(value)}" '
            f'placeholder="{placeholder}">'
            f"<button>Search</button></form>"
        )

    @app.get("/")
    def index() -> str:
        """One page, one box. Searching titles and bodies together.

        There used to be two: `/` filtered titles, `/search` grepped bodies,
        and neither said the other existed — so a search that found nothing
        looked like an empty archive. This runs both and shows the body
        matches as snippets under the list.
        """
        q = (request.args.get("q") or "").strip()
        tag = (request.args.get("tag") or "").strip() or None
        participant = (request.args.get("participant") or "").strip() or None
        date_from = (request.args.get("from") or "").strip() or None
        date_to = (request.args.get("to") or "").strip() or None
        try:
            page_no = max(1, int(request.args.get("page") or 1))
        except (TypeError, ValueError):
            page_no = 1

        # One extra row tells us whether a next page exists without counting
        # the whole archive.
        wanted = page_no * PAGE_SIZE + 1
        entries = store.list_transcripts(
            query=q or None, tag=tag, participant=participant,
            date_from=date_from, date_to=date_to, limit=wanted,
        )
        start = (page_no - 1) * PAGE_SIZE
        window = entries[start:start + PAGE_SIZE]
        has_next = len(entries) > page_no * PAGE_SIZE

        parts = ["<h1>Meeting transcripts</h1>", search_form(q)]
        filtered = bool(q or tag or participant or date_from or date_to)
        if filtered:
            parts.append('<p class="muted">Filtered — <a href="/">clear</a></p>')

        if window:
            rows = "".join(fm_row(fm) for fm in window)
            parts.append(
                "<table><tr><th>When</th><th>Title</th><th>Duration</th>"
                "<th>Tags</th></tr>" + rows + "</table>"
            )
        elif not q:
            parts.append("<p>No transcripts yet. Record a meeting and it will "
                         "show up here.</p>")

        def page_link(n: int, label: str) -> str:
            args = {k: v for k, v in request.args.items() if k != "page"}
            args["page"] = str(n)
            query = "&".join(f"{html.escape(k)}={html.escape(str(v))}"
                             for k, v in args.items())
            return f'<a href="/?{query}">{label}</a>'

        if page_no > 1 or has_next:
            bits = ['<div class="pager">']
            bits.append(page_link(page_no - 1, "← Newer") if page_no > 1
                        else '<span class="muted">← Newer</span>')
            bits.append(f'<span class="muted">Page {page_no}</span>')
            bits.append(page_link(page_no + 1, "Older →") if has_next
                        else '<span class="muted">Older →</span>')
            bits.append("</div>")
            parts.append("".join(bits))

        if q:
            seen_ids = {str(fm.get("id")) for fm in window}
            hits = [h for h in store.search(q, limit=25)
                    if str((h["metadata"] or {}).get("id")) not in seen_ids]
            if hits:
                parts.append(f'<h2>Also found in the text of '
                             f'{len(hits)} transcript(s)</h2>')
            elif not window:
                parts.append(f"<p>Nothing matched "
                             f"{html.escape(q)!r}, in titles or in text.</p>")
            for hit in hits:
                fm = hit["metadata"]
                tid = html.escape(str(fm.get("id") or ""))
                title = html.escape(str(fm.get("title") or "(untitled)"))
                date = html.escape(str(fm.get("date") or ""))
                parts.append(f'<h2><a href="/transcript/{tid}">{title}</a> '
                             f'<span class="muted">{date}</span></h2>')
                for snippet in hit["snippets"]:
                    parts.append(f'<p class="snippet">…{html.escape(snippet)}…</p>')

        return page("Transcripts", "\n".join(parts))

    @app.get("/search")
    def search() -> Any:
        """Kept so old bookmarks land somewhere useful."""
        from flask import redirect

        q = (request.args.get("q") or "").strip()
        return redirect(f"/?q={html.escape(q)}" if q else "/")

    @app.get("/audio/<transcript_id>/<stream>")
    def audio(transcript_id: str, stream: str) -> Any:
        if stream not in ("mic", "system"):
            abort(404)
        record = store.get_transcript(transcript_id)
        if record is None:
            abort(404)
        path = audio_path(record["metadata"], stream)
        if path is None:
            abort(404)
        mimetype = _PLAYABLE_SUFFIXES.get(path.suffix.lower(),
                                          "application/octet-stream")
        # conditional=True gives us Range requests, which is what lets the
        # player seek instead of downloading the whole meeting first.
        return send_file(path, mimetype=mimetype, conditional=True)

    @app.get("/transcript/<transcript_id>/download")
    def download(transcript_id: str) -> Any:
        from flask import Response

        record = store.get_transcript(transcript_id)
        if record is None:
            abort(404)
        path = record["path"]
        return Response(
            Path(path).read_text(encoding="utf-8"),
            mimetype="text/markdown; charset=utf-8",
            headers={"Content-Disposition":
                     f'attachment; filename="{Path(path).name}"'},
        )

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
        players: list[str] = []
        for stream, label in (("mic", "You (microphone)"),
                              ("system", "Everyone else (system audio)")):
            path = audio_path(fm, stream)
            if path is None:
                continue
            if path.suffix.lower() not in _PLAYABLE_SUFFIXES:
                continue
            tid = html.escape(transcript_id)
            players.append(
                f"<div><label>{label}</label>"
                f'<audio controls preload="metadata" '
                f'src="/audio/{tid}/{stream}"></audio></div>'
            )
        player_html = (
            f'<div class="player">{"".join(players)}</div>'
            f'<p class="muted">Click any timestamp to play from there.</p>'
            if players else ""
        )

        toolbar = (
            f'<div class="toolbar">'
            f'<button data-copy="transcript-body">Copy text</button>'
            f'<a href="/transcript/{html.escape(transcript_id)}/download">'
            f"<button>Download .md</button></a>"
            f"</div>"
        )

        body_html = (
            f"<h1>{title}</h1>"
            f'<div class="meta-card"><dl>{"".join(meta_bits)}</dl></div>'
            + player_html
            + toolbar
            + f'<div id="transcript-body">{render_markdown_min(record["body"])}</div>'
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
    audio_dir: str | Path | None = None,
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
        server = make_server(host, port, create_app(store, audio_dir),
                             threaded=True)
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
