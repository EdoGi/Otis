"""Cover for audit finding E13 — the web UI's gaps.

Five, and each of them made the archive harder to use as it grew:

* list and search were separate pages that did not know about each other, so
  a search that found no title match looked like an empty archive;
* the list was a hard ``limit=200`` with nothing after it — transcript 201 was
  unreachable;
* no audio playback at all, so a timestamp was a number you could not act on;
* no way to copy or export a transcript;
* ``color-scheme: light dark`` asked the browser to flip the background, then
  the sheet hardcoded #0a66c2 links and a #c33 badge on top of it.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from src.storage.transcript_store import TranscriptStore
from src.web.server import PAGE_SIZE, create_app, render_markdown_min


def _fm(sid: str, *, title: str, date: str = "2026-06-01", **over) -> dict:
    fm = {
        "id": sid, "title": title, "date": date, "start_time": "10:00",
        "duration_minutes": 30, "language": "en", "tags": [],
        "audio_files": {"mic": None, "system": None}, "audio_available": False,
    }
    fm.update(over)
    return fm


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return path


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
@pytest.fixture()
def many(tmp_path: Path):
    store = TranscriptStore(tmp_path / "transcripts")
    for i in range(PAGE_SIZE * 2 + 5):
        store.save(_fm(f"s-{i:03d}", title=f"Meeting {i:03d}",
                       date=f"2026-06-{(i % 28) + 1:02d}"), "## Transcript\n\nbody\n")
    return create_app(store).test_client()


def test_the_list_is_paged(many) -> None:
    text = many.get("/").get_data(as_text=True)
    assert text.count('href="/transcript/') == PAGE_SIZE


def test_later_pages_are_reachable(many) -> None:
    """Transcript 201 used to be unreachable from the web UI at all."""
    page2 = many.get("/?page=2").get_data(as_text=True)
    assert 'href="/transcript/' in page2
    assert "Page 2" in page2
    page1 = many.get("/").get_data(as_text=True)
    assert page1 != page2


def test_the_first_page_offers_no_newer_link(many) -> None:
    text = many.get("/").get_data(as_text=True)
    assert "Older →" in text
    assert 'href="/?page=0' not in text


def test_the_last_page_offers_no_older_link(many) -> None:
    text = many.get("/?page=3").get_data(as_text=True)
    assert "Page 3" in text
    assert "/?page=4" not in text


def test_a_nonsense_page_number_falls_back_to_the_first(many) -> None:
    assert many.get("/?page=banana").status_code == 200
    assert many.get("/?page=-4").status_code == 200


def test_paging_keeps_the_search(many) -> None:
    text = many.get("/?q=Meeting&page=2").get_data(as_text=True)
    assert "q=Meeting" in text


# ---------------------------------------------------------------------------
# One search box
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path: Path):
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("t-title", title="Budget review"), "## Transcript\n\nnothing\n")
    store.save(_fm("t-body", title="Unrelated", date="2026-06-02"),
               "## Transcript\n\n**[00:05]** Me: we should discuss the budget\n")
    return create_app(store).test_client()


def test_one_box_finds_titles_and_text_together(client) -> None:
    text = client.get("/?q=budget").get_data(as_text=True)
    assert "Budget review" in text      # title match, in the table
    assert "Unrelated" in text          # body match, as a snippet


def test_a_body_match_is_not_repeated_in_the_snippets(client) -> None:
    """It is already in the table above; twice reads as two transcripts."""
    text = client.get("/?q=Unrelated").get_data(as_text=True)
    assert text.count('href="/transcript/t-body"') == 1


def test_a_search_with_no_hits_says_so(client) -> None:
    text = client.get("/?q=zzzznothing").get_data(as_text=True)
    assert "Nothing matched" in text
    assert "No transcripts yet" not in text, "an empty result is not an empty archive"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
@pytest.fixture()
def with_audio(tmp_path: Path):
    audio = tmp_path / "audio"
    _wav(audio / "2026" / "06" / "clip_mic.wav")
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("a-1", title="With audio", audio_available=True,
                   audio_files={"mic": "2026/06/clip_mic.wav", "system": None}),
               "## Transcript\n\n**[01:05]** Me: hello\n")
    return create_app(store, audio_dir=audio).test_client(), audio


def test_a_transcript_with_audio_gets_a_player(with_audio) -> None:
    client, _ = with_audio
    text = client.get("/transcript/a-1").get_data(as_text=True)
    assert '<audio controls' in text
    assert 'src="/audio/a-1/mic"' in text


def test_the_audio_is_served_and_seekable(with_audio) -> None:
    """Range support is what lets the player seek instead of downloading."""
    client, _ = with_audio
    resp = client.get("/audio/a-1/mic")
    assert resp.status_code == 200
    assert resp.headers.get("Accept-Ranges") == "bytes"


def test_timestamps_become_seek_links(with_audio) -> None:
    client, _ = with_audio
    text = client.get("/transcript/a-1").get_data(as_text=True)
    assert 'data-t="65"' in text, "01:05 is 65 seconds in"


def test_an_hour_long_timestamp_is_understood() -> None:
    out = render_markdown_min("**[1:04:31]** Me: still going")
    assert 'data-t="3871"' in out


def test_a_transcript_without_audio_gets_no_player(client) -> None:
    text = client.get("/transcript/t-title").get_data(as_text=True)
    assert "<audio" not in text


def test_audio_is_refused_when_the_transcript_says_it_is_gone(tmp_path) -> None:
    audio = tmp_path / "audio"
    _wav(audio / "clip_mic.wav")
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("gone", title="Deleted audio", audio_available=False,
                   audio_files={"mic": "clip_mic.wav"}), "## Transcript\n\nx\n")
    client = create_app(store, audio_dir=audio).test_client()
    assert client.get("/audio/gone/mic").status_code == 404


def test_frontmatter_cannot_escape_the_audio_directory(tmp_path) -> None:
    """A .md file is editable by anything; it must not become a file reader."""
    audio = tmp_path / "audio"
    audio.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("token", encoding="utf-8")
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("evil", title="Escape", audio_available=True,
                   audio_files={"mic": "../secret.txt"}), "## Transcript\n\nx\n")

    client = create_app(store, audio_dir=audio).test_client()

    assert client.get("/audio/evil/mic").status_code == 404


def test_an_unknown_stream_name_is_refused(with_audio) -> None:
    client, _ = with_audio
    assert client.get("/audio/a-1/passwd").status_code == 404


def test_only_mic_and_system_are_servable(tmp_path) -> None:
    """Frontmatter listing an extra stream must not make it fetchable.

    Path containment already stops it reaching outside audio_dir; this bounds
    what the route serves to the two streams the recorder writes, even for a
    file that is inside.
    """
    audio = tmp_path / "audio"
    _wav(audio / "clip_mic.wav")
    _wav(audio / "extra.wav")
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("multi", title="Extra stream", audio_available=True,
                   audio_files={"mic": "clip_mic.wav", "system": None,
                                "notes": "extra.wav"}),
               "## Transcript\n\nx\n")

    client = create_app(store, audio_dir=audio).test_client()

    assert client.get("/audio/multi/mic").status_code == 200
    assert client.get("/audio/multi/notes").status_code == 404


def test_audio_is_off_when_no_directory_was_configured(tmp_path) -> None:
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("a-1", title="x", audio_available=True,
                   audio_files={"mic": "a.wav"}), "## Transcript\n\nx\n")
    assert create_app(store).test_client().get("/audio/a-1/mic").status_code == 404


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_a_transcript_can_be_downloaded(client) -> None:
    resp = client.get("/transcript/t-title/download")
    assert resp.status_code == 200
    assert "attachment" in resp.headers["Content-Disposition"]
    assert "budget review" in resp.get_data(as_text=True).lower()


def test_downloading_something_that_is_not_there_is_a_404(client) -> None:
    assert client.get("/transcript/nope/download").status_code == 404


def test_the_page_offers_copy_and_download(client) -> None:
    text = client.get("/transcript/t-title").get_data(as_text=True)
    assert 'data-copy="transcript-body"' in text
    assert "/transcript/t-title/download" in text


# ---------------------------------------------------------------------------
# Dark mode
# ---------------------------------------------------------------------------
def test_colours_are_tokens_not_hardcoded(client) -> None:
    text = client.get("/").get_data(as_text=True)
    assert "prefers-color-scheme: dark" in text
    assert "#0a66c2" in text, "the light value still exists as a token"
    # ...but not as the only definition: the dark block must redefine it.
    dark = text.split("prefers-color-scheme: dark", 1)[1]
    assert "--link:" in dark
    assert "--danger-bg:" in dark


def test_body_paints_its_own_background(client) -> None:
    """Otherwise the text flips with the OS and the background does not."""
    text = client.get("/").get_data(as_text=True)
    assert "background: var(--bg)" in text
    assert "color: var(--fg)" in text
