"""Tests for src/storage/transcript_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.storage.transcript_store import (
    TranscriptStore,
    render_with_frontmatter,
    slugify,
    split_frontmatter,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _fm(
    *,
    session_id: str = "11111111-1111-1111-1111-111111111111",
    title: str = "Sprint Planning",
    date: str = "2026-05-09",
    start_time: str = "14:00",
    language: str = "fr",
    participants: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": session_id,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": "14:30",
        "duration_minutes": 30,
        "language": language,
        "app": "zoom.us",
        "participants": participants or ["Alice <alice@example.com>"],
        "tags": tags or [],
        "audio_files": {
            "mic": f"2026/05/{date}_1400_mic.wav",
            "system": f"2026/05/{date}_1400_system.wav",
        },
        "audio_available": True,
        "model": "small",
    }


def _populate(store: TranscriptStore, *frontmatters: dict, body: str = "## Transcript\n\nbody") -> list[Path]:
    return [store.save(fm, body) for fm in frontmatters]


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Sprint Planning", "sprint-planning"),
        ("Sprint  Planning!", "sprint-planning"),
        ("  Café del Mar  ", "café-del-mar"),
        ("", "untitled"),
        ("/", "untitled"),
        ("a" * 100, "a" * 50),
    ],
)
def test_slugify(title: str, expected: str) -> None:
    assert slugify(title) == expected


# ---------------------------------------------------------------------------
# split / render frontmatter
# ---------------------------------------------------------------------------
def test_render_and_split_round_trip() -> None:
    fm = {"id": "x", "title": "Hello"}
    body = "# Hello\n\nbody text\n"
    rendered = render_with_frontmatter(fm, body)
    parsed_fm, parsed_body = split_frontmatter(rendered)
    assert parsed_fm == fm
    assert parsed_body.strip() == body.strip()


def test_split_handles_missing_frontmatter() -> None:
    fm, body = split_frontmatter("# just markdown\n\nno yaml here")
    assert fm == {}
    assert body.startswith("#")


def test_split_handles_corrupt_yaml() -> None:
    bad = "---\n: : :\n---\n\nbody"
    fm, _ = split_frontmatter(bad)
    assert fm == {}


# ---------------------------------------------------------------------------
# Save / read
# ---------------------------------------------------------------------------
def test_save_writes_to_year_month_tree(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    path = store.save(_fm(), body="## Transcript\n\nhello")
    assert path == tmp_path / "2026" / "05" / "2026-05-09_1400_sprint-planning.md"
    assert path.exists()


def test_save_round_trip(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    fm = _fm()
    path = store.save(fm, "## Transcript\n\nhello")
    loaded = store.get_transcript(fm["id"])
    assert loaded is not None
    assert loaded["metadata"]["id"] == fm["id"]
    assert loaded["body"].strip().startswith("## Transcript")
    assert loaded["path"] == path


def test_save_atomic_no_partial_files_on_disk(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(), "## Transcript")
    leftover = list(tmp_path.rglob("*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# list_transcripts
# ---------------------------------------------------------------------------
def test_list_returns_most_recent_first(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    _populate(
        store,
        _fm(session_id="a", date="2026-05-09", start_time="14:00"),
        _fm(session_id="b", date="2026-05-09", start_time="09:00"),
        _fm(session_id="c", date="2026-04-30", start_time="10:00"),
    )
    listed = store.list_transcripts()
    assert [fm["id"] for fm in listed] == ["a", "b", "c"]


def test_list_filters_by_date_range(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    _populate(
        store,
        _fm(session_id="old", date="2026-04-01"),
        _fm(session_id="new", date="2026-05-09"),
    )
    listed = store.list_transcripts(date_from="2026-05-01")
    assert [fm["id"] for fm in listed] == ["new"]


def test_list_filters_by_language_and_tag(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    _populate(
        store,
        _fm(session_id="fr", language="fr", tags=["work"]),
        _fm(session_id="en", language="en", tags=[], date="2026-05-08"),
    )
    by_lang = store.list_transcripts(language="fr")
    assert [fm["id"] for fm in by_lang] == ["fr"]
    by_tag = store.list_transcripts(tag="work")
    assert [fm["id"] for fm in by_tag] == ["fr"]


def test_list_filters_by_participant_substring(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    _populate(
        store,
        _fm(session_id="alice", participants=["Alice <alice@x.com>"]),
        _fm(session_id="bob", participants=["Bob <bob@x.com>"], date="2026-05-08"),
    )
    listed = store.list_transcripts(participant="alice")
    assert [fm["id"] for fm in listed] == ["alice"]


def test_list_recursively_scans_subdirectories(tmp_path: Path) -> None:
    """Files in YYYY/MM/ must be discovered, not just the root."""
    store = TranscriptStore(tmp_path)
    _populate(
        store,
        _fm(session_id="2026", date="2026-05-09"),
        _fm(session_id="2025", date="2025-12-15"),
    )
    listed = store.list_transcripts()
    assert {fm["id"] for fm in listed} == {"2026", "2025"}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search_finds_body_text(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(session_id="a"), "## Transcript\n\nWe shipped the **deadline** feature today.")
    store.save(_fm(session_id="b", date="2026-05-08"), "## Transcript\n\nNothing relevant.")
    hits = store.search("deadline")
    assert len(hits) == 1
    assert hits[0]["metadata"]["id"] == "a"
    assert any("deadline" in s for s in hits[0]["snippets"])


def test_search_returns_empty_for_blank_query(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(), "## Transcript\n\nhello")
    assert store.search("") == []


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------
def test_add_remove_tag(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(), "## Transcript")
    assert store.add_tag("11111111-1111-1111-1111-111111111111", "important")
    fm = store.get_transcript("11111111-1111-1111-1111-111111111111")["metadata"]
    assert "important" in fm["tags"]
    assert store.remove_tag("11111111-1111-1111-1111-111111111111", "important")
    fm = store.get_transcript("11111111-1111-1111-1111-111111111111")["metadata"]
    assert fm["tags"] == []


def test_add_tag_idempotent(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(), "## Transcript")
    sid = "11111111-1111-1111-1111-111111111111"
    store.add_tag(sid, "x")
    store.add_tag(sid, "x")
    fm = store.get_transcript(sid)["metadata"]
    assert fm["tags"] == ["x"]


# ---------------------------------------------------------------------------
# delete (soft)
# ---------------------------------------------------------------------------
def test_delete_moves_to_trash(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    path = store.save(_fm(), "## Transcript")
    target = store.delete_transcript("11111111-1111-1111-1111-111111111111")
    assert target is not None
    assert not path.exists()
    assert target.exists()
    assert target.parent == store.trash_dir


def test_trashed_transcript_not_listed(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    store.save(_fm(session_id="alive"), "## Transcript")
    store.save(_fm(session_id="dead", date="2026-05-08"), "## Transcript")
    store.delete_transcript("dead")
    listed = store.list_transcripts()
    assert {fm["id"] for fm in listed} == {"alive"}


# ---------------------------------------------------------------------------
# audio_available marker
# ---------------------------------------------------------------------------
def test_mark_audio_unavailable_updates_frontmatter(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path)
    fm = _fm(session_id="x")
    fm["audio_available"] = True
    store.save(fm, "## Transcript")
    store.mark_audio_unavailable("x")
    fresh = store.get_transcript("x")["metadata"]
    assert fresh["audio_available"] is False
