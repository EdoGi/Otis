"""Tests for src/mcp/core.py — the logic behind the MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

from src.mcp.core import (
    get_transcript_core,
    jsonify,
    list_transcripts_core,
    search_transcripts_core,
    store_from_config,
)
from src.storage.transcript_store import TranscriptStore


def _fm(sid: str, *, title: str, date: str, language: str = "en",
        tags: list[str] | None = None,
        participants: list[str] | None = None) -> dict:
    return {
        "id": sid,
        "title": title,
        "date": date,
        "start_time": "10:00",
        "end_time": "10:30",
        "duration_minutes": 30,
        "language": language,
        "app": "zoom.us",
        "participants": participants or ["Alice <alice@example.com>"],
        "tags": tags or [],
        "audio_files": {"mic": None, "system": None},
        "audio_available": False,
        "model": "small",
    }


def _seeded_store(tmp_path: Path) -> TranscriptStore:
    store = TranscriptStore(tmp_path / "transcripts")
    store.save(_fm("id-1", title="Sprint Planning", date="2026-05-01"),
               "## Transcript\n\nWe discussed the roadmap and budget.")
    store.save(_fm("id-2", title="Budget Review", date="2026-05-15",
                   language="fr", tags=["finance"],
                   participants=["Bob <bob@corp.com>"]),
               "## Transcript\n\nLe budget est approuvé.")
    store.save(_fm("id-3", title="1:1 with Bob", date="2026-06-01"),
               "## Transcript\n\nCareer growth and the budget freeze.")
    return store


def test_list_returns_newest_first_and_jsonable(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    out = list_transcripts_core(store)
    assert [fm["id"] for fm in out] == ["id-3", "id-2", "id-1"]
    json.dumps(out)  # must not raise


def test_list_filters_combine(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    assert [fm["id"] for fm in list_transcripts_core(store, date_from="2026-05-10",
                                                     date_to="2026-05-31")] == ["id-2"]
    assert [fm["id"] for fm in list_transcripts_core(store, language="fr")] == ["id-2"]
    assert [fm["id"] for fm in list_transcripts_core(store, tag="finance")] == ["id-2"]
    assert [fm["id"] for fm in list_transcripts_core(store, participant="bob")
            ] == ["id-2"]
    assert [fm["id"] for fm in list_transcripts_core(store, query="sprint")] == ["id-1"]
    assert len(list_transcripts_core(store, limit=2)) == 2


def test_list_clamps_pathological_limits(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    assert len(list_transcripts_core(store, limit=0)) >= 1   # clamped up to 1
    assert list_transcripts_core(store, limit=100000)        # clamped, no blowup


def test_search_returns_snippets(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    hits = search_transcripts_core(store, "budget")
    assert len(hits) == 3
    for hit in hits:
        assert any("budget" in s.lower() for s in hit["snippets"])
        assert isinstance(hit["path"], str)
    json.dumps(hits)


def test_get_transcript_roundtrip(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    out = get_transcript_core(store, "id-2")
    assert out["metadata"]["title"] == "Budget Review"
    assert "approuvé" in out["body"]
    json.dumps(out)


def test_get_transcript_unknown_id_returns_error_dict(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    out = get_transcript_core(store, "nope")
    assert "error" in out and "nope" in out["error"]


def test_jsonify_handles_paths_and_dates() -> None:
    from datetime import date, datetime

    out = jsonify({
        "p": Path("/tmp/x"),
        "d": date(2026, 5, 1),
        "ts": datetime(2026, 5, 1, 10, 0),
        "nested": [Path("/a"), {"k": Path("/b")}],
    })
    json.dumps(out)
    assert out["p"] == "/tmp/x"
    assert out["d"] == "2026-05-01"


def test_store_from_config_honours_user_override(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.yaml"
    target = tmp_path / "my-transcripts"
    user_cfg.write_text(f"storage:\n  transcript_dir: {target}\n", encoding="utf-8")
    store = store_from_config(user_config_path=user_cfg)
    assert store.root == target
