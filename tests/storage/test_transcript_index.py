"""Cover for audit finding C5 — every read path re-parsed the whole tree.

``list_transcripts`` (579 ms) and ``completed_session_ids`` (564 ms) at 1 000
transcripts, both on the rumps main thread, both after every transcription.
Measured breakdown: 773 ms of that is YAML parsing and 7.4 ms is walking the
tree, so :class:`_FrontmatterCache` caches the parse and keeps the walk.

That choice is only safe if the cache can never disagree with the disk, which
is what most of this file is about: transcripts are meant to be hand-edited,
``git checkout``-ed and deleted in Finder, and every one of those must be seen
on the very next call.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.storage.transcript_store import (
    TranscriptStore,
    render_with_frontmatter,
    split_frontmatter,
)


def fm(sid: str, **over) -> dict:
    base = {
        "id": sid,
        "title": f"Meeting {sid}",
        "date": "2026-07-01",
        "start_time": "10:30",
        "end_time": "11:00",
        "duration_minutes": 30,
        "language": "en",
        "participants": [],
        "tags": [],
        "audio_available": True,
    }
    base.update(over)
    return base


@pytest.fixture()
def store(tmp_path: Path) -> TranscriptStore:
    s = TranscriptStore(tmp_path / "transcripts")
    for i in range(5):
        s.save(fm(f"sess-{i}"), f"# Body {i}\n\ncontent")
    return s


# ---------------------------------------------------------------------------
# The point of the cache
# ---------------------------------------------------------------------------
def test_an_unchanged_tree_is_never_reparsed(store) -> None:
    store.list_transcripts(limit=50)
    after_first = store._fm_cache.parses
    assert after_first >= 5

    for _ in range(10):
        store.list_transcripts(limit=50)
        store.completed_session_ids()
        store.path_for("sess-3")

    assert store._fm_cache.parses == after_first, (
        "a tree nobody touched must cost one parse per file, ever"
    )


def test_every_read_path_shares_one_cache(store) -> None:
    store.completed_session_ids()
    baseline = store._fm_cache.parses
    assert store.path_for("sess-0") is not None
    assert store.has_completed_transcript("sess-0")
    assert store.get_transcript("sess-0") is not None
    assert store.list_transcripts(limit=50)
    assert store._fm_cache.parses == baseline


def test_only_the_changed_file_is_reparsed(store) -> None:
    store.list_transcripts(limit=50)
    baseline = store._fm_cache.parses

    store.update_frontmatter("sess-2", {"title": "Renamed"})
    store.list_transcripts(limit=50)

    assert store._fm_cache.parses == baseline + 1


# ---------------------------------------------------------------------------
# The cache must never disagree with the disk
# ---------------------------------------------------------------------------
def test_an_edit_made_outside_otis_is_seen_immediately(store) -> None:
    """The user opens the .md in an editor and fixes a title."""
    assert store.list_transcripts(limit=50)  # warm
    path = store.path_for("sess-1")
    assert path is not None
    head, body = split_frontmatter(path.read_text(encoding="utf-8"))
    head["title"] = "Edited By Hand"
    path.write_text(render_with_frontmatter(head, body), encoding="utf-8")

    titles = [t["title"] for t in store.list_transcripts(limit=50)]
    assert "Edited By Hand" in titles


def test_an_edit_that_preserves_file_size_is_still_seen(store) -> None:
    """Validating on size alone would miss this; the stamp carries mtime too."""
    store.list_transcripts(limit=50)
    path = store.path_for("sess-1")
    assert path is not None
    text = path.read_text(encoding="utf-8")
    swapped = text.replace("Meeting sess-1", "Meeting SESS-1")
    assert len(swapped) == len(text)
    path.write_text(swapped, encoding="utf-8")

    titles = [t["title"] for t in store.list_transcripts(limit=50)]
    assert "Meeting SESS-1" in titles


def test_a_file_deleted_outside_otis_disappears(store) -> None:
    """Finder, `rm`, or a `git checkout` of an older tree."""
    store.list_transcripts(limit=50)
    path = store.path_for("sess-4")
    assert path is not None
    path.unlink()

    assert store.path_for("sess-4") is None
    assert "sess-4" not in store.completed_session_ids()
    assert len(store.list_transcripts(limit=50)) == 4


def test_a_file_added_outside_otis_appears(store) -> None:
    store.list_transcripts(limit=50)
    new = store.root / "2026" / "07" / "2026-07-01_0900_dropped-in.md"
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_text(
        render_with_frontmatter(fm("dropped-in"), "# Dropped in\n\nbody"),
        encoding="utf-8",
    )

    assert store.path_for("dropped-in") == new
    assert "dropped-in" in store.completed_session_ids()


def test_deleted_files_do_not_accumulate_in_the_cache(store) -> None:
    store.list_transcripts(limit=50)
    assert len(store._fm_cache._entries) == 5

    for sid in ("sess-0", "sess-1", "sess-2"):
        p = store.path_for(sid)
        assert p is not None
        p.unlink()
    store.list_transcripts(limit=50)

    assert len(store._fm_cache._entries) == 2, "evict on a full pass, or it grows forever"


def test_a_file_vanishing_mid_walk_is_skipped(store, monkeypatch) -> None:
    """Retention trashing a transcript while the menu is being built."""
    real_stat = Path.stat
    doomed = store.path_for("sess-2")

    def flaky_stat(self, *a, **kw):
        if self == doomed:
            raise FileNotFoundError(str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    ids = {t["id"] for t in store.list_transcripts(limit=50)}
    assert "sess-2" not in ids
    assert len(ids) == 4


def test_save_is_visible_to_the_next_read(store) -> None:
    store.list_transcripts(limit=50)
    store.save(fm("brand-new", date="2026-07-02"), "# New\n\nbody")
    assert store.path_for("brand-new") is not None
    assert "brand-new" in store.completed_session_ids()


def test_a_failure_placeholder_rewritten_as_success_flips_completeness(store) -> None:
    store.save(fm("retry-me", status="failed", tags=["failed"]), "# failed\n\nretry")
    assert "retry-me" not in store.completed_session_ids()

    store.save(fm("retry-me"), "# Recovered\n\nbody")
    assert "retry-me" in store.completed_session_ids(), (
        "a stale cache here would keep audio pinned by retention forever"
    )


# ---------------------------------------------------------------------------
# Threads: the menu bar, the retention timer and transcription workers share it
# ---------------------------------------------------------------------------
def test_concurrent_readers_agree(store) -> None:
    seen: list[int] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(25):
                seen.append(len(store.list_transcripts(limit=50)))
                store.completed_session_ids()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert set(seen) == {5}


def test_a_writer_racing_readers_does_not_corrupt_results(store) -> None:
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                for entry in store.list_transcripts(limit=50):
                    assert entry.get("id"), "never yield half-parsed frontmatter"
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(30):
            store.update_frontmatter("sess-1", {"title": f"Rename {i}"})
    finally:
        stop.set()
        for t in threads:
            t.join()

    assert not errors
    assert store.get_transcript("sess-1")["metadata"]["title"] == "Rename 29"
