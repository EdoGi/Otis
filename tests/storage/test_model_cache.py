"""P3 item 27 — reclaim disk from Whisper models nothing uses.

mlx-whisper downloads weights into the HuggingFace hub cache and never removes
them, so switching model once leaves the old one behind forever. On this
machine `small` (459 MB) sat beside the `medium` actually configured — more
reclaimable space than the entire audio store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.model_cache import (
    free_bytes,
    list_cached_models,
    prune_models,
    reclaimable_bytes,
    unused_models,
)

PREFIX = "models--mlx-community--whisper-"


def _model(hub: Path, name: str, *, size: int = 4096, symlinked: bool = False) -> Path:
    root = hub / f"{PREFIX}{name}-mlx"
    blobs, snaps = root / "blobs", root / "snapshots" / "abc123"
    blobs.mkdir(parents=True); snaps.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(b"\0" * size)
    if symlinked:
        # Exactly how the real hub cache is laid out.
        (snaps / "weights.npz").symlink_to(blob)
    return root


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    h = tmp_path / "hub"; h.mkdir()
    return h


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
def test_lists_cached_models_largest_first(hub: Path) -> None:
    _model(hub, "small", size=1000)
    _model(hub, "medium", size=5000)
    names = [m.name for m in list_cached_models(hub)]
    assert names == ["medium", "small"]


def test_symlinked_blobs_are_counted_once(hub: Path) -> None:
    """The hub symlinks snapshots/<rev>/file at blobs/<sha>. Following those
    double-counted every weight file — it reported 963 MB reclaimable where du
    said 459 MB, which would have been a lie about what the user stood to gain.
    """
    _model(hub, "small", size=10_000, symlinked=True)
    (model,) = list_cached_models(hub)
    assert model.size_bytes == 10_000


def test_unrelated_cache_entries_are_ignored(hub: Path) -> None:
    (hub / "models--some--other-model").mkdir(parents=True)
    (hub / "datasets--foo").mkdir(parents=True)
    _model(hub, "small")
    assert [m.name for m in list_cached_models(hub)] == ["small"]


def test_a_missing_hub_is_not_an_error(tmp_path: Path) -> None:
    assert list_cached_models(tmp_path / "nope") == []


def test_reclaimable_excludes_the_active_model(hub: Path) -> None:
    _model(hub, "small", size=1000)
    _model(hub, "medium", size=5000)
    assert reclaimable_bytes("medium", hub) == 1000
    assert reclaimable_bytes("small", hub) == 5000


def test_active_model_matching_is_case_insensitive(hub: Path) -> None:
    _model(hub, "medium", size=5000)
    assert unused_models("MEDIUM", hub) == []


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def test_pruning_removes_only_the_unused(hub: Path) -> None:
    small = _model(hub, "small", size=1000)
    medium = _model(hub, "medium", size=5000)

    removed = prune_models("medium", hub_dir=hub)

    assert [m.name for m in removed] == ["small"]
    assert not small.exists()
    assert medium.exists(), "the configured model must survive"


def test_the_active_model_is_never_removed(hub: Path) -> None:
    medium = _model(hub, "medium", size=5000)
    assert prune_models("medium", hub_dir=hub) == []
    assert medium.exists()


def test_dry_run_deletes_nothing(hub: Path) -> None:
    small = _model(hub, "small", size=1000)
    _model(hub, "medium", size=5000)

    removed = prune_models("medium", hub_dir=hub, dry_run=True)

    assert [m.name for m in removed] == ["small"]
    assert small.exists(), "a dry run must not touch the disk"


def test_pruning_an_empty_cache_is_a_no_op(hub: Path) -> None:
    assert prune_models("medium", hub_dir=hub) == []


def test_a_model_that_cannot_be_removed_is_survivable(hub, monkeypatch) -> None:
    _model(hub, "small", size=1000)
    import src.storage.model_cache as mc

    monkeypatch.setattr(
        mc.shutil, "rmtree",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("read-only")),
    )
    assert prune_models("medium", hub_dir=hub) == []   # reported, not raised


# ---------------------------------------------------------------------------
# Free space
# ---------------------------------------------------------------------------
def test_free_bytes_reads_the_volume(tmp_path: Path) -> None:
    assert free_bytes(tmp_path) > 0


def test_free_bytes_on_a_missing_path_is_zero_not_an_error(tmp_path: Path) -> None:
    assert free_bytes(tmp_path / "does" / "not" / "exist") == 0
