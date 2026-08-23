"""Whisper model cache accounting.

mlx-whisper downloads model weights into the HuggingFace hub cache and never
removes them. Switching model once leaves the old one behind forever: on this
machine ``small`` (459 MB) sat unused beside the ``medium`` (1.4 GB) actually
configured — more reclaimable space than the entire audio store.

Pruning is deliberately an explicit action rather than something that happens
on its own. Deleting a 1.4 GB download the user might switch back to, without
being asked, trades a slow re-download for disk they may not need.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where huggingface_hub keeps downloaded weights.
DEFAULT_HUB_DIR = Path("~/.cache/huggingface/hub").expanduser()

#: Only ever touch directories that are clearly Whisper models we fetched.
_MODEL_PREFIX = "models--mlx-community--whisper-"


@dataclass(frozen=True)
class CachedModel:
    name: str          # "small", "medium", "large-v3"
    path: Path
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


def _dir_size(path: Path) -> int:
    """Bytes actually occupied, counting each blob once.

    The hub cache symlinks ``snapshots/<rev>/model.npz`` at ``blobs/<sha>``.
    ``Path.is_file()`` follows symlinks, so the naive walk counts every weight
    file twice — it reported 963 MB reclaimable where ``du`` says 459 MB, which
    would have been a lie to the user about how much they stood to gain.
    """
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_symlink():
                continue  # the blob it points at is counted on its own
            if p.is_file():
                total += p.stat().st_size
        except OSError:  # pragma: no cover (races / permissions)
            continue
    return total


def list_cached_models(hub_dir: Path | None = None) -> list[CachedModel]:
    """Every Whisper model currently taking up space, largest first."""
    hub = Path(hub_dir) if hub_dir is not None else DEFAULT_HUB_DIR
    if not hub.exists():
        return []
    out: list[CachedModel] = []
    for entry in hub.iterdir():
        if not entry.is_dir() or not entry.name.startswith(_MODEL_PREFIX):
            continue
        name = entry.name[len(_MODEL_PREFIX):].removesuffix("-mlx")
        out.append(CachedModel(name=name, path=entry, size_bytes=_dir_size(entry)))
    return sorted(out, key=lambda m: m.size_bytes, reverse=True)


def unused_models(
    active_model: str, hub_dir: Path | None = None
) -> list[CachedModel]:
    """Cached models other than the one currently configured."""
    active = str(active_model).lower()
    return [m for m in list_cached_models(hub_dir) if m.name.lower() != active]


def reclaimable_bytes(active_model: str, hub_dir: Path | None = None) -> int:
    return sum(m.size_bytes for m in unused_models(active_model, hub_dir))


def prune_models(
    active_model: str, *, hub_dir: Path | None = None, dry_run: bool = False,
) -> list[CachedModel]:
    """Delete cached models other than ``active_model``. Returns what went.

    Refuses to delete the active model even if asked, because re-downloading
    it is the difference between the next meeting transcribing and not.
    """
    victims = unused_models(active_model, hub_dir)
    removed: list[CachedModel] = []
    for model in victims:
        if dry_run:
            removed.append(model)
            continue
        try:
            shutil.rmtree(model.path)
            logger.info("Removed cached model %s (%.0f MB)", model.name, model.size_mb)
            removed.append(model)
        except Exception as exc:
            logger.warning("Could not remove cached model %s: %s", model.name, exc)
    return removed


def free_bytes(path: Path) -> int:
    """Bytes available on the volume holding ``path``. 0 if it can't be read."""
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free)
    except Exception:  # pragma: no cover (unmounted volume)
        return 0


__all__ = [
    "CachedModel",
    "DEFAULT_HUB_DIR",
    "free_bytes",
    "list_cached_models",
    "prune_models",
    "reclaimable_bytes",
    "unused_models",
]
