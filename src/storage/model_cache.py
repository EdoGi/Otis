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


#: Approximate download size per model (MB), for the picker's warning. medium,
#: small and tiny are measured from this Mac's cache; base and large-v3 are
#: estimated from their parameter counts against those three.
DOWNLOAD_MB = {"tiny": 74, "base": 148, "small": 481, "medium": 1525,
               "large": 3100, "large-v3": 3100}


def cached_model_names(hub_dir: Path | None = None) -> set[str]:
    """Names of the Whisper models on disk — a directory listing, no sizing.

    list_cached_models() walks every file to size it; the picker only needs
    to know which exist.
    """
    hub = Path(hub_dir) if hub_dir is not None else DEFAULT_HUB_DIR
    try:
        entries = list(hub.iterdir())
    except OSError:
        return set()
    return {
        e.name[len(_MODEL_PREFIX):].removesuffix("-mlx")
        for e in entries
        if e.is_dir() and e.name.startswith(_MODEL_PREFIX) and _has_complete_snapshot(e)
    }


def _has_complete_snapshot(repo_dir: Path) -> bool:
    """A snapshot holding config.json and weights — not just a repo folder.

    huggingface_hub creates the folder before any weights arrive, so an
    interrupted download (network, quit, a full disk) left a directory that
    looked "cached" while the next transcription downloaded it all again.
    """
    for snap in (repo_dir / "snapshots").glob("*"):
        if (snap / "config.json").exists() and any(snap.glob("weights.*")):
            return True
    return False


def is_model_cached(model: str, cached: set[str]) -> bool:
    """Whether config model ``model`` is among ``cached`` ("large" is v3)."""
    from src.transcription.whisper_engine import _MODEL_REPO_MAP

    key = str(model).lower()
    repo = _MODEL_REPO_MAP.get(key, "")
    name = repo.removeprefix("mlx-community/whisper-").removesuffix("-mlx") or key
    return name in {c.lower() for c in cached}


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
    """Cached models other than the one currently configured.

    Through is_model_cached, so the config alias "large" keeps large-v3: a
    literal name comparison would have listed — and pruned — the active model.
    """
    return [m for m in list_cached_models(hub_dir)
            if not is_model_cached(str(active_model), {m.name})]


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
    "DEFAULT_HUB_DIR",
    "CachedModel",
    "free_bytes",
    "list_cached_models",
    "prune_models",
    "reclaimable_bytes",
    "unused_models",
]
