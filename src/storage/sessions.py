"""Discovering recordings on disk that still need a transcript.

Lifted out of ``src/ui/menubar.py`` so ``otis run`` can use it. The daemon had
no orphan scan at all, so a crash or a reboot left recordings that nothing
would ever look at again — the menu bar's "Generate Transcript" menu was the
only thing that knew how to find them, and only a user who opened it.

Read-only by design: this looks, it never repairs. A listing that rewrites WAV
headers is a listing you cannot run from a menu build (see P4 item 29).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


def find_orphan_sessions(
    *,
    audio_dir: Path,
    store: TranscriptStore | None = None,
    current_recording_session_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return audio sessions on disk without a corresponding transcript.

    Three on-disk layouts are accepted:

    * ``{uuid}_metadata.json`` next to ``{uuid}_*.wav`` — fresh recorder
      output (typically at ``audio_dir`` root).
    * ``YYYY-MM-DD_HHMM_metadata.json`` under ``audio_dir/YYYY/MM/`` —
      already renamed by a successful :class:`TranscriptProcessor` run.
    * Orphan ``{uuid}_*.wav`` at the audio root with **no** metadata.json
      — recorder.stop() crashed mid-write. The caller can still transcribe
      these via a synthesised metadata dict.

    Read-only: this only ever looks. Repairing a crash-truncated header is
    the job of whoever transcribes the session
    (:meth:`MenuBarApp._repair_then_transcribe`), because a listing that
    rewrites files is a listing you cannot run from a menu build.

    ``current_recording_session_id`` excludes the session being actively
    recorded right now — its WAV files are still being written and would
    otherwise show up as a tempting (but corrupted) orphan.
    """
    if not audio_dir.exists():
        return []

    # One store scan up front instead of a full rglob+parse per candidate.
    # Failure placeholders don't count — those sessions must stay retryable.
    completed_ids: set[str] = set()
    if store is not None:
        try:
            completed_ids = store.completed_session_ids()
        except Exception:
            logger.exception("Could not index completed transcripts")

    results: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()
    # We also track resolved WAV paths from Pass 1 so Pass 2 doesn't list
    # the same physical files under a filename-derived "session id" that
    # differs from the metadata's id (e.g. renamed YYYY/MM/HH-MM_mic.wav
    # belongs to a UUID session, but the filename itself looks like a date).
    seen_audio_paths: set[Path] = set()
    if current_recording_session_id:
        seen_session_ids.add(current_recording_session_id)

    # Pass 1 — sessions with explicit metadata.json (preferred).
    for meta_path in sorted(audio_dir.rglob("*_metadata.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        sid = str(meta.get("session_id") or "")
        if not sid or sid in seen_session_ids:
            continue
        mic_relative = meta.get("mic_wav") or f"{sid}_mic.wav"
        sys_relative = meta.get("system_wav")
        mic_path, sys_path = _resolve_session_audio(
            meta_path=meta_path,
            declared_mic=mic_relative,
            declared_sys=sys_relative,
            sid=sid,
        )
        if mic_path is None:
            continue
        seen_audio_paths.add(mic_path)
        if sys_path is not None:
            seen_audio_paths.add(sys_path)
        if sid in completed_ids:
            seen_session_ids.add(sid)
            continue
        seen_session_ids.add(sid)
        results.append({
            "session_id": sid,
            "label": _format_session_label(meta_path, mic_path),
            "metadata_path": meta_path,
            "mic_path": mic_path,
            "system_path": sys_path,
        })
        if len(results) >= limit:
            return results

    # Pass 2 — orphan WAVs anywhere (rglob, in case a partial recording
    # ended up inside the YYYY/MM tree somehow). Skip the actively-recording
    # session, anything already covered by Pass 1, and anything that
    # already has a transcript.
    for mic_path in sorted(audio_dir.rglob("*_mic.wav"), reverse=True):
        if mic_path in seen_audio_paths:
            continue
        sid = mic_path.stem.removesuffix("_mic")
        if sid in seen_session_ids:
            continue
        if sid in completed_ids:
            seen_session_ids.add(sid)
            continue
        seen_session_ids.add(sid)
        sys_path = mic_path.with_name(f"{sid}_system.wav")
        results.append({
            "session_id": sid,
            "label": _format_session_label(mic_path, mic_path) + "  (orphan)",
            "metadata_path": None,
            "mic_path": mic_path,
            "system_path": sys_path if sys_path.exists() else None,
        })
        if len(results) >= limit:
            return results
    return results


def _resolve_session_audio(
    *,
    meta_path: Path,
    declared_mic: str,
    declared_sys: str | None,
    sid: str,
) -> tuple[Path | None, Path | None]:
    """Match the three on-disk layouts retranscribe.py also handles."""
    candidates_mic = [
        meta_path.parent / declared_mic,
        meta_path.with_name(meta_path.name.replace("_metadata.json", "_mic.wav")),
        meta_path.parent / f"{sid}_mic.wav",
    ]
    mic = next((c for c in candidates_mic if c.exists()), None)
    if mic is None:
        return None, None
    sys_path: Path | None = None
    if declared_sys:
        cands = [
            meta_path.parent / declared_sys,
            meta_path.with_name(meta_path.name.replace("_metadata.json", "_system.wav")),
            meta_path.parent / f"{sid}_system.wav",
        ]
        sys_path = next((c for c in cands if c.exists()), None)
    else:
        sibling = meta_path.with_name(
            meta_path.name.replace("_metadata.json", "_system.wav")
        )
        if sibling.exists():
            sys_path = sibling
    return mic, sys_path


def _format_session_label(meta_path: Path | None, mic_path: Path) -> str:
    """Pretty label for the submenu — best-effort timestamp + size."""
    anchor = meta_path or mic_path
    from datetime import datetime
    try:
        mtime = datetime.fromtimestamp(anchor.stat().st_mtime)
        ts = mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts = "unknown"
    try:
        mb = mic_path.stat().st_size / (1024 * 1024)
        size = f"  ({mb:.1f} MB)"
    except Exception:
        size = ""
    return f"{ts}{size}"
