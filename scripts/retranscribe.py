"""Re-run the transcription pipeline against an already-recorded session.

Useful when:

* a previous transcription crashed (ffmpeg missing, OOM, etc.) and the audio
  is still on disk under ``~/Otis/audio/`` as ``{uuid}_*.wav``,
* you switched to a larger Whisper model and want to redo old recordings,
* the merge logic improved and you want fresh transcripts.

Usage::

    python scripts/retranscribe.py                          # all UUID sessions
    python scripts/retranscribe.py 82df2578                 # session prefix
    python scripts/retranscribe.py --model medium 82df2578  # override model
    python scripts/retranscribe.py --language fr            # force language

The script reads the bundled config + ~/.otis/config.yaml, builds the same
pipeline the menu bar uses, and writes one transcript per session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_user_config
from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import (
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
)
from src.transcription.whisper_engine import WhisperEngine


def _list_sessions(audio_dir: Path) -> list[Path]:
    """Return every metadata.json under audio_dir (UUID-named or otherwise)."""
    return sorted(audio_dir.rglob("*_metadata.json"))


def _matches(prefix: str | None, meta_path: Path) -> bool:
    if not prefix:
        return True
    return prefix in meta_path.name


def _resolve_audio_path(
    declared: str | None,
    audio_dir: Path,
    meta_path: Path,
    *,
    suffix: str,
    sid: str,
) -> Path | None:
    """Find the WAV referenced by ``declared`` (relative to ``audio_dir``).

    Falls back to two heuristics if the declared path doesn't exist:

    1. A sibling of ``meta_path`` whose name is the metadata-file prefix
       plus ``suffix`` — this catches the post-rename layout where metadata
       got moved into ``YYYY/MM/`` but its content still references UUIDs.
    2. ``audio_dir/{sid}{suffix}`` — the original UUID layout.
    """
    candidates: list[Path] = []
    if declared:
        candidates.append(audio_dir / declared)
    sibling_prefix = meta_path.name.replace("_metadata.json", "")
    candidates.append(meta_path.with_name(f"{sibling_prefix}{suffix}"))
    candidates.append(audio_dir / f"{sid}{suffix}")

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-transcribe a saved Otis session.")
    parser.add_argument(
        "prefix", nargs="?", help="Optional session-id prefix; matches all if omitted."
    )
    parser.add_argument("--model", default=None, help="Override the Whisper model.")
    parser.add_argument(
        "--language", default=None, help="Force a language (e.g. en, fr). Default: auto."
    )
    args = parser.parse_args()

    cfg = load_user_config()
    audio_dir = Path(cfg.get("storage", "audio_dir", default="~/Otis/audio")).expanduser()
    transcript_dir = Path(
        cfg.get("storage", "transcript_dir", default="~/Otis/transcripts")
    ).expanduser()
    model = args.model or str(cfg.get("transcription", "model", default="small"))
    language = args.language or cfg.get("transcription", "language")

    metadata_files = [m for m in _list_sessions(audio_dir) if _matches(args.prefix, m)]
    if not metadata_files:
        print(f"No sessions found under {audio_dir}.")
        return 1

    print(f"Found {len(metadata_files)} session(s) to retranscribe.")
    print(f"  audio_dir     : {audio_dir}")
    print(f"  transcript_dir: {transcript_dir}")
    print(f"  model         : {model}")
    print(f"  language      : {language or 'auto-detect'}")
    print()

    store = TranscriptStore(transcript_dir)
    engine = WhisperEngine(model_name=model)
    processor = TranscriptProcessor(
        engine=engine, store=store, audio_dir=audio_dir, model_name=model
    )

    succeeded = 0
    for meta_path in metadata_files:
        try:
            recorder_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ✗ {meta_path.name}: could not read metadata: {exc}")
            continue
        sid = recorder_meta.get("session_id") or meta_path.stem.replace("_metadata", "")

        # Resolve mic / system WAV paths in three steps — robust to metadata
        # written before / after the YYYY/MM rename, and to manual user moves.
        mic_path = _resolve_audio_path(
            recorder_meta.get("mic_wav"), audio_dir, meta_path, suffix="_mic.wav", sid=sid
        )
        sys_path = _resolve_audio_path(
            recorder_meta.get("system_wav"), audio_dir, meta_path, suffix="_system.wav", sid=sid
        )
        if mic_path is None:
            print(f"  ✗ {sid[:8]}: mic file missing (looked next to {meta_path.name})")
            continue
        # Rewrite the metadata in-place so subsequent reads (frontmatter, retention)
        # see correct paths.
        recorder_meta["mic_wav"] = str(mic_path.relative_to(audio_dir))
        recorder_meta["system_wav"] = (
            str(sys_path.relative_to(audio_dir)) if sys_path is not None else None
        )

        session = RecordingSession.from_recorder_metadata(recorder_meta, audio_dir=audio_dir)
        # The factory recomputes paths off the dict; force them to what we resolved.
        session = RecordingSession(
            session_id=session.session_id,
            audio_dir=session.audio_dir,
            mic_wav=mic_path,
            system_wav=sys_path,
            metadata_path=meta_path,
            mic_start_monotonic=session.mic_start_monotonic,
            system_start_monotonic=session.system_start_monotonic,
            start_wall_clock=session.start_wall_clock,
            sample_rate=session.sample_rate,
            raw_metadata=recorder_meta,
        )
        if not session.mic_wav.exists():
            print(f"  ✗ {sid[:8]}: mic file missing ({session.mic_wav})")
            continue

        print(f"  • transcribing {sid[:8]} ...", flush=True)
        try:
            result = processor.process(
                session,
                meeting=MeetingSnapshot(),
                language=language,
                on_progress=lambda p: None,
            )
        except Exception as exc:
            print(f"    ✗ failed: {exc}")
            continue
        print(f"    ✓ {result.transcript_path}")
        print(
            f"      mic={result.mic_segments} segs, "
            f"system={result.system_segments} segs, "
            f"echo_dropped={result.echo_dropped}"
        )
        succeeded += 1

    engine.shutdown()
    print(f"\nDone — {succeeded}/{len(metadata_files)} session(s) transcribed.")
    return 0 if succeeded > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
