"""Repair WAV headers left broken by a crash mid-recording.

Python's ``wave`` module patches the RIFF and data chunk sizes only when the
file is closed. If Otis is killed (crash, force-quit, power loss) while the
writer thread is streaming, the header still says "0 frames" even though the
PCM data is sitting right there in the file — and every reader downstream
(``wave.open``, the silence pre-check, Whisper) sees an empty recording.

``repair_wav_header`` rewrites the two size fields from the actual file
size, making the recording readable again. It only touches classic PCM WAVs
(what our recorder writes) and is deliberately conservative: anything it
doesn't fully understand is left untouched.
"""

from __future__ import annotations

import logging
import struct
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

_RIFF_HEADER_BYTES = 12  # "RIFF" + u32 size + "WAVE"


def _find_chunks(data: bytes) -> dict[bytes, tuple[int, int]]:
    """Map chunk id → (offset_of_chunk_header, declared_size).

    ``data`` is the head of the file (enough to cover fmt + any small
    metadata chunks before data). Stops at the ``data`` chunk — its declared
    size may be the broken zero we're here to fix, so we never trust it to
    skip past.
    """
    chunks: dict[bytes, tuple[int, int]] = {}
    pos = _RIFF_HEADER_BYTES
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        chunks[cid] = (pos, size)
        if cid == b"data":
            break
        # Chunks are word-aligned: odd sizes are padded with one byte.
        pos += 8 + size + (size & 1)
    return chunks


def wav_needs_repair(path: Path) -> bool:
    """True iff the WAV header claims 0 frames but PCM bytes exist on disk."""
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnframes() > 0:
                return False
    except (wave.Error, EOFError, OSError):
        # Unreadable header — possibly repairable, let repair decide.
        pass
    except Exception:
        return False
    try:
        head = path.read_bytes()[: 64 * 1024]
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        chunks = _find_chunks(head)
        if b"data" not in chunks or b"fmt " not in chunks:
            return False
        data_offset, declared = chunks[b"data"]
        actual_body = path.stat().st_size - (data_offset + 8)
        return declared == 0 and actual_body > 0
    except Exception:
        return False


def repair_wav_header(path: Path) -> bool:
    """Rewrite RIFF/data sizes from the actual file size. Returns True if fixed.

    Only handles uncompressed PCM (audio format 1) — the only thing our
    recorder produces. Never raises; failures are logged and reported False.
    """
    try:
        size = path.stat().st_size
        if size <= _RIFF_HEADER_BYTES:
            return False
        head = path.read_bytes()[: 64 * 1024]
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        chunks = _find_chunks(head)
        if b"data" not in chunks or b"fmt " not in chunks:
            return False

        fmt_offset, fmt_size = chunks[b"fmt "]
        if fmt_size < 16:
            return False
        audio_format, channels = struct.unpack_from("<HH", head, fmt_offset + 8)
        block_align = struct.unpack_from("<H", head, fmt_offset + 8 + 12)[0]
        if audio_format != 1 or channels < 1 or block_align < 1:
            return False  # not plain PCM — don't guess

        data_offset, declared_size = chunks[b"data"]
        body = size - (data_offset + 8)
        if body <= 0:
            return False
        # Truncate to a whole number of frames (a crash can land mid-frame).
        body -= body % block_align
        if body <= 0 or declared_size == body:
            return False

        with path.open("r+b") as fh:
            fh.seek(4)
            fh.write(struct.pack("<I", (data_offset + 8 + body) - 8))
            fh.seek(data_offset + 4)
            fh.write(struct.pack("<I", body))
        logger.info(
            "Repaired truncated WAV header: %s (data size %d → %d bytes)",
            path.name, declared_size, body,
        )
        return True
    except Exception:
        logger.exception("WAV repair failed for %s", path)
        return False


def repair_if_needed(path: Path | None) -> bool:
    """Convenience guard used by orphan discovery / retranscribe."""
    if path is None or not path.exists():
        return False
    if not wav_needs_repair(path):
        return False
    return repair_wav_header(path)


__all__ = ["repair_if_needed", "repair_wav_header", "wav_needs_repair"]
