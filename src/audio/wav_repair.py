"""Make a damaged WAV readable again, when we can be sure what "damaged" is.

Two shapes of broken header, and it is worth being exact about which one this
recorder can actually produce.

**Declared size zero, data present.** The original reason this module existed:
the belief that a force-quit leaves ``wave`` having never patched the header.
Measured against a real SIGKILL of a live recording, that does not happen —
CPython's ``Wave_write.writeframes`` calls ``_patchheader()`` after *every*
write, so the killed process left 90156 bytes on disk and a header declaring
45056 frames, which is exactly right:

    1fcb09d1..._mic.wav   90156 bytes  header frames=45056  needs_repair=False

So this case is dead for our own files. It is kept because it costs nothing
and other writers do produce it.

**Declared size larger than the file.** This is the shape that can really
reach us, and the one the module used to refuse outright: a WAV whose data
chunk claims more bytes than the file contains. A copy interrupted partway, a
half-finished cloud sync, a truncated restore from a backup. Every reader then
either raises or runs off the end of the data, so the recording is unusable —
and clamping the declared size down to what is actually present loses nothing,
because those bytes do not exist.

The opposite (declared *smaller* than the file) is still refused, and that
asymmetry is deliberate: a short data chunk is how a perfectly valid WAV
carries trailing ``LIST``/``INFO`` chunks after the audio, and "repairing" it
would swallow that metadata into the samples.
"""

from __future__ import annotations

import logging
import struct
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
    """True iff the header disagrees with the file in a way we can fix."""
    # Deliberately no `wave.open(...).getnframes() > 0` fast path. A truncated
    # file still declares its original frame count, so that check called it
    # healthy and short-circuited the one case worth catching. The chunk walk
    # below is the authority: it compares what the header claims against what
    # the file actually holds.
    try:
        with path.open("rb") as fh:
            head = fh.read(64 * 1024)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        chunks = _find_chunks(head)
        if b"data" not in chunks or b"fmt " not in chunks:
            return False
        data_offset, declared = chunks[b"data"]
        actual_body = path.stat().st_size - (data_offset + 8)
        if actual_body <= 0:
            return False
        # Zero: the header was never written. Larger than the file: the file
        # was truncated. Smaller is legitimate — trailing chunks — so it is
        # not ours to touch.
        return declared == 0 or declared > actual_body
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
        with path.open("rb") as fh:
            head = fh.read(64 * 1024)
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
        if declared_size != 0 and declared_size <= body:
            # Either it already agrees, or it declares less than the file
            # holds — which is how a valid WAV carries trailing LIST/INFO
            # chunks after the audio. Rewriting that would fold the metadata
            # into the samples.
            return False
        # Truncate to a whole number of frames (a crash can land mid-frame).
        body -= body % block_align
        if body <= 0:
            return False

        with path.open("r+b") as fh:
            fh.seek(4)
            fh.write(struct.pack("<I", (data_offset + 8 + body) - 8))
            fh.seek(data_offset + 4)
            fh.write(struct.pack("<I", body))
        logger.info(
            "Repaired WAV header: %s (data size %d → %d bytes; %s)",
            path.name, declared_size, body,
            "header never written" if declared_size == 0 else "file truncated",
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
