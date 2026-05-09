"""Transcript post-processor (Phase 4).

Merges the mic and system streams using their monotonic anchors, runs
diarisation/labelling, and emits the final markdown. Implementation lands in
Phase 4.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Stub: implement in Phase 4."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("TranscriptProcessor lands in Phase 4.")
