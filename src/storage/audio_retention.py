"""Audio retention policy (Phase 5).

Sweeps the audio directory and deletes WAV files older than
``audio_retention_days``. Implementation lands in Phase 5.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AudioRetention:
    """Stub: implement in Phase 5."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("AudioRetention lands in Phase 5.")
