"""Print which input device is the default and whether anything is using it.

Usage:
    python scripts/probe_mic.py

Open QuickTime Player → New Audio Recording (don't actually record), then
re-run this script — ``mic_in_use`` should flip to True. Close QuickTime and
re-run, it should flip back to False.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.coreaudio_probe import (
    CoreAudioProbeError,
    get_default_input_device_id,
    is_default_input_running,
)


def main() -> int:
    try:
        device_id = get_default_input_device_id()
    except CoreAudioProbeError as exc:
        print(f"CoreAudio probe failed: {exc}")
        return 1
    print(f"default input device id: {device_id}")
    print(f"mic in use:              {is_default_input_running()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
