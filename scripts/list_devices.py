"""Print every CoreAudio device the recorder can see, with input/output flags.

Usage:
    python scripts/list_devices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import sounddevice as sd

    print("default:", sd.default.device)
    for i, d in enumerate(sd.query_devices()):
        flags = []
        if d["max_input_channels"]:
            flags.append(f"in:{d['max_input_channels']}")
        if d["max_output_channels"]:
            flags.append(f"out:{d['max_output_channels']}")
        print(f"  [{i:>2}] {d['name']!r}  ({'/'.join(flags) or '?'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
