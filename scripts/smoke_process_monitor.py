"""Run the ProcessMonitor for ~15 s and print every detection.

Usage:
    python scripts/smoke_process_monitor.py            # default 15s, default whitelist
    python scripts/smoke_process_monitor.py 30         # 30s
    python scripts/smoke_process_monitor.py 30 zoom.us slack

Open Zoom / Slack / Teams while it's running to see ``DETECTED`` lines. Close
them to see ``ENDED`` lines.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.process_monitor import ProcessMonitor

DEFAULT_WHITELIST = ["zoom.us", "Microsoft Teams", "Webex", "Slack", "FaceTime"]
DEFAULT_BLACKLIST = ["SuperWhisper"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    whitelist = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_WHITELIST

    pm = ProcessMonitor(
        whitelisted_apps=whitelist,
        blacklisted_apps=DEFAULT_BLACKLIST,
        poll_interval_seconds=2.0,
    )
    pm.on_meeting_detected(lambda app: print(f">>> DETECTED: {app}"))
    pm.on_meeting_ended(lambda app: print(f"<<< ENDED:    {app}"))

    print(f"Watching for {whitelist} for {seconds:.0f}s — open/close a meeting app to see events.")
    pm.start()
    try:
        time.sleep(seconds)
    finally:
        pm.stop()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
