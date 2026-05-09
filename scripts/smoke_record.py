"""Phase 1 smoke test: record a few seconds and print the result.

Usage:
    python scripts/smoke_record.py [seconds]   # default 5
    python scripts/smoke_record.py 5 --pause-at 2 --resume-at 3

Outputs the WAV files and metadata to ./smoke_out/ (override with --out).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import wave
from pathlib import Path

# Allow running the script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.blackhole_check import format_setup_instructions, verify_blackhole_setup
from src.audio.devices import DeviceManager
from src.audio.recorder import DualStreamRecorder


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 dual-stream smoke recorder")
    parser.add_argument("seconds", nargs="?", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("smoke_out"))
    parser.add_argument("--pause-at", type=float, default=None,
                        help="Seconds into the recording at which to pause.")
    parser.add_argument("--resume-at", type=float, default=None,
                        help="Seconds into the recording at which to resume.")
    parser.add_argument("--mic", default=None, help="Mic device name (substring) or index.")
    parser.add_argument("--system", default="BlackHole 2ch", help="System device name; '' to disable.")
    parser.add_argument("--no-sleep-wake", action="store_true",
                        help="Skip the NSWorkspace observer (useful when running without GUI).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dm = DeviceManager()
    status = verify_blackhole_setup(dm)
    print(format_setup_instructions(status))
    print()

    rec = DualStreamRecorder(
        audio_dir=args.out,
        sample_rate=16000,
        channels=1,
        mic_device=args.mic,
        system_device=(args.system or None),
        device_manager=dm,
        observe_sleep_wake=not args.no_sleep_wake,
    )

    print(f"Recording {args.seconds:.1f}s into {args.out.resolve()} … say something.")
    session = rec.start()
    print(f"  session_id = {session}")

    start = time.monotonic()
    paused = False
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= args.seconds:
            break
        if (
            args.pause_at is not None
            and not paused
            and elapsed >= args.pause_at
        ):
            print(f"  [pause @ {elapsed:.2f}s]")
            rec.pause()
            paused = True
        if (
            args.resume_at is not None
            and paused
            and elapsed >= args.resume_at
        ):
            print(f"  [resume @ {elapsed:.2f}s]")
            rec.resume()
            paused = False
        time.sleep(0.05)

    meta = rec.stop()
    print("\nRecording stopped. Metadata:")
    print(json.dumps(meta, indent=2))

    for label, key in [("mic", "mic_wav"), ("system", "system_wav")]:
        name = meta.get(key)
        if not name:
            print(f"  {label}: (skipped — no device)")
            continue
        path = args.out / name
        with wave.open(str(path), "rb") as wf:
            seconds = wf.getnframes() / wf.getframerate()
            print(f"  {label}: {path}  {seconds:.2f}s  "
                  f"{wf.getnchannels()}ch  {wf.getframerate()} Hz")

    print("\nPlay back with:")
    if meta.get("mic_wav"):
        print(f"  afplay {args.out / meta['mic_wav']}")
    if meta.get("system_wav"):
        print(f"  afplay {args.out / meta['system_wav']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
