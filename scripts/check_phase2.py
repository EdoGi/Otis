"""Otis self-check — now a thin wrapper around ``otis doctor``.

The original script was written for the Phase 1/2 milestone and never
updated: it accepted Python 3.10, read only the bundled config, judged
BlackHole structurally (green while system audio was not routed), and looked
``pytest`` up on PATH, so it was red whenever the venv was not activated.
``python -m src.main doctor`` replaces it; this file stays so the old command
still works.

Usage:
    .venv/bin/python scripts/check_phase2.py               # doctor only
    .venv/bin/python scripts/check_phase2.py --with-tests  # doctor + pytest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-tests", action="store_true",
                        help="also run the test suite (~35 s)")
    args = parser.parse_args(argv)

    from src import doctor

    code = doctor.main()
    if args.with_tests:
        print("\nRunning the test suite …")
        # The interpreter running this script, not whatever `pytest` is on PATH.
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                              cwd=ROOT, timeout=600)
        code = code or proc.returncode
    return code


if __name__ == "__main__":
    sys.exit(main())
