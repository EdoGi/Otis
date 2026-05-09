"""Force-regenerate the menu-bar icons under ~/.otis/icons/.

Run after pulling new code or swapping ``OtisIcon.png``:

    python scripts/regenerate_icons.py

By default it picks up ``OtisIcon.png`` from the project root. Pass a path to
use a different source image:

    python scripts/regenerate_icons.py path/to/another.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.icons import DEFAULT_SOURCE_PATH, regenerate_icons


def main() -> int:
    icons_dir = Path("~/.otis/icons").expanduser()
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_PATH

    if not source.exists():
        print(f"Source not found: {source}")
        print("(Falls back to programmatic shapes — no avatar.)")
        source = None  # type: ignore[assignment]

    out = regenerate_icons(icons_dir, source_path=source)
    print(f"Wrote {len(out)} icon(s) to {icons_dir}:")
    for state, path in sorted(out.items()):
        retina = path.with_name(f"{state}@2x.png")
        print(f"  {state:>12} → {path.name} + {retina.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
