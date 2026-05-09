"""Programmatic generation of menu-bar icons.

We generate the icons once at first launch (or whenever the icon directory is
missing) so the user doesn't have to ship binary assets in the repo. Each state
gets a 16×16 and a 32×32 PNG. macOS picks the right resolution per display.

Why not template images?
    macOS template images (black-on-transparent, auto-tinted by the OS) would
    look great in the menu bar — but they'd erase the colour signal we use to
    convey state (red = recording, orange = approaching, etc.). We generate
    coloured PNGs and accept the slight inconsistency between light/dark mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


# Each state maps to ((R, G, B), is_filled, glyph)
# Glyph values: "mic" (microphone), "circle" (recording dot), "pause" (||),
# "gear" (processing), "moon" (off-hours).
_PALETTE: Final[dict[str, tuple[tuple[int, int, int], bool, str]]] = {
    "idle":        ((128, 128, 128), False, "mic"),
    "approaching": ((255, 136,   0), True,  "mic"),
    "detected":    ((255, 136,   0), True,  "mic"),     # blink alternates with idle
    "recording":   ((229,  57,  53), True,  "circle"),
    "paused":      ((255, 196,   0), True,  "pause"),
    "processing":  (( 30, 136, 229), True,  "gear"),
    "off_hours":   (( 96,  96,  96), False, "moon"),
}

ICON_SIZES: Final[tuple[int, ...]] = (16, 32)


def ensure_icons(icons_dir: Path) -> dict[str, Path]:
    """Generate every state icon under ``icons_dir`` and return a state→path map.

    The map points to the 32×32 file; macOS finds the matching ``@2x`` variant
    automatically when both ``foo.png`` and ``foo@2x.png`` are present in the
    same directory.
    """
    icons_dir = icons_dir.expanduser()
    icons_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for state, (color, filled, glyph) in _PALETTE.items():
        base_path = icons_dir / f"{state}.png"
        retina_path = icons_dir / f"{state}@2x.png"
        if not base_path.exists():
            _render(state, glyph, color, filled, ICON_SIZES[0]).save(base_path, "PNG")
        if not retina_path.exists():
            _render(state, glyph, color, filled, ICON_SIZES[1]).save(retina_path, "PNG")
        out[state] = base_path
    logger.debug("Icons ready in %s: %s", icons_dir, list(out.keys()))
    return out


def regenerate_icons(icons_dir: Path) -> dict[str, Path]:
    """Force re-creation of every icon (useful after a palette change)."""
    icons_dir = icons_dir.expanduser()
    if icons_dir.exists():
        for p in icons_dir.glob("*.png"):
            p.unlink()
    return ensure_icons(icons_dir)


# ----------------------------------------------------------------------------
# Renderers — kept tiny on purpose. Pillow is only imported when actually
# rendering, so importing this module doesn't pull Pillow on platforms that
# don't need icons (e.g. CI).
# ----------------------------------------------------------------------------
def _render(state: str, glyph: str, color: tuple[int, int, int], filled: bool, size: int):
    """Render one icon to a PIL Image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(1, size // 12)
    rgba = (*color, 255)

    if glyph == "mic":
        _draw_mic(draw, size, rgba, filled, stroke)
    elif glyph == "circle":
        _draw_circle(draw, size, rgba, stroke)
    elif glyph == "pause":
        _draw_pause(draw, size, rgba)
    elif glyph == "gear":
        _draw_gear(draw, size, rgba, stroke)
    elif glyph == "moon":
        _draw_moon(draw, size, rgba)
    else:  # pragma: no cover (defensive)
        raise ValueError(f"Unknown glyph: {glyph}")
    return img


def _draw_mic(draw, size: int, rgba, filled: bool, stroke: int) -> None:
    """A simple mic: rounded rectangle head, vertical stand, base bar."""
    head_w = size * 0.42
    head_h = size * 0.55
    head_x = (size - head_w) / 2
    head_y = size * 0.10
    radius = head_w / 2

    if filled:
        draw.rounded_rectangle(
            [(head_x, head_y), (head_x + head_w, head_y + head_h)],
            radius=radius, fill=rgba,
        )
    else:
        draw.rounded_rectangle(
            [(head_x, head_y), (head_x + head_w, head_y + head_h)],
            radius=radius, outline=rgba, width=stroke,
        )

    stand_x = size / 2
    stand_top = head_y + head_h + size * 0.04
    stand_bot = size * 0.86
    draw.line([(stand_x, stand_top), (stand_x, stand_bot)], fill=rgba, width=stroke)

    base_w = size * 0.32
    draw.line(
        [(stand_x - base_w / 2, stand_bot), (stand_x + base_w / 2, stand_bot)],
        fill=rgba, width=stroke,
    )


def _draw_circle(draw, size: int, rgba, stroke: int) -> None:  # noqa: ARG001
    """Solid disc — the universal "recording" indicator."""
    pad = size * 0.18
    draw.ellipse([(pad, pad), (size - pad, size - pad)], fill=rgba)


def _draw_pause(draw, size: int, rgba) -> None:
    """Two thick vertical bars (||)."""
    bar_w = size * 0.18
    bar_h = size * 0.55
    gap = size * 0.12
    cx = size / 2
    cy = size / 2
    left = (cx - gap / 2 - bar_w, cy - bar_h / 2)
    right = (cx + gap / 2, cy - bar_h / 2)
    draw.rectangle([left, (left[0] + bar_w, left[1] + bar_h)], fill=rgba)
    draw.rectangle([right, (right[0] + bar_w, right[1] + bar_h)], fill=rgba)


def _draw_gear(draw, size: int, rgba, stroke: int) -> None:
    """Cog-like ring — close enough to "processing" at 16 px."""
    pad = size * 0.18
    draw.ellipse([(pad, pad), (size - pad, size - pad)], outline=rgba, width=stroke * 2)
    inner = size * 0.36
    cx = cy = size / 2
    draw.ellipse(
        [(cx - inner / 2, cy - inner / 2), (cx + inner / 2, cy + inner / 2)],
        fill=rgba,
    )


def _draw_moon(draw, size: int, rgba) -> None:
    """Crescent moon for off-hours."""
    pad = size * 0.15
    draw.ellipse([(pad, pad), (size - pad, size - pad)], fill=rgba)
    # Bite out the right side.
    bite_pad_x = size * 0.05
    draw.ellipse(
        [(pad + bite_pad_x, pad), (size - pad + bite_pad_x, size - pad)],
        fill=(0, 0, 0, 0),
    )
