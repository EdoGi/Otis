"""Programmatic generation of menu-bar icons.

Two pipelines:

1. **Avatar-based** (preferred) — when the project ships an ``OtisIcon.png``
   in the repo root, every state icon is the downscaled avatar with a coloured
   "status badge" in the bottom-right corner. The user keeps brand identity
   *and* a clearly visible state cue (red dot for recording, orange for
   approaching, etc.).

2. **Programmatic fallback** — if the source image isn't available (CI, tests,
   broken install) we draw simple geometric shapes (mic / circle / pause / …)
   so the app still has something to show.

Either way: 16 px and 32 px PNGs land in ``~/.otis/icons/`` on first launch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


# State → ((R, G, B) for badge / programmatic colour, programmatic glyph)
_STATE_BADGE_COLORS: Final[dict[str, tuple[int, int, int] | None]] = {
    "idle":        None,                 # avatar shown clean
    "approaching": (255, 136,   0),      # orange
    "detected":    (255, 136,   0),      # blink alternates with idle
    "recording":   (229,  57,  53),      # red
    "paused":      (255, 196,   0),      # yellow
    "processing":  ( 30, 136, 229),      # blue
    "off_hours":   (140, 140, 140),      # neutral gray
}

# Glyph fallback (used only when no source avatar is provided)
_PROGRAMMATIC_GLYPHS: Final[dict[str, tuple[tuple[int, int, int], bool, str]]] = {
    "idle":        ((128, 128, 128), False, "mic"),
    "approaching": ((255, 136,   0), True,  "mic"),
    "detected":    ((255, 136,   0), True,  "mic"),
    "recording":   ((229,  57,  53), True,  "circle"),
    "paused":      ((255, 196,   0), True,  "pause"),
    "processing":  (( 30, 136, 229), True,  "gear"),
    "off_hours":   (( 96,  96,  96), False, "moon"),
}

ICON_SIZES: Final[tuple[int, ...]] = (16, 32)
DEFAULT_SOURCE_PATH = Path(__file__).resolve().parent.parent.parent / "OtisIcon.png"

# Backwards-compat alias for tests that imported _PALETTE before this rewrite.
_PALETTE = _PROGRAMMATIC_GLYPHS


# ---------------------------------------------------------------- public API
def ensure_icons(
    icons_dir: Path,
    *,
    source_path: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Generate every state icon under ``icons_dir`` and return a state→path map.

    Parameters
    ----------
    icons_dir:
        Directory to write the PNGs into (created if absent).
    source_path:
        Optional path to a square source image (e.g. the bundled
        ``OtisIcon.png``). When ``None``, we look at the repo-root default
        first and fall back to programmatic shapes if it's missing.
    force:
        Overwrite existing icons even if they already exist on disk. Used by
        :func:`regenerate_icons`.
    """
    icons_dir = icons_dir.expanduser()
    icons_dir.mkdir(parents=True, exist_ok=True)

    src = source_path if source_path is not None else (
        DEFAULT_SOURCE_PATH if DEFAULT_SOURCE_PATH.exists() else None
    )

    if src is not None and src.exists():
        logger.info("Generating icons from %s", src)
        return _ensure_from_source(src, icons_dir, force=force)

    logger.info("OtisIcon.png not found; using programmatic icons.")
    return _ensure_programmatic(icons_dir, force=force)


def regenerate_icons(
    icons_dir: Path,
    *,
    source_path: Path | None = None,
) -> dict[str, Path]:
    """Force re-creation of every icon (after a palette change or new source)."""
    icons_dir = icons_dir.expanduser()
    if icons_dir.exists():
        for p in icons_dir.glob("*.png"):
            p.unlink()
    return ensure_icons(icons_dir, source_path=source_path, force=True)


# =========================================================================
# Avatar-based pipeline
# =========================================================================
def _ensure_from_source(
    source_path: Path,
    icons_dir: Path,
    *,
    force: bool,
) -> dict[str, Path]:
    from PIL import Image

    base = Image.open(source_path).convert("RGBA")
    out: dict[str, Path] = {}

    for state, badge in _STATE_BADGE_COLORS.items():
        for size in ICON_SIZES:
            target = icons_dir / _icon_filename(state, size)
            if target.exists() and not force:
                continue
            icon = base.resize((size, size), Image.LANCZOS)
            if badge is not None:
                _draw_corner_badge(icon, badge, size)
            elif state == "idle":
                # Slightly fade the idle icon so an active state's full-saturation
                # version visibly pops.
                icon = _fade(icon, alpha_scale=0.85)
            elif state == "off_hours":
                icon = _grayscale(icon, alpha_scale=0.55)
            icon.save(target, "PNG")
        out[state] = icons_dir / f"{state}.png"

    logger.debug("Avatar icons ready: %s", sorted(out.keys()))
    return out


def _draw_corner_badge(
    icon,
    color: tuple[int, int, int],
    size: int,
) -> None:
    """Draw a circular state badge in the bottom-right corner with a white ring.

    Sizing chosen so the avatar stays the dominant visual element: the badge
    diameter is ~36 % of the icon, sitting in the corner. At 32 px that's
    roughly a 12 px dot — large enough to read at a glance, small enough that
    the face/hair are still recognisable.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(icon)
    radius = max(2.0, size * 0.18)
    ring = max(1.0, size * 0.05)
    margin = max(1.0, size * 0.06)
    cx = size - radius - margin - ring
    cy = size - radius - margin - ring

    draw.ellipse(
        [
            (cx - radius - ring, cy - radius - ring),
            (cx + radius + ring, cy + radius + ring),
        ],
        fill=(255, 255, 255, 235),
    )
    draw.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=(*color, 255),
    )


def _fade(icon, *, alpha_scale: float):
    """Return a copy with alpha multiplied by ``alpha_scale`` (0..1)."""
    from PIL import Image

    r, g, b, a = icon.split()
    a = a.point(lambda v: int(v * alpha_scale))
    return Image.merge("RGBA", (r, g, b, a))


def _grayscale(icon, *, alpha_scale: float = 1.0):
    """Return a desaturated copy (used for the off-hours state)."""
    from PIL import Image, ImageOps

    r, g, b, a = icon.split()
    gray = ImageOps.grayscale(Image.merge("RGB", (r, g, b))).convert("RGB")
    gr, gg, gb = gray.split()
    a = a.point(lambda v: int(v * alpha_scale))
    return Image.merge("RGBA", (gr, gg, gb, a))


# =========================================================================
# Programmatic pipeline (CI / no-source fallback)
# =========================================================================
def _ensure_programmatic(icons_dir: Path, *, force: bool) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for state, (color, filled, glyph) in _PROGRAMMATIC_GLYPHS.items():
        for size in ICON_SIZES:
            target = icons_dir / _icon_filename(state, size)
            if target.exists() and not force:
                continue
            _render(state, glyph, color, filled, size).save(target, "PNG")
        out[state] = icons_dir / f"{state}.png"
    logger.debug("Programmatic icons ready: %s", sorted(out.keys()))
    return out


def _render(state: str, glyph: str, color: tuple[int, int, int], filled: bool, size: int):
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
    else:
        raise ValueError(f"Unknown glyph: {glyph}")  # pragma: no cover
    return img


def _draw_mic(draw, size, rgba, filled, stroke):
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


def _draw_circle(draw, size, rgba, _stroke):
    pad = size * 0.18
    draw.ellipse([(pad, pad), (size - pad, size - pad)], fill=rgba)


def _draw_pause(draw, size, rgba):
    bar_w = size * 0.18
    bar_h = size * 0.55
    gap = size * 0.12
    cx = size / 2
    cy = size / 2
    left = (cx - gap / 2 - bar_w, cy - bar_h / 2)
    right = (cx + gap / 2, cy - bar_h / 2)
    draw.rectangle([left, (left[0] + bar_w, left[1] + bar_h)], fill=rgba)
    draw.rectangle([right, (right[0] + bar_w, right[1] + bar_h)], fill=rgba)


def _draw_gear(draw, size, rgba, stroke):
    pad = size * 0.18
    draw.ellipse([(pad, pad), (size - pad, size - pad)], outline=rgba, width=stroke * 2)
    inner = size * 0.36
    cx = cy = size / 2
    draw.ellipse(
        [(cx - inner / 2, cy - inner / 2), (cx + inner / 2, cy + inner / 2)],
        fill=rgba,
    )


def _draw_moon(draw, size, rgba):
    pad = size * 0.15
    draw.ellipse([(pad, pad), (size - pad, size - pad)], fill=rgba)
    bite_pad_x = size * 0.05
    draw.ellipse(
        [(pad + bite_pad_x, pad), (size - pad + bite_pad_x, size - pad)],
        fill=(0, 0, 0, 0),
    )


# =========================================================================
# Helpers
# =========================================================================
def _icon_filename(state: str, size: int) -> str:
    return f"{state}.png" if size == ICON_SIZES[0] else f"{state}@2x.png"
