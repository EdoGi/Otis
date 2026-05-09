"""Tests for src/ui/icons.py.

We don't visually inspect the PNGs; we just verify each state produces a file,
the right resolution, the configured colour shows up somewhere, and re-running
``ensure_icons`` is a no-op (idempotent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.icons import (
    _PALETTE,
    _STATE_BADGE_COLORS,
    ICON_SIZES,
    ensure_icons,
    regenerate_icons,
)


# ============================================================================
# Programmatic pipeline (no source image)
# ============================================================================
def test_ensure_icons_creates_every_state_programmatic(tmp_path: Path) -> None:
    """When no source image is supplied, programmatic shapes are generated."""
    icons = ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    assert set(icons.keys()) == set(_PALETTE.keys())
    for state in _PALETTE:
        assert (tmp_path / f"{state}.png").exists()
        assert (tmp_path / f"{state}@2x.png").exists()


def test_ensure_icons_is_idempotent(tmp_path: Path) -> None:
    """Running twice must NOT regenerate the same files."""
    ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    first_mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.png")}
    ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    second_mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.png")}
    assert first_mtimes == second_mtimes


def test_regenerate_icons_overwrites(tmp_path: Path) -> None:
    ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    target = tmp_path / "idle.png"
    target.write_text("not a png")  # corrupt
    regenerate_icons(tmp_path, source_path=tmp_path / "missing.png")
    data = target.read_bytes()
    assert data.startswith(b"\x89PNG"), "regenerate_icons should rewrite the file as PNG"


def test_each_icon_has_correct_dimensions(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    for state in _PALETTE:
        with Image.open(tmp_path / f"{state}.png") as img:
            assert img.size == (ICON_SIZES[0], ICON_SIZES[0])
        with Image.open(tmp_path / f"{state}@2x.png") as img:
            assert img.size == (ICON_SIZES[1], ICON_SIZES[1])


def test_programmatic_icon_colour_appears_in_pixels(tmp_path: Path) -> None:
    """Each programmatic icon should contain its palette colour somewhere."""
    pytest.importorskip("PIL")
    from PIL import Image

    ensure_icons(tmp_path, source_path=tmp_path / "missing.png")
    for state, (color, _filled, _glyph) in _PALETTE.items():
        with Image.open(tmp_path / f"{state}@2x.png") as img:
            rgba = img.convert("RGBA")
            pixels = list(rgba.getdata())
        match = any(
            all(abs(pixel[i] - color[i]) <= 10 for i in range(3)) and pixel[3] > 0
            for pixel in pixels
        )
        assert match, f"state {state!r}: palette colour {color} never appears"


# ============================================================================
# Avatar-based pipeline (with a source image)
# ============================================================================
def _make_dummy_source(tmp_path: Path) -> Path:
    """Create a 64×64 solid-blue square as a stand-in for OtisIcon.png."""
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGBA", (64, 64), (0, 100, 200, 255)).save(src)
    return src


def test_avatar_pipeline_creates_every_state(tmp_path: Path) -> None:
    icons_dir = tmp_path / "icons"
    src = _make_dummy_source(tmp_path)
    icons = ensure_icons(icons_dir, source_path=src)
    assert set(icons.keys()) == set(_STATE_BADGE_COLORS.keys())
    for state in _STATE_BADGE_COLORS:
        assert (icons_dir / f"{state}.png").exists()
        assert (icons_dir / f"{state}@2x.png").exists()


def test_avatar_state_badge_colour_is_present(tmp_path: Path) -> None:
    """Each state with a badge colour must have that colour in the @2x icon."""
    pytest.importorskip("PIL")
    from PIL import Image

    icons_dir = tmp_path / "icons"
    src = _make_dummy_source(tmp_path)
    ensure_icons(icons_dir, source_path=src)

    for state, color in _STATE_BADGE_COLORS.items():
        if color is None:
            continue
        with Image.open(icons_dir / f"{state}@2x.png") as img:
            pixels = list(img.convert("RGBA").getdata())
        match = any(
            all(abs(p[i] - color[i]) <= 12 for i in range(3)) and p[3] > 200
            for p in pixels
        )
        assert match, f"state {state!r}: badge colour {color} not visible"


def test_avatar_idle_has_no_badge_colour(tmp_path: Path) -> None:
    """The idle icon should be the avatar untouched (no orange/red/yellow blob)."""
    pytest.importorskip("PIL")
    from PIL import Image

    icons_dir = tmp_path / "icons"
    src = _make_dummy_source(tmp_path)
    ensure_icons(icons_dir, source_path=src)

    with Image.open(icons_dir / "idle@2x.png") as img:
        pixels = list(img.convert("RGBA").getdata())
    # No "recording red" pixels in the idle icon.
    red = (229, 57, 53)
    has_red = any(
        all(abs(p[i] - red[i]) <= 12 for i in range(3)) and p[3] > 200
        for p in pixels
    )
    assert not has_red, "idle icon should not contain the recording-red badge colour"
