"""Tests for src/ui/icons.py.

We don't visually inspect the PNGs; we just verify each state produces a file,
the right resolution, the configured colour shows up somewhere, and re-running
``ensure_icons`` is a no-op (idempotent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.icons import _PALETTE, ICON_SIZES, ensure_icons, regenerate_icons


def test_ensure_icons_creates_every_state(tmp_path: Path) -> None:
    icons = ensure_icons(tmp_path)
    assert set(icons.keys()) == set(_PALETTE.keys())
    for state in _PALETTE:
        assert (tmp_path / f"{state}.png").exists()
        assert (tmp_path / f"{state}@2x.png").exists()


def test_ensure_icons_is_idempotent(tmp_path: Path) -> None:
    """Running twice must NOT regenerate the same files."""
    ensure_icons(tmp_path)
    first_mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.png")}
    ensure_icons(tmp_path)
    second_mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.png")}
    assert first_mtimes == second_mtimes


def test_regenerate_icons_overwrites(tmp_path: Path) -> None:
    ensure_icons(tmp_path)
    target = tmp_path / "idle.png"
    target.write_text("not a png")  # corrupt
    regenerate_icons(tmp_path)
    # File now exists and is no longer 'not a png' (it's binary PNG content).
    data = target.read_bytes()
    assert data.startswith(b"\x89PNG"), "regenerate_icons should rewrite the file as PNG"


def test_each_icon_has_correct_dimensions(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    ensure_icons(tmp_path)
    for state in _PALETTE:
        with Image.open(tmp_path / f"{state}.png") as img:
            assert img.size == (ICON_SIZES[0], ICON_SIZES[0])
        with Image.open(tmp_path / f"{state}@2x.png") as img:
            assert img.size == (ICON_SIZES[1], ICON_SIZES[1])


def test_icon_colour_appears_in_pixels(tmp_path: Path) -> None:
    """Each icon should contain at least one pixel close to its palette colour."""
    pytest.importorskip("PIL")
    from PIL import Image

    ensure_icons(tmp_path)
    for state, (color, _filled, _glyph) in _PALETTE.items():
        with Image.open(tmp_path / f"{state}@2x.png") as img:
            rgba = img.convert("RGBA")
            pixels = list(rgba.getdata())
        # At least one pixel within 10/255 of the palette colour, ignoring alpha.
        match = any(
            all(abs(pixel[i] - color[i]) <= 10 for i in range(3)) and pixel[3] > 0
            for pixel in pixels
        )
        assert match, f"state {state!r}: palette colour {color} never appears"
