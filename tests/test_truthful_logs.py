"""Log lines that told the wrong story — the log is how this app gets debugged.

* "Archived …_mic.wav -> …flac (0.0 MB -> 68.6 MB)": the WAV's size was read
  after the WAV had been deleted (2026-10-21, a 145 MB recording).
* "Generating icons from OtisIcon.png" on all 36 launches, while the 14 PNGs
  in ~/.otis/icons have not been rewritten since 2026-05-09 — and the 512 KB
  source was decoded every launch anyway, for nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_the_archive_line_reports_the_source_size(tmp_path: Path, monkeypatch, caplog) -> None:
    import src.transcription.processor as processor

    src = tmp_path / "s_mic.wav"
    src.write_bytes(b"\0" * 2_000_000)
    dst = tmp_path / "2026" / "10" / "s_mic.flac"

    def fake_encode(source: Path, target: Path) -> bool:
        target.write_bytes(b"\0" * 900_000)
        return True

    monkeypatch.setattr(processor, "encode_audio", fake_encode)
    fake_self = SimpleNamespace(_audio_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger=processor.__name__):
        processor.TranscriptProcessor._archive_one(fake_self, src, dst)  # type: ignore[arg-type]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Archived"))
    assert "(2.0 MB -> 0.9 MB)" in line


def test_existing_icons_are_neither_regenerated_nor_decoded(tmp_path: Path, monkeypatch, caplog) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    import src.ui.icons as icons

    source = tmp_path / "OtisIcon.png"
    Image.new("RGBA", (64, 64), (200, 50, 50, 255)).save(source)
    out = tmp_path / "icons"
    icons.ensure_icons(out, source_path=source)

    def no_decode(*a, **k):
        raise AssertionError("decoded the source although every icon exists")

    monkeypatch.setattr(Image, "open", no_decode)
    caplog.clear()  # the first call above generated the icons and said so
    with caplog.at_level(logging.INFO, logger=icons.__name__):
        paths = icons.ensure_icons(out, source_path=source)
    assert set(paths) == set(icons._STATE_BADGE_COLORS)
    assert all(p.exists() for p in paths.values())
    mine = [r.getMessage() for r in caplog.records if r.name == icons.__name__]
    assert not any("Generat" in m for m in mine), mine
