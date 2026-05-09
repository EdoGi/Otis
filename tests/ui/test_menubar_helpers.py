"""Tests for the rumps-free helpers in src/ui/menubar.py.

We deliberately don't try to spin up rumps inside pytest — the live menu-bar
event loop owns the macOS main thread and isn't sensible to drive headless.
Instead we test:

* ``is_within_working_hours`` (pure function),
* ``format_duration_mmss`` (pure function),
* ``write_user_config_override`` (deep-merges + writes YAML).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from src.ui.menubar import (
    format_duration_mmss,
    is_within_working_hours,
    write_user_config_override,
)


# ============================================================================
# format_duration_mmss
# ============================================================================
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (1, "00:01"),
        (59.9, "00:59"),
        (60, "01:00"),
        (93.4, "01:33"),
        (3600, "60:00"),
        (-5, "00:00"),
    ],
)
def test_format_duration_mmss(seconds: float, expected: str) -> None:
    assert format_duration_mmss(seconds) == expected


# ============================================================================
# is_within_working_hours
# ============================================================================
def test_inside_window_normal_workday() -> None:
    """Tuesday at 10:00 with Mon–Fri / 08:00–20:00 → in window."""
    tuesday_10am = datetime(2026, 5, 12, 10, 0)
    assert is_within_working_hours(
        tuesday_10am,
        working_days=[0, 1, 2, 3, 4],
        start_hhmm="08:00",
        end_hhmm="20:00",
    )


def test_outside_window_evening() -> None:
    tuesday_22h = datetime(2026, 5, 12, 22, 0)
    assert not is_within_working_hours(
        tuesday_22h,
        working_days=[0, 1, 2, 3, 4],
        start_hhmm="08:00",
        end_hhmm="20:00",
    )


def test_outside_window_weekend() -> None:
    saturday_10am = datetime(2026, 5, 16, 10, 0)
    assert not is_within_working_hours(
        saturday_10am,
        working_days=[0, 1, 2, 3, 4],
        start_hhmm="08:00",
        end_hhmm="20:00",
    )


def test_inside_window_when_saturday_is_enabled() -> None:
    saturday_10am = datetime(2026, 5, 16, 10, 0)
    assert is_within_working_hours(
        saturday_10am,
        working_days=[0, 1, 2, 3, 4, 5],
        start_hhmm="08:00",
        end_hhmm="20:00",
    )


def test_boundary_inclusive_at_start() -> None:
    tuesday_8am = datetime(2026, 5, 12, 8, 0)
    assert is_within_working_hours(
        tuesday_8am,
        working_days=[0, 1, 2, 3, 4],
        start_hhmm="08:00",
        end_hhmm="20:00",
    )


def test_malformed_hours_falls_back_to_open() -> None:
    """Bad config strings shouldn't shut detection off."""
    tuesday_10am = datetime(2026, 5, 12, 10, 0)
    assert is_within_working_hours(
        tuesday_10am,
        working_days=[0, 1, 2, 3, 4],
        start_hhmm="not a time",
        end_hhmm="also not",
    )


# ============================================================================
# write_user_config_override
# ============================================================================
def test_writes_fresh_user_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_user_config_override(cfg_path, {"transcription": {"model": "large-v3"}})
    data = yaml.safe_load(cfg_path.read_text())
    assert data == {"transcription": {"model": "large-v3"}}


def test_deep_merges_with_existing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "transcription": {"model": "small", "language": None},
                "app": {"working_days": [0, 1, 2, 3, 4]},
            }
        )
    )
    write_user_config_override(cfg_path, {"transcription": {"model": "large-v3"}})
    data = yaml.safe_load(cfg_path.read_text())
    # The new value replaced the old one, but sibling keys survived.
    assert data["transcription"]["model"] == "large-v3"
    assert data["transcription"]["language"] is None
    assert data["app"]["working_days"] == [0, 1, 2, 3, 4]


def test_creates_parent_directory(tmp_path: Path) -> None:
    cfg_path = tmp_path / "nested" / "dirs" / "config.yaml"
    write_user_config_override(cfg_path, {"app": {"name": "Otis"}})
    assert cfg_path.exists()


def test_corrupt_existing_config_is_overwritten(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(":::not yaml")
    write_user_config_override(cfg_path, {"transcription": {"model": "tiny"}})
    data = yaml.safe_load(cfg_path.read_text())
    assert data == {"transcription": {"model": "tiny"}}
