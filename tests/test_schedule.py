"""Tests for src/schedule.py."""

from __future__ import annotations

from datetime import datetime

from src.config import Config
from src.schedule import is_within_working_hours, working_hours_from_config


def test_working_hours_from_config_reads_app_block() -> None:
    cfg = Config(
        {"app": {"working_days": [0, 2], "working_hours": {"start": "07:30", "end": "18:00"}}}
    )
    kwargs = working_hours_from_config(cfg)
    assert kwargs == {
        "working_days": [0, 2],
        "start_hhmm": "07:30",
        "end_hhmm": "18:00",
    }
    # The dict unpacks straight into the checker.
    monday_morning = datetime(2026, 6, 8, 9, 0)  # a Monday
    assert is_within_working_hours(monday_morning, **kwargs)


def test_working_hours_from_config_defaults_when_missing() -> None:
    kwargs = working_hours_from_config(Config({}))
    assert kwargs == {
        "working_days": [0, 1, 2, 3, 4],
        "start_hhmm": "08:00",
        "end_hhmm": "20:00",
    }


def test_menubar_reexport_still_works() -> None:
    from src.ui.menubar import is_within_working_hours as reexported

    assert reexported is is_within_working_hours
