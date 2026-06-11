"""Working-hours window logic, shared by the menu-bar app and the daemon.

Lives in its own module (rather than ``src.ui.menubar``) so the headless
daemon can use it without any conceptual dependency on the UI layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, time as dtime
from typing import Any

from src.config import Config


def is_within_working_hours(
    now: datetime,
    *,
    working_days: Iterable[int],
    start_hhmm: str,
    end_hhmm: str,
) -> bool:
    """Return True iff ``now`` falls inside the configured work window.

    ``working_days`` are Python weekday() values (0 = Monday … 6 = Sunday).
    ``start_hhmm`` / ``end_hhmm`` are ``"HH:MM"`` strings in local time.
    """
    if now.weekday() not in set(working_days):
        return False
    try:
        sh, sm = map(int, start_hhmm.split(":"))
        eh, em = map(int, end_hhmm.split(":"))
    except (ValueError, AttributeError):
        return True  # malformed config → don't block
    start = dtime(sh, sm)
    end = dtime(eh, em)
    cur = now.time()
    return start <= cur <= end


def working_hours_from_config(cfg: Config) -> dict[str, Any]:
    """Extract the kwargs for :func:`is_within_working_hours` from a Config."""
    return {
        "working_days": cfg.get("app", "working_days", default=[0, 1, 2, 3, 4]),
        "start_hhmm": cfg.get("app", "working_hours", "start", default="08:00"),
        "end_hhmm": cfg.get("app", "working_hours", "end", default="20:00"),
    }


__all__ = ["is_within_working_hours", "working_hours_from_config"]
