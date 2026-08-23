"""Working-hours window logic, shared by the menu-bar app and the daemon.

Lives in its own module (rather than ``src.ui.menubar``) so the headless
daemon can use it without any conceptual dependency on the UI layer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from datetime import time as dtime
from typing import Any

from src.config import Config

logger = logging.getLogger(__name__)


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
    try:
        sh, sm = map(int, start_hhmm.split(":"))
        eh, em = map(int, end_hhmm.split(":"))
        start = dtime(sh, sm)
        end = dtime(eh, em)
    except (ValueError, AttributeError):
        logger.warning(
            "Unparseable working hours (%r-%r); treating every hour as a "
            "working hour rather than silently disabling detection.",
            start_hhmm, end_hhmm,
        )
        return True
    days = set(working_days)
    cur = now.time()

    if start <= end:
        return now.weekday() in days and start <= cur <= end

    # Overnight window, e.g. 22:00-02:00. `start <= cur <= end` is never true
    # for those bounds, so this silently reported "outside working hours" all
    # day and detection never ran. The window is two pieces, and the piece
    # after midnight belongs to the day it *started* on: at 01:00 on Tuesday
    # the user is still inside Monday's window.
    if cur >= start:
        return now.weekday() in days
    yesterday = (now.weekday() - 1) % 7
    return cur <= end and yesterday in days


def working_hours_from_config(cfg: Config) -> dict[str, Any]:
    """Extract the kwargs for :func:`is_within_working_hours` from a Config."""
    return {
        "working_days": cfg.get("app", "working_days", default=[0, 1, 2, 3, 4]),
        "start_hhmm": cfg.get("app", "working_hours", "start", default="08:00"),
        "end_hhmm": cfg.get("app", "working_hours", "end", default="20:00"),
    }


__all__ = ["is_within_working_hours", "working_hours_from_config"]
