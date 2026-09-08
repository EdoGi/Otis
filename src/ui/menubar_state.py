"""Menu-bar constants, display state, and the pure helpers around them.

Split out of ``src/ui/menubar.py`` (audit item 42), which had grown past 2 200
lines mixing rumps wiring, a state machine, settings persistence and disk
scanning. Nothing here touches rumps or the app, so it is testable on its own
and readable without the event loop in your head.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from src.detection.detector import MeetingContext
from src.health import HealthState

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("French", "fr"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Spanish", "es"),
    ("German", "de"),
]

WHISPER_MODELS: list[str] = ["tiny", "base", "small", "medium", "large-v3"]

WEEKDAY_NAMES: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Hour presets offered in Settings → Working Hours.
HOUR_START_PRESETS: list[str] = [f"{h:02d}:00" for h in range(6, 13)]   # 06:00-12:00
HOUR_END_PRESETS: list[str] = [f"{h:02d}:00" for h in range(14, 23)]    # 14:00-22:00

# Polling cadences for main-thread timers (rumps.Timer).
QUEUE_DRAIN_INTERVAL = 0.1   # 100 ms — main-thread event drain
BLINK_INTERVAL = 0.5         # 500 ms — DETECTED state blink
DURATION_INTERVAL = 1.0      # 1 s — recording duration tick
SCHEDULE_CHECK_INTERVAL = 60  # 60 s — off-hours window check

# Suffix shown on the "Status" menu title when something is degraded.
_GLYPH_FOR: dict[HealthState, str] = {
    HealthState.OK: "",
    HealthState.UNKNOWN: "·",
    HealthState.WARN: "!",
    HealthState.FAIL: "✕",
}


# ============================================================================
# Helpers used both by the live app and by tests.
# ============================================================================
def format_duration_mmss(seconds: float) -> str:
    """``93.4 -> "01:33"`` — used in the menu-bar title during recording."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_user_config_override(
    user_config_path: Path,
    overrides: dict[str, Any],
) -> None:
    """Merge ``overrides`` into the user's YAML config and persist it.

    Existing keys in the file are preserved; only the keys we touch are
    written. Used by settings toggles (Whisper model, working days, etc.).
    """

    user_config_path = user_config_path.expanduser()
    user_config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if user_config_path.exists():
        try:
            loaded = yaml.safe_load(user_config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Could not parse %s (%s); rewriting from scratch.",
                user_config_path,
                exc,
            )
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
        elif loaded is not None:
            logger.warning(
                "%s did not contain a mapping (got %s); rewriting from scratch.",
                user_config_path,
                type(loaded).__name__,
            )

    merged = _deep_merge_dicts(existing, overrides)
    user_config_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


# ============================================================================
# State manager — the part of the app that has no rumps dependency, so it's
# unit-testable.
# ============================================================================
class UiState(StrEnum):
    """Menu-bar display states.

    This was a bare ``class UiState(str)`` of class attributes whose docstring
    said "str so .value works" — ``.value`` was exactly what did not work,
    because nothing here was an enum member. Now it is one, next to
    NotificationType which was already a proper ``str, Enum``. Comparisons
    against plain strings still hold, so nothing downstream changes.
    """

    IDLE = "idle"
    APPROACHING = "approaching"
    DETECTED = "detected"
    #: Opening the capture devices. Transient, but not instant: measured on
    #: this Mac, the first start in a process takes 5.3 s of cold CoreAudio
    #: enumeration and later ones 200-270 ms.
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    #: Draining the capture queues and writing metadata, ~250 ms.
    SAVING = "saving"
    #: Recording is safely on disk and Whisper has NOT started. We are holding
    #: off because a call is still in progress. Distinct from PROCESSING
    #: because nothing is running: Start works, detection works, and quitting
    #: discards nothing.
    DEFERRED = "deferred"
    PROCESSING = "processing"
    OFF_HOURS = "off_hours"


@dataclass
class UiSnapshot:
    """Pure-data snapshot of what the menu bar wants to show right now."""

    state: str = UiState.IDLE
    recording_started_monotonic: float | None = None
    # When the current pause began. While paused, elapsed time freezes; on
    # resume, recording_started_monotonic is shifted forward by the pause
    # length so the MM:SS counter only counts recorded time.
    paused_at_monotonic: float | None = None
    current_meeting: MeetingContext | None = None
    selected_language_code: str | None = None
    show_off_hours: bool = False

    def title_text(self) -> str:
        """The text rumps shows next to the icon. Empty for compact icons-only.

        Took a ``blink_filled`` argument that nothing passed and nothing read.
        """
        if self.state == UiState.RECORDING and self.recording_started_monotonic is not None:
            elapsed = time.monotonic() - self.recording_started_monotonic
            return format_duration_mmss(elapsed)
        if self.state == UiState.OFF_HOURS:
            return "off-hours"
        if self.state == UiState.STARTING:
            return "starting…"
        if self.state == UiState.SAVING:
            return "saving…"
        if self.state == UiState.DEFERRED:
            return "waiting"
        return ""


#: Offered in the whitelist editor whether or not they are running, so a user
#: who unticks one can ever tick it back on.
KNOWN_MEETING_APPS: tuple[str, ...] = (
    "zoom.us",
    "Microsoft Teams",
    "Webex",
    "Slack",
    "FaceTime",
    "Discord",
    "Google Meet",
    "Whereby",
)

#: Cap on how many *running* apps get appended, so the submenu stays a menu.
MAX_RUNNING_CANDIDATES = 20


def running_app_names() -> list[str]:
    """Names of applications running right now, best effort."""
    try:
        from src.detection.process_monitor import _nsworkspace_apps

        apps = _nsworkspace_apps()
    except Exception:  # pragma: no cover (non-macOS)
        return []
    if not apps:
        return []
    names: list[str] = []
    for app in apps:
        bundle = str(app.get("bundle_id") or "").lower()
        # Apple's own apps are almost never what someone wants to whitelist,
        # and they are most of the list. FaceTime is offered above regardless.
        if bundle.startswith("com.apple."):
            continue
        name = str(app.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def whitelist_candidates(
    enabled: Iterable[str], *, running: Iterable[str] | None = None,
) -> list[str]:
    """Everything the whitelist editor should offer, in a stable order.

    The old menu was built from the whitelist itself, with every entry ticked.
    So unticking Slack removed it from the config, and after a restart it was
    not in the list any more — unreachable except by hand-editing YAML. There
    was no way to add an app either.

    Offering enabled + known + running fixes both directions: what you turned
    off stays visible to turn back on, and anything you are actually running
    can be added without leaving the menu.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(name: str) -> None:
        key = name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(name.strip())

    for name in enabled:            # yours first — the ones you chose
        add(str(name))
    for name in KNOWN_MEETING_APPS:  # then what we ship
        add(name)
    extra = 0
    for name in sorted(running or []):
        if extra >= MAX_RUNNING_CANDIDATES:
            break
        before = len(out)
        add(str(name))
        extra += len(out) - before
    return out
