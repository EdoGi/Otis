"""What Otis understands in a config file, and what to say when it doesn't.

Audit E10: the config had no schema, no validation, and no versioning. A typo
was simply ignored — ``transcription.deffer_while_in_call`` is accepted and
does nothing, so the user believes they turned something off that is still on.
An invalid Whisper model raised ``ValueError`` out of ``build_pipeline``,
outside any handler, which for a Login Item means the app never appears and
the traceback goes to a stdout nobody sees.

Two levels, deliberately:

* **Warnings** for keys we do not recognise. Never fatal — a config written
  for a newer Otis, or carrying a section a user keeps for their own notes,
  must not stop the app from starting.
* **Errors** for values that cannot work, where guessing would be worse than
  stopping: a model name that does not exist, a working-hours string that is
  not a time, a port outside its range.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: Every key Otis reads, as dotted paths. A trailing ``.*`` means "anything
#: below here is the user's own business" (calendar accounts, for instance).
KNOWN_KEYS: frozenset[str] = frozenset({
    "app.name",
    "app.working_days",
    "app.working_hours.start",
    "app.working_hours.end",
    "audio.mic_device",
    "audio.system_audio_device",
    "audio.sample_rate",
    "audio.channels",
    "detection.process_monitor.enabled",
    "detection.process_monitor.poll_interval_seconds",
    "detection.process_monitor.whitelisted_apps",
    "detection.process_monitor.blacklisted_apps",
    "detection.calendar.enabled",
    "detection.calendar.poll_interval_seconds",
    "detection.calendar.pre_meeting_alert_minutes",
    "detection.calendar.provider",
    "detection.calendar.accounts.*",
    "transcription.model",
    "transcription.language",
    "transcription.defer_while_in_call",
    "transcription.suggest_titles",
    "transcription.dedup_echoes",
    "transcription.idle_timeout_seconds",
    "storage.transcript_dir",
    "storage.audio_dir",
    "storage.audio_retention_days",
    "storage.audio_format",
    "storage.queue_dir",
    "web.port",
    "web.host",
})

VALID_MODELS = ("tiny", "base", "small", "medium", "large", "large-v3")
VALID_AUDIO_FORMATS = ("flac", "wav")
_HHMM = re.compile(r"^\d{1,2}:\d{2}$")


class ConfigError(ValueError):
    """A config value that cannot work, phrased for the person who wrote it."""


def _flatten(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.extend(_flatten(value, path))
            else:
                out.append((path, value))
    return out


def _is_known(path: str) -> bool:
    if path in KNOWN_KEYS:
        return True
    for known in KNOWN_KEYS:
        if not known.endswith(".*"):
            continue
        base = known[:-2]
        # The key itself counts too: `accounts` is a list, so it flattens to a
        # leaf rather than to `accounts.<something>`.
        if path == base or path.startswith(base + "."):
            return True
    return False


def _did_you_mean(value: str, options: tuple[str, ...]) -> str:
    close = difflib.get_close_matches(str(value).lower(), options, n=1, cutoff=0.6)
    if close:
        return f" Did you mean {close[0]!r}?"
    return f" Valid values: {', '.join(options)}."


def unknown_keys(data: dict[str, Any]) -> list[str]:
    """Dotted paths present in ``data`` that Otis does not read."""
    return sorted(
        path for path, _ in _flatten(data)
        if not _is_known(path) and not path.startswith("mcp.")
    )


def validate(data: dict[str, Any]) -> list[str]:
    """Check a merged config. Returns warnings; raises on unusable values."""
    model = _get(data, "transcription", "model")
    if model is not None and str(model) not in VALID_MODELS:
        raise ConfigError(
            f"transcription.model is {model!r}, which is not a Whisper model."
            + _did_you_mean(model, VALID_MODELS)
        )

    fmt = _get(data, "storage", "audio_format")
    if fmt is not None and str(fmt).lower() not in VALID_AUDIO_FORMATS:
        raise ConfigError(
            f"storage.audio_format is {fmt!r}."
            + _did_you_mean(fmt, VALID_AUDIO_FORMATS)
        )

    for edge in ("start", "end"):
        value = _get(data, "app", "working_hours", edge)
        if value is not None and not _HHMM.match(str(value)):
            raise ConfigError(
                f"app.working_hours.{edge} is {value!r}; it must look like "
                f'"09:00". An unparseable time used to disable the schedule '
                f"silently."
            )

    port = _get(data, "web", "port")
    if port is not None:
        try:
            port_num = int(port)
        except (TypeError, ValueError):
            raise ConfigError(f"web.port is {port!r}; it must be a number.") from None
        if not (1 <= port_num <= 65535):
            raise ConfigError(f"web.port is {port_num}; it must be 1-65535.")

    days = _get(data, "app", "working_days")
    if days is not None:
        if not isinstance(days, (list, tuple)) or any(
            not isinstance(d, int) or not (0 <= d <= 6) for d in days
        ):
            raise ConfigError(
                "app.working_days must be a list of 0-6 (0 = Monday). "
                f"Got {days!r}."
            )

    warnings = [
        f"Unknown config key {path!r} — Otis does not read it, so setting it "
        f"has no effect."
        for path in unknown_keys(data)
    ]
    for line in warnings:
        logger.warning("%s", line)
    return warnings


def _get(data: dict[str, Any], *path: str) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


__all__ = [
    "KNOWN_KEYS",
    "VALID_AUDIO_FORMATS",
    "VALID_MODELS",
    "ConfigError",
    "unknown_keys",
    "validate",
]
