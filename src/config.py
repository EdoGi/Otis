"""Configuration loader for Otis.

Loads YAML config, applies sensible defaults, and expands ``~`` in any path-like
string values. Designed to be safe to import on non-macOS platforms (no native
imports here).

The returned :class:`Config` object supports both dotted-attribute access
(``cfg.audio.sample_rate``) and dict-style access (``cfg["audio"]``,
``cfg.get("audio", "sample_rate")``).
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
USER_CONFIG_PATH = Path("~/.otis/config.yaml")

# In-code defaults: a safety net if the bundled YAML is missing or partial.
DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "Otis",
        "launch_at_login": True,
        "working_days": [0, 1, 2, 3, 4],
        "working_hours": {"start": "08:00", "end": "20:00"},
    },
    "audio": {
        "mic_device": None,
        "system_audio_device": "BlackHole 2ch",
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
    },
    "detection": {
        "process_monitor": {
            "enabled": True,
            "poll_interval_seconds": 5,
            "whitelisted_apps": [
                "zoom.us",
                "Microsoft Teams",
                "Webex",
                "Slack",
                "FaceTime",
            ],
            "blacklisted_apps": ["SuperWhisper"],
        },
        "calendar": {
            "enabled": True,
            "poll_interval_seconds": 60,
            "pre_meeting_alert_minutes": 2,
            "provider": "google",
            "accounts": [
                {
                    "label": "personal",
                    "credentials_path": "~/.otis/credentials.json",
                    "token_path": "~/.otis/google_token.json",
                    "calendar_ids": ["primary"],
                }
            ],
        },
        "mic_activation": {
            "enabled": True,
            "trigger_apps_only": True,
        },
    },
    "transcription": {
        "engine": "mlx-whisper",
        "model": "small",
        "language": None,
    },
    "storage": {
        "transcript_dir": "~/Otis/transcripts",
        "audio_dir": "~/Otis/audio",
        "audio_retention_days": 30,
    },
    "web": {
        "port": 8765,
        "host": "127.0.0.1",
    },
    "mcp": {
        "enabled": True,
        "port": 8766,
    },
}

# Keys whose values are filesystem paths and should be expanded.
_PATH_KEYS = {
    "credentials_path",
    "token_path",
    "transcript_dir",
    "audio_dir",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand_paths(node: Any, parent_key: str | None = None) -> Any:
    """Walk the config tree expanding ``~`` for known path keys."""
    if isinstance(node, dict):
        return {k: _expand_paths(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_paths(v, parent_key) for v in node]
    if isinstance(node, str) and parent_key in _PATH_KEYS:
        return os.path.expanduser(node)
    return node


class _AttrDict(dict):
    """Dict subclass that also exposes its string keys as attributes.

    Nested dicts and dicts inside lists are wrapped recursively so that
    ``cfg.audio.sample_rate`` and ``cfg.detection.process_monitor.whitelisted_apps[0]``
    both work without surprising the user.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)


class Config:
    """Wraps the loaded config tree with helpers for typed access.

    Three equivalent ways to read a value::

        cfg.audio.sample_rate                 # attribute style
        cfg["audio"]["sample_rate"]           # dict style
        cfg.get("audio", "sample_rate")       # dotted lookup with default
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: _AttrDict = _AttrDict(data)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        """Convenience classmethod equivalent to :func:`load_config`."""
        return load_config(path)

    @property
    def raw(self) -> _AttrDict:
        return self._data

    def get(self, *path: str, default: Any = None) -> Any:
        """Dotted-path lookup, e.g. ``cfg.get("audio", "sample_rate")``."""
        node: Any = self._data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, so this won't
        # interfere with ``self._data`` set in __init__.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._data, name)
        except AttributeError:
            raise AttributeError(f"Config has no key {name!r}") from None

    def __repr__(self) -> str:
        return f"Config({list(self._data.keys())})"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load YAML config, deep-merge with defaults, expand ``~`` in path values.

    * ``path=None`` → load the bundled ``config/default_config.yaml``. If the
      bundled file is missing (broken install), fall back to in-code DEFAULTS
      with a logged warning.
    * Explicit ``path`` that does not exist → raise :class:`FileNotFoundError`
      with a clear, actionable message.
    """
    explicit = path is not None
    cfg_path = Path(path) if explicit else DEFAULT_CONFIG_PATH
    file_data: dict[str, Any] = {}

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config root in {cfg_path} must be a mapping, got {type(loaded).__name__}."
            )
        file_data = loaded
    elif explicit:
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. "
            f"Pass an existing YAML file or omit --config to use the bundled default."
        )
    else:
        logger.warning(
            "Bundled config %s not found; using built-in defaults. "
            "(Did the install copy config/default_config.yaml?)",
            cfg_path,
        )

    merged = _deep_merge(DEFAULTS, file_data)
    expanded = _expand_paths(merged)
    return Config(expanded)


def load_user_config(
    path: str | os.PathLike[str] | None = None,
    *,
    user_config_path: str | os.PathLike[str] | None = None,
) -> Config:
    """Load the bundled config, then merge ``~/.otis/config.yaml`` on top.

    Used by the menu-bar app so settings the user toggles in the UI (Whisper
    model, working days, app whitelist) take effect on next launch without
    them having to edit the bundled YAML.
    """
    cfg = load_config(path)
    user_path = Path(user_config_path) if user_config_path is not None else USER_CONFIG_PATH
    user_path = user_path.expanduser()
    if user_path.exists():
        try:
            user_data = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Could not read user config %s (%s); ignoring.", user_path, exc)
            return cfg
        if isinstance(user_data, dict) and user_data:
            merged = _deep_merge(cfg.raw, user_data)
            return Config(_expand_paths(merged))
    return cfg
