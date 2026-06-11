"""Tests for src/config.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import DEFAULTS, Config, load_config, load_user_config


def test_load_default_config_returns_known_keys() -> None:
    cfg = load_config()
    for key in ("app", "audio", "detection", "transcription", "storage", "web"):
        assert key in cfg
    # Dead keys were removed in the Phase-6 cleanup.
    assert "mcp" not in cfg
    assert cfg.get("audio", "sample_rate") == 16000
    assert cfg.get("audio", "system_audio_device") == "BlackHole 2ch"


def test_config_classmethod_load_works() -> None:
    """Reviewer expects Config.load(path)-style entry point."""
    cfg = Config.load()
    assert isinstance(cfg, Config)
    assert cfg.get("audio", "sample_rate") == 16000


def test_config_supports_attribute_access() -> None:
    """cfg.audio.sample_rate must work, not just cfg.get(...)."""
    cfg = load_config()
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.system_audio_device == "BlackHole 2ch"
    assert cfg.transcription.model == "small"
    assert cfg.detection.process_monitor.poll_interval_seconds == 5
    # Lists of primitives stay lists, but list-of-dicts entries would be wrapped.
    assert cfg.app.working_days == [0, 1, 2, 3, 4]


def test_attribute_access_raises_for_unknown_keys() -> None:
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.audio.nonexistent_field


def test_load_config_raises_clear_error_for_missing_explicit_path(tmp_path: Path) -> None:
    """Missing user-supplied config should raise FileNotFoundError, not silently default."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError) as exc:
        load_config(missing)
    assert str(missing) in str(exc.value)


def test_load_config_default_path_falls_back_to_in_code_defaults(monkeypatch, tmp_path: Path) -> None:
    """If the *bundled* config is missing (broken install), fall back to DEFAULTS."""
    from src import config as config_mod
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")
    cfg = load_config()
    assert cfg.get("app", "name") == DEFAULTS["app"]["name"]
    assert cfg.get("transcription", "model") == "small"


def test_load_config_deep_merges_user_overrides(tmp_path: Path) -> None:
    p = tmp_path / "user.yaml"
    p.write_text(
        """
audio:
  sample_rate: 44100
transcription:
  model: large-v3
""".strip()
    )
    cfg = load_config(p)
    # Override applied
    assert cfg.get("audio", "sample_rate") == 44100
    assert cfg.get("transcription", "model") == "large-v3"
    # Sibling defaults preserved
    assert cfg.get("audio", "system_audio_device") == "BlackHole 2ch"
    assert cfg.get("transcription", "language") is None


def test_load_config_expands_tilde_in_path_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config()
    transcript_dir = cfg.get("storage", "transcript_dir")
    accounts = cfg.get("detection", "calendar", "accounts")
    assert transcript_dir == os.path.expanduser("~/Otis/transcripts")
    assert "~" not in transcript_dir
    # Multi-account schema: every credentials_path / token_path inside the
    # ``accounts`` list must be expanded too.
    assert accounts and len(accounts) >= 1
    for entry in accounts:
        assert entry["credentials_path"] == os.path.expanduser(
            "~/.otis/credentials.json"
        )
        assert "~" not in entry["credentials_path"]
        assert "~" not in entry["token_path"]


def test_config_get_returns_default_when_path_missing() -> None:
    cfg = Config({"a": {"b": 1}})
    assert cfg.get("a", "b") == 1
    assert cfg.get("a", "missing") is None
    assert cfg.get("a", "missing", default="fallback") == "fallback"
    assert cfg.get("missing", "deep") is None


def test_config_load_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_load_user_config_returns_bundled_when_user_file_missing(tmp_path: Path) -> None:
    """No ~/.otis/config.yaml → identical to load_config()."""
    cfg = load_user_config(user_config_path=tmp_path / "missing.yaml")
    assert cfg.audio.sample_rate == 16000
    assert cfg.transcription.model == "small"


def test_load_user_config_overrides_only_specified_keys(tmp_path: Path) -> None:
    """User config takes precedence; sibling defaults still apply."""
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        "transcription:\n  model: large-v3\napp:\n  working_days: [0, 1]\n"
    )
    cfg = load_user_config(user_config_path=user_path)
    assert cfg.transcription.model == "large-v3"
    assert cfg.transcription.language is None  # default preserved
    assert cfg.app.working_days == [0, 1]
    assert cfg.audio.sample_rate == 16000


def test_load_user_config_corrupt_yaml_falls_back(tmp_path: Path) -> None:
    """A broken user config doesn't crash the app."""
    user_path = tmp_path / "user.yaml"
    user_path.write_text(":::not yaml")
    cfg = load_user_config(user_config_path=user_path)
    assert cfg.transcription.model == "small"


def test_config_working_days_uses_python_weekday_convention() -> None:
    """Working days must use weekday() (0=Monday) not isoweekday() (1=Monday)."""
    cfg = load_config()
    days = cfg.get("app", "working_days")
    assert days == [0, 1, 2, 3, 4]  # Mon–Fri, weekday() convention
    assert min(days) == 0  # never 1, which would be isoweekday()
    assert 6 not in days  # Sunday excluded under weekday() convention


# ---------------------------------------------------------------------------
# apply_overrides — the in-memory half of settings persistence
# ---------------------------------------------------------------------------
def test_apply_overrides_deep_merges_into_live_tree() -> None:
    cfg = Config({"app": {"working_hours": {"start": "08:00", "end": "20:00"},
                          "working_days": [0, 1, 2, 3, 4]}})
    cfg.apply_overrides({"app": {"working_hours": {"start": "09:00"}}})
    assert cfg.get("app", "working_hours", "start") == "09:00"
    # Sibling keys survive the merge.
    assert cfg.get("app", "working_hours", "end") == "20:00"
    assert cfg.get("app", "working_days") == [0, 1, 2, 3, 4]


def test_apply_overrides_replaces_lists() -> None:
    cfg = Config({"app": {"working_days": [0, 1, 2, 3, 4]}})
    cfg.apply_overrides({"app": {"working_days": [5, 6]}})
    assert cfg.get("app", "working_days") == [5, 6]


def test_apply_overrides_expands_path_keys() -> None:
    cfg = Config({"storage": {"audio_dir": "/tmp/a"}})
    cfg.apply_overrides({"storage": {"audio_dir": "~/elsewhere"}})
    assert cfg.get("storage", "audio_dir").startswith("/")
    assert "~" not in cfg.get("storage", "audio_dir")


def test_apply_overrides_keeps_attribute_access() -> None:
    cfg = Config({"transcription": {"model": "small"}})
    cfg.apply_overrides({"transcription": {"model": "medium"}})
    assert cfg.transcription.model == "medium"
