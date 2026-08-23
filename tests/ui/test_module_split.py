"""Cover for audit item 42 — menubar.py was 2 200 lines of four concerns.

rumps wiring, a state machine, settings persistence and disk scanning, in one
file. It is three now, and this file guards the seams: the public import path
must not change, the mixin must actually be wired in, and the state module
must stay free of the app.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_the_public_import_path_is_unchanged() -> None:
    """Everything that used to come from src.ui.menubar still does.

    A lint --fix will happily delete re-export imports it thinks are unused,
    which is exactly how this broke while it was being written.
    """
    import src.ui.menubar as menubar

    missing = [name for name in menubar.__all__ if not hasattr(menubar, name)]
    assert missing == []


def test_the_settings_mixin_is_actually_wired_in() -> None:
    from src.ui.menubar import MenuBarApp
    from src.ui.menubar_settings import SettingsMenuMixin

    assert issubclass(MenuBarApp, SettingsMenuMixin)
    for name in ("_build_settings_submenu", "_on_model_picked",
                 "_on_whitelist_toggled", "_update_working_hours_menu"):
        assert hasattr(MenuBarApp, name), name


def test_the_state_module_does_not_import_the_app() -> None:
    """It is the layer everything else depends on; a cycle would defeat it."""
    source = Path("src/ui/menubar_state.py").read_text(encoding="utf-8")
    assert "from src.ui.menubar import" not in source
    assert "import rumps" not in source


def test_the_state_module_holds_no_rumps_wiring() -> None:
    """No dependency on rumps, so it is importable and testable without a Mac.

    Checked against real imports rather than the word "rumps" in the text —
    the docstrings quite reasonably mention it.
    """
    import src.ui.menubar_state as state

    source = Path("src/ui/menubar_state.py").read_text(encoding="utf-8")
    code = [ln for ln in source.splitlines() if not ln.strip().startswith("#")]
    assert not [ln for ln in code if "import rumps" in ln]
    assert not [ln for ln in code if "rumps." in ln]
    assert "rumps" not in {m.__name__ for m in vars(state).values()
                           if inspect.ismodule(m)}


def test_no_file_is_back_over_two_thousand_lines() -> None:
    for path in ("src/ui/menubar.py", "src/ui/menubar_state.py",
                 "src/ui/menubar_settings.py"):
        n = len(Path(path).read_text(encoding="utf-8").splitlines())
        assert n < 2000, f"{path} is {n} lines"


# ---------------------------------------------------------------------------
# The behaviour the split must not have changed
# ---------------------------------------------------------------------------
def test_uistate_is_a_real_enum() -> None:
    """It was `class UiState(str)` whose docstring claimed `.value` worked."""
    from enum import Enum

    from src.ui.menubar import UiState

    assert issubclass(UiState, Enum)
    assert UiState.RECORDING.value == "recording"
    assert UiState.IDLE == "idle", "comparisons against plain strings still hold"


def test_title_text_takes_no_dead_argument() -> None:
    from src.ui.menubar import UiSnapshot

    params = inspect.signature(UiSnapshot.title_text).parameters
    assert list(params) == ["self"], "blink_filled was never read"


def test_the_app_still_assembles(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    bar = MenuBarApp(
        config=Config({"storage": {"audio_dir": str(tmp_path / "a"),
                                   "transcript_dir": str(tmp_path / "t")},
                       "detection": {"process_monitor":
                                     {"whitelisted_apps": ["zoom.us"]}}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    # The settings submenu is built by the mixin; if it were not wired in,
    # construction would have raised long before here.
    assert bar._mi["settings_root"] is not None
    assert bar._whitelist_items
    assert bar._model_items
