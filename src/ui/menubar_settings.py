"""The Settings submenu: building it, and reacting to what is picked.

Split out of ``src/ui/menubar.py`` (audit item 42). A mixin rather than a
separate object because every one of these reads or writes app state — the
live config, the running engine, the process monitor — and threading that
through a collaborator would buy separation on paper and indirection in
practice.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from src.ui.menubar_state import (
    HOUR_END_PRESETS,
    HOUR_START_PRESETS,
    SUPPORTED_LANGUAGES,
    WEEKDAY_NAMES,
    WHISPER_MODELS,
    running_app_names,
    whitelist_candidates,
    write_user_config_override,
)
from src.ui.notifications import NotificationType

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.ui.menubar_state import UiSnapshot
    from src.ui.notifications import NotificationManager

logger = logging.getLogger(__name__)


class SettingsMenuMixin:
    """Settings menu behaviour for :class:`~src.ui.menubar.MenuBarApp`."""

    if TYPE_CHECKING:
        # Provided by MenuBarApp. Declared so the checker knows what the mixin
        # reads and writes; nothing here is assigned — that is the host's job.
        _config: Config
        _mi: dict[str, Any]
        _notifications: NotificationManager
        _user_config_path: Path
        _transcribe: Any
        _snapshot: UiSnapshot
        _lock: threading.RLock
        _lang_items: dict[str, Any]
        _detector: MeetingDetector

        def _on_schedule_tick(self, _timer: Any) -> None: ...

    def _build_settings_submenu(self, root: Any) -> None:
        import rumps

        # Whisper model
        model_root = rumps.MenuItem(
            f"Whisper Model: {self._config.get('transcription', 'model', default='small')}"
        )
        self._model_items: dict[str, rumps.MenuItem] = {}
        for m in WHISPER_MODELS:
            it = rumps.MenuItem(m, callback=self._on_model_picked)
            # The title may carry a "download" note; the model name rides here.
            it._otis_model = m
            self._model_items[m] = it
            model_root.add(it)
        self._mi["settings_model_root"] = model_root
        self._update_model_menu()

        # Working days
        days_root = rumps.MenuItem("Working Days")
        self._day_items: dict[int, rumps.MenuItem] = {}
        active_days = set(self._config.get("app", "working_days", default=[0, 1, 2, 3, 4]))
        for idx, name in enumerate(WEEKDAY_NAMES):
            it = rumps.MenuItem(name, callback=self._on_day_toggled)
            it._otis_day_index = idx
            it.state = idx in active_days
            self._day_items[idx] = it
            days_root.add(it)
        self._mi["settings_days_root"] = days_root

        # Working hours — Start / End submenus with hourly presets.
        hours_root = rumps.MenuItem(self._working_hours_title())
        self._hour_items: dict[tuple[str, str], rumps.MenuItem] = {}
        for which, presets in (("start", HOUR_START_PRESETS), ("end", HOUR_END_PRESETS)):
            sub = rumps.MenuItem(which.capitalize())
            for hhmm in presets:
                it = rumps.MenuItem(hhmm, callback=self._on_working_hour_picked)
                it._otis_which = which
                it._otis_hhmm = hhmm
                self._hour_items[(which, hhmm)] = it
                sub.add(it)
            hours_root.add(sub)
        self._mi["settings_hours_root"] = hours_root
        self._update_working_hours_menu()

        # App whitelist
        wl_root = rumps.MenuItem("App Whitelist")
        self._whitelist_items: dict[str, rumps.MenuItem] = {}
        enabled = [
            str(a) for a in self._config.get(
                "detection", "process_monitor", "whitelisted_apps", default=[]
            ) or []
        ]
        for app_name in whitelist_candidates(enabled, running=running_app_names()):
            it = rumps.MenuItem(app_name, callback=self._on_whitelist_toggled)
            it._otis_app_name = app_name
            it.state = app_name in enabled
            self._whitelist_items[app_name] = it
            wl_root.add(it)
        self._mi["settings_whitelist_root"] = wl_root

        root.add(model_root)
        root.add(days_root)
        root.add(hours_root)
        root.add(wl_root)

    # =====================================================================
    # Detector callback wiring (background-thread side)
    # =====================================================================
    def _persist_setting(self, overrides: dict[str, Any]) -> None:
        """Persist a settings change to disk AND the live config.

        Without the in-memory half, anything that re-reads ``self._config``
        (the schedule tick, the next transcription) would keep seeing the
        stale pre-toggle value until restart.
        """
        write_user_config_override(self._user_config_path, overrides)
        self._config.apply_overrides(overrides)
    def _on_language_picked(self, sender: Any) -> None:
        code = getattr(sender, "_otis_lang_code", None)
        with self._lock:
            self._snapshot.selected_language_code = code
        self._update_language_menu(code)
        # Persist so the choice survives restart and the transcription
        # pipeline picks it up via load_user_config().
        self._persist_setting({"transcription": {"language": code}})
    def _on_model_picked(self, sender: Any) -> None:
        model = getattr(sender, "_otis_model", sender.title)
        if model not in WHISPER_MODELS:
            return
        from src.storage.model_cache import DOWNLOAD_MB, cached_model_names, is_model_cached

        if not is_model_cached(model, cached_model_names()):
            # Not a modal: the menu already said so; this confirms what follows.
            self._notifications.notify(
                NotificationType.SETTING_SAVED,
                f"Whisper model: {model}",
                f"Not downloaded yet — about {_size_label(DOWNLOAD_MB.get(model))} will "
                f"download from huggingface.co before the next transcription.",
                force=True,
            )
        self._persist_setting({"transcription": {"model": model}})
        self._mi["settings_model_root"].title = f"Whisper Model: {model}"
        self._update_model_menu(active=model)
        # Applied on a worker: switching takes the engine's run lock, which is
        # held for the whole of a transcription — minutes. The checkmark and
        # the config are already correct; this makes the engine agree.
        threading.Thread(
            target=self._apply_model_live,
            args=(model,),
            name="otis-model-switch",
            daemon=True,
        ).start()
    def _apply_model_live(self, model: str) -> None:
        engine = self._engine()
        if engine is None:
            self._notifications.notify(
                NotificationType.SETTING_SAVED,
                f"Whisper model: {model}",
                "Takes effect on restart.",
            )
            return
        try:
            changed = engine.set_model(model)
        except Exception as exc:
            logger.exception("Could not switch the Whisper model")
            self._notifications.notify(
                NotificationType.ERROR, "Could not switch model", str(exc), force=True
            )
            return
        if changed:
            self._notifications.notify(
                NotificationType.SETTING_SAVED,
                f"Whisper model: {model}",
                "In effect from the next transcription.",
            )
    def _engine(self) -> Any | None:
        """The live WhisperEngine, if this front end has one.

        ``None`` in tests and in record-only setups, where "takes effect on
        restart" is the honest thing to say.
        """
        pipeline = getattr(self._transcribe, "pipeline", None)
        processor = getattr(pipeline, "processor", None)
        return getattr(processor, "_engine", None)
    def _on_day_toggled(self, sender: Any) -> None:
        idx = getattr(sender, "_otis_day_index", None)
        if idx is None:
            return
        sender.state = not sender.state
        active = sorted(i for i, item in self._day_items.items() if item.state)
        self._persist_setting({"app": {"working_days": active}})
        # Re-evaluate the window right away — a day toggle can flip us
        # into/out of OFF_HOURS without waiting up to 60 s for the tick.
        self._on_schedule_tick(None)
    def _on_whitelist_toggled(self, sender: Any) -> None:
        sender.state = not sender.state
        active = [name for name, item in self._whitelist_items.items() if item.state]
        self._persist_setting(
            {"detection": {"process_monitor": {"whitelisted_apps": active}}}
        )
        applied = self._apply_whitelist_live(active)
        self._notifications.notify(
            NotificationType.SETTING_SAVED,
            "Whitelist saved",
            "Now watching: " + (", ".join(active) or "nothing")
            if applied
            else "Restart Otis for this to take effect.",
        )
    def _apply_whitelist_live(self, active: list[str]) -> bool:
        """Push the new whitelist into the running monitor. True if it landed."""
        monitor = getattr(self._detector, "process_monitor", None)
        setter = getattr(monitor, "set_whitelist", None)
        if setter is None:
            return False
        try:
            setter(active)
            return True
        except Exception:
            logger.exception("Could not apply the whitelist live")
            return False
    def _update_language_menu(self, code: str | None) -> None:
        title = "Auto-detect"
        for label, c in SUPPORTED_LANGUAGES:
            if c == code:
                title = label
                break
        self._mi["language_root"].title = f"Language: {title}"
        for label, item in self._lang_items.items():
            item.state = (label == title)
    def _update_model_menu(self, active: str | None = None) -> None:
        """Tick the active model and mark the ones that would download.

        The picker used to offer every model as if equal; choosing an uncached
        one silently downloaded ~150 MB (base) to ~3 GB (large-v3) inside the
        next transcription.
        """
        from src.storage.model_cache import DOWNLOAD_MB, cached_model_names, is_model_cached

        active = active or self._config.get("transcription", "model", default="small")
        cached = cached_model_names()
        for name, item in self._model_items.items():
            item.state = (name == active)
            title = name if is_model_cached(name, cached) else \
                f"{name}  (download ~{_size_label(DOWNLOAD_MB.get(name))})"
            if item.title != title:
                item.title = title

    def _working_hours_title(self) -> str:
        start = self._config.get("app", "working_hours", "start", default="08:00")
        end = self._config.get("app", "working_hours", "end", default="20:00")
        return f"Working Hours: {start} → {end}"
    def _update_working_hours_menu(self) -> None:
        current = {
            "start": self._config.get("app", "working_hours", "start", default="08:00"),
            "end": self._config.get("app", "working_hours", "end", default="20:00"),
        }
        for (which, hhmm), item in self._hour_items.items():
            item.state = (hhmm == current[which])
    def _on_working_hour_picked(self, sender: Any) -> None:
        which = getattr(sender, "_otis_which", None)
        hhmm = getattr(sender, "_otis_hhmm", None)
        if which not in ("start", "end") or not hhmm:
            return
        self._persist_setting({"app": {"working_hours": {which: hhmm}}})
        self._mi["settings_hours_root"].title = self._working_hours_title()
        self._update_working_hours_menu()
        # Apply immediately instead of waiting up to 60 s for the next tick.
        self._on_schedule_tick(None)

    # =====================================================================
    # Recording lifecycle (called from main thread)
    # =====================================================================


def _size_label(mb: float | None) -> str:
    if not mb:
        return "an unknown amount"
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"

