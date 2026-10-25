"""rumps/AppKit helpers the menu bar needs that rumps does not provide.

Split out of menubar.py to keep it under the size limit audit item 42 set; each
helper works around a specific rumps behaviour that hurt Otis in real use.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("src.ui.menubar")

#: Where 'Fix System Audio…' sends the user when the output bypasses BlackHole.
SOUND_SETTINGS_URL = "x-apple.systempreferences:com.apple.Sound-Settings.extension"


_METER_LABELS = {"mic": "Mic input", "system": "System input"}


def _meter_row(label: str, bar: str) -> str:
    return f"   {_METER_LABELS.get(label, label)}  {bar}"


def _forget_menu_item(item: Any) -> None:
    """Drop rumps' global reference to a MenuItem we are discarding.

    rumps.MenuItem.set_callback stores every item in NSApp._ns_to_py_and_callback
    and nothing ever removes it, so a discarded item lives for the process.
    """
    try:
        import rumps.rumps as rumps_impl

        rumps_impl.NSApp._ns_to_py_and_callback.pop(item._menuitem, None)
    except Exception:  # pragma: no cover (rumps internals changed)
        pass


def _start_timer(timer: Any) -> None:
    """Start a rumps.Timer so it also fires while a modal dialog is open.

    rumps adds every Timer in NSDefaultRunLoopMode only, and NSAlert.runModal
    runs the loop in NSModalPanelRunLoopMode, so while any alert was open the
    100 ms drain, the schedule tick and the duration tick all stopped (probe:
    11 fires per second -> 0). An unanswered 'Fix System Audio…' alert froze an
    instance on 2026-09-10 — for three days — and again on 2026-10-14.

    Event-tracking mode (a menu held open) is left out on purpose: the drain
    rebuilds Status submenu items, which must not change under an open menu.
    """
    was_running = bool(getattr(timer, "is_alive", lambda: False)())
    timer.start()
    if was_running:
        return
    try:
        from AppKit import NSModalPanelRunLoopMode
        from Foundation import NSRunLoop

        NSRunLoop.currentRunLoop().addTimer_forMode_(timer._nstimer, NSModalPanelRunLoopMode)
    except Exception:  # pragma: no cover (no AppKit, or rumps internals changed)
        logger.debug("Timer stays default-mode only", exc_info=True)


def _alert(**kwargs: Any) -> Any:
    """rumps.alert, brought to the front first.

    Otis is an accessory app that rumps activates once, at launch, so a later
    alert could open behind the frontmost window — invisible, while it blocked
    the menu. Only deliberate confirmations still use one.
    """
    import rumps

    try:
        from AppKit import NSApplication

        nsapp = NSApplication.sharedApplication()
        if hasattr(nsapp, "activate"):
            nsapp.activate()
        else:  # pragma: no cover (macOS < 14)
            nsapp.activateIgnoringOtherApps_(True)
    except Exception:  # pragma: no cover
        logger.debug("Could not activate before an alert", exc_info=True)
    return rumps.alert(**kwargs)
