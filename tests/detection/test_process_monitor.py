"""Tests for src/detection/process_monitor.py.

The monitor is fully decoupled from psutil and CoreAudio in its constructor —
both are passed in. So these tests don't need any system access; they just
exercise the matching, debouncing, and event logic.
"""

from __future__ import annotations

import time
from typing import Any, Iterable


from src.detection.process_monitor import ProcessMonitor


def _procs(*names_with_pid: tuple[str, int]) -> Iterable[dict[str, Any]]:
    return [{"pid": pid, "name": name} for name, pid in names_with_pid]


# ----------------------------------------------------------------- whitelist
# NOTE: the browser-via-mic tests that used to live here were removed with
# the heuristic itself (audit A3). Detection is calendar-first; a browser
# being open is no longer evidence of a meeting. See test_process_matching.py
# for the exact phantom process names that used to trigger it.


def test_detects_zoom_when_running() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us", "Microsoft Teams"],
        process_iter=lambda: _procs(("zoom.us", 123)),
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == ["zoom.us"]


def test_detects_each_whitelisted_app() -> None:
    apps = ["zoom.us", "Microsoft Teams", "Webex", "Slack", "FaceTime"]
    for app in apps:
        detected: list[str] = []
        pm = ProcessMonitor(
            whitelisted_apps=apps,
            process_iter=lambda app=app: _procs((app, 1)),
        )
        pm.on_meeting_detected(detected.append)
        pm.poll_once()
        assert detected == [app.lower()], f"failed to detect {app}"


def test_whitelist_is_case_insensitive() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["Zoom.us"],
        process_iter=lambda: _procs(("ZOOM.US", 1)),
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == ["zoom.us"]


# ----------------------------------------------------------------- blacklist
def test_blacklist_blocks_match() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["whisper"],
        blacklisted_apps=["SuperWhisper"],
        process_iter=lambda: _procs(("SuperWhisper", 1)),
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == []


# ------------------------------------------------------------------ browsers
def test_debounce_suppresses_redetect_within_window() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: _procs(("zoom.us", 1)),
        debounce_seconds=30.0,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    pm.poll_once()
    pm.poll_once()
    assert detected == ["zoom.us"]


def test_debounce_lets_redetect_after_window() -> None:
    """When debounce is tiny, a second cycle re-fires."""
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: _procs(("zoom.us", 1)),
        debounce_seconds=0.0,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    time.sleep(0.01)
    pm.poll_once()
    assert detected == ["zoom.us", "zoom.us"]


def test_debounce_does_not_collapse_different_apps() -> None:
    """Debounce is per-app: Zoom and Slack starting close together still each fire once."""
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us", "slack"],
        process_iter=lambda: _procs(("zoom.us", 1), ("Slack", 2)),
        debounce_seconds=30.0,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    pm.poll_once()
    assert sorted(detected) == ["slack", "zoom.us"]


# ---------------------------------------------------------------------- end
def test_meeting_ended_fires_when_app_disappears() -> None:
    state = {"alive": True}

    def proc_iter():
        if state["alive"]:
            return _procs(("zoom.us", 1))
        return []

    detected: list[str] = []
    ended: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=proc_iter,
    )
    pm.on_meeting_detected(detected.append)
    pm.on_meeting_ended(ended.append)

    pm.poll_once()
    assert detected == ["zoom.us"] and ended == []
    state["alive"] = False
    pm.poll_once()
    assert ended == ["zoom.us"]


def test_active_meetings_snapshot() -> None:
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us", "slack"],
        process_iter=lambda: _procs(("zoom.us", 1), ("Slack", 2)),
    )
    pm.poll_once()
    assert set(pm.active_meetings()) == {"zoom.us", "slack"}


# ---------------------------------------------------------------- background
def test_start_stop_lifecycle() -> None:
    """The background thread must start and stop cleanly."""
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        poll_interval_seconds=0.01,
        process_iter=lambda: _procs(("zoom.us", 1)),
    )
    detected: list[str] = []
    pm.on_meeting_detected(detected.append)
    pm.start()
    # Give the thread time for a couple of cycles.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not detected:
        time.sleep(0.02)
    pm.stop()
    assert detected == ["zoom.us"]
    assert not pm.is_running()


def test_callback_exception_does_not_break_loop() -> None:
    """A misbehaving subscriber must not stop other callbacks from firing."""

    def bad(_: str) -> None:
        raise RuntimeError("boom")

    fine_calls: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: _procs(("zoom.us", 1)),
    )
    pm.on_meeting_detected(bad)
    pm.on_meeting_detected(fine_calls.append)
    pm.poll_once()
    assert fine_calls == ["zoom.us"]
