"""Tests for src/detection/process_monitor.py.

The monitor is fully decoupled from psutil and CoreAudio in its constructor —
both are passed in. So these tests don't need any system access; they just
exercise the matching, debouncing, and event logic.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterable

import pytest

from src.detection.process_monitor import ProcessMonitor


def _procs(*names_with_pid: tuple[str, int]) -> Iterable[dict[str, Any]]:
    return [{"pid": pid, "name": name} for name, pid in names_with_pid]


# ----------------------------------------------------------------- whitelist
def test_detects_zoom_when_running() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us", "Microsoft Teams"],
        process_iter=lambda: _procs(("zoom.us", 123)),
        mic_probe=lambda: False,
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
            mic_probe=lambda: False,
        )
        pm.on_meeting_detected(detected.append)
        pm.poll_once()
        assert detected == [app.lower()], f"failed to detect {app}"


def test_whitelist_is_case_insensitive() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["Zoom.us"],
        process_iter=lambda: _procs(("ZOOM.US", 1)),
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == []


# ------------------------------------------------------------------ browsers
def test_browser_only_counts_when_mic_active() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=[],  # browser detection alone
        process_iter=lambda: _procs(("Google Chrome", 99)),
        browser_apps=["Google Chrome"],
        mic_probe=lambda: True,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == ["google chrome"]


def test_browser_alone_with_mic_idle_is_ignored() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=[],
        process_iter=lambda: _procs(("Google Chrome", 99)),
        browser_apps=["Google Chrome"],
        mic_probe=lambda: False,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == []


def test_mic_activation_disabled_skips_browser_detection() -> None:
    """When mic_activation_enabled=False, browsers must never fire — even if
    the mic probe says active. Use case: a dictation tool (e.g. SuperWhisper)
    holds the mic open continuously, making the mic-active heuristic useless.
    """
    detected: list[str] = []
    probe_calls: list[bool] = []

    def probe() -> bool:
        probe_calls.append(True)
        return True

    pm = ProcessMonitor(
        whitelisted_apps=[],
        process_iter=lambda: _procs(("Safari", 1), ("Arc", 2)),
        browser_apps=["Safari", "Arc"],
        mic_probe=probe,
        mic_activation_enabled=False,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == []
    # And we shouldn't even bother probing the mic when the feature is off.
    assert probe_calls == []


def test_mic_activation_disabled_still_fires_whitelisted_apps() -> None:
    """Disabling mic activation must NOT disable detection of real meeting
    apps like Zoom — only the mic-based browser fallback is affected.
    """
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: _procs(("zoom.us", 11), ("Safari", 22)),
        browser_apps=["Safari"],
        mic_probe=lambda: True,
        mic_activation_enabled=False,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == ["zoom.us"]


def test_browser_mic_probe_failure_treated_as_inactive() -> None:
    """Mic probe raising must not prevent the monitor from running."""
    detected: list[str] = []

    def boom() -> bool:
        raise RuntimeError("CoreAudio offline")

    pm = ProcessMonitor(
        whitelisted_apps=[],
        process_iter=lambda: _procs(("Safari", 7)),
        browser_apps=["Safari"],
        mic_probe=boom,
    )
    pm.on_meeting_detected(detected.append)
    pm.poll_once()
    assert detected == []


# ------------------------------------------------------------------ debounce
def test_debounce_suppresses_redetect_within_window() -> None:
    detected: list[str] = []
    pm = ProcessMonitor(
        whitelisted_apps=["zoom.us"],
        process_iter=lambda: _procs(("zoom.us", 1)),
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
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
        mic_probe=lambda: False,
    )
    pm.on_meeting_detected(bad)
    pm.on_meeting_detected(fine_calls.append)
    pm.poll_once()
    assert fine_calls == ["zoom.us"]
