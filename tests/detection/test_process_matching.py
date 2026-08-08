"""Regression cover for audit finding A3 — detection fired on phantom apps.

``process_monitor`` matched whitelist and browser entries with a raw
``entry in process_name`` substring test. Enumerated against the real process
table on the user's Mac:

    'arc'    matched  TrialARChivingService, seARChpartyd, seARChpartyuseragent
    'safari' matched  SAFARIWidgetExtension, com.apple.SAFARIPlatformSupport.Helper

Neither Arc nor Safari was running. Those Apple daemons run 24/7, so the
browser set was never empty — which is how 398 "Meeting detected" toasts were
produced against 8 real recordings.

Matching is now exact, or on a word boundary so helper processes still count.
"""

from __future__ import annotations

import pytest

from src.detection.process_monitor import ProcessMonitor, app_name_matches


# ---------------------------------------------------------------------------
# The matcher itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entry,process_name",
    [
        ("arc", "searchpartyd"),
        ("arc", "searchpartyuseragent"),
        ("arc", "TrialArchivingService"),
        ("safari", "SafariWidgetExtension"),
        ("safari", "com.apple.SafariPlatformSupport.Helper"),
        ("slack", "slackd"),
        ("zoom.us", "zoomus-helper-thing"),
        ("teams", "teamviewer"),
    ],
)
def test_phantom_matches_are_rejected(entry: str, process_name: str) -> None:
    assert app_name_matches(entry, process_name) is False, (
        f"{entry!r} must not match {process_name!r}"
    )


@pytest.mark.parametrize(
    "entry,process_name",
    [
        ("arc", "Arc"),
        ("safari", "Safari"),
        ("slack", "Slack"),
        ("zoom.us", "zoom.us"),
        ("microsoft teams", "Microsoft Teams"),
        ("facetime", "FaceTime"),
        ("webex", "Webex"),
        # Helper processes belong to the same app and should still count.
        ("slack", "Slack Helper"),
        ("microsoft teams", "Microsoft Teams Helper (Renderer)"),
        ("brave browser", "Brave Browser"),
    ],
)
def test_real_matches_are_accepted(entry: str, process_name: str) -> None:
    assert app_name_matches(entry, process_name) is True, (
        f"{entry!r} should match {process_name!r}"
    )


def test_matching_is_case_insensitive() -> None:
    assert app_name_matches("SLACK", "slack") is True
    assert app_name_matches("slack", "SLACK") is True


def test_empty_entry_never_matches() -> None:
    assert app_name_matches("", "anything") is False
    assert app_name_matches("slack", "") is False


def test_regex_metacharacters_in_a_name_are_literal() -> None:
    """'zoom.us' must not let '.' act as a wildcard."""
    assert app_name_matches("zoom.us", "zoomXus") is False
    assert app_name_matches("zoom.us", "zoom.us") is True


# ---------------------------------------------------------------------------
# End to end through poll_once
# ---------------------------------------------------------------------------
def make_monitor(**kw) -> ProcessMonitor:
    defaults = dict(
        whitelisted_apps=["zoom.us", "Microsoft Teams", "Slack"],
        blacklisted_apps=["SuperWhisper"],
        poll_interval_seconds=5,
    )
    defaults.update(kw)
    return ProcessMonitor(**defaults)


def procs(*names: str) -> list[dict]:
    return [{"pid": 100 + i, "name": n} for i, n in enumerate(names)]


def test_apple_daemons_do_not_trigger_detection() -> None:
    """The exact live process table that produced 398 phantom detections."""
    detected: list[str] = []
    mon = make_monitor(
        process_iter=lambda: iter(procs(
            "searchpartyd",
            "searchpartyuseragent",
            "TrialArchivingService",
            "SafariWidgetExtension",
            "com.apple.SafariPlatformSupport.Helper",
        )),
        mic_probe=lambda: True,  # dictation tool holding the mic
    )
    mon.on_meeting_detected(lambda name, _pids=None: detected.append(name))

    assert mon.poll_once() == set()
    assert detected == []


def test_a_real_meeting_app_still_detects() -> None:
    detected: list[str] = []
    mon = make_monitor(
        process_iter=lambda: iter(procs("zoom.us", "searchpartyd")),
        mic_probe=lambda: False,
    )
    mon.on_meeting_detected(lambda name, _pids=None: detected.append(name))

    assert mon.poll_once() == {"zoom.us"}
    assert detected == ["zoom.us"]


def test_blacklist_still_wins() -> None:
    mon = make_monitor(
        process_iter=lambda: iter(procs("SuperWhisper", "zoom.us")),
        mic_probe=lambda: False,
    )
    assert mon.poll_once() == {"zoom.us"}
