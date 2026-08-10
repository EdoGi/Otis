"""Tests for the process-enumeration source (audit finding C1).

``psutil.process_iter(attrs=["pid","name"])`` cost 24.8 ms every 5 s — 0.5% of
a core, all day, and essentially the app's entire idle CPU. The reason is
subtle: ``Process.name()`` falls back to ``cmdline()`` whenever the name is 15+
characters, and that is a second sysctl copying the target's whole argv and
environment. 261 of 571 processes on this Mac qualified.

``NSWorkspace.runningApplications()`` answers the question we actually have —
which *applications* are running — in 0.58 ms over 93 entries. Measured
end-to-end, poll_once went 24.8 ms → 1.43 ms.
"""

from __future__ import annotations

import pytest

from src.detection.process_monitor import (
    ProcessMonitor,
    _default_process_iter,
    _nsworkspace_apps,
    _psutil_processes,
)


def make_monitor(processes, **kw) -> ProcessMonitor:
    defaults = dict(
        whitelisted_apps=["zoom.us", "Microsoft Teams"],
        blacklisted_apps=["SuperWhisper"],
        poll_interval_seconds=5,
        process_iter=lambda: iter(processes),
    )
    defaults.update(kw)
    return ProcessMonitor(**defaults)


# ---------------------------------------------------------------------------
# Bundle identifiers — locale-proof whitelisting
# ---------------------------------------------------------------------------
def test_a_bundle_id_can_match_when_the_name_is_localized() -> None:
    """NSWorkspace reports localized names ("Aperçu", "Accessibilité"), so a
    French Mac must still detect an app whitelisted by bundle id."""
    mon = make_monitor(
        [{"pid": 1, "name": "Réunion Zoom", "bundle_id": "us.zoom.xos"}],
        whitelisted_apps=["us.zoom.xos"],
    )
    assert mon.poll_once() == {"us.zoom.xos"}


def test_name_matching_still_works_without_a_bundle_id() -> None:
    mon = make_monitor([{"pid": 1, "name": "zoom.us"}])
    assert mon.poll_once() == {"zoom.us"}


def test_blacklist_applies_to_bundle_ids_too() -> None:
    mon = make_monitor(
        [{"pid": 1, "name": "Dictation", "bundle_id": "com.superwhisper.app"}],
        whitelisted_apps=["dictation"],
        blacklisted_apps=["com.superwhisper.app"],
    )
    assert mon.poll_once() == set()


def test_a_bundle_id_does_not_create_phantom_matches() -> None:
    """The Safari support helper carries com.apple.SafariPlatformSupport.Helper
    as its bundle id — matching must not resurrect the phantom."""
    mon = make_monitor(
        [{
            "pid": 1,
            "name": "AutoFill (iTerm2)",
            "bundle_id": "com.apple.SafariPlatformSupport.Helper",
        }],
        whitelisted_apps=["safari"],
    )
    assert mon.poll_once() == set()


def test_missing_bundle_id_key_is_tolerated() -> None:
    """The psutil fallback yields dicts with no bundle_id at all."""
    mon = make_monitor([{"pid": 1, "name": "zoom.us"}, {"pid": 2, "name": "Finder"}])
    assert mon.poll_once() == {"zoom.us"}


def test_empty_bundle_id_is_ignored() -> None:
    mon = make_monitor([{"pid": 1, "name": "zoom.us", "bundle_id": ""}])
    assert mon.poll_once() == {"zoom.us"}


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
def test_falls_back_to_psutil_when_appkit_is_unavailable(monkeypatch) -> None:
    import src.detection.process_monitor as pm

    monkeypatch.setattr(pm, "_nsworkspace_apps", lambda: None)
    out = list(_default_process_iter())
    assert out, "the fallback must still enumerate something"
    assert all("name" in p for p in out)


def test_psutil_fallback_yields_pid_and_name() -> None:
    out = list(_psutil_processes())
    assert out
    sample = out[0]
    assert "pid" in sample and "name" in sample


@pytest.mark.skipif(
    _nsworkspace_apps() is None, reason="AppKit/NSWorkspace unavailable"
)
class TestNSWorkspaceSource:
    def test_returns_applications_with_the_expected_shape(self) -> None:
        apps = _nsworkspace_apps()
        assert apps is not None and apps
        for app in apps:
            assert isinstance(app["pid"], int)
            assert isinstance(app["name"], str) and app["name"]
            assert isinstance(app["bundle_id"], str)

    def test_the_phantom_daemons_are_not_applications(self) -> None:
        """searchpartyd and friends are processes, not apps — NSWorkspace
        doesn't list them at all, so they can't be detected by mistake."""
        names = {a["name"].lower() for a in (_nsworkspace_apps() or [])}
        for phantom in (
            "searchpartyd",
            "searchpartyuseragent",
            "trialarchivingservice",
            "safariwidgetextension",
        ):
            assert phantom not in names

    def test_enumerates_far_fewer_entries_than_psutil(self) -> None:
        apps = _nsworkspace_apps() or []
        procs = list(_psutil_processes())
        assert len(apps) < len(procs), (
            "the whole point is asking a narrower question"
        )
