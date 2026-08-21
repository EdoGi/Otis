"""Regressions found by adversarially verifying the P0/P1 commits.

Each test here corresponds to a defect that the P0/P1 suite passed over — in
several cases because the test asserted the shape of the code rather than its
behaviour. They are grouped by the audit finding they belong to so the
connection back to the original claim stays visible.
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import pytest

from src.detection.process_monitor import app_name_matches, bundle_id_matches
from src.health import HealthRegistry, HealthState
from src.storage.audio_retention import AudioRetentionManager
from src.storage.transcript_store import TranscriptStore, transcript_is_complete


# ---------------------------------------------------------------------------
# Retention must not delete a half-lost meeting's audio
# ---------------------------------------------------------------------------
def _wav(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)


def _age(p: Path, days: float) -> None:
    t = time.time() - days * 86400
    os.utime(p, (t, t))


def _fm(sid: str, **extra) -> dict:
    fm = {
        "id": sid, "title": "M", "date": "2026-07-01", "start_time": "10:30",
        "end_time": "11:00", "duration_minutes": 30, "language": "en", "app": None,
        "participants": [], "tags": [], "audio_files": {"mic": None, "system": None},
        "audio_available": True, "model": "small",
    }
    fm.update(extra)
    return fm


@pytest.mark.parametrize("system_audio", ["silent", "missing"])
def test_a_half_captured_meeting_is_not_complete(system_audio: str) -> None:
    """P0 taught transcripts to record system_audio. Retention still read any
    non-`failed` transcript as a success, so the audio you would re-transcribe
    once routing is fixed was deleted at day 30 — the audit's own scenario."""
    assert transcript_is_complete(_fm("s", system_audio=system_audio)) is False
    assert transcript_is_complete(_fm("s", system_audio="ok")) is True
    assert transcript_is_complete(_fm("s", status="failed")) is False


def test_retention_keeps_audio_for_a_silent_system_stream(tmp_path: Path) -> None:
    audio = tmp_path / "audio"; audio.mkdir()
    store = TranscriptStore(tmp_path / "t")
    sid = "half-lost"
    _wav(audio / f"{sid}_mic.wav"); _wav(audio / f"{sid}_system.wav")
    (audio / f"{sid}_metadata.json").write_text(json.dumps({"session_id": sid}))
    for p in audio.iterdir():
        _age(p, 45)
    store.save(_fm(sid, system_audio="silent"), "# call\n\n_(no speech detected)_")

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert (audio / f"{sid}_mic.wav").exists()
    assert sid in report.protected_session_ids


def test_deleting_relocated_audio_marks_the_transcript(tmp_path: Path) -> None:
    """Relocated files are date-named, so the UUID is gone from the filename.
    The session id only survives in the sibling metadata — which is rewritten
    at transcription time and so is usually NOT past the horizon itself."""
    audio = tmp_path / "audio"; audio.mkdir()
    store = TranscriptStore(tmp_path / "t")
    base = audio / "2026" / "07"
    mic = base / "2026-07-01_1030_mic.wav"
    meta = base / "2026-07-01_1030_metadata.json"
    _wav(mic)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"session_id": "sid-relocated"}), encoding="utf-8")
    _age(mic, 45); _age(meta, 1)          # metadata rewritten yesterday
    store.save(_fm("sid-relocated"), "# ok\n\nbody")

    report = AudioRetentionManager(
        audio_dir=audio, transcript_store=store, retention_days=30
    ).cleanup_now()

    assert not mic.exists()
    assert report.transcripts_marked == ["sid-relocated"], (
        "the transcript would keep advertising audio that no longer exists"
    )


# ---------------------------------------------------------------------------
# Matching: dots belong to a name, and bundle ids have their own rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entry,name",
    [("arc", "com.apple.Arc.helper"),
     ("safari", "com.apple.SafariPlatformSupport.Helper"),
     ("teams", "com.microsoft.teams2")],
)
def test_a_dotted_process_name_is_not_a_word_boundary(entry: str, name: str) -> None:
    """psutil reports reverse-DNS process names. Treating "." as a boundary
    let a bare word match the middle of one, reopening the phantom class."""
    assert app_name_matches(entry, name) is False


@pytest.mark.parametrize(
    "entry,name",
    [("Microsoft Teams (work or school)",
      "Microsoft Teams (work or school) Helper (Renderer)"),
     ("webex+", "Webex+ Helper")],
)
def test_an_entry_ending_in_a_non_word_char_still_matches(entry, name) -> None:
    r"""\b after ")" demands a following word char, which never comes before a
    space — so these real app names only ever matched on exact equality."""
    assert app_name_matches(entry, name) is True


def test_bundle_ids_match_by_hierarchy_not_by_word() -> None:
    assert bundle_id_matches("us.zoom.xos", "us.zoom.xos") is True
    assert bundle_id_matches("us.zoom.xos", "us.zoom.xos.renderer") is True
    assert bundle_id_matches("arc", "com.apple.Arc.helper") is False
    assert bundle_id_matches("us.zoom", "com.us.zoom") is False


# ---------------------------------------------------------------------------
# Health: a first report renders but does not announce
# ---------------------------------------------------------------------------
def test_a_first_report_is_delivered_so_the_panel_can_render() -> None:
    reg = HealthRegistry()
    seen: list[tuple[HealthState, HealthState]] = []
    reg.on_change(lambda rep, prev: seen.append((prev, rep.state)))
    reg.report("system_audio", label="System audio", state=HealthState.OK)
    assert seen == [(HealthState.UNSET, HealthState.OK)], (
        "the panel is rendered from this callback; it must still fire"
    )


def test_unset_is_distinct_from_unknown() -> None:
    """UNKNOWN is a component saying 'I could not tell'. UNSET is 'nobody has
    reported yet'. Conflating them made a clean startup look like a recovery."""
    assert HealthState.UNSET is not HealthState.UNKNOWN
    reg = HealthRegistry()
    reg.report("a", label="A", state=HealthState.UNKNOWN)
    assert reg.overall() is HealthState.UNKNOWN
    assert reg.is_degraded() is False


# ---------------------------------------------------------------------------
# The fast app source is opt-in, because it needs a live main run loop
# ---------------------------------------------------------------------------
def test_the_default_process_source_is_the_one_that_always_works() -> None:
    """Cocoa only refreshes NSWorkspace.runningApplications while the main run
    loop runs. `otis run` is headless, so defaulting to it would have made the
    daemon permanently blind to any app launched after startup — verified on
    this machine by launching TextEdit from a background thread and never
    seeing it appear."""
    from src.detection.process_monitor import (
        _default_process_iter,
        _psutil_processes,
        nsworkspace_process_iter,
    )

    assert _default_process_iter is not nsworkspace_process_iter
    sample = list(_default_process_iter())
    assert sample and all("name" in p for p in sample)
    # It really is the psutil source: processes, not applications. The exact
    # count drifts between two enumerations as processes come and go, so
    # compare the magnitude rather than an exact figure.
    from src.detection.process_monitor import _nsworkspace_apps

    apps = _nsworkspace_apps()
    if apps is not None:
        assert len(sample) > len(apps) * 1.5, (
            f"{len(sample)} entries looks like the application source, not psutil"
        )


def test_the_menu_bar_opts_into_the_fast_source() -> None:
    """rumps drives a real run loop, so the menu bar — where battery actually
    matters — keeps the ~20x saving."""
    import inspect

    import src.main as main_mod

    src = inspect.getsource(main_mod)
    assert "process_iter=nsworkspace_process_iter" in src


def test_the_daemon_does_not_opt_in() -> None:
    import inspect

    import src.daemon as daemon_mod

    assert "nsworkspace_process_iter" not in inspect.getsource(daemon_mod)


# ---------------------------------------------------------------------------
# Detection must not be latched as started when it declined
# ---------------------------------------------------------------------------
def test_detector_start_reports_whether_everything_came_up() -> None:
    from src.detection.detector import MeetingDetector

    class Declining:
        def start(self):
            pass

        def stop(self):
            pass

        def is_running(self):
            return False

        def on_upcoming_meeting(self, cb):
            pass

        def on_meeting_should_end(self, cb):
            pass

    class Fine(Declining):
        def is_running(self):
            return True

    assert MeetingDetector(
        process_monitor=None, calendar_pollers=[Fine()]
    ).start() is True
    assert MeetingDetector(
        process_monitor=None, calendar_pollers=[Fine(), Declining()]
    ).start() is False


def test_daemon_retries_when_detection_declined_to_start() -> None:
    """One wedged socket at 08:00 used to mean no calendar detection until the
    working-hours window reopened the next morning."""
    from datetime import datetime

    from src.config import Config
    from src.daemon import OtisDaemon

    class FlakyDetector:
        def __init__(self):
            self.attempts = 0

        def start(self):
            self.attempts += 1
            return self.attempts > 1        # declines the first time

        def stop(self):
            pass

        def on_meeting_approaching(self, cb): pass
        def on_meeting_detected(self, cb): pass
        def on_meeting_ended(self, cb): pass
        def on_process_disappeared(self, cb): pass

    det = FlakyDetector()
    daemon = OtisDaemon(
        Config({"app": {"working_days": list(range(7)),
                        "working_hours": {"start": "00:00", "end": "23:59"}}}),
        notify=lambda _m: None,
        detector=det,                       # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 8, 10, 0),
    )

    daemon._apply_schedule()
    assert daemon._detector_started is False, "a declined start must not latch"
    daemon._apply_schedule()
    assert daemon._detector_started is True, "the next tick must retry"
    assert det.attempts == 2
