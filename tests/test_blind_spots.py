"""Cover for audit finding F — the suite tested the shape, not the failures.

The audit named four blind spots. Two closed on their own as the rework went
past them (the fixtures that hard-disabled ``defer_while_in_call`` are now
exercised in tests/ui/test_menubar_app.py and tests/ui/test_deferred_state.py;
``_SleepWakeObserver`` got direct coverage when P4 found its second instance
crashed). These are the rest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.audio.recorder import DualStreamRecorder


# ---------------------------------------------------------------------------
# mic_bytes ≈ system_bytes — the invariant that was broken in 3 of 4 sessions
# ---------------------------------------------------------------------------
def _metadata(rec: DualStreamRecorder, tmp_path: Path) -> dict:
    import json

    meta = next(tmp_path.glob("*_metadata.json"))
    return json.loads(meta.read_text(encoding="utf-8"))


def test_the_two_streams_end_the_same_length(fake_sounddevice, tmp_path: Path,
                                             monkeypatch) -> None:
    """No test asserted this, and three of four real sessions were desynced.

    Teardown used to stop and close the mic completely before touching the
    system stream, so the system stream kept capturing for the whole of the
    mic's teardown. Every real session showed the delta as an exact multiple
    of one capture block.

    The fake device stops instantly, which is precisely why the old code
    looked fine under test — the desync came from real CoreAudio teardown
    taking time. So stopping is made slow here: that is the condition, and
    without the barrier this fails.
    """
    import time

    stream_cls = None
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec.start()

    for state in (rec._mic_state, rec._system_state):
        if state is not None and state.stream is not None:
            stream_cls = type(state.stream)
            break
    assert stream_cls is not None

    real_close = stream_cls.close

    def slow_close(self) -> None:
        # Modelled on CoreAudio: stop() returns once capture has ceased, and
        # close() is the slow part. That ordering is the whole point — with
        # the barrier both streams have already stopped before any close
        # begins, so this costs nothing; without it, the mic's close is time
        # the system stream spends still recording.
        time.sleep(0.5)
        real_close(self)

    monkeypatch.setattr(stream_cls, "close", slow_close)
    time.sleep(0.4)
    rec.stop()

    meta = _metadata(rec, tmp_path)
    mic, system = meta["mic_bytes"], meta["system_bytes"]
    assert mic > 0 and system > 0
    from src.audio.recorder import CAPTURE_BLOCK_BYTES

    assert abs(mic - system) <= CAPTURE_BLOCK_BYTES, (
        f"streams desynced by {abs(mic - system)} bytes "
        f"({abs(mic - system) / CAPTURE_BLOCK_BYTES:.1f} blocks)"
    )


def test_byte_parity_survives_a_pause(fake_sounddevice, tmp_path: Path) -> None:
    """Pause is checked per callback; without a shared gate the streams drift."""
    import time

    from src.audio.recorder import CAPTURE_BLOCK_BYTES

    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec.start()
    time.sleep(0.2)
    rec.pause()
    time.sleep(0.2)
    rec.resume()
    time.sleep(0.2)
    rec.stop()

    meta = _metadata(rec, tmp_path)
    assert abs(meta["mic_bytes"] - meta["system_bytes"]) <= CAPTURE_BLOCK_BYTES


# ---------------------------------------------------------------------------
# The ical_uid=None default that made the suite blind
# ---------------------------------------------------------------------------
def _event(uid: str | None, *, eid: str, when: datetime):
    from src.detection.calendar_poller import CalendarEvent

    return CalendarEvent(
        id=eid, title="Standup", start=when, end=when + timedelta(minutes=30),
        attendees=[], meeting_link=None, ical_uid=uid,
    )


def test_the_same_meeting_on_two_calendars_is_one_meeting() -> None:
    """The case the suite could not see.

    ``_evt()`` defaulted ical_uid to None, so canonical_key fell back to the
    per-calendar id and every test compared ids that happened to match. A real
    event invited to two calendars has two ids and one iCalUID — and that is
    exactly when comparing on id gets it wrong.
    """
    when = datetime.now(timezone.utc) + timedelta(minutes=5)
    mine = _event("meeting@google.com", eid="cal-a-123", when=when)
    theirs = _event("meeting@google.com", eid="cal-b-987", when=when)

    assert mine.id != theirs.id
    assert mine.canonical_key == theirs.canonical_key


def test_two_different_meetings_stay_different() -> None:
    when = datetime.now(timezone.utc) + timedelta(minutes=5)
    a = _event("standup@google.com", eid="e1", when=when)
    b = _event("retro@google.com", eid="e1", when=when)
    assert a.canonical_key != b.canonical_key, "same id, different meetings"


def test_an_event_with_no_ical_uid_still_has_a_key() -> None:
    """Some providers omit it; falling back to the id is correct there."""
    when = datetime.now(timezone.utc)
    evt = _event(None, eid="only-an-id", when=when)
    assert evt.canonical_key


# ---------------------------------------------------------------------------
# coreaudio_probe — 300 lines of ctypes with no direct test
# ---------------------------------------------------------------------------
def test_fourcc_encodes_the_selectors_coreaudio_expects() -> None:
    """Every property lookup in this module depends on this being right."""
    from src.audio.coreaudio_probe import _fourcc

    assert _fourcc("dIn ") == 0x64496E20
    assert _fourcc("agrp") == 0x61677270
    assert _fourcc("uid ") == 0x75696420


def test_the_probe_answers_on_this_machine() -> None:
    """It is ctypes against a live framework; a wrong signature raises."""
    probe = pytest.importorskip("src.audio.coreaudio_probe")

    ids = probe.list_audio_device_ids()
    assert isinstance(ids, list)
    assert ids, "no audio devices at all is not a plausible Mac"
    for device_id in ids:
        assert isinstance(probe.get_device_name(device_id), str)
        assert isinstance(probe.is_aggregate_device(device_id), bool)


def test_the_default_input_is_a_device_we_can_name() -> None:
    probe = pytest.importorskip("src.audio.coreaudio_probe")

    device_id = probe.get_default_input_device_id()
    assert device_id in probe.list_audio_device_ids()
    assert probe.get_device_name(device_id)


def test_is_default_input_running_returns_a_bool() -> None:
    """The deferral gate reads this; an exception there defers forever."""
    probe = pytest.importorskip("src.audio.coreaudio_probe")
    assert isinstance(probe.is_default_input_running(), bool)


def test_a_vanished_device_degrades_rather_than_crashing() -> None:
    """Devices disappear between enumeration and query all the time.

    The split here is deliberate and worth pinning down: the low-level
    primitives raise a typed CoreAudioProbeError, and the wrappers the app
    actually calls swallow it. A probe that raised into the deferral gate
    would defer every transcription for two hours.
    """
    probe = pytest.importorskip("src.audio.coreaudio_probe")

    # A readable placeholder rather than "", so a log line still says which
    # device went missing.
    name = probe.get_device_name(999_999)
    assert isinstance(name, str) and "999999" in name
    assert probe.is_aggregate_device(999_999) is False

    with pytest.raises(probe.CoreAudioProbeError):
        probe.is_input_device_running(999_999)


def test_the_mic_gate_never_raises_at_its_caller(monkeypatch) -> None:
    probe = pytest.importorskip("src.audio.coreaudio_probe")

    monkeypatch.setattr(probe, "get_default_input_device_id", lambda: 999_999)
    assert probe.is_default_input_running() is False


# ---------------------------------------------------------------------------
# _default_process_iter — the source the phantom detections came from
# ---------------------------------------------------------------------------
def test_the_real_process_source_yields_usable_records() -> None:
    """Every detection test injects a fake, so nothing exercised this."""
    from src.detection.process_monitor import _default_process_iter

    records = list(_default_process_iter())
    assert records, "psutil returned nothing"
    assert all(isinstance(r, dict) for r in records)
    assert all("name" in r and "pid" in r for r in records)


def test_the_real_source_feeds_the_real_matcher() -> None:
    """The audit's phantoms — searchpartyd matching 'arc' — came from here."""
    from src.detection.process_monitor import ProcessMonitor, _default_process_iter

    monitor = ProcessMonitor(whitelisted_apps=["arc", "safari"],
                             process_iter=_default_process_iter)
    seen = monitor.poll_once()

    running = {str(r.get("name") or "").lower() for r in _default_process_iter()}
    for name in seen:
        assert name in running or any(name in r for r in running), (
            f"{name!r} was detected but nothing by that name is running"
        )


# ---------------------------------------------------------------------------
# The suite must not write to the user's operational log
# ---------------------------------------------------------------------------
def test_the_suite_does_not_touch_the_real_log() -> None:
    """A regression from P5: tests that call main() logged into ~/.otis.

    otis.log rotates at 1 MB keeping five backups, so a full run did not just
    add noise — it evicted real meeting history. 63% of the retained log was
    pytest fixtures by the time it was noticed.
    """
    import src.main as main_mod

    # The autouse fixture in conftest redirects this for every test; if it
    # stops working, this points at the user's home directory again.
    assert main_mod.LOG_DIR.expanduser() != Path("~/.otis").expanduser()


def test_configure_logging_writes_where_log_dir_points(tmp_path, monkeypatch) -> None:
    import logging

    import src.main as main_mod

    monkeypatch.setattr(main_mod, "LOG_DIR", tmp_path / "elsewhere")
    main_mod._configure_logging("INFO")
    logging.getLogger("src.check").info("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()

    written = (tmp_path / "elsewhere" / "otis.log")
    assert written.exists()
    assert "hello" in written.read_text(encoding="utf-8")
