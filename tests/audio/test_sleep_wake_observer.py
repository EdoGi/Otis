"""Regression: the sleep/wake observer burned a full CPU core for every recording.

``_SleepWakeObserver`` ran a dedicated thread looping on
``NSRunLoop.currentRunLoop().runUntilDate_(now + 0.5 s)``. A run loop with no
input sources returns from ``runUntilDate_`` immediately, so the loop never
slept: measured 2.95 s of CPU in 3.01 s of wall time (98% of a core) on the
user's 8 GB M1, from recorder start to recorder stop — every meeting, for the
whole meeting.

The thread bought nothing either. NSWorkspace posts its sleep/wake
notifications on the main thread and a notification is delivered on the thread
that posts it, so the handlers always ran on the main thread, never on
'otis-sleepwake'. The observer now just registers with NSWorkspace's center,
and the wake handler — which may reopen PortAudio streams — hands its work to
a short-lived thread so it cannot stall the menu bar.
"""

from __future__ import annotations

import resource
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("AppKit")

from src.audio.recorder import DualStreamRecorder, _SleepWakeObserver


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def test_a_started_observer_costs_no_cpu() -> None:
    observer = _SleepWakeObserver(on_sleep=lambda: None, on_wake=lambda: None)
    assert observer.start()
    try:
        before = _process_cpu_seconds()
        time.sleep(1.0)
        used = _process_cpu_seconds() - before
    finally:
        observer.stop()
    # The spinning loop used ~0.98 s of CPU per wall second.
    assert used < 0.3, f"observer burned {used:.2f}s of CPU in 1s idle"


def test_the_observer_starts_no_thread() -> None:
    before = {t.name for t in threading.enumerate()}
    observer = _SleepWakeObserver(on_sleep=lambda: None, on_wake=lambda: None)
    assert observer.start()
    try:
        after = {t.name for t in threading.enumerate()}
    finally:
        observer.stop()
    assert "otis-sleepwake" not in after - before


def test_workspace_notifications_reach_the_handlers() -> None:
    from AppKit import NSWorkspace

    fired: list[str] = []
    observer = _SleepWakeObserver(
        on_sleep=lambda: fired.append("sleep"),
        on_wake=lambda: fired.append("wake"),
    )
    assert observer.start()
    center = NSWorkspace.sharedWorkspace().notificationCenter()
    try:
        center.postNotificationName_object_("NSWorkspaceWillSleepNotification", None)
        center.postNotificationName_object_("NSWorkspaceDidWakeNotification", None)
    finally:
        observer.stop()
    assert fired == ["sleep", "wake"]


def test_a_stopped_observer_no_longer_fires() -> None:
    from AppKit import NSWorkspace

    fired: list[str] = []
    observer = _SleepWakeObserver(on_sleep=lambda: fired.append("sleep"), on_wake=lambda: None)
    assert observer.start()
    observer.stop()
    NSWorkspace.sharedWorkspace().notificationCenter().postNotificationName_object_(
        "NSWorkspaceWillSleepNotification", None
    )
    assert fired == []


def test_wake_work_runs_off_the_posting_thread(tmp_path: Path, monkeypatch) -> None:
    """The wake path can reopen PortAudio streams; it must not block the UI thread."""
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    ran_on: list[threading.Thread] = []
    done = threading.Event()

    def fake_resume(*, source: str = "user") -> None:
        ran_on.append(threading.current_thread())
        done.set()

    monkeypatch.setattr(rec, "resume", fake_resume)
    rec._handle_system_wake()
    assert done.wait(5.0)
    assert ran_on and ran_on[0] is not threading.current_thread()


def test_sleep_pauses_synchronously(tmp_path: Path, monkeypatch) -> None:
    """Sleep stays synchronous: macOS gives sleep handlers little time, and a
    pause that ran after the matching wake would leave the recording paused."""
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    ran_on: list[threading.Thread] = []
    monkeypatch.setattr(rec, "pause", lambda *, source="user": ran_on.append(threading.current_thread()))
    rec._handle_system_sleep()
    assert ran_on == [threading.current_thread()]


def test_a_wake_resume_waits_for_a_stop_in_progress(tmp_path: Path, monkeypatch) -> None:
    """Review finding: moving the wake resume off the main thread let a Stop
    click overlap the stream rebuild. stop() skipped the stream the revive had
    set to None, finished, and the revive then started a new InputStream that
    nothing owned — mic left open, its queue filling with no writer. The
    resume now holds the same lock stop() does."""
    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    ran = threading.Event()
    monkeypatch.setattr(rec, "resume", lambda *, source="user": ran.set())

    with rec._stop_lock:  # a stop() in progress
        rec._handle_system_wake()
        assert not ran.wait(0.3), "resume ran while stop() held its lock"
    assert ran.wait(5.0)


# ---------------------------------------------------------------------------
# The capture watchdog's rebuild path (review round 2, deferred at the time)
# ---------------------------------------------------------------------------
class _FakeStream:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        self.calls.append("close")


def _stream_state(stream):
    from types import SimpleNamespace

    return SimpleNamespace(label="mic", stream=stream, last_progress_monotonic=0.0)


def test_a_rebuild_overlapping_stop_closes_the_stream_it_opened(tmp_path: Path, monkeypatch) -> None:
    """The watchdog's rebuild set st.stream = None and then opened a new stream
    outside any lock; a stop() landing in that window skipped the stream,
    finished, and the new one kept capturing with no owner. The rebuild now
    opens into a local and attaches it only if the recording is still live."""
    from src.audio.recorder import RecorderState

    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec._state = RecorderState.RECORDING
    old, new = _FakeStream(), _FakeStream()
    st = _stream_state(old)

    def open_while_stop_lands(_st, sd=None):
        rec._state = RecorderState.STOPPING  # stop() got the state lock meanwhile
        return new

    monkeypatch.setattr(rec, "_open_input_stream", open_while_stop_lands)
    assert rec._revive_stream(st) is False
    assert new.calls == ["stop", "close"], "the orphan must be shut down"
    assert st.stream is None
    assert old.calls == ["stop", "close"]


def test_a_rebuild_during_a_live_recording_attaches_the_new_stream(tmp_path: Path, monkeypatch) -> None:
    from src.audio.recorder import RecorderState

    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec._state = RecorderState.RECORDING
    old, new = _FakeStream(), _FakeStream()
    st = _stream_state(old)
    monkeypatch.setattr(rec, "_open_input_stream", lambda _st, sd=None: new)
    assert rec._revive_stream(st) is True
    assert st.stream is new and new.calls == []
    assert old.calls == ["stop", "close"]


def test_no_rebuild_once_the_recording_has_stopped(tmp_path: Path, monkeypatch) -> None:
    from src.audio.recorder import RecorderState

    rec = DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)
    rec._state = RecorderState.STOPPED
    opened: list[bool] = []
    monkeypatch.setattr(rec, "_open_input_stream", lambda _st, sd=None: opened.append(True))
    st = _stream_state(None)
    assert rec._revive_stream(st) is False
    assert opened == []
