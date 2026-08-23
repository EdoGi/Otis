"""Cover for audit C5's recorder half — stop() held the state lock throughout.

``stop()`` ran the whole teardown inside ``_state_lock``: two writer joins of
up to 5 s each, plus a metadata write. For that whole window nothing else
could read or change recorder state.

Worse than a stall, it could eat an error. ``_note_writer_failure`` runs *on*
the writer thread and takes the same lock, so a writer dying as we stopped —
the disk-full case it exists for — blocked against the very thread joining it.
The join burned its 5 s timeout, logged "writer thread did not exit", and the
failure was never reported.

Teardown now runs outside the lock, with the recorder parked in STOPPING so
pause/resume and the stall watchdog stand down.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from src.audio.recorder import DualStreamRecorder, RecorderState


def _recorder(tmp_path: Path) -> DualStreamRecorder:
    return DualStreamRecorder(audio_dir=tmp_path, observe_sleep_wake=False)


def test_state_is_readable_while_the_teardown_runs(
    fake_sounddevice, tmp_path: Path, monkeypatch
) -> None:
    """The UI polls state; a slow flush must not make that block."""
    rec = _recorder(tmp_path)
    rec.start()

    in_teardown = threading.Event()
    release = threading.Event()
    real = rec._teardown_streams

    def slow_teardown() -> None:
        in_teardown.set()
        assert release.wait(5.0)
        real()

    monkeypatch.setattr(rec, "_teardown_streams", slow_teardown)
    stopper = threading.Thread(target=rec.stop, daemon=True)
    stopper.start()
    assert in_teardown.wait(5.0)

    try:
        # The whole point: these must not block behind the teardown.
        observed: list[RecorderState] = []
        reader = threading.Thread(target=lambda: observed.append(rec.state), daemon=True)
        reader.start()
        reader.join(2.0)
        assert not reader.is_alive(), "state read blocked on the teardown"
        assert observed == [RecorderState.STOPPING]

        acquired = rec._state_lock.acquire(timeout=2.0)
        assert acquired, "the state lock was held across the teardown"
        rec._state_lock.release()
    finally:
        release.set()
        stopper.join(10.0)

    assert rec.state is RecorderState.STOPPED


def test_pause_during_teardown_is_ignored_not_blocked(
    fake_sounddevice, tmp_path: Path, monkeypatch
) -> None:
    rec = _recorder(tmp_path)
    rec.start()
    in_teardown = threading.Event()
    release = threading.Event()
    real = rec._teardown_streams

    def slow_teardown() -> None:
        in_teardown.set()
        assert release.wait(5.0)
        real()

    monkeypatch.setattr(rec, "_teardown_streams", slow_teardown)
    stopper = threading.Thread(target=rec.stop, daemon=True)
    stopper.start()
    assert in_teardown.wait(5.0)

    try:
        pauser = threading.Thread(target=rec.pause, daemon=True)
        pauser.start()
        pauser.join(2.0)
        assert not pauser.is_alive(), "pause() blocked on the teardown"
        assert rec.state is RecorderState.STOPPING, "pause must not take effect"
    finally:
        release.set()
        stopper.join(10.0)


def test_a_writer_failing_during_stop_can_report_it(
    fake_sounddevice, tmp_path: Path, monkeypatch
) -> None:
    """The deadlock this change removes, reproduced through the real lock.

    The writer thread's failure path needs the state lock. If stop() holds it
    while joining that thread, the report never lands.
    """
    rec = _recorder(tmp_path)
    rec.start()
    reported = threading.Event()
    real = rec._teardown_streams

    def teardown_with_a_dying_writer() -> None:
        def writer_side() -> None:
            # Exactly what _note_writer_failure does: take the state lock from
            # a thread stop() is about to join.
            with rec._state_lock:
                reported.set()

        t = threading.Thread(target=writer_side, daemon=True)
        t.start()
        t.join(3.0)
        real()

    monkeypatch.setattr(rec, "_teardown_streams", teardown_with_a_dying_writer)
    rec.stop()

    assert reported.is_set(), "the writer could not report while stop() ran"


def test_stopping_is_a_distinct_state(
    fake_sounddevice, tmp_path: Path, monkeypatch
) -> None:
    """The stall watchdog stands down on it; STOPPED would be a lie mid-flush."""
    rec = _recorder(tmp_path)
    rec.start()
    seen: list[RecorderState] = []
    real = rec._teardown_streams

    def note_state() -> None:
        seen.append(rec.state)
        real()

    monkeypatch.setattr(rec, "_teardown_streams", note_state)
    rec.stop()

    assert seen == [RecorderState.STOPPING]
    assert rec.state is RecorderState.STOPPED


def test_concurrent_stops_tear_down_once(
    fake_sounddevice, tmp_path: Path, monkeypatch
) -> None:
    rec = _recorder(tmp_path)
    rec.start()
    calls: list[int] = []
    real = rec._teardown_streams

    def counting() -> None:
        calls.append(1)
        time.sleep(0.05)
        real()

    monkeypatch.setattr(rec, "_teardown_streams", counting)
    results: list[dict] = []
    threads = [
        threading.Thread(target=lambda: results.append(rec.stop()), daemon=True)
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert calls == [1], f"teardown ran {len(calls)} times"
    assert len(results) == 4
    assert rec.state is RecorderState.STOPPED


def test_the_watchdog_stands_down_once_stopping(
    fake_sounddevice, tmp_path: Path
) -> None:
    """It must not try to revive a stream that is being torn down."""
    rec = _recorder(tmp_path)
    rec.start()
    with rec._state_lock:
        rec._state = RecorderState.STOPPING
    rec._check_streams_alive()  # must be a no-op, and must not raise
    with rec._state_lock:
        rec._state = RecorderState.RECORDING
    rec.stop()
