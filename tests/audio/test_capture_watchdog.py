"""P2 item 20 — a dead capture device must not be silent about it.

Before this, `on_device_error` fired from exactly one place: the writer
thread's exception handler. That means *disk* failures only. A device that was
unplugged, or that re-enumerated while the Mac slept, simply stopped calling
the callback — no exception, no error, `stop()` reporting success, and the rest
of the meeting written as nothing at all.

The signal is deliberately "no bytes", not "no level": a quiet room still
delivers blocks of zeros, and the level monitor was taught to ignore gaps so a
pause wouldn't look like a dead stream.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.audio.recorder import (
    STREAM_STALL_SECONDS,
    DualStreamRecorder,
    RecorderState,
)
from src.health import SYSTEM_AUDIO, HealthState, get_health, reset_health


@pytest.fixture(autouse=True)
def _clean_health():
    reset_health()
    yield
    reset_health()


@pytest.fixture()
def recorder(tmp_path: Path, fake_sounddevice):
    errors: list[tuple[str, Exception]] = []
    rec = DualStreamRecorder(
        audio_dir=tmp_path / "audio",
        observe_sleep_wake=False,
        on_device_error=lambda label, exc: errors.append((label, exc)),
    )
    rec._errors = errors  # type: ignore[attr-defined]
    return rec


def _freeze_stream(st, *, seconds_ago: float) -> None:
    """Pretend this stream last delivered audio `seconds_ago`."""
    st.last_bytes_seen = st.bytes_written
    st.last_progress_monotonic = time.monotonic() - seconds_ago


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_a_stalled_stream_is_reported(recorder) -> None:
    recorder.start()
    st = recorder._system_state
    _freeze_stream(st, seconds_ago=STREAM_STALL_SECONDS + 1)

    recorder._check_streams_alive()

    assert st.stalled is True
    assert any(label == "system" for label, _ in recorder._errors), (
        "a dead device must reach on_device_error, not only disk failures"
    )
    rep = get_health().get(SYSTEM_AUDIO)
    assert rep is not None and rep.state is HealthState.FAIL
    assert "capture stopped" in rep.detail
    recorder.stop()


def test_it_reports_once_not_every_tick(recorder) -> None:
    recorder.start()
    st = recorder._system_state
    _freeze_stream(st, seconds_ago=STREAM_STALL_SECONDS + 1)

    for _ in range(10):
        recorder._check_streams_alive()

    assert len([1 for label, _ in recorder._errors if label == "system"]) == 1
    recorder.stop()


def test_a_brief_gap_is_not_a_stall(recorder) -> None:
    recorder.start()
    _freeze_stream(recorder._system_state, seconds_ago=STREAM_STALL_SECONDS / 3)
    recorder._check_streams_alive()
    assert recorder._system_state.stalled is False
    assert recorder._errors == []
    recorder.stop()


def test_recovery_clears_the_stall(recorder) -> None:
    recorder.start()
    st = recorder._system_state
    _freeze_stream(st, seconds_ago=STREAM_STALL_SECONDS + 1)
    recorder._check_streams_alive()
    assert st.stalled is True

    st.bytes_written += 4096          # audio started flowing again
    recorder._check_streams_alive()

    assert st.stalled is False
    assert get_health().get(SYSTEM_AUDIO).state is HealthState.OK
    recorder.stop()


# ---------------------------------------------------------------------------
# The watchdog must not fire when no audio is expected
# ---------------------------------------------------------------------------
def test_a_paused_recording_is_not_a_stall(recorder) -> None:
    """The callback drops frames while paused, so bytes stop by design. A
    25-minute pause must not look like a dead device."""
    recorder.start()
    recorder.pause()
    _freeze_stream(recorder._system_state, seconds_ago=26 * 60)

    recorder._check_streams_alive()

    assert recorder._system_state.stalled is False
    assert recorder._errors == []
    recorder.stop()


def test_resuming_from_sleep_does_not_backdate_the_stall(recorder) -> None:
    recorder.start()
    recorder.pause(source="sleep")
    _freeze_stream(recorder._system_state, seconds_ago=3 * 3600)

    recorder.resume(source="wake")
    recorder._check_streams_alive()

    assert recorder._system_state.stalled is False, (
        "the sleep itself must not be counted against the stream"
    )
    recorder.stop()


def test_a_writer_failure_is_not_double_reported(recorder) -> None:
    recorder.start()
    st = recorder._system_state
    st.error = OSError(28, "No space left on device")
    _freeze_stream(st, seconds_ago=STREAM_STALL_SECONDS + 1)

    recorder._check_streams_alive()

    assert st.stalled is False, "the writer already reported this one"
    recorder.stop()


# ---------------------------------------------------------------------------
# Revival
# ---------------------------------------------------------------------------
def test_a_stalled_stream_is_rebuilt(recorder) -> None:
    recorder.start()
    st = recorder._system_state
    original = st.stream
    _freeze_stream(st, seconds_ago=STREAM_STALL_SECONDS + 1)

    recorder._check_streams_alive()

    assert st.stream is not None
    assert st.stream is not original, "the dead stream object should be replaced"
    assert recorder.state is RecorderState.RECORDING
    recorder.stop()


def test_revival_keeps_the_same_wav_and_writer(recorder) -> None:
    """The gap is missing audio; the rest of the meeting must still land in the
    same file rather than being lost."""
    recorder.start()
    st = recorder._system_state
    path, writer, q = st.wav_path, st.writer_thread, st.queue

    assert recorder._revive_stream(st) is True

    assert st.wav_path == path
    assert st.writer_thread is writer
    assert st.queue is q
    recorder.stop()


def test_revival_failure_is_survivable(recorder, monkeypatch) -> None:
    recorder.start()
    st = recorder._mic_state

    def boom(_st, sd=None):
        raise RuntimeError("device really is gone")

    monkeypatch.setattr(recorder, "_open_input_stream", boom)
    assert recorder._revive_stream(st) is False
    recorder.stop()  # must not raise
