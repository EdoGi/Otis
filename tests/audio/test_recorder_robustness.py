"""Regression cover for audit findings B7 and B8 — recording failures.

Four ways the recorder shipped a broken session and said nothing:

* teardown ran stop()+close() fully for the mic before touching the system
  stream, so the two files ended different lengths — every real session on this
  machine showed the delta as an exact multiple of one capture block, and three
  of four were desynced;
* the capture queue was unbounded, so a merely *slow* writer (Time Machine,
  Spotlight, a network volume) grew it at 32 KB/s per stream for the whole
  meeting. The drop-on-full handler existed but was documented as unreachable;
* a dead writer left the recorder in RECORDING — the UI kept showing a live
  session while every subsequent frame was silently discarded;
* BlackHole failing to open tore down the mic recording too, defeating the
  graceful-degradation path built for exactly that case.
"""

from __future__ import annotations

import queue
import threading
import wave
from pathlib import Path

import pytest

from src.audio.recorder import (
    CAPTURE_BLOCK_BYTES,
    DualStreamRecorder,
    RecorderState,
    _looks_like_disk_full,
)
from src.health import reset_health


@pytest.fixture(autouse=True)
def _clean_health():
    reset_health()
    yield
    reset_health()


@pytest.fixture()
def recorder(tmp_path: Path, fake_sounddevice) -> DualStreamRecorder:  # noqa: ARG001
    return DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)


# ---------------------------------------------------------------------------
# B8 — cross-stream barrier
# ---------------------------------------------------------------------------
def test_both_streams_stop_before_either_closes(recorder) -> None:
    """The barrier: no stream may still be capturing while another is being
    closed, which is the window the two files drifted in."""
    order: list[str] = []
    recorder.start()

    for st in (recorder._mic_state, recorder._system_state):
        if st is None or st.stream is None:
            continue
        label = st.label
        real_stop, real_close = st.stream.stop, st.stream.close
        st.stream.stop = lambda _l=label, _f=real_stop: (order.append(f"stop:{_l}"), _f())
        st.stream.close = lambda _l=label, _f=real_close: (order.append(f"close:{_l}"), _f())

    recorder.stop()

    # sounddevice's close() calls stop() internally, so look at the first
    # occurrence of each event per stream: every stream must have been halted
    # before the first close begins.
    def first(event: str) -> int:
        return order.index(event)

    labels = {e.split(":", 1)[1] for e in order}
    last_stop = max(first(f"stop:{label}") for label in labels)
    first_close = min(first(f"close:{label}") for label in labels)
    assert last_stop < first_close, (
        f"a stream was closed while another was still capturing: {order}"
    )


def test_desync_beyond_one_block_is_reported(recorder, caplog) -> None:
    recorder.start()
    recorder._mic_state.bytes_written = 1_000_000
    recorder._system_state.bytes_written = 1_000_000 + CAPTURE_BLOCK_BYTES * 3
    with caplog.at_level("WARNING"):
        recorder.stop()
    assert any("apart" in r.message for r in caplog.records), (
        "a multi-block drift must not ship silently"
    )


def test_sub_block_difference_is_not_flagged(recorder, caplog) -> None:
    recorder.start()
    recorder._mic_state.bytes_written = 1_000_000
    recorder._system_state.bytes_written = 1_000_000 + 128
    with caplog.at_level("WARNING"):
        recorder.stop()
    assert not any("apart" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# B7 — bounded capture queue
# ---------------------------------------------------------------------------
def test_capture_queue_is_bounded(recorder) -> None:
    recorder.start()
    for st in (recorder._mic_state, recorder._system_state):
        if st is None:
            continue
        assert st.queue.maxsize > 0, "an unbounded queue grows for the whole meeting"
    recorder.stop()


def test_queue_size_covers_a_useful_buffer(recorder) -> None:
    """Big enough to ride out a hiccup, small enough to bound memory."""
    size = recorder._queue_maxsize()
    assert size >= 8
    assert size * CAPTURE_BLOCK_BYTES < 8 * 1024 * 1024, "cap the buffer well under 8 MB"


def test_full_queue_drops_instead_of_growing(recorder) -> None:
    recorder.start()
    st = recorder._mic_state
    # Fill it past capacity the way a stalled writer would.
    try:
        while True:
            st.queue.put_nowait(b"\x00" * CAPTURE_BLOCK_BYTES)
    except queue.Full:
        pass
    assert st.queue.full()
    recorder.stop()


# ---------------------------------------------------------------------------
# B7 — a dead writer must leave RECORDING
# ---------------------------------------------------------------------------
def test_disk_full_takes_the_recorder_out_of_recording(recorder) -> None:
    recorder.start()
    assert recorder.state is RecorderState.RECORDING

    boom = OSError(28, "No space left on device")
    for st in (recorder._mic_state, recorder._system_state):
        if st is not None:
            st.error = boom
    recorder._note_writer_failure(recorder._mic_state, boom)

    assert recorder.state is RecorderState.ERROR, (
        "the UI must stop claiming a recording is in progress"
    )


def test_one_dead_writer_does_not_stop_the_other(recorder) -> None:
    recorder.start()
    boom = OSError(28, "No space left on device")
    recorder._system_state.error = boom
    recorder._note_writer_failure(recorder._system_state, boom)

    assert recorder.state is RecorderState.RECORDING, (
        "the mic is still capturing; don't throw the session away"
    )


def test_writer_failure_reaches_the_health_panel(recorder) -> None:
    from src.health import SYSTEM_AUDIO, HealthState, get_health

    recorder.start()
    boom = OSError(28, "No space left on device")
    recorder._system_state.error = boom
    recorder._note_writer_failure(recorder._system_state, boom)

    rep = get_health().get(SYSTEM_AUDIO)
    assert rep is not None and rep.state is HealthState.FAIL
    assert "disk is full" in rep.detail


@pytest.mark.parametrize(
    "exc,expected",
    [
        (OSError(28, "No space left on device"), True),
        (OSError("[Errno 28] No space left on device"), True),
        (RuntimeError("ENOSPC while flushing"), True),
        (OSError(13, "Permission denied"), False),
        (RuntimeError("device disconnected"), False),
    ],
)
def test_disk_full_detection(exc: BaseException, expected: bool) -> None:
    assert _looks_like_disk_full(exc) is expected


# ---------------------------------------------------------------------------
# B7 — BlackHole failure must not kill the mic
# ---------------------------------------------------------------------------
def test_system_stream_failure_degrades_to_mic_only(tmp_path, fake_sounddevice) -> None:  # noqa: ARG001
    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)
    real_open = rec._open_stream

    def open_but_fail_system(st):
        if st.label == "system":
            raise RuntimeError("PaError: device unavailable")
        return real_open(st)

    rec._open_stream = open_but_fail_system  # type: ignore[method-assign]

    session_id = rec.start()

    assert session_id, "the mic recording must survive a broken BlackHole"
    assert rec.state is RecorderState.RECORDING
    assert rec._system_state is None
    metadata = rec.stop()
    assert metadata["system_wav"] is None
    assert Path(metadata["mic_wav"]).name.endswith("_mic.wav")


def test_system_stream_failure_notifies(tmp_path, fake_sounddevice) -> None:  # noqa: ARG001
    errors: list[tuple[str, Exception]] = []
    rec = DualStreamRecorder(
        audio_dir=tmp_path / "audio",
        observe_sleep_wake=False,
        on_device_error=lambda label, exc: errors.append((label, exc)),
    )
    real_open = rec._open_stream

    def open_but_fail_system(st):
        if st.label == "system":
            raise RuntimeError("PaError: device unavailable")
        return real_open(st)

    rec._open_stream = open_but_fail_system  # type: ignore[method-assign]
    rec.start()
    rec.stop()

    assert errors and errors[0][0] == "system"


def test_mic_failure_still_aborts_the_session(tmp_path, fake_sounddevice) -> None:  # noqa: ARG001
    """Mic-only is a usable recording; system-only is not what was asked for."""
    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)

    def always_fail(st):
        raise RuntimeError("PaError: mic unavailable")

    rec._open_stream = always_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        rec.start()
    assert rec.state is RecorderState.ERROR


def test_writer_thread_records_a_real_wav(recorder) -> None:
    """Sanity: the barrier changes didn't break the normal path."""
    recorder.start()
    st = recorder._mic_state
    st.queue.put_nowait(b"\x00\x01" * 800)
    threading.Event().wait(0.05)
    meta = recorder.stop()
    with wave.open(str(recorder._audio_dir / Path(meta["mic_wav"]).name)) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
