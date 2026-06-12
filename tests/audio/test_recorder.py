"""Tests for src/audio/recorder.py.

These tests use the ``fake_sounddevice`` fixture so no real audio hardware is
required. The fake stream feeds 10 ms blocks of silence so the writer thread
exercises the same code path it would in production.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import pytest

from src.audio.devices import DeviceManager
from src.audio.recorder import DualStreamRecorder, RecorderState


def _make_recorder(tmp_path: Path) -> DualStreamRecorder:
    return DualStreamRecorder(
        audio_dir=tmp_path,
        sample_rate=16000,
        channels=1,
        mic_device=None,                # default mic
        system_device="BlackHole",      # substring match
        device_manager=DeviceManager(),
        observe_sleep_wake=False,       # avoid pyobjc in tests
    )


def test_init_does_not_touch_audio(tmp_path: Path) -> None:
    """Constructing a recorder should not require sounddevice."""
    rec = DualStreamRecorder(
        audio_dir=tmp_path,
        device_manager=None,
        observe_sleep_wake=False,
    )
    assert rec.state == RecorderState.IDLE
    assert rec.session_id is None


def test_start_stop_creates_wavs_and_metadata(fake_sounddevice, tmp_path: Path) -> None:  # noqa: ARG001
    rec = _make_recorder(tmp_path)
    session = rec.start()
    assert rec.state == RecorderState.RECORDING
    assert session == rec.session_id

    # Let the fake stream feed a couple of blocks so the writer stamps anchors.
    time.sleep(0.05)

    metadata = rec.stop()
    assert rec.state == RecorderState.STOPPED

    mic_path = tmp_path / f"{session}_mic.wav"
    sys_path = tmp_path / f"{session}_system.wav"
    meta_path = tmp_path / f"{session}_metadata.json"
    assert mic_path.exists()
    assert sys_path.exists()
    assert meta_path.exists()

    # Verify WAV headers reflect 16 kHz mono 16-bit PCM.
    with wave.open(str(mic_path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2

    on_disk = json.loads(meta_path.read_text())
    assert metadata == on_disk


def test_metadata_has_required_keys_and_monotonic_anchors(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    rec = _make_recorder(tmp_path)
    rec.start()
    time.sleep(0.05)
    meta = rec.stop()

    # Required structure
    assert "session_id" in meta
    assert "mic_start_monotonic" in meta
    assert "system_start_monotonic" in meta
    assert "start_wall_clock" in meta
    assert "pauses" in meta
    assert isinstance(meta["pauses"], list)

    # Monotonic anchors must be set and look like time.monotonic() values.
    assert meta["mic_start_monotonic"] is not None
    assert meta["system_start_monotonic"] is not None
    assert meta["mic_start_monotonic"] > 0
    assert meta["system_start_monotonic"] > 0

    # Phase 4 cares about the delta between the two anchors. It should be a
    # plausible small offset (not zero unless by sheer luck — but always finite).
    delta = abs(meta["mic_start_monotonic"] - meta["system_start_monotonic"])
    assert delta < 5.0, f"unexpected anchor skew: {delta}s"

    # Wall clock must round-trip ISO-8601.
    from datetime import datetime
    datetime.fromisoformat(meta["start_wall_clock"])


def test_pause_and_resume_logged_in_metadata(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    rec = _make_recorder(tmp_path)
    rec.start()
    time.sleep(0.03)

    rec.pause()
    assert rec.state == RecorderState.PAUSED
    time.sleep(0.03)

    rec.resume()
    assert rec.state == RecorderState.RECORDING
    time.sleep(0.03)

    meta = rec.stop()
    assert len(meta["pauses"]) == 1
    p = meta["pauses"][0]
    assert "paused_at" in p
    assert "resumed_at" in p
    assert p["resumed_at"] > p["paused_at"]


def test_callback_stops_queueing_after_writer_death(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    """Once the writer thread dies, the audio callback must drop frames
    instead of growing the queue without bound."""
    rec = _make_recorder(tmp_path)
    rec.start()
    time.sleep(0.05)

    mic_state = rec._mic_state
    assert mic_state is not None
    # Simulate writer death the way the writer loop reports it.
    mic_state.error = RuntimeError("disk full")
    time.sleep(0.05)  # fake stream keeps firing the callback
    size_after_death = mic_state.queue.qsize()
    time.sleep(0.1)
    assert mic_state.queue.qsize() <= size_after_death + 1  # no unbounded growth
    rec.stop()


def test_on_device_error_setter_fires_on_writer_crash(
    fake_sounddevice, tmp_path: Path, monkeypatch  # noqa: ARG001
) -> None:
    """The late-bound callback (set after construction, before start) must
    be invoked when the writer loop crashes."""
    import wave as wave_module

    failures: list[tuple[str, Exception]] = []
    rec = _make_recorder(tmp_path)
    rec.on_device_error = lambda label, exc: failures.append((label, exc))
    assert rec.on_device_error is not None

    real_open = wave_module.open

    def exploding_open(path, mode="rb"):
        if "_mic.wav" in str(path):
            raise OSError("simulated disk failure")
        return real_open(path, mode)

    monkeypatch.setattr("src.audio.recorder.wave.open", exploding_open)
    rec.start()
    time.sleep(0.1)
    rec.stop()

    assert failures, "on_device_error was never invoked"
    assert failures[0][0] == "mic"
    assert "disk failure" in str(failures[0][1])


def test_wake_does_not_resume_user_initiated_pause(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    """User pauses, lid closes, Mac wakes — recording must stay paused."""
    rec = _make_recorder(tmp_path)
    rec.start()
    rec.pause()  # user
    rec.pause(source="sleep")  # no-op (already paused), source stays "user"
    rec.resume(source="wake")
    assert rec.state == RecorderState.PAUSED
    # The user can still resume explicitly.
    rec.resume()
    assert rec.state == RecorderState.RECORDING
    rec.stop()


def test_wake_resumes_sleep_initiated_pause(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    rec = _make_recorder(tmp_path)
    rec.start()
    rec.pause(source="sleep")
    assert rec.state == RecorderState.PAUSED
    rec.resume(source="wake")
    assert rec.state == RecorderState.RECORDING
    meta = rec.stop()
    assert len(meta["pauses"]) == 1


def test_user_resume_clears_pause_source(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    """After a full user pause/resume cycle, sleep/wake behaves normally again."""
    rec = _make_recorder(tmp_path)
    rec.start()
    rec.pause()
    rec.resume()
    rec.pause(source="sleep")
    rec.resume(source="wake")
    assert rec.state == RecorderState.RECORDING
    rec.stop()


def test_pause_while_idle_is_a_noop(tmp_path: Path) -> None:
    rec = DualStreamRecorder(
        audio_dir=tmp_path,
        device_manager=None,
        observe_sleep_wake=False,
    )
    rec.pause()  # must not raise
    rec.resume()
    assert rec.state == RecorderState.IDLE


def test_double_start_raises(fake_sounddevice, tmp_path: Path) -> None:  # noqa: ARG001
    rec = _make_recorder(tmp_path)
    rec.start()
    try:
        with pytest.raises(RuntimeError):
            rec.start()
    finally:
        rec.stop()


def test_stop_before_start_returns_gracefully(tmp_path: Path) -> None:
    """Calling stop() without start() should be a no-op, not crash."""
    rec = DualStreamRecorder(
        audio_dir=tmp_path,
        device_manager=None,
        observe_sleep_wake=False,
    )
    result = rec.stop()
    assert result == {}
    assert rec.state == RecorderState.IDLE


def test_records_mic_only_when_blackhole_missing(
    fake_sounddevice_no_blackhole, tmp_path: Path, caplog: pytest.LogCaptureFixture  # noqa: ARG001
) -> None:
    """Graceful degradation: warn but keep recording the mic, no system.wav."""
    rec = DualStreamRecorder(
        audio_dir=tmp_path,
        system_device="BlackHole 2ch",
        device_manager=DeviceManager(),
        observe_sleep_wake=False,
    )
    with caplog.at_level("WARNING"):
        session = rec.start()
    time.sleep(0.05)
    meta = rec.stop()

    mic_path = tmp_path / f"{session}_mic.wav"
    sys_path = tmp_path / f"{session}_system.wav"
    meta_path = tmp_path / f"{session}_metadata.json"

    assert mic_path.exists(), "mic wav must exist"
    assert not sys_path.exists(), "system wav must NOT exist when BlackHole missing"
    assert meta_path.exists()

    # Metadata still has the keys, but system fields are null.
    assert meta["mic_start_monotonic"] is not None
    assert meta["system_start_monotonic"] is None
    assert meta["system_device"] is None
    assert meta["system_wav"] is None

    # And we logged a clear warning.
    assert any("system audio" in r.message.lower() or "blackhole" in r.message.lower()
               for r in caplog.records)


def test_open_pause_window_closed_on_stop(fake_sounddevice, tmp_path: Path) -> None:  # noqa: ARG001
    """Stopping while paused should close the open pause window in metadata."""
    rec = _make_recorder(tmp_path)
    rec.start()
    time.sleep(0.03)
    rec.pause()
    time.sleep(0.03)
    meta = rec.stop()
    assert len(meta["pauses"]) == 1
    p = meta["pauses"][0]
    assert p["resumed_at"] >= p["paused_at"]


def test_streams_open_with_low_power_capture_settings(
    fake_sounddevice, tmp_path: Path  # noqa: ARG001
) -> None:
    """Large capture blocks are the difference between ~19% and ~6% CPU
    during a meeting — never silently regress to PortAudio's ~9 ms default."""
    from src.audio.recorder import CAPTURE_BLOCKSIZE_FRAMES, CAPTURE_LATENCY

    rec = _make_recorder(tmp_path)
    rec.start()
    try:
        for st in (rec._mic_state, rec._system_state):
            assert st is not None
            assert st.stream.blocksize == CAPTURE_BLOCKSIZE_FRAMES
            assert st.stream.latency == CAPTURE_LATENCY
    finally:
        rec.stop()
    assert CAPTURE_BLOCKSIZE_FRAMES >= 2048  # ≥128 ms at 16 kHz
