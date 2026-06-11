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
