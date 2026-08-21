"""P2 item 21 — device capability checks and honest error classification.

Two problems. The recorder passed samplerate=16000, channels=1 unconditionally
and never consulted the capability table it already collects. And when an open
failed, it decided what went wrong by sniffing for the substring "denied" —
so an unrelated PaError was reported as a microphone-permission problem,
sending the user to System Settings to fix something that wasn't broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.devices import AudioDevice
from src.audio.recorder import (
    DeviceOpenError,
    DualStreamRecorder,
    PermissionDeniedError,
    _is_permission_error,
    _warn_if_device_cannot_do,
)


def _device(**kw) -> AudioDevice:
    base = dict(index=0, name="Test Mic", max_input_channels=1,
                max_output_channels=0, default_sample_rate=48000.0, hostapi=0)
    base.update(kw)
    return AudioDevice(**base)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    ["Permission denied", "Operation not permitted",
     "Microphone access was refused", "access denied by the system"],
)
def test_real_permission_errors_are_recognised(message: str) -> None:
    assert _is_permission_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Invalid sample rate",
        "Device unavailable",
        # The bare word "denied" used to be enough to claim a permissions
        # problem, whatever it was actually about.
        "connection denied by peer",
        "The request was denied because the device is in use",
        "PaErrorCode -9997: Invalid number of channels",
    ],
)
def test_unrelated_failures_are_not_called_permission_problems(message: str) -> None:
    assert _is_permission_error(RuntimeError(message)) is False


def test_an_open_failure_names_the_device_and_the_settings(
    tmp_path: Path, fake_sounddevice, monkeypatch
) -> None:
    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("PaErrorCode -9997")

    monkeypatch.setattr(fake_sounddevice, "InputStream", Boom)

    with pytest.raises(DeviceOpenError) as excinfo:
        rec.start()
    text = str(excinfo.value)
    assert "16000" in text and "1ch" in text, f"unhelpful message: {text}"
    assert "PaErrorCode -9997" in text, "the underlying cause must survive"


def test_a_permission_failure_still_says_so(
    tmp_path: Path, fake_sounddevice, monkeypatch
) -> None:
    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)

    class Denied:
        def __init__(self, *a, **k):
            raise RuntimeError("Permission denied")

    monkeypatch.setattr(fake_sounddevice, "InputStream", Denied)

    with pytest.raises(PermissionDeniedError) as excinfo:
        rec.start()
    assert "System Settings" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Capability warnings
# ---------------------------------------------------------------------------
def test_requesting_more_channels_than_the_device_has_warns(caplog) -> None:
    with caplog.at_level("WARNING"):
        _warn_if_device_cannot_do(_device(max_input_channels=1), 16000, 2)
    assert any("input channel" in r.message for r in caplog.records)


def test_a_capable_device_is_silent(caplog) -> None:
    with caplog.at_level("WARNING"):
        _warn_if_device_cannot_do(_device(max_input_channels=2), 16000, 1)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_resampling_device_is_not_a_warning(caplog) -> None:
    """CoreAudio resamples 48k -> 16k transparently; that is normal, not a
    problem, and must not cry wolf on every single recording."""
    with caplog.at_level("WARNING"):
        _warn_if_device_cannot_do(_device(default_sample_rate=48000.0), 16000, 1)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_missing_capability_table_is_tolerated(caplog) -> None:
    """Never let an advisory check stop a recording."""
    class Bare:
        name = "Mystery Device"

    with caplog.at_level("WARNING"):
        _warn_if_device_cannot_do(Bare(), 16000, 1)  # must not raise
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_the_check_never_blocks_a_recording(tmp_path, fake_sounddevice, caplog) -> None:
    """BlackHole 2ch declares two input channels and we open it as mono. That
    is fine, and the check must not turn it into a failure."""
    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)
    with caplog.at_level("WARNING"):
        sid = rec.start()
    assert sid
    rec.stop()
