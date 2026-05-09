"""Tests for src/audio/devices.py."""

from __future__ import annotations

import pytest

from src.audio.devices import DeviceManager, DeviceNotFoundError


def test_lists_all_devices(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    names = [d.name for d in dm.devices]
    assert "MacBook Pro Microphone" in names
    assert "BlackHole 2ch" in names
    assert "MacBook Pro Speakers" in names


def test_input_devices_filter(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    input_names = [d.name for d in dm.input_devices()]
    assert "MacBook Pro Microphone" in input_names
    assert "BlackHole 2ch" in input_names
    # Output-only devices are excluded
    assert "MacBook Pro Speakers" not in input_names


def test_default_input_returns_mic(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    default = dm.default_input()
    assert default is not None
    assert default.name == "MacBook Pro Microphone"


def test_get_default_mic_alias(fake_sounddevice) -> None:  # noqa: ARG001
    """The review references DeviceManager.get_default_mic() — must be an alias."""
    dm = DeviceManager()
    assert dm.get_default_mic() is dm.default_input()


def test_list_devices_returns_full_list(fake_sounddevice) -> None:  # noqa: ARG001
    """The review references DeviceManager.list_devices()."""
    dm = DeviceManager()
    listed = dm.list_devices()
    assert len(listed) == len(dm.devices)
    assert {d.name for d in listed} == {d.name for d in dm.devices}


def test_default_input_falls_back_when_coreaudio_returns_minus_one(
    monkeypatch,
) -> None:
    """If CoreAudio reports no default input, use the first input device."""
    import sys

    from tests.conftest import make_fake_sd_module

    mod = make_fake_sd_module(default_input=-1)
    monkeypatch.setitem(sys.modules, "sounddevice", mod)

    dm = DeviceManager()
    fallback = dm.default_input()
    assert fallback is not None
    assert fallback.name == "MacBook Pro Microphone"


def test_detects_blackhole_when_present(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    assert dm.is_blackhole_installed()
    bh = dm.find_blackhole()
    assert bh is not None
    assert "blackhole" in bh.name.lower()


def test_detects_missing_blackhole(fake_sounddevice_no_blackhole) -> None:  # noqa: ARG001
    dm = DeviceManager()
    assert not dm.is_blackhole_installed()
    assert dm.find_blackhole() is None


def test_finds_multi_output_device(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    multi = dm.find_multi_output_device()
    assert multi is not None
    assert "multi-output" in multi.name.lower()


def test_no_multi_output_when_not_configured(fake_sounddevice_no_multi_output) -> None:  # noqa: ARG001
    dm = DeviceManager()
    assert dm.find_multi_output_device() is None


def test_finds_user_renamed_multi_output_via_coreaudio(monkeypatch) -> None:
    """A user-renamed multi-output (e.g. 'Otis BT') must be detected."""
    import sys
    from tests.conftest import make_fake_sd_module

    sd = make_fake_sd_module(
        devices=[
            {"name": "MacBook Pro Microphone", "max_input_channels": 1,
             "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "BlackHole 2ch", "max_input_channels": 2,
             "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "MacBook Pro Speakers", "max_input_channels": 0,
             "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0},
            # Custom-named multi-output — would slip past the substring heuristic.
            {"name": "Otis BT", "max_input_channels": 0,
             "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)

    # Pretend CoreAudio reports "Otis BT" as an aggregate device.
    import src.audio.coreaudio_probe as probe
    monkeypatch.setattr(probe, "list_aggregate_devices", lambda: [(99, "Otis BT")])

    dm = DeviceManager()
    multi = dm.find_multi_output_device()
    assert multi is not None
    assert multi.name == "Otis BT"


def test_falls_back_to_name_heuristic_when_coreaudio_fails(monkeypatch, fake_sounddevice) -> None:  # noqa: ARG001
    """If CoreAudio enumeration fails, the substring heuristic still catches the default name."""
    import src.audio.coreaudio_probe as probe

    def boom():
        raise RuntimeError("coreaudio offline")

    monkeypatch.setattr(probe, "list_aggregate_devices", boom)

    dm = DeviceManager()
    multi = dm.find_multi_output_device()
    assert multi is not None
    # The fake sd fixture's default-named "Multi-Output" device is still found.
    assert "multi-output" in multi.name.lower()


def test_require_raises_clear_error_when_missing(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    with pytest.raises(DeviceNotFoundError) as excinfo:
        dm.require("Nonexistent Device")
    assert "Nonexistent" in str(excinfo.value)


def test_find_by_name_is_case_insensitive(fake_sounddevice) -> None:  # noqa: ARG001
    dm = DeviceManager()
    assert dm.find_by_name("blackhole") is not None
    assert dm.find_by_name("BLACKHOLE") is not None
    assert dm.find_by_name("BlackHole") is not None
