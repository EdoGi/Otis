"""Tests for src/audio/blackhole_check.py."""

from __future__ import annotations

from src.audio.blackhole_check import (
    SETUP_INSTRUCTIONS,
    format_setup_instructions,
    verify_blackhole_setup,
)
from src.audio.devices import DeviceManager


def test_status_ok_when_blackhole_and_multi_output_present(fake_sounddevice) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    assert status.installed
    assert status.multi_output_configured
    assert status.ok
    assert status.issues == []
    assert status.blackhole_device_name and "blackhole" in status.blackhole_device_name.lower()
    assert status.multi_output_device_name and "multi-output" in status.multi_output_device_name.lower()


def test_status_reports_missing_blackhole(fake_sounddevice_no_blackhole) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    assert not status.installed
    assert not status.ok
    assert any("BlackHole" in i for i in status.issues)


def test_status_reports_missing_multi_output(fake_sounddevice_no_multi_output) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    assert status.installed
    assert not status.multi_output_configured
    assert not status.ok
    assert any("Multi-Output" in i or "Aggregate" in i for i in status.issues)


def test_format_setup_instructions_when_ok(fake_sounddevice) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    msg = format_setup_instructions(status)
    assert "✓" in msg or "configured" in msg.lower()


def test_format_setup_instructions_when_missing(fake_sounddevice_no_blackhole) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    msg = format_setup_instructions(status)
    assert "brew install blackhole-2ch" in msg
    assert "Audio MIDI Setup" in SETUP_INSTRUCTIONS


def test_status_to_dict_round_trips(fake_sounddevice) -> None:  # noqa: ARG001
    status = verify_blackhole_setup(DeviceManager())
    d = status.to_dict()
    assert d["installed"] is True
    assert d["multi_output_configured"] is True
    assert d["issues"] == []
