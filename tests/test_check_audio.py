"""Regression: `otis check-audio` said ✓ while system audio recorded silence.

check-audio ran only the structural check (does BlackHole exist, does any
aggregate exist) and exited 0 on it. It never asked whether the Mac's current
output feeds BlackHole — the question that cost two months of one-sided
transcripts. Measured on 2026-10-25: check-audio exit 0 with "✓" while
check_output_routing() said NOT_ROUTED (output: 'Altoparlanti MacBook Pro'),
the same verdict the menu bar logged at most launches.
"""

from __future__ import annotations

import pytest

import src.main as main_mod
from src.audio.blackhole_check import BlackHoleStatus, RoutingState, RoutingStatus


class _FakeDevices:
    def __init__(self) -> None:
        self.devices: list = []


@pytest.fixture
def structural_ok(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "DeviceManager", _FakeDevices)
    monkeypatch.setattr(
        main_mod, "verify_blackhole_setup",
        lambda dm=None: BlackHoleStatus(
            installed=True, multi_output_configured=True,
            blackhole_device_name="BlackHole 2ch", multi_output_device_name="Otis",
        ),
    )


def _routing(monkeypatch, status: RoutingStatus) -> list[int]:
    calls: list[int] = []

    def fake(probe=None):
        calls.append(1)
        return status

    monkeypatch.setattr(main_mod, "check_output_routing", fake)
    return calls


def test_not_routed_fails_and_says_why(structural_ok, monkeypatch, capsys) -> None:
    _routing(monkeypatch, RoutingStatus(
        state=RoutingState.NOT_ROUTED, default_output_name="Speakers",
        suggested_device="Otis",
    ))
    assert main_mod.main(["check-audio"]) == 1
    out = capsys.readouterr().out
    assert "Speakers" in out
    assert "Otis" in out


def test_routed_passes(structural_ok, monkeypatch, capsys) -> None:
    _routing(monkeypatch, RoutingStatus(
        state=RoutingState.OK, default_output_name="Otis", includes_blackhole=True,
        is_aggregate=True, member_names=["Speakers", "BlackHole 2ch"],
    ))
    assert main_mod.main(["check-audio"]) == 0
    assert "✓" in capsys.readouterr().out


@pytest.mark.parametrize("state", [RoutingState.INAUDIBLE, RoutingState.UNKNOWN])
def test_inaudible_or_unknown_is_its_own_exit_code(structural_ok, monkeypatch, state) -> None:
    _routing(monkeypatch, RoutingStatus(state=state, default_output_name="Otis BT"))
    assert main_mod.main(["check-audio"]) == 3


def test_the_structural_line_no_longer_reads_as_a_full_pass(structural_ok, monkeypatch, capsys) -> None:
    _routing(monkeypatch, RoutingStatus(
        state=RoutingState.NOT_ROUTED, default_output_name="Speakers",
    ))
    main_mod.main(["check-audio"])
    out = capsys.readouterr().out
    assert "✓" not in out, "a ✓ next to a failing routing check is the bug"


def test_missing_blackhole_fails_without_probing_routing(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "DeviceManager", _FakeDevices)
    monkeypatch.setattr(
        main_mod, "verify_blackhole_setup",
        lambda dm=None: BlackHoleStatus(installed=False, multi_output_configured=False,
                                        issues=["BlackHole is not installed."]),
    )
    calls = _routing(monkeypatch, RoutingStatus(state=RoutingState.OK))
    assert main_mod.main(["check-audio"]) == 1
    assert calls == []
