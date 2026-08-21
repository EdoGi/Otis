"""Tests for src/health.py — the component health registry.

The registry is the thing whose absence let three failures run silently for
months, so its contract is tested hard: state transitions fire exactly once,
callbacks can't take the app down, and ``overall()`` reports the worst state.
"""

from __future__ import annotations

import threading

import pytest

from src.health import (
    HealthRegistry,
    HealthState,
    describe_health,
    get_health,
    reset_health,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def registry(clock: FakeClock) -> HealthRegistry:
    return HealthRegistry(clock=clock)


# ---------------------------------------------------------------------------
# Basic reporting
# ---------------------------------------------------------------------------
def test_unknown_until_reported(registry: HealthRegistry) -> None:
    assert registry.get("system_audio") is None
    assert registry.overall() is HealthState.OK
    assert registry.is_degraded() is False


def test_report_records_state_detail_and_time(registry, clock) -> None:
    clock.advance(5)
    rep = registry.report(
        "system_audio", label="System audio", state=HealthState.FAIL,
        detail="No signal", hint="Set output to Otis",
    )
    assert rep.state is HealthState.FAIL
    assert rep.label == "System audio"
    assert rep.detail == "No signal"
    assert rep.hint == "Set output to Otis"
    assert rep.updated_at == 1005.0
    assert registry.get("system_audio") == rep


def test_overall_is_the_worst_component(registry) -> None:
    registry.report("a", label="A", state=HealthState.OK)
    assert registry.overall() is HealthState.OK
    registry.report("b", label="B", state=HealthState.WARN)
    assert registry.overall() is HealthState.WARN
    registry.report("c", label="C", state=HealthState.FAIL)
    assert registry.overall() is HealthState.FAIL
    # Recovering the worst one drops us back to the next worst.
    registry.report("c", label="C", state=HealthState.OK)
    assert registry.overall() is HealthState.WARN


def test_unknown_does_not_count_as_degraded(registry) -> None:
    registry.report("a", label="A", state=HealthState.UNKNOWN)
    assert registry.overall() is HealthState.UNKNOWN
    assert registry.is_degraded() is False


def test_is_degraded_for_warn_and_fail(registry) -> None:
    registry.report("a", label="A", state=HealthState.WARN)
    assert registry.is_degraded() is True
    registry.report("a", label="A", state=HealthState.FAIL)
    assert registry.is_degraded() is True
    registry.report("a", label="A", state=HealthState.OK)
    assert registry.is_degraded() is False


def test_snapshot_uses_canonical_order_then_insertion(registry) -> None:
    registry.report("zzz_custom", label="Custom", state=HealthState.OK)
    registry.report("calendar", label="Calendar", state=HealthState.OK)
    registry.report("system_audio", label="System audio", state=HealthState.OK)
    keys = [r.key for r in registry.snapshot()]
    # Canonical components lead, in declared order; unknown keys trail.
    assert keys.index("system_audio") < keys.index("calendar")
    assert keys[-1] == "zzz_custom"


# ---------------------------------------------------------------------------
# Change callbacks — the anti-log-spam contract
# ---------------------------------------------------------------------------
def test_callback_fires_on_transition_only(registry) -> None:
    seen: list[tuple[str, HealthState, HealthState]] = []
    registry.on_change(lambda rep, prev: seen.append((rep.key, prev, rep.state)))

    registry.report("calendar", label="Calendar", state=HealthState.FAIL, detail="no token")
    registry.report("calendar", label="Calendar", state=HealthState.FAIL, detail="no token")
    registry.report("calendar", label="Calendar", state=HealthState.FAIL, detail="no token")

    assert seen == [("calendar", HealthState.UNSET, HealthState.FAIL)]


def test_callback_fires_again_on_recovery(registry) -> None:
    seen: list[HealthState] = []
    registry.on_change(lambda rep, prev: seen.append(rep.state))
    registry.report("calendar", label="Calendar", state=HealthState.FAIL)
    registry.report("calendar", label="Calendar", state=HealthState.OK)
    assert seen == [HealthState.FAIL, HealthState.OK]


def test_detail_change_alone_does_not_refire(registry) -> None:
    seen: list[str] = []
    registry.on_change(lambda rep, prev: seen.append(rep.detail))
    registry.report("disk", label="Disk", state=HealthState.WARN, detail="10 GB free")
    registry.report("disk", label="Disk", state=HealthState.WARN, detail="9 GB free")
    assert seen == ["10 GB free"]
    # ...but the stored detail is still refreshed.
    assert registry.get("disk").detail == "9 GB free"


def test_callback_exception_does_not_propagate(registry) -> None:
    def boom(rep, prev):
        raise RuntimeError("notification backend down")

    registry.on_change(boom)
    # Must not raise — a broken notifier can't take down a recording.
    rep = registry.report("system_audio", label="System audio", state=HealthState.FAIL)
    assert rep.state is HealthState.FAIL


def test_multiple_callbacks_all_fire_even_if_one_raises(registry) -> None:
    seen: list[str] = []
    registry.on_change(lambda rep, prev: seen.append("first"))
    registry.on_change(lambda rep, prev: (_ for _ in ()).throw(RuntimeError("x")))
    registry.on_change(lambda rep, prev: seen.append("third"))
    registry.report("a", label="A", state=HealthState.WARN)
    assert seen == ["first", "third"]


# ---------------------------------------------------------------------------
# Thread safety — reports arrive from poller, recorder and UI threads
# ---------------------------------------------------------------------------
def test_concurrent_reports_do_not_lose_transitions(registry) -> None:
    seen: list[HealthState] = []
    lock = threading.Lock()

    def record(rep, prev):
        with lock:
            seen.append(rep.state)

    registry.on_change(record)

    def hammer(key: str) -> None:
        for _ in range(50):
            registry.report(key, label=key, state=HealthState.WARN)

    threads = [threading.Thread(target=hammer, args=(f"c{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 8 components, each transitioning UNKNOWN -> WARN exactly once.
    assert len(seen) == 8
    assert len(registry.snapshot()) == 8


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_describe_health_is_single_line_per_component(registry) -> None:
    registry.report("system_audio", label="System audio", state=HealthState.FAIL,
                    detail="silent", hint="Set output to Otis")
    registry.report("calendar", label="Calendar", state=HealthState.OK, detail="2 accounts")
    lines = describe_health(registry)
    assert any("System audio" in ln and "silent" in ln for ln in lines)
    assert any("Calendar" in ln for ln in lines)
    assert len(lines) == 2


def test_describe_health_empty_registry(registry) -> None:
    assert describe_health(registry) == []


# ---------------------------------------------------------------------------
# Process-global accessor
# ---------------------------------------------------------------------------
def test_get_health_is_a_singleton() -> None:
    reset_health()
    try:
        a = get_health()
        b = get_health()
        assert a is b
        a.report("system_audio", label="System audio", state=HealthState.WARN)
        assert get_health().get("system_audio").state is HealthState.WARN
    finally:
        reset_health()


def test_reset_health_clears_state() -> None:
    reset_health()
    get_health().report("calendar", label="Calendar", state=HealthState.FAIL)
    reset_health()
    assert get_health().get("calendar") is None
