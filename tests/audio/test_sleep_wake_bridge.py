"""Regression: the second recorder in a process lost sleep/wake handling.

``_SleepWakeObserver._run`` declared its Objective-C bridge class inline. ObjC
registers class names process-wide, so the first recorder defined it and every
one after that raised

    objc.error: _Bridge is overriding existing Objective-C class

inside its own observer thread — printed to stderr, watched by nobody. The
menu bar builds a fresh recorder per meeting, so sleep/wake worked for the
first recording of a session and was dead for the rest, taking P2's wake-time
stream revival with it: the exact case where a lid closed mid-meeting leaves a
stream producing silence for the remainder.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("Foundation")

from src.audio.recorder import sleep_wake_bridge_class


def test_the_bridge_class_is_defined_only_once() -> None:
    """Two recorders in one process is the normal case, not an edge case."""
    first = sleep_wake_bridge_class()
    second = sleep_wake_bridge_class()
    assert first is second


def test_repeated_calls_never_raise() -> None:
    """Each call used to be a fresh `class _Bridge(NSObject)` declaration."""
    for _ in range(5):
        assert sleep_wake_bridge_class() is not None


def test_concurrent_callers_get_the_same_class() -> None:
    seen: list[object] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def build() -> None:
        try:
            barrier.wait(5.0)
            seen.append(sleep_wake_bridge_class())
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert not errors
    assert len({id(c) for c in seen}) == 1


def test_handlers_are_per_instance_not_baked_into_the_class() -> None:
    """A shared class means each recorder's callbacks live on its own bridge."""
    cls = sleep_wake_bridge_class()
    fired: list[str] = []

    a = cls.alloc().init()
    a.otis_on_sleep = lambda: fired.append("a-sleep")
    a.otis_on_wake = lambda: fired.append("a-wake")
    b = cls.alloc().init()
    b.otis_on_sleep = lambda: fired.append("b-sleep")
    b.otis_on_wake = lambda: fired.append("b-wake")

    a.workspaceWillSleep_(None)
    b.workspaceDidWake_(None)

    assert fired == ["a-sleep", "b-wake"]


def test_a_bridge_with_no_handlers_is_harmless() -> None:
    """The window between alloc().init() and assigning the callbacks."""
    bare = sleep_wake_bridge_class().alloc().init()
    bare.workspaceWillSleep_(None)
    bare.workspaceDidWake_(None)


def test_a_raising_handler_does_not_escape_into_the_run_loop() -> None:
    cls = sleep_wake_bridge_class()
    bridge = cls.alloc().init()
    bridge.otis_on_sleep = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    bridge.workspaceWillSleep_(None)  # must not raise
