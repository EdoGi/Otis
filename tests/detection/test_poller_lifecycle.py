"""Regression cover for audit finding B9 — thread leak and a dead backoff.

Two bugs in the poll loop's lifecycle:

* ``stop()`` joined for 2 s and then set ``_thread = None`` regardless of
  whether the thread had actually exited. ``start()`` only checked
  ``self._thread``, so a wedged poller was forgotten and the next off-hours
  cycle stacked another one on top — one leaked thread and socket per cycle.
* the exponential backoff in ``_run`` only grew when ``_poll_cycle`` *raised*,
  but ``fetch_today_events`` swallows everything, so it never did. An offline
  laptop retried at full rate: ~960 doomed requests per account over an
  8-hour flight, and twice that in warning lines.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.detection.calendar_poller import (
    MAX_BACKOFF_SECONDS,
    GoogleCalendarPoller,
)


@pytest.fixture()
def poller(tmp_path) -> GoogleCalendarPoller:
    return GoogleCalendarPoller(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        calendar_ids=["primary"],
        poll_interval_seconds=0.01,
    )


# ---------------------------------------------------------------------------
# Thread lifecycle
# ---------------------------------------------------------------------------
def test_a_wedged_thread_is_not_duplicated(poller) -> None:
    release = threading.Event()

    def wedged(_stop_event: Any) -> None:
        release.wait(timeout=10)

    poller._run = wedged  # type: ignore[method-assign]
    try:
        poller.start()
        assert poller.is_running()

        poller.stop()  # joins 2 s, thread is still stuck
        assert poller._lingering, "a thread that outlived stop() must be tracked"

        poller.start()  # must NOT stack a second poller
        assert poller._thread is None, "started a second poller over a wedged one"
    finally:
        release.set()


def test_start_works_again_once_the_thread_finishes(poller) -> None:
    release = threading.Event()

    def wedged(_stop_event: Any) -> None:
        release.wait(timeout=10)

    poller._run = wedged  # type: ignore[method-assign]
    poller.start()
    poller.stop()
    assert poller._lingering

    release.set()
    poller._lingering[0].join(timeout=5)

    poller._run = lambda _e: None  # type: ignore[method-assign]
    poller.start()
    assert poller._lingering == [], "a finished thread must stop blocking start()"


def test_clean_stop_leaves_nothing_lingering(poller) -> None:
    poller._run = lambda _e: None  # type: ignore[method-assign]
    poller.start()
    poller.stop()
    assert poller._lingering == []
    assert poller._thread is None


def test_stop_is_safe_before_start(poller) -> None:
    poller.stop()  # must not raise
    assert poller._lingering == []


# ---------------------------------------------------------------------------
# Backoff actually engages
# ---------------------------------------------------------------------------
class FailingService:
    """Every calendar fetch raises — the offline case."""

    def events(self) -> FailingService:
        return self

    def list(self, **_kw: Any) -> FailingService:
        return self

    def execute(self, **_kw: Any) -> dict[str, Any]:
        raise ConnectionError("offline")


def test_a_failed_fetch_is_reported_not_swallowed(poller) -> None:
    poller._service = FailingService()
    assert poller.fetch_today_events() == []
    assert poller._last_cycle_failed is True


def test_a_successful_fetch_clears_the_failure_flag(poller) -> None:
    class OkService:
        def events(self):
            return self

        def list(self, **_kw):
            return self

        def execute(self, **_kw):
            return {"items": []}

    poller._service = FailingService()
    poller.fetch_today_events()
    assert poller._last_cycle_failed is True

    poller._service = OkService()
    poller.fetch_today_events()
    assert poller._last_cycle_failed is False


def test_backoff_grows_while_offline_and_is_capped(poller) -> None:
    """Drive the real loop with a stop event that records each wait."""
    waits: list[float] = []
    poller._service = FailingService()

    class RecordingEvent:
        def __init__(self) -> None:
            self.n = 0

        def is_set(self) -> bool:
            return self.n > 25

        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            self.n += 1
            return False

    poller._run(RecordingEvent())  # type: ignore[arg-type]

    assert waits, "the loop must wait between cycles"
    assert waits[0] >= poller._poll_interval
    assert waits == sorted(waits), "backoff must be non-decreasing while failing"
    assert waits[-1] > waits[0], "backoff never grew — this is the dead-code bug"
    assert max(waits) <= MAX_BACKOFF_SECONDS


def test_backoff_resets_after_recovery(poller) -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def events(self):
            return self

        def list(self, **_kw):
            return self

        def execute(self, **_kw):
            self.calls += 1
            if self.calls <= 3:
                raise ConnectionError("offline")
            return {"items": []}

    waits: list[float] = []
    flaky = Flaky()
    poller._service = flaky
    # A run of failures discards the cached service so the next cycle rebuilds
    # it — mirror that here instead of leaving the poller with no service at
    # all, which would fail on missing credentials rather than on the network.
    poller._build_service = lambda: flaky  # type: ignore[method-assign]

    class RecordingEvent:
        def __init__(self) -> None:
            self.n = 0

        def is_set(self) -> bool:
            return self.n > 6

        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            self.n += 1
            return False

    poller._run(RecordingEvent())  # type: ignore[arg-type]

    assert waits[-1] == pytest.approx(poller._poll_interval), (
        "once the network is back, polling must return to its normal cadence"
    )
