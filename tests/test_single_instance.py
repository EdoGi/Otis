"""Regression: every click on Otis.app started another full copy of Otis.

The applet LaunchServices tracks exits seconds after spawning Python, so macOS
never knew Otis was running. 8 of 36 real launches (2026-08-24..10-21) found
port 8765 held by an earlier copy; on 2026-10-14 three ran at once. Each copy
ran its own detector, pollers and startup toasts, and drained the same queue.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import src.main as main_mod
from src import single_instance
from src.single_instance import LOCK_NAME, SURFACE_NAME

REPO = Path(__file__).resolve().parent.parent

_HOLDER = """
import sys, time
sys.path.insert(0, {repo!r})
from pathlib import Path
from src.single_instance import acquire
lock = acquire(Path(sys.argv[1]), command=sys.argv[2])
print("held" if lock else "busy", flush=True)
time.sleep(60)
"""


@pytest.fixture
def holder():
    """Hold a lock from another process, as a running Otis would."""
    procs: list[subprocess.Popen] = []

    def _hold(path: Path, command: str = "ui") -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER.format(repo=str(REPO)), str(path), command],
            stdout=subprocess.PIPE, text=True,
        )
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"
        procs.append(proc)
        return proc

    yield _hold
    for proc in procs:
        proc.kill()
        proc.wait(5)


# ---------------------------------------------------------------------------
# The lock itself
# ---------------------------------------------------------------------------
def test_a_second_process_cannot_take_a_held_lock(tmp_path: Path, holder) -> None:
    holder(tmp_path / LOCK_NAME)
    assert single_instance.acquire(tmp_path / LOCK_NAME, command="ui") is None


def test_the_lock_is_freed_when_its_holder_dies(tmp_path: Path, holder) -> None:
    proc = holder(tmp_path / LOCK_NAME)
    proc.kill()  # no cleanup code runs, as with a crash or kill -9
    proc.wait(5)
    lock = single_instance.acquire(tmp_path / LOCK_NAME, command="ui")
    assert lock is not None
    lock.release()


def test_the_holder_is_recorded(tmp_path: Path, holder) -> None:
    proc = holder(tmp_path / LOCK_NAME, command="run")
    who = single_instance.read_holder(tmp_path / LOCK_NAME)
    assert who.pid == proc.pid
    assert who.command == "run"


def test_release_lets_the_same_process_take_it_again(tmp_path: Path) -> None:
    first = single_instance.acquire(tmp_path / LOCK_NAME, command="ui")
    assert first is not None
    first.release()
    again = single_instance.acquire(tmp_path / LOCK_NAME, command="ui")
    assert again is not None
    again.release()


# ---------------------------------------------------------------------------
# main(): a second launch defers to the first
# ---------------------------------------------------------------------------
@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(main_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main_mod, "SURFACE_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(main_mod, "STARTUP_GRACE_SECONDS", 0.0)  # tests opt in
    return tmp_path


@pytest.fixture
def no_start(monkeypatch) -> list[str]:
    """Fail loudly if main() would build a second app."""
    started: list[str] = []
    monkeypatch.setattr(main_mod, "_run_menubar", lambda cfg, **kw: started.append("ui") or 0)
    monkeypatch.setattr(main_mod, "_run_daemon", lambda cfg: started.append("run") or 0)
    return started


def test_ui_launch_while_running_does_not_start_a_second_app(
    run_dir: Path, holder, no_start, monkeypatch
) -> None:
    holder(run_dir / LOCK_NAME)
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: None)
    assert main_mod.main(["ui"]) == 0
    assert no_start == []


def test_ui_launch_while_running_asks_the_running_copy_to_show_itself(
    run_dir: Path, holder, no_start, monkeypatch
) -> None:
    holder(run_dir / LOCK_NAME)
    surface = run_dir / SURFACE_NAME
    seen: list[bool] = []

    def running_copy() -> None:  # what the first copy's drain timer does
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if single_instance.take_surface_request(surface):
                seen.append(True)
                return
            time.sleep(0.02)

    t = threading.Thread(target=running_copy)
    t.start()
    unresponsive: list[object] = []
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: unresponsive.append(a))
    monkeypatch.setattr(main_mod, "SURFACE_WAIT_SECONDS", 3.0)
    assert main_mod.main(["ui"]) == 0
    t.join(5)
    assert seen == [True]
    assert unresponsive == []  # it answered, so no "not responding" message
    assert not surface.exists()


def test_an_unresponsive_running_copy_is_reported_not_ignored(
    run_dir: Path, holder, no_start, monkeypatch
) -> None:
    proc = holder(run_dir / LOCK_NAME)
    old = time.time() - 3600  # taken an hour ago: past any startup grace
    os.utime(run_dir / LOCK_NAME, (old, old))
    told: list[tuple] = []
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: told.append(a))
    monkeypatch.setattr(main_mod, "_bring_forward", lambda pid: None)
    assert main_mod.main(["ui"]) == 0
    assert no_start == []
    assert told, "the user must hear that the running copy is stuck"
    assert str(proc.pid) in " ".join(map(str, told[0]))
    # main() replaces the root handlers, so read what reached otis.log.
    assert "did not respond" in (run_dir / "otis.log").read_text(encoding="utf-8")
    # A stale request must not pop the menu open later, when it unfreezes.
    assert not (run_dir / SURFACE_NAME).exists()


def test_daemon_refuses_to_run_beside_another_otis(run_dir: Path, holder, no_start) -> None:
    holder(run_dir / LOCK_NAME)
    assert main_mod.main(["run"]) == 1
    assert no_start == []


def test_ui_launch_beside_the_daemon_says_so(run_dir: Path, holder, no_start, monkeypatch) -> None:
    holder(run_dir / LOCK_NAME, command="run")
    told: list[tuple] = []
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: told.append(a))
    assert main_mod.main(["ui"]) == 1
    assert no_start == []
    assert told and "daemon" in " ".join(map(str, told[0])).lower()


def test_a_free_lock_is_taken_and_released_around_the_app(run_dir: Path, monkeypatch) -> None:
    during: list[object] = []

    def fake_menubar(cfg, **kw):
        during.append(single_instance.acquire(run_dir / LOCK_NAME, command="probe"))
        return 0

    monkeypatch.setattr(main_mod, "_run_menubar", fake_menubar)
    assert main_mod.main(["ui"]) == 0
    assert during == [None], "the app must hold the lock while it runs"
    after = single_instance.acquire(run_dir / LOCK_NAME, command="probe")
    assert after is not None
    after.release()


def test_check_audio_runs_while_otis_is_running(run_dir: Path, holder, monkeypatch) -> None:
    """Diagnostics must work exactly when Otis is up — never take the lock."""
    holder(run_dir / LOCK_NAME)
    monkeypatch.setattr(main_mod, "_print_audio_status", lambda cfg: 42)
    monkeypatch.setattr(main_mod, "_defer_to_running_instance",
                        lambda *a: pytest.fail("check-audio deferred to the running copy"))
    # 42 proves the lock-free path ran: deferring also returns 0 (review finding).
    assert main_mod.main(["check-audio"]) == 42
    assert not (run_dir / SURFACE_NAME).exists()


# ---------------------------------------------------------------------------
# The running copy answers a surface request
# ---------------------------------------------------------------------------
def test_the_menu_bar_consumes_a_surface_request(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    sent: list[tuple[str, str, str]] = []
    surface = tmp_path / SURFACE_NAME
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        health=HealthRegistry(),
        surface_request_path=surface,
    )
    single_instance.request_surface(surface)
    app._drain_main_queue(None)
    assert not surface.exists()
    assert any("already running" in t.lower() for t, _s, _m in sent)


def test_the_menu_opens_after_the_drain_returns(tmp_path: Path) -> None:
    """Review finding: performClick_ ran menu tracking inside the drain timer's
    own callout. A timer never re-fires while its callout is on the stack, so
    an alert picked from that menu (About, the quit confirmation) froze the
    drain again — the very freeze 49f951f removed. The click is now scheduled
    for a later run-loop pass."""
    pytest.importorskip("rumps")
    from types import SimpleNamespace

    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    calls: list[tuple] = []

    class Button:
        def performClick_(self, sender):
            calls.append(("performClick_", sender))

        def performSelector_withObject_afterDelay_(self, sel, obj, delay):
            calls.append(("later", sel, obj, delay))

    surface = tmp_path / SURFACE_NAME
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons", user_config_path=tmp_path / "config.yaml",
        health=HealthRegistry(), surface_request_path=surface,
    )
    button = Button()
    app._app._nsapp = SimpleNamespace(nsstatusitem=SimpleNamespace(button=lambda: button))
    single_instance.request_surface(surface)
    app._drain_main_queue(None)
    assert calls == [("later", "performClick:", None, 0.0)]


def test_a_click_during_a_cold_start_is_not_called_a_freeze(
    run_dir: Path, holder, no_start, monkeypatch
) -> None:
    """Review finding: the lock is taken before config, pipeline, web bind and
    menu construction, and requests are only answered once the run loop turns.
    A second click in that window waited 2.5 s and was told the healthy,
    still-starting copy was frozen and should be killed."""
    holder(run_dir / LOCK_NAME)  # just taken: the lock file is seconds old
    told: list[tuple] = []
    forward: list[int] = []
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: told.append(a))
    monkeypatch.setattr(main_mod, "_bring_forward", forward.append)
    monkeypatch.setattr(main_mod, "STARTUP_GRACE_SECONDS", 0.4)
    assert main_mod.main(["ui"]) == 0
    assert told and "still starting" in told[-1][0].lower()
    assert "dialog" in told[-1][1], "a bad-config dialog also looks like a slow start"
    assert len(forward) == 1, "raise it in case a dialog is what it is stuck on"
    assert (run_dir / SURFACE_NAME).exists(), "keep it: the menu opens once the copy is ready"


def _surface_app(tmp_path: Path):
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    sent: list[tuple[str, str, str]] = []
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons", user_config_path=tmp_path / "config.yaml",
        health=HealthRegistry(), surface_request_path=tmp_path / SURFACE_NAME,
    )
    app._sent = sent  # type: ignore[attr-defined]
    return app


def test_with_a_dialog_open_the_running_copy_raises_it_instead(tmp_path: Path, monkeypatch) -> None:
    """Review round 2: the drain also runs during a modal, so it answered the
    request and toasted "the menu is open now" — but the scheduled click only
    fires after the dialog closes. Now the dialog comes forward and the toast
    says to answer it; no menu pops open later by surprise."""
    pytest.importorskip("rumps")
    from types import SimpleNamespace

    app = _surface_app(tmp_path)
    raised: list[str] = []
    clicks: list[tuple] = []
    window = SimpleNamespace(makeKeyAndOrderFront_=lambda _s: raised.append("front"))
    monkeypatch.setattr(app, "_modal_window", lambda: window)
    monkeypatch.setattr(app, "_activate_app", lambda: raised.append("activate"))
    button = SimpleNamespace(performSelector_withObject_afterDelay_=lambda *a: clicks.append(a))
    app._app._nsapp = SimpleNamespace(nsstatusitem=SimpleNamespace(button=lambda: button))

    single_instance.request_surface(tmp_path / SURFACE_NAME)
    app._drain_main_queue(None)
    assert raised == ["activate", "front"]
    assert clicks == []
    assert any("dialog" in m for _t, _s, m in app._sent)


def test_a_stale_surface_request_is_dropped_when_the_lock_is_taken(run_dir: Path, monkeypatch) -> None:
    """Review round 2: a request left by a launch whose target died before
    answering made the *next* fresh start pop its menu open and toast
    "already running". Any request present when a copy takes the lock was
    written for an earlier holder."""
    (run_dir / SURFACE_NAME).write_text("1234\n")
    seen: list[bool] = []
    monkeypatch.setattr(main_mod, "_run_menubar",
                        lambda cfg, **kw: seen.append((run_dir / SURFACE_NAME).exists()) or 0)
    assert main_mod.main(["ui"]) == 0
    assert seen == [False]


def test_the_bad_config_dialog_comes_to_the_front(monkeypatch) -> None:
    """Review round 2: rumps.alert does not activate the app, so the dialog
    could open behind other windows while holding the lock."""
    import src.ui.rumps_support as rs

    shown: list[dict] = []
    monkeypatch.setattr(rs, "_alert", lambda **kw: shown.append(kw) or 1)
    assert main_mod._report_bad_config(ValueError("bad model"), gui=True) == 2
    assert shown and "bad model" in shown[0]["message"]
