"""Regression: a launch that failed, a crash and a Quit all left no trace.

Otis.app runs ``run.sh`` with stdout/stderr sent to /dev/null, main() had no
excepthook, and the entry-point logger was named ``otis`` — outside the
``src`` hierarchy that _configure_logging sets to INFO — so "Logging to",
"Loaded config" and "Web UI listening" appeared 0 times across 36 real
launches. A menu Quit runs NSApp.terminate_, which calls exit() without
running ``finally`` blocks or atexit, so nothing marked it either. On
2026-10-21 the process simply stopped logging at 13:42; whether the user quit
or it died could not be told from the log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import src.main as main_mod

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch) -> Path:
    import src.audio.blackhole_check as bc

    # MenuBarApp.run() checks BlackHole at launch; keep real PortAudio out.
    monkeypatch.setattr(
        bc, "verify_blackhole_setup",
        lambda device_manager=None: bc.BlackHoleStatus(installed=True, multi_output_configured=True),
    )
    monkeypatch.setattr(main_mod, "LOG_DIR", tmp_path)
    # main() installs process-wide hooks; put the originals back afterwards.
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    return tmp_path


def _log_text(log_dir: Path) -> str:
    return (log_dir / "otis.log").read_text(encoding="utf-8")


def test_entrypoint_info_lines_reach_the_log_file(log_dir: Path) -> None:
    main_mod._configure_logging("INFO")
    main_mod.logger.info("entrypoint marker")
    assert "entrypoint marker" in _log_text(log_dir)


def test_every_line_carries_the_pid(log_dir: Path) -> None:
    """Concurrent copies interleave in one file; only a pid separates them."""
    main_mod._configure_logging("INFO")
    main_mod.logger.warning("pid marker")
    line = next(ln for ln in _log_text(log_dir).splitlines() if "pid marker" in ln)
    assert f" {os.getpid()} " in line


def test_a_launch_is_marked_with_pid_and_command(log_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "_run_menubar", lambda cfg, **kw: 0)
    assert main_mod.main(["ui"]) == 0
    text = _log_text(log_dir)
    assert "Otis starting" in text
    assert f"pid={os.getpid()}" in text
    assert "command=ui" in text


def test_an_exception_escaping_startup_is_logged(log_dir: Path, monkeypatch) -> None:
    def boom(cfg, **kw):
        raise RuntimeError("menu bar construction exploded")

    monkeypatch.setattr(main_mod, "_run_menubar", boom)
    with pytest.raises(RuntimeError):
        main_mod.main(["ui"])
    text = _log_text(log_dir)
    assert "failed to start" in text
    assert "menu bar construction exploded" in text
    assert "Traceback" in text


def test_an_uncaught_thread_exception_is_logged(log_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "_run_menubar", lambda cfg, **kw: 0)
    main_mod.main(["ui"])

    def worker() -> None:
        raise ValueError("worker thread died")

    t = threading.Thread(target=worker, name="otis-test-worker")
    t.start()
    t.join()
    text = _log_text(log_dir)
    assert "worker thread died" in text
    assert "otis-test-worker" in text


def test_releasing_model_memory_never_imports_mlx_whisper(monkeypatch) -> None:
    """Quit/shutdown called the release path, which imported mlx-whisper
    (with scipy and numba) just to clear a model that was never loaded:
    2.6 s measured, on every shutdown of a session with no transcription."""
    import src.transcription.whisper_engine as we

    imported: list[bool] = []
    monkeypatch.delitem(sys.modules, "mlx_whisper.transcribe", raising=False)
    monkeypatch.setattr(we, "_import_model_holder", lambda: imported.append(True))
    we._release_mlx_model_cache()
    assert imported == []


# ---------------------------------------------------------------------------
# Quit
# ---------------------------------------------------------------------------
def _headless_app(tmp_path: Path, detector):
    rumps = pytest.importorskip("rumps")
    from src.config import Config
    from src.health import HealthRegistry
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    class NoLoopApp(rumps.App):
        def run(self, **kwargs):  # the real one blocks until Quit
            return None

    from src.audio.blackhole_check import RoutingState, RoutingStatus

    # tmp audio dir and an injected routing check: run() must not scan the
    # user's ~/Otis/audio or ask this Mac's CoreAudio (review finding).
    return MenuBarApp(
        config=Config({"app": {"working_days": []}}),
        detector=detector,
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
        audio_dir=tmp_path / "audio",
        health=HealthRegistry(),
        rumps_app_factory=NoLoopApp,
        routing_check=lambda: RoutingStatus(state=RoutingState.UNKNOWN),
    )


class _Detector:
    def __init__(self) -> None:
        self.calendar_pollers: list = []
        self.stopped = 0

    def __getattr__(self, name):  # on_meeting_* registration etc.
        return lambda *a, **k: None

    def stop(self) -> None:
        self.stopped += 1


def test_quit_is_logged_and_stops_detection(tmp_path: Path, log_dir: Path) -> None:
    import rumps

    detector = _Detector()
    app = _headless_app(tmp_path, detector)
    main_mod._configure_logging("INFO")
    app.run()
    try:
        assert app._on_before_quit in rumps.events.before_quit.callbacks
        before = detector.stopped  # run()'s own teardown already stopped it once
        app._on_before_quit()
    finally:
        rumps.events.before_quit.unregister(app._on_before_quit)
    assert detector.stopped == before + 1
    assert "Otis quitting" in _log_text(log_dir)


def test_quit_handler_survives_a_failing_step(tmp_path: Path, log_dir: Path) -> None:
    """rumps swallows handler exceptions to stderr (/dev/null); a raise
    must neither escape nor skip the log line."""
    detector = _Detector()

    def bad_stop() -> None:
        raise RuntimeError("detector stop failed")

    detector.stop = bad_stop  # type: ignore[method-assign]
    app = _headless_app(tmp_path, detector)
    main_mod._configure_logging("INFO")
    app._on_before_quit()
    assert "Otis quitting" in _log_text(log_dir)


# ---------------------------------------------------------------------------
# run.sh: what happens before Python logging exists
# ---------------------------------------------------------------------------
def test_run_sh_keeps_a_launch_log_when_not_on_a_terminal(tmp_path: Path) -> None:
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "run.sh"), "--help"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, timeout=60,
    )
    assert proc.returncode == 0
    launch_log = tmp_path / ".otis" / "launch.log"
    assert launch_log.exists(), "the applet discards output; run.sh must keep it"
    text = launch_log.read_text(encoding="utf-8")
    assert "Otis launching" in text
    assert "usage: otis" in text  # python's own output lands there too


# ---------------------------------------------------------------------------
# kill / SIGTERM
# ---------------------------------------------------------------------------
def test_sigterm_quits_through_the_normal_quit_path(tmp_path: Path, log_dir: Path, monkeypatch) -> None:
    """A plain `kill` (or launchd at shutdown — the Python process is not the
    app LaunchServices knows) ended the menu bar instantly: no quit line, no
    before_quit, a running recording's WAV left unfinalised. It now asks AppKit
    to terminate, which runs the same handler a menu Quit does."""
    import signal

    import rumps

    quits: list[bool] = []
    monkeypatch.setattr(rumps, "quit_application", lambda *a: quits.append(True))
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    app = _headless_app(tmp_path, _Detector())
    main_mod._configure_logging("INFO")
    try:
        app.run()
        assert signal.getsignal(signal.SIGTERM) == app._on_terminate_signal
        app._on_terminate_signal(signal.SIGTERM, None)
    finally:
        rumps.events.before_quit.unregister(app._on_before_quit)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert quits == [True]
    assert "SIGTERM" in _log_text(log_dir)


def test_a_frozen_main_thread_cannot_make_kill_a_no_op(tmp_path: Path, log_dir: Path) -> None:
    """Review finding: the SIGTERM handler runs only when the main thread runs
    Python. Frozen in native code — the case where the second launch says "not
    responding" and suggests killing it — `kill PID` then did nothing at all,
    where before the handler existed it ended the process. A watchdog thread
    now sees the signal through the wakeup fd and forces the exit if the normal
    quit has not finished within the grace period."""
    import os
    import signal

    app = _headless_app(tmp_path, _Detector())
    main_mod._configure_logging("INFO")
    exits: list[int] = []
    asked: list[object] = []
    r, w = os.pipe()
    t = threading.Thread(target=app._escalate_unhandled_quit, args=(r,),
                         kwargs={"grace": 0.05, "exit_fn": exits.append,
                                 "call_on_main": asked.append})
    t.start()
    os.write(w, bytes([signal.SIGCHLD]))  # not a quit signal: ignored
    os.write(w, bytes([signal.SIGTERM]))
    t.join(5)
    # First the graceful request (runs even while a menu is open), then force.
    assert asked == [app._quit_once]
    assert exits == [143]
    assert "forcing exit" in _log_text(log_dir)


def test_one_signal_quits_once(tmp_path: Path, log_dir: Path, monkeypatch) -> None:
    import rumps

    quits: list[bool] = []
    monkeypatch.setattr(rumps, "quit_application", lambda *a: quits.append(True))
    app = _headless_app(tmp_path, _Detector())
    app._quit_once()  # the watchdog's request
    app._quit_once()  # the Python handler, running late
    assert quits == [True]


def test_the_not_responding_toast_suggests_force_quit(tmp_path: Path, monkeypatch) -> None:
    told: list[tuple] = []
    monkeypatch.setattr(main_mod, "_notify_from_second_launch", lambda *a: told.append(a))
    monkeypatch.setattr(main_mod, "_bring_forward", lambda pid: None)
    monkeypatch.setattr(main_mod, "SURFACE_WAIT_SECONDS", 0.05)
    from src import single_instance

    lock_path = tmp_path / single_instance.LOCK_NAME
    lock_path.write_text("99999\nui\n", encoding="utf-8")
    import os
    import time as _time

    old = _time.time() - 3600  # an old lock: not a copy that is still starting
    os.utime(lock_path, (old, old))
    main_mod._defer_to_running_instance("ui", tmp_path)
    assert told and "Force Quit" in told[-1][1]


def test_run_sh_leaves_piped_cli_output_alone(tmp_path: Path) -> None:
    """Review finding: gating on "stderr is not a TTY" also caught pipes, so
    `./scripts/run.sh doctor | tee out.txt` (or any tool) printed nothing —
    everything went to launch.log. Only the applet's case, where output would
    be thrown away, is redirected now."""
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "run.sh"), "--help"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0
    assert "usage: otis" in proc.stdout
    assert not (tmp_path / ".otis" / "launch.log").exists()


def test_an_unwritable_launch_log_never_blocks_a_launch(tmp_path: Path) -> None:
    """Review finding: under `set -e` a failed redirect stopped run.sh before
    Python started — the no-trace launch failure the log exists to prevent."""
    (tmp_path / ".otis").mkdir()
    log = tmp_path / ".otis" / "launch.log"
    log.write_text("")
    log.chmod(0o444)
    env = {**os.environ, "HOME": str(tmp_path)}
    try:
        proc = subprocess.run(
            ["bash", str(REPO / "scripts" / "run.sh"), "--help"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, timeout=60,
        )
    finally:
        log.chmod(0o644)
    assert proc.returncode == 0


def test_run_sh_keeps_stdout_when_only_stderr_is_silenced(tmp_path: Path) -> None:
    """Review round 2: the gate checked only stderr, so a user's own
    `run.sh doctor 2>/dev/null` sent the report to launch.log. Only the
    applet's case — both streams discarded — is redirected now."""
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "run.sh"), "--help"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=env, timeout=60,
    )
    assert "usage: otis" in proc.stdout
    assert not (tmp_path / ".otis" / "launch.log").exists()


@pytest.mark.parametrize(("target", "level"), [("pipe", "INFO"), ("file", "WARNING")])
def test_console_level_follows_where_stderr_goes(tmp_path: Path, log_dir: Path,
                                                 monkeypatch, target, level) -> None:
    """Into a file (run.sh's launch.log) only warnings go, so it does not
    duplicate otis.log; a pipe — `run.sh run | tee`, a tool — keeps INFO."""
    import logging

    if target == "pipe":
        r, w = os.pipe()
        stream = os.fdopen(w, "w")
    else:
        stream = open(tmp_path / "stderr.txt", "w")
    monkeypatch.setattr(sys, "stderr", stream)
    try:
        main_mod._configure_logging("INFO")
        console = next(h for h in logging.getLogger().handlers
                       if type(h) is logging.StreamHandler)
        assert logging.getLevelName(console.level if console.level else logging.INFO) == level
    finally:
        stream.close()
        if target == "pipe":
            os.close(r)
