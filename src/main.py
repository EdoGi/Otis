"""Otis entry point.

Two top-level commands:

* ``otis check-audio`` (or ``otis --check-audio`` for back-compat) — print the
  CoreAudio device list, the BlackHole / multi-output status and whether the
  current output actually feeds BlackHole. Exits 0 only when it does.
* ``otis run`` — start the foreground daemon: detection + auto-record on
  meeting detection. Stays in the foreground; Ctrl-C to stop.

Later phases will replace the daemon with a proper menu-bar app and add the
transcription, storage, web, and MCP servers.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from src import single_instance
from src.audio.blackhole_check import (
    RoutingState,
    check_output_routing,
    format_setup_instructions,
    verify_blackhole_setup,
)
from src.audio.devices import DeviceManager
from src.config import Config, load_user_config
from src.config_schema import ConfigError, validate
from src.daemon import OtisDaemon

# Inside the ``src`` hierarchy on purpose: _configure_logging sets the level on
# ``src`` alone, and under the old name "otis" every INFO line from here —
# "Logging to", "Loaded config", "Web UI listening" — was dropped (0 of each
# across 36 real launches).
logger = logging.getLogger("src.main")

#: Kept open for the life of the process so faulthandler can write a Python
#: stack for a native crash (a segfault in CoreAudio/PortAudio/MLX).
_crash_log: Any = None

#: Where otis.log lives. A module constant so the test suite can point it
#: somewhere disposable: tests that call main() or _configure_logging would
#: otherwise write into the user's real log and, at 1 MB with five rotations,
#: push their actual meeting history out of it.
LOG_DIR = Path("~/.otis")

#: How long a second launch waits for the running menu bar to answer its
#: "show yourself" request. Its drain timer polls every 100 ms, so a live copy
#: answers in well under a second; one that has not answered after this long
#: is not servicing its run loop (an unanswered modal did this twice).
SURFACE_WAIT_SECONDS = 2.5

#: Commands that own the mic, the queue and port 8765, and so must not run
#: twice. check-audio, prune-models and the MCP server stay lock-free: they are
#: most useful exactly while Otis is running.
_EXCLUSIVE_COMMANDS = frozenset({"ui", "run"})


def _configure_logging(level: str) -> None:
    """Set up console + ``LOG_DIR/otis.log`` file logging.

    The file handler is critical for the menu-bar app: when launched from a
    Login Item there's no terminal attached, so the log file is the only way
    to see what detection saw and what the menu-bar reacted to.
    """
    from logging.handlers import RotatingFileHandler

    numeric = getattr(logging, level.upper(), logging.INFO)
    # Milliseconds so a slow launch step can be attributed (the 8 s gap at the
    # 2026-10-20 login launch could not be), and the pid because copies of
    # Otis used to run concurrently and interleave in this one file.
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(process)d %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handlers go on the root so anything that logs is captured, but the
    # *level* is set on `src` alone. --log-level DEBUG used to turn on debug
    # for every library in the process: urllib3 logs each request, googleapi
    # logs whole payloads, and the 1 MB log file rotated away the Otis lines
    # the flag was turned on to read.
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(numeric)
    # Clear any handlers added by a previous call (re-runs in tests / REPL).
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console. Launched from Otis.app, stderr is run.sh's launch.log, which
    # exists for what happens before or outside this logging (a missing venv,
    # an import error, a native crash). Copying every INFO line there too would
    # grow it without bound next to the rotated otis.log, so into a file only
    # warnings and worse go. Terminals and pipes get everything.
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    if _stderr_is_a_log_file():
        console.setLevel(logging.WARNING)
    root.addHandler(console)

    # File — keep five 1 MB rotations under ~/.otis/otis.log.
    try:
        log_dir = LOG_DIR.expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "otis.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info("Logging to %s (level=%s).", log_file, level.upper())
    except Exception as exc:  # pragma: no cover (e.g. read-only HOME)
        logger.warning("Could not enable file logging: %s", exc)


def _stderr_is_a_log_file() -> bool:
    """True when stderr is a regular file — run.sh's launch.log."""
    import stat

    try:
        return stat.S_ISREG(os.fstat(sys.stderr.fileno()).st_mode)
    except Exception:  # pytest's capture, a closed stderr
        return False


def _install_crash_logging() -> None:
    """Route anything uncaught into otis.log instead of the discarded stderr.

    Errors that escape a rumps callback are still printed by rumps itself;
    since run.sh now keeps a launch.log, those land there.
    """
    global _crash_log

    def log_uncaught(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        if getattr(exc, "_otis_logged", False):
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def log_uncaught_thread(args: Any) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        logger.critical(
            "Uncaught exception in thread %s", name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = log_uncaught
    import threading

    threading.excepthook = log_uncaught_thread

    # A native crash takes the Python logging down with it; faulthandler
    # writes the Python stack straight to a file descriptor. Leave an
    # existing destination alone (pytest installs its own).
    import faulthandler

    if not faulthandler.is_enabled():
        try:
            log_dir = LOG_DIR.expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            _crash_log = open(log_dir / "crash.log", "a", encoding="utf-8")
            faulthandler.enable(file=_crash_log, all_threads=True)
        except Exception:  # pragma: no cover
            logger.debug("Could not enable faulthandler", exc_info=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="otis", description="Otis meeting transcriber")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (defaults to bundled config/default_config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser(
        "check-audio",
        help="Print BlackHole status and audio device list.",
    )
    sub.add_parser(
        "doctor",
        help="Check everything Otis depends on (read-only, no network).",
    )
    sub.add_parser(
        "run",
        help="Start the headless daemon (detection + auto-record on detection).",
    )
    sub.add_parser(
        "ui",
        help="Start the menu-bar app (default — manual recording controls).",
    )
    prune = sub.add_parser(
        "prune-models",
        help="Delete cached Whisper models other than the configured one.",
    )
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )

    # Legacy back-compat: ``--check-audio`` without a subcommand.
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _prune_models(cfg: Config, *, dry_run: bool = False) -> int:
    """Reclaim disk from Whisper models the configured setup no longer uses.

    mlx-whisper never removes a downloaded model, so switching model once
    leaves the old weights behind forever. Deliberately an explicit command:
    silently deleting a 1.4 GB download the user might switch back to trades a
    long re-download for disk they may not need.
    """
    from src.storage.model_cache import list_cached_models, prune_models

    active = str(cfg.get("transcription", "model", default="small"))
    cached = list_cached_models()
    if not cached:
        print("No Whisper models are cached.")
        return 0

    print(f"Configured model: {active}\n")
    for model in cached:
        mark = "keep  " if model.name.lower() == active.lower() else "remove"
        print(f"  {mark}  {model.name:10s} {model.size_mb:7.0f} MB")

    removed = prune_models(active, dry_run=dry_run)
    if not removed:
        print("\nNothing to remove.")
        return 0
    total = sum(m.size_mb for m in removed)
    if dry_run:
        print(f"\nWould free {total:.0f} MB. Re-run without --dry-run to do it.")
    else:
        print(f"\nFreed {total:.0f} MB.")
    return 0


#: check-audio's exit code per routing verdict. 2 is taken by config errors.
_ROUTING_EXIT_CODES = {
    RoutingState.OK: 0,
    RoutingState.NOT_ROUTED: 1,
    RoutingState.INAUDIBLE: 3,
    RoutingState.UNKNOWN: 3,
}


def _print_audio_status(_cfg: Config) -> int:
    """List devices, then answer both questions: is it installed, is it wired?

    This used to stop at the structural check and exit 0 on it, printing a ✓
    while the Mac's output bypassed BlackHole (measured 2026-10-25). Exit
    codes: 0 routed, 1 not installed or not routed, 3 routed with no speakers
    or not determinable.
    """
    device_manager = DeviceManager()
    status = verify_blackhole_setup(device_manager)

    print("Audio devices:")
    for d in device_manager.devices:
        flags = []
        if d.is_input:
            flags.append("in")
        if d.is_output:
            flags.append("out")
        print(f"  [{d.index:>2}] {d.name!r}  ({'/'.join(flags) or '?'})")
    print()
    print(format_setup_instructions(status))
    if not status.ok:
        return 1

    routing = check_output_routing()
    print()
    print(f"{'✓' if routing.ok else '✗'} Routing: {routing.message()}")
    return _ROUTING_EXIT_CODES.get(routing.state, 3)


def _run_daemon(cfg: Config) -> int:
    from src.pipeline import build_pipeline

    pipeline = build_pipeline(cfg)
    daemon = OtisDaemon(cfg, transcription_pipeline=pipeline)
    return daemon.run()


def _run_menubar(cfg: Config, *, surface_request_path: Path | None = None) -> int:
    """Build the menu-bar app from config and start its event loop."""
    from src.detection.calendar_poller import build_pollers_from_config
    from src.detection.detector import MeetingDetector
    from src.detection.process_monitor import (
        ProcessMonitor,
        nsworkspace_process_iter,
    )
    from src.pipeline import (
        build_pipeline,
        make_recorder_factory,
        make_transcription_handler,
    )
    from src.storage.audio_retention import AudioRetentionManager
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager, NotificationType

    # ---- detection ---------------------------------------------------------
    pm_cfg = cfg.get("detection", "process_monitor", default={}) or {}
    process_monitor = None
    if pm_cfg.get("enabled", True):
        process_monitor = ProcessMonitor(
            whitelisted_apps=list(pm_cfg.get("whitelisted_apps", [])),
            blacklisted_apps=list(pm_cfg.get("blacklisted_apps", [])),
            poll_interval_seconds=float(pm_cfg.get("poll_interval_seconds", 5)),
            # rumps runs a real main run loop, which is what keeps
            # NSWorkspace.runningApplications fresh — so the menu bar can use
            # the ~20x cheaper application source. `otis run` cannot; see
            # nsworkspace_process_iter().
            process_iter=nsworkspace_process_iter,
        )

    cal_cfg = cfg.get("detection", "calendar", default={}) or {}
    calendar_pollers = []
    if cal_cfg.get("enabled", True):
        try:
            calendar_pollers = build_pollers_from_config(cal_cfg)
        except Exception as exc:
            logger.warning("Calendar pollers disabled: %s", exc)

    detector = MeetingDetector(
        process_monitor=process_monitor,
        calendar_pollers=calendar_pollers,
    )

    # ---- recording + transcription pipeline --------------------------------
    pipeline = build_pipeline(cfg)
    recorder_factory = make_recorder_factory(cfg)

    # ---- audio retention ---------------------------------------------------
    retention = AudioRetentionManager(
        audio_dir=pipeline.audio_dir,
        transcript_store=pipeline.store,
        retention_days=int(cfg.get("storage", "audio_retention_days", default=30)),
    )
    retention.start_periodic()

    notifications = NotificationManager()

    # The progress sink is late-bound: the handler is built before the app
    # exists, but transcriptions can only start once app.run() is live.
    app_ref: list[Any] = []

    def on_progress_pct(pct: int) -> None:
        if app_ref:
            app_ref[0].notify_transcription_progress(pct)

    transcription_handler = make_transcription_handler(
        pipeline, on_progress_pct=on_progress_pct
    )

    # ---- local web UI -------------------------------------------------------
    web_host = str(cfg.get("web", "host", default="127.0.0.1"))
    web_port = int(cfg.get("web", "port", default=8765))
    try:
        from src.web.server import serve_in_background

        serve_in_background(pipeline.store, host=web_host, port=web_port,
                            audio_dir=pipeline.audio_dir)
        logger.info("Web UI listening on http://%s:%d", web_host, web_port)
    except OSError as exc:
        logger.warning(
            "Web UI disabled: could not bind %s:%d (%s)", web_host, web_port, exc
        )
        notifications.notify(
            NotificationType.ERROR,
            "Web UI unavailable",
            f"Port {web_port} is busy — transcripts still save normally.",
            force=True,
        )
    except Exception:
        logger.exception("Web UI failed to start; continuing without it")

    app = MenuBarApp(
        config=cfg,
        detector=detector,
        recorder_factory=recorder_factory,
        notifications=notifications,
        transcription_handler=transcription_handler,
        transcript_store=pipeline.store,
        surface_request_path=surface_request_path,
    )
    app_ref.append(app)
    try:
        app.run()
    finally:
        retention.stop()
        pipeline.shutdown()
    return 0


def _report_bad_config(exc: BaseException, *, gui: bool = False) -> int:
    """Tell the user their config is wrong, somewhere they will see it."""
    message = str(exc)
    logger.error("Config error: %s", message)
    print(f"Otis could not start — your config is not usable.\n\n{message}\n",
          file=sys.stderr)
    print("Edit ~/.otis/config.yaml, or delete it to fall back to the "
          "bundled defaults.", file=sys.stderr)
    # A dialog only on the menu-bar path — that is the Login Item case the
    # audit describes, where there is no terminal to read the text above. On
    # any CLI subcommand a modal that blocks until someone clicks it would be
    # worse than the traceback it replaces.
    if gui:
        try:
            import rumps

            rumps.alert(
                title="Otis could not start",
                message=f"{message}\n\nEdit ~/.otis/config.yaml, or delete it "
                        f"to use the bundled defaults.",
                ok="OK",
            )
        except Exception:  # pragma: no cover (headless / no rumps)
            logger.debug("Could not show a config dialog", exc_info=True)
    return 2


def _bring_forward(pid: int) -> None:
    """Activate another process so a modal it is stuck in comes to the front."""
    try:
        from AppKit import (
            NSApplicationActivateIgnoringOtherApps,
            NSRunningApplication,
        )

        running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if running is not None:
            running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
    except Exception:  # pragma: no cover (no AppKit)
        logger.debug("Could not bring pid %s forward", pid, exc_info=True)


def _notify_from_second_launch(title: str, message: str) -> None:
    """Best-effort toast from a process that is about to exit."""
    try:
        from src.ui.notifications import NotificationManager, NotificationType

        NotificationManager().notify(NotificationType.ERROR, title, message, force=True)
    except Exception:  # pragma: no cover (headless)
        logger.debug("Could not post the second-launch notification", exc_info=True)


def _defer_to_running_instance(command: str, run_dir: Path) -> int:
    """Another Otis holds the lock: never start a second copy, but never be silent."""
    holder = single_instance.read_holder(run_dir / single_instance.LOCK_NAME)
    who = f"pid {holder.pid}" if holder.pid else "unknown pid"
    logger.warning(
        "Otis is already running (%s, %s); not starting a second copy for '%s'.",
        who, holder.command or "?", command,
    )
    if command == "run":
        print(f"Otis is already running ({who}). Quit it before starting the "
              f"headless daemon.", file=sys.stderr)
        return 1
    if holder.command == "run":
        _notify_from_second_launch(
            "Otis daemon is running",
            f"The headless daemon ({who}) is recording meetings. Stop it before "
            f"opening the menu-bar app.",
        )
        return 1

    surface = run_dir / single_instance.SURFACE_NAME
    single_instance.request_surface(surface)
    deadline = time.monotonic() + SURFACE_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not surface.exists():
            logger.info("The running copy (%s) answered; it is showing its menu.", who)
            return 0
        time.sleep(0.05)

    # Nobody consumed the request: that copy is alive (it holds the lock) but
    # its main run loop is not turning. Say so, and try to raise whatever it is
    # stuck behind. Withdraw the request so its menu does not pop open later.
    single_instance.take_surface_request(surface)
    logger.warning(
        "The running copy (%s) did not respond within %.1fs; its main thread "
        "looks frozen.", who, SURFACE_WAIT_SECONDS,
    )
    if holder.pid:
        _bring_forward(holder.pid)
    _notify_from_second_launch(
        "Otis is running but not responding",
        f"The copy already running ({who}) did not answer. If a dialog is open, "
        f"answer it; otherwise Force Quit it in Activity Monitor (or "
        f"`kill -9 {holder.pid or 'PID'}`) and open Otis again.",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    _install_crash_logging()

    # Every mode honours ~/.otis/config.yaml — the settings toggled in the
    # menu bar (model, whitelist, working hours) apply to the headless
    # daemon too, instead of silently reverting to bundled defaults.
    command = args.command or ("check-audio" if args.check_audio else "ui")
    logger.info(
        "Otis starting: pid=%d ppid=%d command=%s python=%s cwd=%s",
        os.getpid(), os.getppid(), command, sys.executable, os.getcwd(),
    )

    if command == "doctor":
        # Loads and reports the config itself: a broken config is a finding
        # here, not a reason to stop.
        from src import doctor

        return doctor.main(args.config, run_dir=LOG_DIR)
    if command not in _EXCLUSIVE_COMMANDS:
        return _dispatch(args, command)

    # Before the config, the pipeline, the web bind and any toast: a duplicate
    # launch should cost ~100 ms and touch nothing the running copy owns.
    run_dir = LOG_DIR.expanduser()
    lock = single_instance.acquire(run_dir / single_instance.LOCK_NAME, command=command)
    if lock is None:
        return _defer_to_running_instance(command, run_dir)
    try:
        return _dispatch(args, command, surface_request_path=run_dir / single_instance.SURFACE_NAME)
    except Exception as exc:
        # Launched from Otis.app nothing reads stderr, so this line is the
        # only record of why the menu bar never appeared.
        logger.critical("Otis failed to start or crashed (%s)", command, exc_info=True)
        exc._otis_logged = True  # type: ignore[attr-defined]  # excepthook: don't log it twice
        raise
    finally:
        lock.release()


def _dispatch(
    args: argparse.Namespace, command: str, *, surface_request_path: Path | None = None
) -> int:
    try:
        cfg = load_user_config(args.config)
        validate(dict(cfg.raw))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        # Audit E9: an invalid model raised out of build_pipeline with no
        # handler, so launching from a Login Item produced no menu bar and a
        # traceback on a stdout nobody reads. Say what is wrong, where.
        return _report_bad_config(exc, gui=(command == "ui"))
    logger.info("Loaded config; storage at %s", cfg.get("storage", "transcript_dir"))

    if command == "prune-models":
        return _prune_models(cfg, dry_run=bool(getattr(args, "dry_run", False)))
    if command == "check-audio":
        return _print_audio_status(cfg)
    if command == "run":
        return _run_daemon(cfg)
    if command == "ui":
        return _run_menubar(cfg, surface_request_path=surface_request_path)

    print("Otis — usage:")
    print("  otis                    start the menu-bar app (default)")
    print("  otis ui                 same — explicit")
    print("  otis check-audio        verify BlackHole + audio devices")
    print("  otis doctor             check everything Otis depends on")
    print("  otis run                headless detection + auto-record daemon")
    print("  otis --help             full options")
    return 0


if __name__ == "__main__":
    sys.exit(main())
