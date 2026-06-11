"""Otis entry point.

Two top-level commands:

* ``otis check-audio`` (or ``otis --check-audio`` for back-compat) — print the
  CoreAudio device list and the BlackHole / multi-output status. Exits 0 if
  the audio setup is healthy.
* ``otis run`` — start the foreground daemon: detection + auto-record on
  meeting detection. Stays in the foreground; Ctrl-C to stop.

Later phases will replace the daemon with a proper menu-bar app and add the
transcription, storage, web, and MCP servers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.audio.blackhole_check import format_setup_instructions, verify_blackhole_setup
from src.audio.devices import DeviceManager
from src.config import Config, load_user_config
from src.daemon import OtisDaemon

logger = logging.getLogger("otis")


def _configure_logging(level: str) -> None:
    """Set up console + ~/.otis/otis.log file logging.

    The file handler is critical for the menu-bar app: when launched from a
    Login Item there's no terminal attached, so the log file is the only way
    to see what detection saw and what the menu-bar reacted to.
    """
    from logging.handlers import RotatingFileHandler

    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric)
    # Clear any handlers added by a previous call (re-runs in tests / REPL).
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # File — keep five 1 MB rotations under ~/.otis/otis.log.
    try:
        log_dir = Path("~/.otis").expanduser()
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
        "run",
        help="Start the headless daemon (detection + auto-record on detection).",
    )
    sub.add_parser(
        "ui",
        help="Start the menu-bar app (default — manual recording controls).",
    )

    # Legacy back-compat: ``--check-audio`` without a subcommand.
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _print_audio_status(_cfg: Config) -> int:
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
    return 0 if status.ok else 1


def _run_daemon(cfg: Config) -> int:
    from src.pipeline import build_pipeline

    pipeline = build_pipeline(cfg)
    daemon = OtisDaemon(cfg, transcription_pipeline=pipeline)
    return daemon.run()


def _run_menubar(cfg: Config) -> int:
    """Build the menu-bar app from config and start its event loop."""
    from src.detection.calendar_poller import build_pollers_from_config
    from src.detection.detector import MeetingDetector
    from src.detection.process_monitor import ProcessMonitor
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
    mic_cfg = cfg.get("detection", "mic_activation", default={}) or {}
    process_monitor = None
    if pm_cfg.get("enabled", True):
        process_monitor = ProcessMonitor(
            whitelisted_apps=list(pm_cfg.get("whitelisted_apps", [])),
            blacklisted_apps=list(pm_cfg.get("blacklisted_apps", [])),
            poll_interval_seconds=float(pm_cfg.get("poll_interval_seconds", 5)),
            mic_activation_enabled=bool(mic_cfg.get("enabled", True)),
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

        serve_in_background(pipeline.store, host=web_host, port=web_port)
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
    )
    app_ref.append(app)
    try:
        app.run()
    finally:
        retention.stop()
        pipeline.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # Every mode honours ~/.otis/config.yaml — the settings toggled in the
    # menu bar (model, whitelist, working hours) apply to the headless
    # daemon too, instead of silently reverting to bundled defaults.
    cfg = load_user_config(args.config)
    logger.info("Loaded config; storage at %s", cfg.get("storage", "transcript_dir"))

    command = args.command or ("check-audio" if args.check_audio else "ui")

    if command == "check-audio":
        return _print_audio_status(cfg)
    if command == "run":
        return _run_daemon(cfg)
    if command == "ui":
        return _run_menubar(cfg)

    print("Otis — usage:")
    print("  otis                    start the menu-bar app (default)")
    print("  otis ui                 same — explicit")
    print("  otis check-audio        verify BlackHole + audio devices")
    print("  otis run                headless detection + auto-record daemon")
    print("  otis --help             full options")
    return 0


if __name__ == "__main__":
    sys.exit(main())
