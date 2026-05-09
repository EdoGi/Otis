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

from src.audio.blackhole_check import format_setup_instructions, verify_blackhole_setup
from src.audio.devices import DeviceManager
from src.config import Config, load_config, load_user_config
from src.daemon import OtisDaemon

logger = logging.getLogger("otis")


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
    daemon = OtisDaemon(cfg)
    return daemon.run()


def _run_menubar(cfg: Config) -> int:
    """Build the menu-bar app from config and start its event loop."""
    from pathlib import Path

    from src.audio.recorder import DualStreamRecorder
    from src.detection.calendar_poller import build_pollers_from_config
    from src.detection.detector import MeetingDetector
    from src.detection.process_monitor import ProcessMonitor
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    pm_cfg = cfg.get("detection", "process_monitor", default={}) or {}
    process_monitor = None
    if pm_cfg.get("enabled", True):
        process_monitor = ProcessMonitor(
            whitelisted_apps=list(pm_cfg.get("whitelisted_apps", [])),
            blacklisted_apps=list(pm_cfg.get("blacklisted_apps", [])),
            poll_interval_seconds=float(pm_cfg.get("poll_interval_seconds", 5)),
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

    def recorder_factory(_cfg: Config) -> DualStreamRecorder:
        audio_dir = Path(_cfg.get("storage", "audio_dir", default="~/Otis/audio"))
        return DualStreamRecorder(
            audio_dir=audio_dir,
            sample_rate=int(_cfg.get("audio", "sample_rate", default=16000)),
            channels=int(_cfg.get("audio", "channels", default=1)),
            mic_device=_cfg.get("audio", "mic_device"),
            system_device=_cfg.get("audio", "system_audio_device", default="BlackHole 2ch"),
        )

    app = MenuBarApp(
        config=cfg,
        detector=detector,
        recorder_factory=recorder_factory,
        notifications=NotificationManager(),
    )
    app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # Menu-bar app picks up user overrides; the headless modes use the bundled
    # defaults to keep behaviour predictable when running over SSH / scripts.
    if args.command == "ui" or (args.command is None and not args.check_audio):
        cfg = load_user_config(args.config)
    else:
        cfg = load_config(args.config)
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
