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
from src.config import Config, load_config, load_user_config
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
    daemon = OtisDaemon(cfg)
    return daemon.run()


def _run_menubar(cfg: Config) -> int:
    """Build the menu-bar app from config and start its event loop."""
    from pathlib import Path

    from src.audio.recorder import DualStreamRecorder
    from src.detection.calendar_poller import build_pollers_from_config
    from src.detection.detector import MeetingDetector
    from src.detection.process_monitor import ProcessMonitor
    from src.storage.audio_retention import AudioRetentionManager
    from src.storage.transcript_store import TranscriptStore
    from src.transcription.processor import (
        MeetingSnapshot,
        RecordingSession,
        TranscriptProcessor,
    )
    from src.transcription.whisper_engine import WhisperEngine
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    # ---- detection ---------------------------------------------------------
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

    # ---- audio recording ---------------------------------------------------
    audio_dir = Path(cfg.get("storage", "audio_dir", default="~/Otis/audio")).expanduser()

    def recorder_factory(_cfg: Config) -> DualStreamRecorder:
        return DualStreamRecorder(
            audio_dir=audio_dir,
            sample_rate=int(_cfg.get("audio", "sample_rate", default=16000)),
            channels=int(_cfg.get("audio", "channels", default=1)),
            mic_device=_cfg.get("audio", "mic_device"),
            system_device=_cfg.get("audio", "system_audio_device", default="BlackHole 2ch"),
        )

    # ---- transcription pipeline -------------------------------------------
    transcript_dir = Path(
        cfg.get("storage", "transcript_dir", default="~/Otis/transcripts")
    ).expanduser()
    store = TranscriptStore(transcript_dir)
    engine = WhisperEngine(
        model_name=str(cfg.get("transcription", "model", default="small")),
    )
    processor = TranscriptProcessor(
        engine=engine,
        store=store,
        audio_dir=audio_dir,
        model_name=engine.model_name,
    )

    # ---- audio retention ---------------------------------------------------
    retention = AudioRetentionManager(
        audio_dir=audio_dir,
        transcript_store=store,
        retention_days=int(cfg.get("storage", "audio_retention_days", default=30)),
    )
    retention.start_periodic()

    # ---- transcription handler the menu bar calls on Stop ------------------
    notifications = NotificationManager()

    def transcription_handler(metadata: dict[str, Any]) -> None:
        """Bridge from the menu bar's Stop & Transcribe to the real pipeline."""
        session = RecordingSession.from_recorder_metadata(metadata, audio_dir=audio_dir)
        meeting_dict = metadata.get("_meeting") or {}
        meeting = MeetingSnapshot(
            title=meeting_dict.get("title"),
            app=meeting_dict.get("app"),
            participants=list(meeting_dict.get("participants") or []),
            meeting_link=meeting_dict.get("meeting_link"),
            calendar_event_id=meeting_dict.get("calendar_event_id"),
        )
        language = metadata.get("_language")
        # process() is synchronous; the menu bar already runs us in a worker
        # thread, so we stay there to keep PROCESSING state coherent.
        processor.process(session, meeting=meeting, language=language)

    app = MenuBarApp(
        config=cfg,
        detector=detector,
        recorder_factory=recorder_factory,
        notifications=notifications,
        transcription_handler=transcription_handler,
        transcript_store=store,
    )
    try:
        app.run()
    finally:
        retention.stop()
        engine.shutdown()
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
