"""``otis doctor`` — check the things that actually break Otis on this Mac.

It replaces ``scripts/check_phase2.py``, written for the Phase 1/2 milestone
and never updated. That script checked long-settled things (openai-whisper
absent, the Phase-2 Google imports) and missed every failure this Mac has
actually had:

* system audio not routed through BlackHole — its BlackHole check was the
  structural one, green while the output bypassed BlackHole;
* the user's own config — it loaded only the bundled defaults, so a config
  that stops Otis from starting passed;
* a second Otis already running, or another app on port 8765;
* the configured Whisper model not cached (the first meeting downloads it);
* the notification identity file a uv-built venv needs, and afconvert for
  FLAC archives;
* its Python gate said ≥3.10 while the project needs 3.12, and it shelled
  out to ``pytest`` from PATH, so it was red whenever the venv was not
  activated.

Every check is read-only and makes no network request. Exit code 1 if any
check fails; warnings do not fail.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

#: The project's floor (pyproject.toml requires-python); a test keeps them equal.
MIN_PYTHON = (3, 12)

#: Imported by the app at runtime. mlx_whisper is only *located*, never
#: imported: importing it pulls in scipy and numba (~2 s).
REQUIRED_MODULES = ("sounddevice", "numpy", "yaml", "rumps", "flask", "mlx_whisper", "mcp")

AFCONVERT = Path("/usr/bin/afconvert")


class Verdict(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    label: str
    verdict: Verdict
    detail: str = ""
    fix: str = ""


_GLYPH = {Verdict.OK: "✓", Verdict.WARN: "!", Verdict.FAIL: "✗"}


# ---------------------------------------------------------------------------
# Individual checks. Each takes what it needs and returns one Check, so they
# can be tested without the machine they normally inspect.
# ---------------------------------------------------------------------------
def check_python(version: tuple[int, ...] | None = None) -> Check:
    v = tuple(version or sys.version_info[:3])
    shown = ".".join(str(p) for p in v)
    if v[:2] >= MIN_PYTHON:
        return Check("Python", Verdict.OK, f"{shown} ({sys.executable})")
    need = ".".join(str(p) for p in MIN_PYTHON)
    return Check("Python", Verdict.FAIL, f"{shown}; Otis needs {need}+",
                 "Rebuild the venv with scripts/setup.sh.")


def check_modules(find: Callable[[str], Any] = importlib.util.find_spec) -> Check:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            if find(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    if missing:
        return Check("Python packages", Verdict.FAIL, "missing: " + ", ".join(missing),
                     "Run scripts/setup.sh.")
    return Check("Python packages", Verdict.OK, f"{len(REQUIRED_MODULES)} present")


def check_mcp_version(version: Callable[[str], str] = importlib.metadata.version) -> Check:
    """mcp 2.x removed mcp.server.fastmcp, which src/mcp/server.py imports."""
    try:
        v = version("mcp")
    except importlib.metadata.PackageNotFoundError:
        return Check("MCP library", Verdict.WARN, "not installed", "Run scripts/setup.sh.")
    major = int(v.split(".")[0]) if v.split(".")[0].isdigit() else 0
    if major >= 2:
        return Check("MCP library", Verdict.FAIL, f"mcp {v}; Otis's server needs mcp<2",
                     "pip install 'mcp<2' in the venv.")
    return Check("MCP library", Verdict.OK, f"mcp {v}")


def check_config(config_path: Path | None) -> tuple[Check, Any]:
    """Load the config exactly the way main() does. Returns (check, cfg)."""
    from src.config import load_user_config
    from src.config_schema import ConfigError, validate

    try:
        cfg = load_user_config(config_path)
        warnings = validate(dict(cfg.raw))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        return (Check("Config", Verdict.FAIL, str(exc),
                      "Edit ~/.otis/config.yaml, or delete it to use the defaults."), None)
    model = cfg.get("transcription", "model", default="small")
    language = cfg.get("transcription", "language") or "auto"
    detail = f"model={model}, language={language}"
    if warnings:
        return (Check("Config", Verdict.WARN, f"{detail}; " + "; ".join(warnings)), cfg)
    return Check("Config", Verdict.OK, detail), cfg


def check_notification_identity(prefix: Path | None = None) -> Check:
    """uv's CPython is not a framework build: rumps needs a bundle id here."""
    plist = Path(prefix or sys.prefix) / "bin" / "Info.plist"
    if not plist.exists():
        return Check("Notifications", Verdict.WARN, f"{plist} missing",
                     "Re-run scripts/setup.sh — it writes this file.")
    try:
        import plistlib

        bundle_id = plistlib.loads(plist.read_bytes()).get("CFBundleIdentifier")
    except Exception as exc:
        return Check("Notifications", Verdict.WARN, f"{plist} unreadable: {exc}")
    if not bundle_id:
        return Check("Notifications", Verdict.WARN, f"{plist} has no CFBundleIdentifier",
                     "Re-run scripts/setup.sh.")
    return Check("Notifications", Verdict.OK, f"bundle id {bundle_id!r}")


def check_afconvert(path: Path = AFCONVERT) -> Check:
    if path.exists():
        return Check("FLAC encoder", Verdict.OK, str(path))
    return Check("FLAC encoder", Verdict.WARN, f"{path} missing",
                 "Recordings will be archived as uncompressed WAV.")


def check_model_cached(model: str, cached_names: list[str]) -> Check:
    from src.transcription.whisper_engine import _MODEL_REPO_MAP

    repo = _MODEL_REPO_MAP.get(model, "")
    wanted = repo.removeprefix("mlx-community/whisper-").removesuffix("-mlx") or model
    if wanted in cached_names:
        return Check("Whisper model", Verdict.OK, f"{model} cached")
    return Check("Whisper model", Verdict.WARN,
                 f"{model} is not downloaded yet (cached: {', '.join(cached_names) or 'none'})",
                 "The next transcription downloads it from huggingface.co first.")


def check_blackhole(verify: Callable[[], Any]) -> Check:
    status = verify()
    if status.ok:
        return Check("BlackHole", Verdict.OK,
                     f"{status.blackhole_device_name} + {status.multi_output_device_name}")
    return Check("BlackHole", Verdict.FAIL, "; ".join(status.issues) or "not set up",
                 "See scripts/setup_blackhole.sh.")


def check_routing(route: Callable[[], Any]) -> Check:
    from src.audio.blackhole_check import RoutingState

    status = route()
    if status.state is RoutingState.OK:
        return Check("System audio routing", Verdict.OK, status.message())
    if status.state is RoutingState.NOT_ROUTED:
        return Check("System audio routing", Verdict.FAIL, status.message())
    return Check("System audio routing", Verdict.WARN, status.message())


def check_running_instance(run_dir: Path) -> Check:
    from src import single_instance

    lock_path = run_dir / single_instance.LOCK_NAME
    lock = single_instance.acquire(lock_path, command="doctor")
    if lock is not None:
        lock.release()
        return Check("Otis process", Verdict.OK, "not running")
    holder = single_instance.read_holder(lock_path)
    return Check("Otis process", Verdict.OK,
                 f"running (pid {holder.pid}, {holder.command or 'unknown command'})")


def check_port(host: str, port: int, *, otis_running: bool) -> Check:
    """Port 8765 is Otis's when Otis runs; otherwise someone else has it."""
    label = f"Web UI port {port}"
    if otis_running:
        return Check(label, Verdict.OK, "held by the running Otis")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return Check(label, Verdict.WARN, f"{host}:{port} is in use by another program",
                         "Otis will start without its web UI; free the port or set web.port.")
    return Check(label, Verdict.OK, "free")


def check_disk(path: Path) -> Check:
    from src.storage.audio_retention import LOW_DISK_CRITICAL_BYTES, LOW_DISK_WARN_BYTES

    target = path if path.exists() else Path.home()
    free = shutil.disk_usage(target).free
    detail = f"{free / 1000**3:.0f} GB free"
    if free < LOW_DISK_CRITICAL_BYTES:
        return Check("Disk", Verdict.FAIL, detail, "Free space before the next meeting.")
    if free < LOW_DISK_WARN_BYTES:
        return Check("Disk", Verdict.WARN, detail, "Consider lowering audio_retention_days.")
    return Check("Disk", Verdict.OK, detail)


def check_calendar_accounts(cfg: Any) -> list[Check]:
    """Token files only — no OAuth refresh, no request to Google."""
    from src.detection.calendar_poller import build_pollers_from_config

    cal_cfg = cfg.get("detection", "calendar", default={}) or {}
    if not cal_cfg.get("enabled", True):
        return [Check("Calendar", Verdict.OK, "disabled in config")]
    try:
        pollers = build_pollers_from_config(cal_cfg)
    except Exception as exc:
        return [Check("Calendar", Verdict.WARN, f"not configured: {exc}")]
    checks = []
    for poller in pollers:
        label = f"Calendar ({poller.label})" if poller.label else "Calendar"
        if poller.token_path.exists():
            # Present is all this can say offline; a revoked token still
            # looks present until the next poll tries it.
            checks.append(Check(label, Verdict.OK, "token present (not verified online)"))
        else:
            checks.append(Check(
                label, Verdict.WARN, "disconnected — no meeting titles or reminders",
                "Status → Reconnect Calendar… in the menu, or scripts/setup_google_cal.sh.",
            ))
    return checks or [Check("Calendar", Verdict.WARN, "no accounts configured")]


def check_crash_log(run_dir: Path) -> Check:
    crash = run_dir / "crash.log"
    if crash.exists() and crash.stat().st_size > 0:
        from datetime import datetime

        when = datetime.fromtimestamp(crash.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return Check("Crash log", Verdict.WARN, f"{crash} has entries (last write {when})",
                     "A native crash was recorded; the Python stack is in that file.")
    return Check("Crash log", Verdict.OK, "empty")


# ---------------------------------------------------------------------------
def run_checks(config_path: Path | None = None, *, run_dir: Path | None = None) -> list[Check]:
    from src.audio.blackhole_check import check_output_routing, verify_blackhole_setup
    from src.storage.model_cache import list_cached_models

    run_dir = (run_dir or Path("~/.otis")).expanduser()
    checks = [check_python(), check_modules(), check_mcp_version()]
    config_check, cfg = check_config(config_path)
    checks.append(config_check)
    checks += [check_notification_identity(), check_afconvert()]
    checks.append(check_blackhole(verify_blackhole_setup))
    if checks[-1].verdict is Verdict.OK:
        checks.append(check_routing(check_output_routing))
    running = check_running_instance(run_dir)
    checks.append(running)
    checks.append(check_crash_log(run_dir))
    if cfg is None:
        return checks
    model = str(cfg.get("transcription", "model", default="small"))
    checks.append(check_model_cached(model, [m.name for m in list_cached_models()]))
    checks.append(check_port(
        str(cfg.get("web", "host", default="127.0.0.1")),
        int(cfg.get("web", "port", default=8765)),
        otis_running=running.detail.startswith("running"),
    ))
    checks.append(check_disk(Path(str(cfg.get("storage", "audio_dir",
                                              default="~/Otis/audio"))).expanduser()))
    checks += check_calendar_accounts(cfg)
    return checks


def render(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        line = f"  {_GLYPH[c.verdict]} {c.label}"
        if c.detail:
            line += f" — {c.detail}"
        lines.append(line)
        if c.fix and c.verdict is not Verdict.OK:
            lines.append(f"      → {c.fix}")
    fails = sum(c.verdict is Verdict.FAIL for c in checks)
    warns = sum(c.verdict is Verdict.WARN for c in checks)
    lines.append("")
    lines.append(f"{fails} failed, {warns} warning(s), "
                 f"{len(checks) - fails - warns} ok.")
    return "\n".join(lines)


def main(config_path: Path | None = None, *, run_dir: Path | None = None) -> int:
    checks = run_checks(config_path, run_dir=run_dir)
    print("Otis doctor\n")
    print(render(checks))
    return 1 if any(c.verdict is Verdict.FAIL for c in checks) else 0


__all__ = ["MIN_PYTHON", "Check", "Verdict", "main", "render", "run_checks"]
