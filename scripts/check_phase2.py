"""End-to-end Phase 1+2 self-check.

Runs every check from the review docs in one place and prints a tidy
``✓ / ✗`` report. Exits 0 if everything is green, 1 otherwise.

Usage:
    python scripts/check_phase2.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def section(self, title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))

    def check(self, label: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {mark} {label}{DIM} — {detail}{RESET}" if detail else f"  {mark} {label}")
        if not ok:
            self.failures.append(label)

    def warn(self, label: str, detail: str) -> None:
        print(f"  {YELLOW}!{RESET} {label}{DIM} — {detail}{RESET}")
        self.warnings.append(label)

    def summary(self) -> int:
        print()
        if self.failures:
            print(f"{RED}{len(self.failures)} check(s) failed:{RESET}")
            for f in self.failures:
                print(f"  • {f}")
            return 1
        if self.warnings:
            print(f"{GREEN}All checks passed{RESET} ({len(self.warnings)} warning(s)).")
        else:
            print(f"{GREEN}All checks passed.{RESET}")
        return 0


# ============================================================================
# Individual checks
# ============================================================================
def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    return v.major == 3 and v.minor >= 10, f"running {sys.executable} ({v.major}.{v.minor}.{v.micro})"


def check_in_venv() -> tuple[bool, str]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return in_venv, f"sys.prefix={sys.prefix}"


def check_import(module: str) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        version = getattr(mod, "__version__", "unknown")
        return True, f"{module} {version}"

    return _check


def check_no_openai_whisper() -> tuple[bool, str]:
    try:
        importlib.import_module("whisper")
        return False, "the openai-whisper package is installed (forbidden — must use mlx-whisper)"
    except ImportError:
        return True, "openai-whisper not installed"


def check_blackhole() -> tuple[bool, str]:
    from src.audio.blackhole_check import verify_blackhole_setup

    status = verify_blackhole_setup()
    if status.ok:
        return True, f"installed + multi-output ({status.multi_output_device_name})"
    issues = "; ".join(status.issues) or "see --check-audio"
    return False, issues


def check_default_mic() -> tuple[bool, str]:
    from src.audio.devices import DeviceManager

    mic = DeviceManager().get_default_mic()
    if mic is None:
        return False, "no input devices found"
    return True, f"using {mic.name!r} (idx {mic.index})"


def check_coreaudio_probe() -> tuple[bool, str]:
    from src.audio.coreaudio_probe import (
        get_default_input_device_id,
        is_default_input_running,
    )

    dev_id = get_default_input_device_id()
    running = is_default_input_running()
    return True, f"device id={dev_id}, in_use={running}"


def check_config_loads() -> tuple[bool, str]:
    from src.config import Config

    cfg = Config.load()
    sr = cfg.audio.sample_rate
    if sr != 16000:
        return False, f"audio.sample_rate={sr} (expected 16000)"
    return True, f"sample_rate={sr}, calendar_provider={cfg.detection.calendar.provider}"


def check_calendar_accounts_configured() -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        from src.config import Config

        cfg = Config.load()
        accounts = cfg.get("detection", "calendar", "accounts")
        if not accounts:
            return False, "no detection.calendar.accounts configured"
        labels = [a["label"] for a in accounts]
        return True, f"{len(accounts)} account(s): {labels}"

    return _check


def check_calendar_token(label: str, token_path: str) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        path = Path(os.path.expanduser(token_path))
        if not path.exists():
            return False, f"token not found at {path}"
        # Try to actually authenticate + fetch.
        from src.detection.calendar_poller import GoogleCalendarPoller

        poller = GoogleCalendarPoller(
            credentials_path="~/.otis/credentials.json",
            token_path=str(path),
        )
        try:
            poller.authenticate(headless=True)
        except Exception as exc:
            return False, f"auth failed: {exc}"
        try:
            events = poller.fetch_today_events()
        except Exception as exc:
            return False, f"fetch failed: {exc}"
        return True, f"token OK, {len(events)} event(s) today"

    return _check


def check_pytest() -> tuple[bool, str]:
    if shutil.which("pytest") is None:
        return False, "pytest not found in PATH"
    proc = subprocess.run(
        ["pytest", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    last_lines = (proc.stdout or proc.stderr).strip().splitlines()
    summary_line = last_lines[-1] if last_lines else "(no output)"
    return proc.returncode == 0, summary_line


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    r = Reporter()

    r.section("Environment")
    r.check("Python ≥3.10", check_python_version)
    r.check("Running inside a virtualenv", check_in_venv)

    r.section("Phase 1 — Python deps")
    for mod in ("sounddevice", "numpy", "yaml", "psutil", "objc"):
        r.check(f"import {mod}", check_import(mod))
    r.check("openai-whisper NOT installed", check_no_openai_whisper)

    r.section("Phase 1 — audio")
    r.check("Config loads with expected defaults", check_config_loads)
    r.check("Default mic resolved", check_default_mic)
    r.check("BlackHole + Multi-Output device", check_blackhole)
    r.check("CoreAudio probe works", check_coreaudio_probe)

    r.section("Phase 2 — detection deps")
    for mod in ("googleapiclient.discovery", "google.auth.transport.requests",
                "google_auth_oauthlib.flow"):
        r.check(f"import {mod}", check_import(mod))

    r.section("Phase 2 — calendar accounts")
    r.check("detection.calendar.accounts populated", check_calendar_accounts_configured())
    try:
        from src.config import Config
        accounts = Config.load().get("detection", "calendar", "accounts") or []
    except Exception:
        accounts = []
    if not accounts:
        r.warn("Calendar tokens", "skipped — accounts list is empty")
    else:
        for entry in accounts:
            label = entry["label"]
            token_path = entry["token_path"]
            r.check(f"Account '{label}' authenticates + fetches", check_calendar_token(label, token_path))

    r.section("Tests")
    r.check("pytest -q passes", check_pytest)

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
