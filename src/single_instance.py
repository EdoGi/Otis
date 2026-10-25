"""One Otis per user: an ``flock`` held for the life of the process.

Why this exists: Otis.app is an AppleScript applet that spawns
``scripts/run.sh`` and exits a few seconds later, so LaunchServices never sees
the long-running Python process. Its usual "second click brings the running
app forward" behaviour is lost, and every click started a full second copy:
8 of the 36 real launches between 2026-08-24 and 2026-10-21 found port 8765
already held by another Otis. Each copy ran its own detector, calendar
pollers, retention sweep and startup error toasts, and all of them drained the
same ``~/.otis/queue`` — so a copy started mid-transcription would re-run the
job the first one was still working on.

The lock is ``fcntl.flock`` on ``~/.otis/otis.lock``. The kernel drops it when
the process ends for any reason — Quit (``NSApp.terminate_`` calls ``exit()``),
a crash, ``kill -9`` — so a stale lock cannot outlive its owner. The file
keeps the holder's pid and command purely for messages.

A second launch does not start; it asks the running copy to show itself by
creating a ``surface`` request file that the menu bar's 100 ms drain timer
consumes. If nobody consumes it, the running copy is frozen (an unanswered
modal did this twice) and the second launch says so instead of failing
silently.
"""

from __future__ import annotations

import fcntl
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_NAME = "otis.lock"
SURFACE_NAME = "surface"


@dataclass(frozen=True)
class Holder:
    """Who holds the lock, as written into the lock file (best effort)."""

    pid: int | None
    command: str | None


class InstanceLock:
    """A held lock. Keep a reference for as long as the process should own it."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd: int | None = fd

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def acquire(path: Path, *, command: str) -> InstanceLock | None:
    """Take the lock at ``path``, or return None if another process holds it.

    Never blocks. The descriptor is opened by ``os.open``, which is not
    inherited by child processes (PEP 446), so ``afconvert`` or ``osascript``
    children cannot accidentally keep it alive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n{command}\n".encode())
    except OSError:  # pragma: no cover (full disk) — the lock itself still holds
        logger.debug("Could not record the lock holder", exc_info=True)
    return InstanceLock(path, fd)


def read_holder(path: Path) -> Holder:
    """Read the pid/command a holder wrote. Never raises."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return Holder(None, None)
    pid: int | None = None
    if lines:
        try:
            pid = int(lines[0])
        except ValueError:
            pid = None
    command = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
    return Holder(pid, command)


def request_surface(path: Path) -> None:
    """Ask the running menu bar to show itself (consumed by its drain timer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def take_surface_request(path: Path) -> bool:
    """Consume a pending request. True if there was one. Never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover
        logger.debug("Could not consume the surface request", exc_info=True)
        return False
    return True
