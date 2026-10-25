"""The native Otis.app launcher that replaces the AppleScript applet.

The osacompile applet spent 4-14 s before it even ran run.sh (live: 4.1 s on
2026-10-21, 14.5 s at the 2026-10-20 login, 7.5 s on 2026-10-25; a bare
applet takes 0.7-2.7 s on this Mac), with nothing visible meanwhile — the
largest slice of "slow to open". The replacement is a tiny C program that
spawns run.sh detached and exits. Its stderr goes to ~/.otis/launch.log, so
even a launch that dies before run.sh is read (a missing file, a TCC denial
on ~/Documents) leaves a line — the applet sent everything to /dev/null.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "launcher" / "otis_launcher.c"

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")


def _build(tmp_path: Path, run_sh: Path) -> Path:
    exe = tmp_path / "Otis"
    subprocess.run(
        ["clang", "-O2", "-Wall", "-Werror", f'-DOTIS_RUN_SH="{run_sh}"',
         "-o", str(exe), str(SOURCE)],
        check=True, capture_output=True, text=True,
    )
    return exe


def _run(exe: Path, home: Path) -> tuple[subprocess.CompletedProcess, float]:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    t0 = time.monotonic()
    proc = subprocess.run([str(exe)], env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    return proc, time.monotonic() - t0


def _wait_for(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.02)
    return False


def test_it_starts_the_script_detached_and_returns_at_once(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    script = tmp_path / "run.sh"
    # Sleeps first: proves the launcher does not wait for Otis to finish.
    script.write_text(f'#!/bin/bash\nsleep 1\necho "$OTIS_LAUNCHED_BY_APP $$" > "{marker}"\n')
    exe = _build(tmp_path, script)
    proc, took = _run(exe, tmp_path)
    assert proc.returncode == 0
    assert took < 2.0, f"launcher took {took:.2f}s (the applet took 4-14 s)"
    assert _wait_for(marker), "the script never ran"
    flag, _pid = marker.read_text().split()
    assert flag == "1", "run.sh must know the app started it"


def test_a_script_that_cannot_run_leaves_a_line_in_the_launch_log(tmp_path: Path) -> None:
    exe = _build(tmp_path, tmp_path / "missing" / "run.sh")
    proc, _took = _run(exe, tmp_path)
    assert proc.returncode == 0
    log = tmp_path / ".otis" / "launch.log"
    assert _wait_for(log)
    assert "run.sh" in log.read_text()


def test_the_child_is_in_its_own_session(tmp_path: Path) -> None:
    """Detached (setsid), so nothing about the launcher's exit reaches it."""
    out = tmp_path / "sid.txt"
    script = tmp_path / "run.sh"
    # exec keeps bash's pid, so "sid == pid" means bash was the session leader.
    script.write_text("#!/bin/bash\nexec /usr/bin/python3 -c "
                      f"'import os; print(os.getsid(0), os.getpid())' > \"{out}\"\n")
    exe = _build(tmp_path, script)
    _run(exe, tmp_path)
    assert _wait_for(out)
    sid, pid = out.read_text().split()
    assert sid == pid, "the child should lead its own session"


def test_run_sh_keeps_its_launch_log_when_the_app_started_it(tmp_path: Path) -> None:
    """The launcher hands run.sh a real stderr (the log), so the old
    "both streams are /dev/null" gate alone would no longer fire."""
    env = {**os.environ, "HOME": str(tmp_path), "OTIS_LAUNCHED_BY_APP": "1"}
    with open(tmp_path / "err.txt", "w") as err:
        proc = subprocess.run(["bash", str(REPO / "scripts" / "run.sh"), "--help"],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=err, env=env, timeout=60)
    assert proc.returncode == 0
    text = (tmp_path / ".otis" / "launch.log").read_text()
    assert "Otis launching" in text and "usage: otis" in text


@pytest.mark.skipif(shutil.which("iconutil") is None, reason="needs macOS iconutil")
def test_build_app_produces_a_signed_native_bundle(tmp_path: Path) -> None:
    import plistlib

    env = {**os.environ, "OTIS_DIST_DIR": str(tmp_path)}
    subprocess.run(["bash", str(REPO / "scripts" / "build_app.sh")], env=env,
                   check=True, capture_output=True, text=True, timeout=180)
    app = tmp_path / "Otis.app"
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "Otis"
    assert info["CFBundleIdentifier"] == "com.tenderstrike.otis"
    assert info["LSUIElement"] is True
    assert "OSAAppletStayOpen" not in info, "no AppleScript left in the bundle"
    exe = app / "Contents" / "MacOS" / "Otis"
    assert "Mach-O" in subprocess.run(["file", str(exe)], capture_output=True, text=True).stdout
    assert not (app / "Contents" / "Resources" / "Scripts").exists()
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    # The compiled-in path is this checkout's run.sh.
    assert str(REPO / "scripts" / "run.sh").encode() in exe.read_bytes()
