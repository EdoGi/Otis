"""Shared pytest fixtures.

Most tests don't have access to a real CoreAudio stack, so we install a fake
``sounddevice`` module into ``sys.modules`` before the code under test imports
it. Each test gets a fresh fake via the ``fake_sounddevice`` fixture.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeInputStream:
    """Stand-in for ``sounddevice.InputStream``.

    Synthesises 10 ms chunks of silence on a daemon thread so the writer thread
    actually receives data and stamps a monotonic anchor.
    """

    samplerate: int
    channels: int
    dtype: str
    device: int
    callback: Callable[..., None]
    blocksize: int = 0
    latency: str | float | None = None
    _running: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def close(self) -> None:
        self.stop()

    def _loop(self) -> None:
        # Lazy import — keeps numpy optional for environments that don't need it.
        try:
            import numpy as np

            block = np.zeros((self.samplerate // 100, self.channels), dtype="int16")
        except Exception:
            block = bytes(self.samplerate // 100 * self.channels * 2)

        while self._running.is_set():
            try:
                self.callback(block, len(block) if hasattr(block, "__len__") else 0, None, None)
            except Exception:  # pragma: no cover
                pass
            time.sleep(0.01)


@dataclass
class FakeDevice:
    name: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: float = 48000.0
    hostapi: int = 0


def make_fake_sd_module(
    *,
    devices: list[dict[str, Any]] | None = None,
    default_input: int | None = None,
    default_output: int | None = None,
) -> types.ModuleType:
    """Build a fake ``sounddevice`` module bound to the given device list."""
    if devices is None:
        devices = [
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "BlackHole 2ch",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "Otis Multi-Output Device",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
        ]

    if default_input is None:
        default_input = next(
            (i for i, d in enumerate(devices) if d["max_input_channels"] > 0), -1
        )
    if default_output is None:
        default_output = next(
            (i for i, d in enumerate(devices) if d["max_output_channels"] > 0), -1
        )

    mod = types.ModuleType("sounddevice")

    def query_devices(_query: Any = None) -> list[dict[str, Any]]:
        return [dict(d) for d in devices]

    class _Default:
        device = (default_input, default_output)

    mod.query_devices = query_devices  # type: ignore[attr-defined]
    mod.default = _Default()  # type: ignore[attr-defined]
    mod.InputStream = FakeInputStream  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``sounddevice`` module for the duration of one test."""
    mod = make_fake_sd_module()
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return mod


@pytest.fixture
def fake_sounddevice_no_blackhole(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Same as ``fake_sounddevice`` but BlackHole isn't installed."""
    mod = make_fake_sd_module(
        devices=[
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return mod


@pytest.fixture
def fake_sounddevice_no_multi_output(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """BlackHole installed but no multi-output device created."""
    mod = make_fake_sd_module(
        devices=[
            {
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "BlackHole 2ch",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return mod


@pytest.fixture(autouse=True)
def _never_write_the_users_log(tmp_path_factory, monkeypatch):
    """Keep the suite out of ``~/.otis/otis.log``.

    Tests that go through ``main()`` or call ``_configure_logging`` directly
    used to append to the real file. otis.log rotates at 1 MB keeping five
    backups, so a full run did not merely add noise — it evicted the user's
    actual meeting history. Measured before this fixture: 63% of the retained
    log was pytest fixtures.
    """
    import src.main as main_mod

    monkeypatch.setattr(
        main_mod, "LOG_DIR", tmp_path_factory.mktemp("otis-log"), raising=True
    )
    # main() installs process-wide excepthooks that log into LOG_DIR; put the
    # originals back so one test's hooks never outlive it.
    import threading

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)


@pytest.fixture(autouse=True)
def _never_read_the_users_config(tmp_path_factory, monkeypatch):
    """Keep the suite off ``~/.otis/config.yaml`` (a review finding).

    Tests that go through main() load the user config; an invalid one made
    every 'ui' test take the bad-config path — and rumps.alert, a modal that
    blocks the run until someone clicks it. The suite's result depended on
    the developer's personal settings. Tests that need a config set
    USER_CONFIG_PATH themselves.
    """
    import src.config as config_mod

    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH",
                        tmp_path_factory.mktemp("otis-cfg") / "config.yaml")
    try:
        import rumps
    except ImportError:  # pragma: no cover (not macOS)
        return
    # Never a real modal under test; tests asserting "no modal" install a
    # raising stub of their own on top of this.
    monkeypatch.setattr(rumps, "alert", lambda *a, **k: 1)


@pytest.fixture(autouse=True)
def _microphone_is_authorized(monkeypatch):
    """Never let a test raise the real macOS microphone prompt (it would be
    asked on behalf of the Python running pytest). Tests of the permission
    flow install their own stubs."""
    from src.audio import mic_permission

    monkeypatch.setattr(mic_permission, "microphone_authorization",
                        lambda: mic_permission.AUTHORIZED)
    monkeypatch.setattr(mic_permission, "request_microphone_access", lambda cb: True)


@pytest.fixture(autouse=True)
def _restore_signal_handling():
    """MenuBarApp.run() (and OtisDaemon.run()) install SIGTERM/SIGHUP/SIGINT
    handlers and a wakeup fd for the whole process. Left behind, a later
    SIGTERM to the pytest run would call a dead app's quit — NSApp.terminate_,
    exit status 0, no summary — and a cancelled gate would read green."""
    import signal

    sigs = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    saved = {s: signal.getsignal(s) for s in sigs}
    wakeup = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(wakeup)
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)
    signal.set_wakeup_fd(wakeup)
