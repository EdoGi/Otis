"""Cover for audit finding C5 — the menus were built by scanning disk inline.

``Recent Transcripts`` and ``Generate Transcript`` were both rebuilt on the
rumps main thread: at startup before ``app.run()``, and again after every
transcription. Between them that is a store scan plus two rglobs of the audio
tree plus a JSON parse per session — ~1.8 s of frozen menu bar at 1 000
transcripts — and ``find_orphan_sessions`` additionally called
``repair_if_needed`` on every recording it passed, so drawing a menu could
read 64 KB per file and rewrite WAV headers.

Scanning now happens on ``otis-menu-scan`` and the main thread only paints
what the worker hands it.
"""

from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.config import Config
from src.detection.detector import MeetingDetector
from src.storage.transcript_store import TranscriptStore
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager

MAIN = threading.current_thread()


def _wav(path: Path, *, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * frames)


def _session(audio: Path, sid: str) -> None:
    _wav(audio / f"{sid}_mic.wav")
    _wav(audio / f"{sid}_system.wav")
    (audio / f"{sid}_metadata.json").write_text(
        json.dumps({"session_id": sid}), encoding="utf-8"
    )


@pytest.fixture()
def app(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir()
    store = TranscriptStore(tmp_path / "transcripts")
    cfg = Config({
        "app": {"working_days": [0, 1, 2, 3, 4],
                "working_hours": {"start": "00:00", "end": "23:59"}},
        "transcription": {"model": "small", "language": None,
                          "defer_while_in_call": False},
        "storage": {"audio_dir": str(audio), "transcript_dir": str(store.root)},
        "detection": {"process_monitor": {"whitelisted_apps": ["zoom.us"]}},
        "web": {"host": "127.0.0.1", "port": 8765},
    })
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        transcript_store=store,
        audio_dir=audio,
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    bar._test_audio = audio        # type: ignore[attr-defined]
    bar._test_store = store        # type: ignore[attr-defined]
    return bar


def _settle(app, *, timeout: float = 5.0) -> None:
    """Wait for the scan worker to finish, then paint its result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t = app._scan_thread
        if t is None and not app._scan_pending:
            break
        time.sleep(0.005)
    else:  # pragma: no cover - only on a hang
        pytest.fail("scan worker never settled")
    app._drain_main_queue(None)


def _labels(app, key: str) -> list[str]:
    return [str(k) for k in app._mi[key].keys()]


# ---------------------------------------------------------------------------
# The main thread must not touch disk
# ---------------------------------------------------------------------------
def test_building_the_app_does_not_scan_disk(app, monkeypatch) -> None:
    """Construction happens before the drain timer exists to render anything."""
    assert _labels(app, "recent_root") == ["(scanning…)"]
    assert _labels(app, "generate_root") == ["(scanning…)"]


def test_the_scan_never_runs_on_the_main_thread(app, monkeypatch) -> None:
    threads: list[str] = []
    real = TranscriptStore.list_transcripts

    def spy(self, **kw):
        threads.append(threading.current_thread().name)
        return real(self, **kw)

    monkeypatch.setattr(TranscriptStore, "list_transcripts", spy)
    _session(app._test_audio, "orphan-1")

    app._request_menu_rescan()
    _settle(app)

    assert threads, "the scan must actually have run"
    assert MAIN.name not in threads
    assert set(threads) == {"otis-menu-scan"}


def test_a_slow_scan_does_not_block_the_caller(app, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow(self, **kw):
        started.set()
        release.wait(5.0)
        return []

    monkeypatch.setattr(TranscriptStore, "list_transcripts", slow)
    t0 = time.perf_counter()
    app._request_menu_rescan()
    elapsed = time.perf_counter() - t0

    assert started.wait(5.0)
    assert elapsed < 0.05, f"request blocked the caller for {elapsed*1000:.0f} ms"
    release.set()
    _settle(app)


def test_rendering_a_seeded_store_stays_off_the_main_thread(app) -> None:
    """The plan's responsiveness check: paint must not depend on store size."""
    store = app._test_store
    for i in range(300):
        store.save({"id": f"s-{i}", "title": f"Meeting {i}", "date": "2026-07-01",
                    "start_time": f"{i % 24:02d}:00"}, "# body\n\ntext")
    app._request_menu_rescan()
    _settle(app)

    t0 = time.perf_counter()
    app._render_recent_menu(app._scan_menus()[0])
    paint_ms = (time.perf_counter() - t0) * 1000
    # Painting five rows is in-memory work; the disk cost moved to the worker.
    assert paint_ms < 50, f"main-thread paint took {paint_ms:.1f} ms"


# ---------------------------------------------------------------------------
# What the worker finds ends up on screen
# ---------------------------------------------------------------------------
def test_scan_results_are_rendered_after_draining(app) -> None:
    _session(app._test_audio, "orphan-1")
    app._test_store.save(
        {"id": "done-1", "title": "Kickoff", "date": "2026-07-01",
         "start_time": "10:30"},
        "# Kickoff\n\nbody",
    )

    app._request_menu_rescan()
    _settle(app)

    assert any("Kickoff" in lbl for lbl in _labels(app, "recent_root"))
    assert _labels(app, "generate_root") != ["(scanning…)"]
    assert _labels(app, "generate_root") != ["(no orphan recordings)"]


def test_empty_reads_as_empty_not_as_still_scanning(app) -> None:
    app._request_menu_rescan()
    _settle(app)
    assert _labels(app, "recent_root") == ["(no transcripts yet)"]
    assert _labels(app, "generate_root") == ["(no orphan recordings)"]


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------
def test_bursts_of_requests_coalesce(app, monkeypatch) -> None:
    """25 requests must cost one worker and one extra pass, not 25 threads.

    Counting *scans* alone is not enough: the pending flag collapses those on
    its own, so a version that spawns a thread per request still scans once
    while leaking 24 threads. Count the workers.
    """
    scans: list[int] = []
    workers: set[int] = set()
    gate = threading.Event()
    real_scan = MenuBarApp._scan_menus
    real_loop = MenuBarApp._scan_loop

    def counting(self, **kw):
        scans.append(1)
        gate.wait(2.0)
        return real_scan(self, **kw)

    def spy_loop(self):
        workers.add(threading.get_ident())
        return real_loop(self)

    monkeypatch.setattr(MenuBarApp, "_scan_menus", counting)
    monkeypatch.setattr(MenuBarApp, "_scan_loop", spy_loop)
    for _ in range(25):
        app._request_menu_rescan()
    gate.set()
    _settle(app)

    assert len(workers) <= 2, f"{len(workers)} threads for 25 requests"
    # One scan in flight, plus at most one more for requests that landed
    # while it was working.
    assert len(scans) <= 2, f"{len(scans)} scans for 25 requests"


def test_a_failing_scan_leaves_the_worker_usable(app, monkeypatch) -> None:
    calls: list[int] = []

    def boom(self, **kw):
        calls.append(1)
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(MenuBarApp, "_scan_menus", boom)
    app._request_menu_rescan()
    _settle(app)
    assert calls == [1]
    assert app._scan_thread is None, "a crashed scan must not pin the worker"

    monkeypatch.undo()
    app._test_store.save(
        {"id": "after", "title": "Recovered", "date": "2026-07-02",
         "start_time": "09:00"}, "# Recovered\n\nbody")
    app._request_menu_rescan()
    _settle(app)
    assert any("Recovered" in lbl for lbl in _labels(app, "recent_root"))


def test_a_store_error_does_not_lose_the_orphan_list(app, monkeypatch) -> None:
    _session(app._test_audio, "orphan-1")

    def broken(self, **kw):
        raise OSError("transcripts unreadable")

    monkeypatch.setattr(TranscriptStore, "list_transcripts", broken)
    app._request_menu_rescan()
    _settle(app)

    assert _labels(app, "recent_root") == ["(no transcripts yet)"]
    assert _labels(app, "generate_root") != ["(no orphan recordings)"]


# ---------------------------------------------------------------------------
# Repair moved to the point of use
# ---------------------------------------------------------------------------
def test_scanning_never_repairs(app, monkeypatch) -> None:
    """A listing that rewrites headers is a listing you can't run from a menu."""
    import src.ui.menubar as mb

    repaired: list[Path] = []
    monkeypatch.setattr(mb, "repair_if_needed",
                        lambda p: repaired.append(p) or False)
    for sid in ("a", "b", "c"):
        _session(app._test_audio, sid)

    app._request_menu_rescan()
    _settle(app)

    assert repaired == [], f"menu build touched {len(repaired)} files"


def test_the_chosen_session_is_repaired_before_transcription(app, monkeypatch) -> None:
    import src.ui.menubar as mb

    repaired: list[Path] = []
    monkeypatch.setattr(mb, "repair_if_needed",
                        lambda p: repaired.append(p) or False)
    transcribed: list[dict] = []
    monkeypatch.setattr(MenuBarApp, "_run_transcription",
                        lambda self, meta: transcribed.append(meta))

    mic = app._test_audio / "pick-me_mic.wav"
    sysw = app._test_audio / "pick-me_system.wav"
    app._repair_then_transcribe({"session_id": "pick-me"}, mic, sysw)

    assert repaired == [mic, sysw]
    assert len(transcribed) == 1


def test_a_repair_failure_does_not_stop_the_transcription(app, monkeypatch) -> None:
    import src.ui.menubar as mb

    monkeypatch.setattr(mb, "repair_if_needed",
                        lambda p: (_ for _ in ()).throw(OSError("read-only fs")))
    transcribed: list[dict] = []
    monkeypatch.setattr(MenuBarApp, "_run_transcription",
                        lambda self, meta: transcribed.append(meta))

    app._repair_then_transcribe({"session_id": "x"},
                                app._test_audio / "x_mic.wav", None)
    assert len(transcribed) == 1
