"""Cover for audit finding E2 — the model picker only wrote config.

The engine is built once at startup. Picking a different model persisted the
setting and moved the checkmark, and nothing else: the engine kept running the
old one. So the menu said `medium`, the transcript's frontmatter said `small`
because that is what actually ran — and a *failure* placeholder for the same
session read config directly and said `medium`. Two files disagreeing about
what produced them.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.transcription.whisper_engine import WhisperEngine


def _engine(model: str = "small") -> WhisperEngine:
    return WhisperEngine(model_name=model,
                         transcribe_fn=lambda *a, **k: {"segments": []},
                         report_health=False)


# ---------------------------------------------------------------------------
# The engine half
# ---------------------------------------------------------------------------
def test_switching_changes_what_will_run() -> None:
    engine = _engine("small")
    try:
        assert engine.set_model("medium") is True
        assert engine.model_name == "medium"
        assert "whisper-medium" in engine.model_path or engine.model_path.endswith(
            "whisper-medium-mlx"
        )
    finally:
        engine.shutdown()


def test_switching_to_the_same_model_is_a_no_op() -> None:
    engine = _engine("small")
    try:
        assert engine.set_model("small") is False
    finally:
        engine.shutdown()


def test_an_unknown_model_is_refused() -> None:
    engine = _engine("small")
    try:
        with pytest.raises(ValueError):
            engine.set_model("enormous")
        assert engine.model_name == "small"
    finally:
        engine.shutdown()


def test_the_resolved_path_is_recomputed_not_reused() -> None:
    """A stale path would keep loading the old weights under the new name."""
    engine = _engine("small")
    try:
        before = engine.model_path
        engine.set_model("medium")
        assert engine.model_path != before
    finally:
        engine.shutdown()


def test_a_switch_waits_for_a_running_transcription(tmp_path: Path) -> None:
    """Swapping the model out from under a decode is not a thing to do."""
    import wave

    started = threading.Event()
    release = threading.Event()

    def slow(_audio, **kwargs):
        started.set()
        release.wait(5.0)
        return {"segments": [], "language": "en"}

    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x11\x11" * 16000)

    engine = WhisperEngine(model_name="small", transcribe_fn=slow,
                           report_health=False)
    switched = threading.Event()
    try:
        t = threading.Thread(target=lambda: engine.transcribe(wav), daemon=True)
        t.start()
        assert started.wait(5.0)

        s = threading.Thread(
            target=lambda: (engine.set_model("medium"), switched.set()), daemon=True
        )
        s.start()
        assert not switched.wait(0.3), "switched mid-transcription"
        assert engine.model_name == "small"

        release.set()
        assert switched.wait(5.0)
        assert engine.model_name == "medium"
        t.join(5.0)
        s.join(5.0)
    finally:
        release.set()
        engine.shutdown()


# ---------------------------------------------------------------------------
# What the transcript records
# ---------------------------------------------------------------------------
def test_frontmatter_follows_the_engine(tmp_path: Path) -> None:
    """It cached the name at construction, so it recorded the wrong model."""
    from src.storage.transcript_store import TranscriptStore
    from src.transcription.processor import TranscriptProcessor

    engine = _engine("small")
    try:
        processor = TranscriptProcessor(
            engine=engine,
            store=TranscriptStore(tmp_path / "t"),
            audio_dir=tmp_path / "a",
        )
        assert processor._model_name == "small"
        engine.set_model("medium")
        assert processor._model_name == "medium"
    finally:
        engine.shutdown()


def test_an_explicitly_pinned_model_is_still_honoured(tmp_path: Path) -> None:
    """retranscribe.py names the model that produced an old recording."""
    from src.storage.transcript_store import TranscriptStore
    from src.transcription.processor import TranscriptProcessor

    engine = _engine("small")
    try:
        processor = TranscriptProcessor(
            engine=engine,
            store=TranscriptStore(tmp_path / "t"),
            audio_dir=tmp_path / "a",
            model_name="large-v3",
        )
        engine.set_model("medium")
        assert processor._model_name == "large-v3"
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# The menu half
# ---------------------------------------------------------------------------
def test_the_picker_reaches_the_live_engine(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.pipeline import make_transcription_handler
    from src.storage.transcript_store import TranscriptStore
    from src.transcription.processor import TranscriptProcessor
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    engine = _engine("small")

    class Pipeline:
        audio_dir = tmp_path / "a"
        store = TranscriptStore(tmp_path / "t")
        processor = TranscriptProcessor(engine=engine, store=store,
                                        audio_dir=tmp_path / "a")

    handler = make_transcription_handler(Pipeline())  # type: ignore[arg-type]
    cfg = Config({"transcription": {"model": "small"},
                  "storage": {"audio_dir": str(tmp_path / "a"),
                              "transcript_dir": str(tmp_path / "t")}})
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: None),
        transcription_handler=handler,
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    try:
        assert bar._engine() is engine

        # Through the click handler, not _apply_model_live directly: a picker
        # that persists config and never spawns the worker passes the direct
        # test happily, which is exactly the bug.
        sender = type("S", (), {"title": "medium"})()
        bar._on_model_picked(sender)
        for t in threading.enumerate():
            if t.name == "otis-model-switch":
                t.join(10.0)

        assert engine.model_name == "medium", "the picker did not reach the engine"
        assert bar._config.get("transcription", "model") == "medium"
    finally:
        engine.shutdown()


def test_a_front_end_with_no_engine_says_restart(tmp_path: Path) -> None:
    pytest.importorskip("rumps")
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    notes: list[tuple] = []
    cfg = Config({"storage": {"audio_dir": str(tmp_path / "a"),
                              "transcript_dir": str(tmp_path / "t")}})
    bar = MenuBarApp(
        config=cfg,
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda *a: notes.append(a)),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    assert bar._engine() is None
    bar._apply_model_live("medium")
    assert any("restart" in str(n).lower() for n in notes)


# ---------------------------------------------------------------------------
# Models that are not downloaded yet
# ---------------------------------------------------------------------------
def _hub_with(tmp_path: Path, *names: str) -> Path:
    hub = tmp_path / "hub"
    for name in names:
        (hub / f"models--mlx-community--whisper-{name}-mlx").mkdir(parents=True)
    return hub


def test_cached_model_names_is_a_cheap_listing(tmp_path: Path) -> None:
    from src.storage.model_cache import cached_model_names, is_model_cached

    hub = _hub_with(tmp_path, "medium", "large-v3")
    names = cached_model_names(hub)
    assert names == {"medium", "large-v3"}
    assert is_model_cached("medium", names)
    assert is_model_cached("large", names), "'large' resolves to the v3 repo"
    assert not is_model_cached("base", names)


def test_the_picker_marks_models_that_would_download(tmp_path: Path, monkeypatch) -> None:
    """The picker offered every model as if equal; choosing an uncached one
    silently downloaded ~145 MB (base) to ~3 GB (large-v3) inside the next
    transcription, holding up that meeting's transcript."""
    pytest.importorskip("rumps")
    import src.storage.model_cache as mc
    from src.config import Config
    from src.detection.detector import MeetingDetector
    from src.ui.menubar import MenuBarApp
    from src.ui.notifications import NotificationManager

    monkeypatch.setattr(mc, "DEFAULT_HUB_DIR", _hub_with(tmp_path, "medium", "small", "tiny"))
    sent: list[tuple[str, str, str]] = []
    bar = MenuBarApp(
        config=Config({"transcription": {"model": "medium"}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons",
        user_config_path=tmp_path / "config.yaml",
    )
    titles = {name: item.title for name, item in bar._model_items.items()}
    assert titles["medium"] == "medium"
    assert "download" in titles["large-v3"] and "GB" in titles["large-v3"]
    assert "download" in titles["base"]

    # Picking a labelled item still selects the model, and says what follows.
    monkeypatch.setattr(bar, "_apply_model_live", lambda model: None)
    bar._on_model_picked(bar._model_items["large-v3"])
    assert bar._config.get("transcription", "model") == "large-v3"
    assert any("download" in m.lower() for _t, _s, m in sent)
