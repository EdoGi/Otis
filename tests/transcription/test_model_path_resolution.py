"""Cover for audit finding C6 — the README's "no phone-home" was not true.

mlx-whisper hands whatever it is given straight to ``snapshot_download``,
which contacts huggingface.co on every load even when all four model files are
already cached. Measured on this Mac: 265.9 ms for the normal call against
0.1 ms resolving the identical path locally. The engine drops the model after
``idle_timeout_seconds``, so a normal day of meetings repeated that request
once per meeting — leaking the timing of every meeting transcribed.

Worth being precise about what this does *not* fix: transcribing already
worked with the endpoint unreachable, because the hub falls back to its cache
once the request fails. What changes is that no request is made at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.transcription.whisper_engine import WhisperEngine, resolve_model_path

REPO = "mlx-community/whisper-medium-mlx"
CACHED = "/cache/hub/models--mlx-community--whisper-medium-mlx/snapshots/abc123"


@pytest.fixture()
def hub(monkeypatch):
    """Stand in for huggingface_hub, recording how it was called."""
    hub_mod = pytest.importorskip("huggingface_hub")
    calls: list[dict[str, Any]] = []

    def fake(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            return CACHED
        return "/downloaded/snapshot"

    monkeypatch.setattr(hub_mod, "snapshot_download", fake)
    return calls


def test_a_cached_model_is_resolved_without_touching_the_network(hub) -> None:
    assert resolve_model_path(REPO) == CACHED
    assert len(hub) == 1
    assert hub[0]["local_files_only"] is True


def test_an_uncached_model_is_downloaded_once(hub, monkeypatch) -> None:
    """A fresh machine must still be able to fetch the model."""
    hub_mod = pytest.importorskip("huggingface_hub")

    def fake(**kwargs):
        hub.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return "/downloaded/snapshot"

    monkeypatch.setattr(hub_mod, "snapshot_download", fake)

    assert resolve_model_path(REPO) == "/downloaded/snapshot"
    assert [c.get("local_files_only") for c in hub] == [True, None]


def test_an_unreachable_hub_falls_back_to_the_repo_id(hub, monkeypatch) -> None:
    """mlx-whisper's own error explains the problem better than ours would."""
    hub_mod = pytest.importorskip("huggingface_hub")

    def always_fails(**kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(hub_mod, "snapshot_download", always_fails)
    assert resolve_model_path(REPO) == REPO


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def test_the_engine_resolves_once_and_remembers(hub) -> None:
    """The model is reloaded after every idle timeout; the path is not."""
    engine = WhisperEngine(model_name="medium",
                           transcribe_fn=lambda *a, **k: {"segments": []})
    try:
        assert engine.model_path == CACHED
        for _ in range(5):
            assert engine.model_path == CACHED
        assert len(hub) == 1, f"resolved {len(hub)} times"
    finally:
        engine.shutdown()


def test_transcribe_is_given_the_local_path_not_the_repo_id(hub, tmp_path) -> None:
    seen: list[Any] = []

    def spy(_audio, **kwargs):
        seen.append(kwargs.get("path_or_hf_repo"))
        return {"segments": [], "language": "en"}

    wav = tmp_path / "clip.wav"
    import wave
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b"\x11\x11" * 16000)

    engine = WhisperEngine(model_name="medium", transcribe_fn=spy,
                           report_health=False)
    try:
        engine.transcribe(wav)
    finally:
        engine.shutdown()

    assert seen == [CACHED], "the repo id would make mlx call the hub again"


def test_resolution_is_lazy(hub) -> None:
    """Constructing an engine must not reach for the network."""
    engine = WhisperEngine(model_name="medium",
                           transcribe_fn=lambda *a, **k: {"segments": []})
    try:
        assert hub == [], "constructing the engine resolved the model"
    finally:
        engine.shutdown()


def test_a_missing_hub_library_is_not_fatal(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("gone")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    assert resolve_model_path(REPO) == REPO
