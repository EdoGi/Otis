"""Regression cover for audit findings A1 (escalation) and B4/C4 (locking).

A1 — the engine already detected that the system stream was pure silence and
logged it at INFO on every single transcription for two months. The signal
existed; nothing consumed it. Now it reaches the health registry and the
result object, so the menu bar and the transcript frontmatter can both say so.

B4/C4 — ``_call_transcribe`` ran outside every lock, and the model-cache
release read ``_inflight`` under the lock then released it before clearing
mlx's ModelHolder. Two concurrent transcriptions could run on one engine, and
a release could land between ModelHolder's check and its return.
"""

from __future__ import annotations

import math
import struct
import threading
import wave
from pathlib import Path

import pytest

from src.health import SYSTEM_AUDIO, HealthRegistry, HealthState, reset_health
from src.transcription.whisper_engine import WhisperEngine


@pytest.fixture(autouse=True)
def _clean_health():
    reset_health()
    yield
    reset_health()


def _write_wav(path: Path, *, seconds: float, amplitude: int, rate: int = 16000) -> None:
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        s = int(amplitude * math.sin(2 * math.pi * 440.0 * i / rate)) if amplitude else 0
        frames += struct.pack("<h", s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))


def silent_wav(tmp_path: Path, name: str, seconds: float = 1.0) -> Path:
    p = tmp_path / name
    _write_wav(p, seconds=seconds, amplitude=0)
    return p


def audible_wav(tmp_path: Path, name: str, seconds: float = 1.0) -> Path:
    p = tmp_path / name
    _write_wav(p, seconds=seconds, amplitude=16000)
    return p


def fake_transcribe(*_a, **_k):
    return {"text": "hello", "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en"}


# ---------------------------------------------------------------------------
# A1 — a silent stream is now reported, not just logged
# ---------------------------------------------------------------------------
def test_silent_system_stream_marks_result_and_fails_health(tmp_path) -> None:
    engine = WhisperEngine(model_name="small", transcribe_fn=fake_transcribe)
    wav = silent_wav(tmp_path, "2026-08-07_1201_system.wav")

    result = engine.transcribe(wav, stream="system")

    assert result.was_silent is True
    assert result.segments == []
    from src.health import get_health

    rep = get_health().get(SYSTEM_AUDIO)
    assert rep is not None
    assert rep.state is HealthState.FAIL
    assert "silen" in rep.detail.lower()
    assert rep.hint and "output" in rep.hint.lower()


def test_stream_is_inferred_from_the_filename(tmp_path) -> None:
    """The processor passes stream= explicitly, but retranscribe.py and the
    orphan path don't — the filename still carries the answer."""
    engine = WhisperEngine(model_name="small", transcribe_fn=fake_transcribe)
    wav = silent_wav(tmp_path, "2026-08-07_1201_system.wav")

    engine.transcribe(wav)  # no stream= argument

    from src.health import get_health

    assert get_health().get(SYSTEM_AUDIO).state is HealthState.FAIL


def test_silent_mic_is_a_warning_not_a_failure(tmp_path) -> None:
    """A silent mic is usually 'nobody spoke', not a broken setup."""
    from src.health import MICROPHONE, get_health

    engine = WhisperEngine(model_name="small", transcribe_fn=fake_transcribe)
    engine.transcribe(silent_wav(tmp_path, "x_mic.wav"), stream="mic")

    rep = get_health().get(MICROPHONE)
    assert rep is not None
    assert rep.state is HealthState.WARN


def test_audible_system_stream_reports_healthy(tmp_path) -> None:
    from src.health import get_health

    engine = WhisperEngine(model_name="small", transcribe_fn=fake_transcribe)
    result = engine.transcribe(audible_wav(tmp_path, "x_system.wav"), stream="system")

    assert result.was_silent is False
    assert get_health().get(SYSTEM_AUDIO).state is HealthState.OK


def test_silence_then_recovery_flips_health_back(tmp_path) -> None:
    from src.health import get_health

    engine = WhisperEngine(model_name="small", transcribe_fn=fake_transcribe)
    engine.transcribe(silent_wav(tmp_path, "a_system.wav"), stream="system")
    assert get_health().get(SYSTEM_AUDIO).state is HealthState.FAIL
    engine.transcribe(audible_wav(tmp_path, "b_system.wav"), stream="system")
    assert get_health().get(SYSTEM_AUDIO).state is HealthState.OK


def test_health_reporting_can_be_disabled(tmp_path) -> None:
    """retranscribe.py works on archived files; it must not rewrite live health."""
    from src.health import get_health

    engine = WhisperEngine(
        model_name="small", transcribe_fn=fake_transcribe, report_health=False
    )
    engine.transcribe(silent_wav(tmp_path, "x_system.wav"), stream="system")
    assert get_health().get(SYSTEM_AUDIO) is None


# ---------------------------------------------------------------------------
# B4 — two transcriptions must not run concurrently on one engine
# ---------------------------------------------------------------------------
def test_concurrent_transcriptions_are_serialised(tmp_path) -> None:
    live = 0
    peak = 0
    guard = threading.Lock()
    started = threading.Event()

    def slow_transcribe(*_a, **_k):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        started.set()
        # Long enough that a second thread would certainly overlap if allowed.
        threading.Event().wait(0.05)
        with guard:
            live -= 1
        return {"text": "x", "segments": [], "language": "en"}

    engine = WhisperEngine(model_name="small", transcribe_fn=slow_transcribe)
    wavs = [audible_wav(tmp_path, f"s{i}_mic.wav") for i in range(4)]

    threads = [threading.Thread(target=engine.transcribe, args=(w,)) for w in wavs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert peak == 1, f"{peak} transcriptions ran at once; the engine is not serialised"


def test_serialisation_still_releases_the_lock_on_error(tmp_path) -> None:
    """A raising transcribe must not wedge every future call."""
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"text": "ok", "segments": [], "language": "en"}

    engine = WhisperEngine(model_name="small", transcribe_fn=flaky)
    wav = audible_wav(tmp_path, "x_mic.wav")

    with pytest.raises(RuntimeError):
        engine.transcribe(wav)
    # Second call must not hang or deadlock.
    result = engine.transcribe(wav)
    assert result.segments == []


# ---------------------------------------------------------------------------
# C4 — the release path must be observable when it breaks
# ---------------------------------------------------------------------------
def test_release_logs_when_mlx_internals_move(caplog, monkeypatch) -> None:
    """The old bare `except Exception: return` meant a renamed ModelHolder
    silently disabled memory release AND skipped mx.clear_cache()."""
    import src.transcription.whisper_engine as we

    def broken_import():
        raise AttributeError("ModelHolder moved upstream")

    monkeypatch.setattr(we, "_import_model_holder", broken_import)
    with caplog.at_level("WARNING"):
        we._release_mlx_model_cache()

    assert any("ModelHolder" in r.message or "model cache" in r.message.lower()
               for r in caplog.records), "a silent no-op release is the bug"


def test_registry_injection_keeps_tests_isolated(tmp_path) -> None:
    registry = HealthRegistry()
    engine = WhisperEngine(
        model_name="small", transcribe_fn=fake_transcribe, health=registry
    )
    engine.transcribe(silent_wav(tmp_path, "x_system.wav"), stream="system")

    assert registry.get(SYSTEM_AUDIO).state is HealthState.FAIL
    from src.health import get_health

    assert get_health().get(SYSTEM_AUDIO) is None
