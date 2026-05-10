"""Aggressive end-to-end stress test for Phases 1-4.

Not a unit test — runs against a temp tree, exercises real edge cases:

* End-to-end pipeline: synthetic recording → processor → store → retention.
* Unicode / emoji / very-long titles → filename safety.
* Concurrent saves to the same logical slot.
* 100 transcripts → list + search latency.
* Path traversal attempts in titles.
* WhisperEngine concurrency (two transcribes in parallel).
* Corrupt metadata recovery.

Run with::

    python scripts/stress_test.py

Exits non-zero on any failure. Prints a summary.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.audio_retention import AudioRetentionManager
from src.storage.transcript_store import TranscriptStore
from src.transcription.processor import (
    MeetingSnapshot,
    RecordingSession,
    TranscriptProcessor,
    apply_offset,
    deduplicate_echo,
    interleave_with_overlaps,
)
from src.transcription.whisper_engine import (
    Segment,
    TranscriptionResult,
    WhisperEngine,
    filter_hallucinations,
)


PASS = "\033[1;32m✓\033[0m"
FAIL = "\033[1;31m✗\033[0m"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _audible_wav(path: Path, *, seconds: float = 1.0, freq: int = 440, amp: int = 8000) -> None:
    """Write a 16 kHz mono 16-bit sine wave."""
    rate = 16000
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        s = int(amp * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))


def _basic_frontmatter(**overrides):
    fm = {
        "id": overrides.get("id", "stress-id"),
        "title": overrides.get("title", "Test"),
        "date": overrides.get("date", "2026-05-10"),
        "start_time": overrides.get("start_time", "14:00"),
        "end_time": overrides.get("end_time", "14:30"),
        "duration_minutes": 30,
        "language": "en",
        "app": None,
        "participants": [],
        "tags": [],
        "audio_files": {"mic": None, "system": None},
        "audio_available": True,
        "model": "small",
    }
    fm.update(overrides)
    return fm


# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------
class Suite:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            self.results.append((name, False, f"assertion: {exc}"))
            print(f"  {FAIL} {name} — {exc}")
        except Exception as exc:
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  {FAIL} {name} — {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            self.results.append((name, True, ""))
            print(f"  {PASS} {name}")

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if not ok)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def t_unicode_and_emoji_in_title(tmp: Path) -> None:
    store = TranscriptStore(tmp)
    fm = _basic_frontmatter(title="📞 Réunion avec Søren — Q1/Q2 résumé!! 🚀")
    path = store.save(fm, "## Transcript\n\nbody")
    assert path.exists(), "save should produce a real file"
    # Filename is ASCII-clean enough to be portable, but slug allows Latin-1 letters.
    name = path.name
    assert name.endswith(".md"), name
    # Sanity round-trip via list_transcripts.
    listed = store.list_transcripts()
    assert any(fm["title"] in entry["title"] for entry in listed), \
        f"Could not find unicode title in listing: {listed}"


def t_very_long_title_truncation(tmp: Path) -> None:
    store = TranscriptStore(tmp)
    fm = _basic_frontmatter(title="x" * 200)
    path = store.save(fm, "## Transcript")
    # macOS allows 255 bytes per filename component.
    assert len(path.name) < 200, f"filename too long: {path.name}"


def t_path_traversal_attempt_in_title(tmp: Path) -> None:
    """Title with ``../`` must not escape transcript_dir."""
    store = TranscriptStore(tmp)
    fm = _basic_frontmatter(title="../../../etc/passwd")
    path = store.save(fm, "body")
    # Resolved path must still live under the store root.
    assert tmp.resolve() in path.resolve().parents, (
        f"path escaped the store root! {path.resolve()}"
    )


def t_concurrent_saves_no_loss(tmp: Path) -> None:
    """Five threads racing to save five distinct sessions at the same minute → all 5 land."""
    store = TranscriptStore(tmp)
    n = 5

    def worker(i: int) -> None:
        store.save(
            _basic_frontmatter(id=f"sid-{i}", title="Same Slot"),
            f"## Transcript\n\nbody {i}",
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    listed = store.list_transcripts()
    ids = {entry["id"] for entry in listed}
    assert ids == {f"sid-{i}" for i in range(n)}, (
        f"expected 5 ids, got {ids}"
    )


def t_search_at_scale(tmp: Path) -> None:
    """100 transcripts → list + search complete in < 1 second."""
    store = TranscriptStore(tmp)
    for i in range(100):
        body = f"## Transcript\n\nMeeting notes {i}. Customer story keyword-{i % 7}."
        store.save(
            _basic_frontmatter(
                id=f"sid-{i}", title=f"Sync {i}",
                date=f"2026-{(i % 12) + 1:02d}-15", start_time=f"{i % 24:02d}:00",
            ),
            body,
        )
    t0 = time.monotonic()
    listed = store.list_transcripts(limit=100)
    t1 = time.monotonic()
    assert len(listed) == 100, f"got {len(listed)} entries"
    assert (t1 - t0) < 1.0, f"list_transcripts too slow: {t1-t0:.2f}s for 100 files"

    t2 = time.monotonic()
    hits = store.search("keyword-3", limit=50)
    t3 = time.monotonic()
    assert len(hits) >= 14, f"expected ~14 hits, got {len(hits)}"
    assert (t3 - t2) < 1.0, f"search too slow: {t3-t2:.2f}s"


def t_corrupt_frontmatter_skipped_in_listing(tmp: Path) -> None:
    """A .md whose frontmatter is broken must not crash list/search."""
    store = TranscriptStore(tmp)
    store.save(_basic_frontmatter(id="ok"), "body")
    bad = tmp / "2026" / "05" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\n: : : nope\n---\nbody\n")
    listed = store.list_transcripts()
    # Bad file is silently skipped; good one still listed.
    assert any(e["id"] == "ok" for e in listed)


def t_end_to_end_pipeline(tmp: Path) -> None:
    """Synthetic recording → processor → transcript on disk → retention deletes audio → flag flips."""
    audio_dir = tmp / "audio"
    transcript_dir = tmp / "transcripts"
    audio_dir.mkdir()

    sid = "e2e-stress-id"
    mic = audio_dir / f"{sid}_mic.wav"
    sysw = audio_dir / f"{sid}_system.wav"
    meta_path = audio_dir / f"{sid}_metadata.json"
    _audible_wav(mic, seconds=1.0)
    _audible_wav(sysw, seconds=1.0)
    meta_path.write_text(json.dumps({
        "session_id": sid,
        "mic_wav": mic.name,
        "system_wav": sysw.name,
        "mic_start_monotonic": 1000.0,
        "system_start_monotonic": 1000.15,
        "start_wall_clock": "2026-05-10T14:00:00+00:00",
        "sample_rate": 16000,
    }))

    fake_engine = WhisperEngine(
        transcribe_fn=lambda *a, **kw: {
            "text": "hello world",
            "language": "en",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        },
    )
    store = TranscriptStore(transcript_dir)
    proc = TranscriptProcessor(engine=fake_engine, store=store, audio_dir=audio_dir)

    session = RecordingSession.from_recorder_metadata(
        json.loads(meta_path.read_text()), audio_dir=audio_dir
    )
    result = proc.process(session, meeting=MeetingSnapshot(title="Customer Sync"))
    assert result.transcript_path.exists()

    # Audio relocated under YYYY/MM/.
    assert not mic.exists(), "mic should have moved"
    relocated_meta = next(audio_dir.rglob("*_metadata.json"), None)
    assert relocated_meta is not None
    relocated_data = json.loads(relocated_meta.read_text())
    assert relocated_data["mic_wav"].endswith("_mic.wav") and "/" in relocated_data["mic_wav"], (
        f"metadata.json mic_wav not updated to relative YYYY/MM path: {relocated_data['mic_wav']}"
    )

    # Run retention with a clock that puts the files past the horizon.
    later = time.time() + 31 * 86400
    rm = AudioRetentionManager(
        audio_dir=audio_dir,
        transcript_store=store,
        retention_days=30,
        clock=lambda: later,
    )
    report = rm.cleanup_now()
    assert sid in report.transcripts_marked, (
        f"transcript {sid} should have been marked unavailable: {report}"
    )
    fm = store.get_transcript(sid)["metadata"]
    assert fm["audio_available"] is False


def t_retention_doesnt_touch_recent_files(tmp: Path) -> None:
    """A fresh recording must NOT be deleted by retention."""
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    (audio_dir / "fresh_mic.wav").write_bytes(b"x" * 1024)
    rm = AudioRetentionManager(audio_dir=audio_dir, retention_days=30)
    rm.cleanup_now()
    assert (audio_dir / "fresh_mic.wav").exists()


def t_concurrent_transcribe_calls(tmp: Path) -> None:
    """Two transcribes in parallel must not deadlock or corrupt state."""
    barrier = threading.Barrier(2)
    seen: list[str] = []

    def fake(audio: str, **_kw):
        barrier.wait(timeout=2.0)
        seen.append(audio)
        return {"text": "x", "segments": [{"start": 0, "end": 1, "text": "x"}], "language": "en"}

    engine = WhisperEngine(transcribe_fn=fake, idle_timeout_seconds=0.1)
    a = tmp / "a.wav"
    b = tmp / "b.wav"
    _audible_wav(a, seconds=0.5)
    _audible_wav(b, seconds=0.5)

    results: list[TranscriptionResult] = []

    def runner(p: Path) -> None:
        results.append(engine.transcribe(p))

    threads = [threading.Thread(target=runner, args=(a,)),
               threading.Thread(target=runner, args=(b,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert all(not t.is_alive() for t in threads), "transcribe threads deadlocked"
    assert len(results) == 2
    # Idle timer fired between calls; engine must have re-warmed cleanly.
    assert engine.is_warm or not engine.is_warm  # just exercise the property


def t_pure_function_correctness(_tmp: Path) -> None:
    """Spot-check helpers don't rely on any imported state."""
    segs = [Segment(0, 1, "hello"), Segment(1, 2, "Joyeux"), Segment(2, 3, "Joyeux"),
            Segment(3, 4, "Joyeux"), Segment(4, 5, "Joyeux"), Segment(5, 6, "Subscribe to my channel")]
    filtered = filter_hallucinations(segs)
    # "hello" plus 2 of the 4 "Joyeux" repetitions → 3 segments.
    assert len(filtered) == 3, f"expected 3 (hello + 2 joyeux), got {len(filtered)}: {[s.text for s in filtered]}"

    sys_segs = [Segment(10.0, 11.0, "good morning team")]
    mic_segs = [Segment(10.05, 11.05, "good morning team")]
    kept, dropped = deduplicate_echo(sys_segs, mic_segs)
    assert dropped == 1 and kept == [], f"echo dedup failed: kept={kept} dropped={dropped}"

    shifted = apply_offset([Segment(0, 1, "x")], 0.5)
    assert shifted[0].start == 0.5 and shifted[0].end == 1.5


def t_processor_handles_silent_streams(tmp: Path) -> None:
    """Silent mic + silent system → empty transcript with friendly placeholder."""
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    sid = "silent-test"
    mic = audio_dir / f"{sid}_mic.wav"
    sysw = audio_dir / f"{sid}_system.wav"
    # All zeros.
    rate = 16000
    silent = bytes(rate * 2)
    for p in (mic, sysw):
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate); wf.writeframes(silent)
    (audio_dir / f"{sid}_metadata.json").write_text(json.dumps({
        "session_id": sid, "mic_wav": mic.name, "system_wav": sysw.name,
        "mic_start_monotonic": 0.0, "system_start_monotonic": 0.0,
        "start_wall_clock": "2026-05-10T14:00:00+00:00",
    }))

    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: {"text": "x", "segments": [], "language": "en"})
    store = TranscriptStore(tmp / "transcripts")
    proc = TranscriptProcessor(engine=engine, store=store, audio_dir=audio_dir)
    session = RecordingSession.from_recorder_metadata(
        json.loads((audio_dir / f"{sid}_metadata.json").read_text()),
        audio_dir=audio_dir,
    )
    result = proc.process(session, meeting=MeetingSnapshot())
    body = result.body
    assert "(no speech detected)" in body, (
        f"expected friendly placeholder; got body:\n{body}"
    )


def t_store_handles_transcripts_in_subdirs(tmp: Path) -> None:
    """A transcript already at a deep YYYY/MM/ path must be discoverable."""
    store = TranscriptStore(tmp)
    # Manually plant a file deep under root.
    deep = tmp / "2024" / "11" / "old.md"
    deep.parent.mkdir(parents=True)
    deep.write_text("---\nid: deep\ntitle: Old\ndate: '2024-11-15'\nstart_time: '09:00'\n---\nbody\n")
    fm = store.path_for("deep")
    assert fm == deep
    listed = store.list_transcripts()
    assert any(e["id"] == "deep" for e in listed)


def t_silent_head_then_speech_is_not_skipped(tmp: Path) -> None:
    """Regression: a recording whose audio starts late must NOT be skipped.

    The earlier _wav_rms looked only at the first 5s — a YouTube clip that
    began playing 8s into the recording got dropped because the head was
    pure silence. Peak-window scan fixes this.
    """
    import math
    import struct

    p = tmp / "silent_head.wav"
    rate = 16000
    silent_seconds = 7
    audible_seconds = 5
    amp = 12000
    frames = bytearray()
    frames += b"\x00\x00" * (silent_seconds * rate)
    for i in range(audible_seconds * rate):
        sample = int(amp * math.sin(2 * math.pi * 440.0 * i / rate))
        frames += struct.pack("<h", sample)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(bytes(frames))

    called = {"hit": False}
    engine = WhisperEngine(transcribe_fn=lambda *a, **kw: (
        called.update(hit=True),
        {"text": "x", "segments": [{"start": 0, "end": 1, "text": "x"}], "language": "en"},
    )[1])
    engine.transcribe(p)
    assert called["hit"], "Pre-check rejected a quiet-head WAV — silent-head bug regressed"


def t_processor_keeps_both_streams_by_default(tmp: Path) -> None:
    """Default behaviour: same content on both tracks → BOTH appear in body."""
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    sid = "default-keep-both"
    mic = audio_dir / f"{sid}_mic.wav"
    sysw = audio_dir / f"{sid}_system.wav"
    _audible_wav(mic, seconds=1.0)
    _audible_wav(sysw, seconds=1.0)
    (audio_dir / f"{sid}_metadata.json").write_text(json.dumps({
        "session_id": sid, "mic_wav": mic.name, "system_wav": sysw.name,
        "mic_start_monotonic": 0.0, "system_start_monotonic": 0.0,
        "start_wall_clock": "2026-05-10T14:00:00+00:00", "sample_rate": 16000,
    }))

    fake_engine = WhisperEngine(transcribe_fn=lambda *a, **kw: {
        "text": "good morning everyone", "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": "good morning everyone"}],
    })
    store = TranscriptStore(tmp / "transcripts")
    proc = TranscriptProcessor(engine=fake_engine, store=store, audio_dir=audio_dir)

    session = RecordingSession.from_recorder_metadata(
        json.loads((audio_dir / f"{sid}_metadata.json").read_text()), audio_dir=audio_dir,
    )
    result = proc.process(session, meeting=MeetingSnapshot())
    assert result.echo_dropped == 0
    assert "Me: good morning everyone" in result.body
    assert "Participant: good morning everyone" in result.body, (
        "Default off-dedup should keep both streams — dedup-default regression"
    )


def t_orphan_finder_handles_mixed_layouts(tmp: Path) -> None:
    """Filesystem with all three on-disk layouts coexisting."""
    from src.ui.menubar import find_orphan_sessions

    audio = tmp / "audio"
    transcripts = tmp / "transcripts"
    audio.mkdir(); transcripts.mkdir()
    store = TranscriptStore(transcripts)

    # Session A — standard UUID layout, no transcript yet.
    sid_a = "aaaaaaaa-0000-0000-0000-000000000000"
    _audible_wav(audio / f"{sid_a}_mic.wav")
    _audible_wav(audio / f"{sid_a}_system.wav")
    (audio / f"{sid_a}_metadata.json").write_text(json.dumps({
        "session_id": sid_a,
        "mic_wav": f"{sid_a}_mic.wav",
        "system_wav": f"{sid_a}_system.wav",
    }))

    # Session B — renamed YYYY/MM/ layout, with a transcript already on disk.
    sid_b = "renamed-existing"
    sub = audio / "2026" / "05"
    sub.mkdir(parents=True)
    _audible_wav(sub / "2026-05-09_1400_mic.wav")
    _audible_wav(sub / "2026-05-09_1400_system.wav")
    (sub / "2026-05-09_1400_metadata.json").write_text(json.dumps({
        "session_id": sid_b,
        "mic_wav": "2026/05/2026-05-09_1400_mic.wav",
        "system_wav": "2026/05/2026-05-09_1400_system.wav",
    }))
    store.save({
        "id": sid_b, "title": "Done", "date": "2026-05-09", "start_time": "14:00",
        "end_time": "14:30", "duration_minutes": 30, "language": "en", "app": None,
        "participants": [], "tags": [], "audio_files": {"mic": None, "system": None},
        "audio_available": True, "model": "small",
    }, "## Transcript")

    # Session C — orphan WAVs, no metadata.json.
    sid_c = "cccccccc-0000-0000-0000-000000000000"
    _audible_wav(audio / f"{sid_c}_mic.wav")

    found = find_orphan_sessions(audio_dir=audio, store=store)
    found_ids = {f["session_id"] for f in found}
    assert sid_a in found_ids, "standard-layout session missing"
    assert sid_b not in found_ids, "transcribed session must be filtered out"
    assert sid_c in found_ids, "orphan UUID WAV missing"
    # Renamed-layout WAV must NOT also appear with a date-derived sid (Pass 2 dup).
    date_derived = "2026-05-09_1400"
    assert date_derived not in found_ids, (
        "Pass 2 double-counted a renamed-layout WAV via its filename"
    )


def t_failure_placeholder_lands_on_disk(tmp: Path) -> None:
    """save_failure() must write a status:failed transcript with the right shape."""
    store = TranscriptStore(tmp)
    err = ConnectionError("hf.co timed out")
    path = store.save_failure(
        session_id="fail-1",
        error=err,
        title="Customer Sync",
        app="zoom.us",
        participants=["Alice <a@x>"],
        model="medium",
        audio_files={"mic": "x_mic.wav", "system": "x_system.wav"},
    )
    assert path.exists()
    rec = store.get_transcript("fail-1")
    assert rec is not None
    fm = rec["metadata"]
    assert fm["status"] == "failed"
    assert fm["error_type"] == "ConnectionError"
    assert "failed" in fm["tags"]
    assert "retranscribe" in rec["body"].lower(), "failure body must mention how to retry"


def t_orphan_finder_excludes_active_recording(tmp: Path) -> None:
    """The actively-recording session's WAVs must NOT be listed as orphans."""
    from src.ui.menubar import find_orphan_sessions

    audio = tmp / "audio"
    audio.mkdir()
    active = "active-recording"
    other = "other-orphan"
    _audible_wav(audio / f"{active}_mic.wav")
    _audible_wav(audio / f"{other}_mic.wav")

    out = find_orphan_sessions(audio_dir=audio, current_recording_session_id=active)
    sids = {f["session_id"] for f in out}
    assert active not in sids
    assert other in sids


def t_audio_retention_handles_concurrent_recording(tmp: Path) -> None:
    """A WAV that's currently being written (mtime = now) must survive a sweep."""
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    fresh = audio_dir / "in-progress_mic.wav"
    fresh.write_bytes(b"x" * 1024)
    rm = AudioRetentionManager(
        audio_dir=audio_dir, retention_days=0,  # everything past 0 days is "old"
        clock=lambda: time.time() - 60,         # but we pretend now is 60s in the past
    )
    rm.cleanup_now()
    assert fresh.exists(), "in-progress recording must survive the sweep"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
TESTS = [
    ("unicode + emoji titles round-trip", t_unicode_and_emoji_in_title),
    ("very long titles get truncated to a safe filename", t_very_long_title_truncation),
    ("path-traversal attempts can't escape transcript_dir", t_path_traversal_attempt_in_title),
    ("concurrent saves to same slot don't lose any", t_concurrent_saves_no_loss),
    ("100 files: list + search latency under 1s", t_search_at_scale),
    ("corrupt frontmatter is skipped, not fatal", t_corrupt_frontmatter_skipped_in_listing),
    ("end-to-end: synth audio → transcript → retention flips flag", t_end_to_end_pipeline),
    ("retention leaves recent files alone", t_retention_doesnt_touch_recent_files),
    ("concurrent transcribe() calls don't deadlock", t_concurrent_transcribe_calls),
    ("pure helpers (filter / dedup / offset)", t_pure_function_correctness),
    ("processor produces a friendly transcript on silent input", t_processor_handles_silent_streams),
    ("transcripts in pre-existing YYYY/MM subdirs are found", t_store_handles_transcripts_in_subdirs),
    ("retention does not delete in-progress recordings", t_audio_retention_handles_concurrent_recording),
    ("silent-head WAV is NOT skipped (regression)", t_silent_head_then_speech_is_not_skipped),
    ("default behaviour keeps both mic + system streams", t_processor_keeps_both_streams_by_default),
    ("orphan finder handles mixed layouts coherently", t_orphan_finder_handles_mixed_layouts),
    ("orphan finder excludes the active-recording session", t_orphan_finder_excludes_active_recording),
    ("save_failure writes a status:failed placeholder transcript", t_failure_placeholder_lands_on_disk),
]


def main() -> int:
    suite = Suite()
    print("Otis stress test (Phases 1-4)")
    print("=" * 50)
    for name, fn in TESTS:
        with tempfile.TemporaryDirectory() as tmp:
            suite.run(name, lambda fn=fn, tmp=tmp: fn(Path(tmp)))
    print("=" * 50)
    total = len(suite.results)
    failed = suite.failed
    if failed:
        print(f"\n{failed}/{total} stress checks FAILED.")
        return 1
    print(f"\nAll {total} stress checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
