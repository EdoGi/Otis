"""The medium-severity findings the P0/P1 verification pass deferred.

Three of the seven closed on their own as the rework went past them — the
per-calendar `ev.id` keyspace (P1 item 14), `_state_entered_at` being set
outside `_transition`, and the titler guessing proper nouns (P1 item 15).
These are the four that were still real.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# A meeting Whisper found nothing in was recorded as 0 minutes long
# ---------------------------------------------------------------------------
def test_a_transcript_with_no_segments_still_knows_how_long_it_was() -> None:
    """Everyone on mute, or a failed capture: no segments, 45 minutes of audio.

    duration came from the last segment's end, so the frontmatter, the web UI
    and any time accounting all said 0 min.
    """
    from src.transcription.processor import _recorded_seconds

    # 44.7 minutes, the real 2026-08-07 session's byte count.
    assert _recorded_seconds({"mic_bytes": 85_753_856, "sample_rate": 16000}) == \
        pytest.approx(2679.8, abs=1.0)


def test_the_longer_of_the_two_streams_wins() -> None:
    from src.transcription.processor import _recorded_seconds

    assert _recorded_seconds(
        {"mic_bytes": 32_000, "system_bytes": 64_000, "sample_rate": 16000}
    ) == pytest.approx(2.0)


def test_no_bytes_at_all_is_still_zero() -> None:
    from src.transcription.processor import _recorded_seconds

    assert _recorded_seconds({}) == 0.0
    assert _recorded_seconds({"mic_bytes": 0}) == 0.0


# ---------------------------------------------------------------------------
# shutdown() left a timer armed behind it
# ---------------------------------------------------------------------------
def test_a_transcription_finishing_after_shutdown_does_not_rearm() -> None:
    """transcribe()'s finally re-armed the idle timer on a shut-down engine.

    It would then fire minutes later and clear mlx's cache under whoever owns
    the process next.
    """
    from src.transcription.whisper_engine import WhisperEngine

    engine = WhisperEngine(model_name="small",
                           transcribe_fn=lambda *a, **k: {"segments": []},
                           report_health=False)
    engine.shutdown()
    engine._reset_idle_timer()   # what the finally block does

    assert engine._idle_timer is None


def test_shutdown_cancels_a_live_timer() -> None:
    from src.transcription.whisper_engine import WhisperEngine

    engine = WhisperEngine(model_name="small",
                           transcribe_fn=lambda *a, **k: {"segments": []},
                           report_health=False)
    engine._reset_idle_timer()
    assert engine._idle_timer is not None
    engine.shutdown()
    assert engine._idle_timer is None


# ---------------------------------------------------------------------------
# A nested aggregate reported "not routed" while audio was reaching BlackHole
# ---------------------------------------------------------------------------
def _probe(tree: dict[int, list[int]], names: dict[int, str], default: int):
    from src.audio.blackhole_check import RoutingProbe

    return RoutingProbe(
        default_output_id=lambda: default,
        device_name=lambda did: names.get(did, f"<{did}>"),
        device_uid=lambda did: names.get(did, ""),
        aggregate_members=lambda did: tree.get(did, []),
        list_devices=lambda: list(names),
    )


def test_blackhole_inside_a_nested_aggregate_is_found() -> None:
    """One level was walked, so this said system audio was broken. It wasn't."""
    from src.audio.blackhole_check import RoutingState, check_output_routing

    tree = {1: [2, 3], 3: [4, 5]}
    names = {1: "Otis", 2: "Speakers", 3: "Inner Aggregate",
             4: "BlackHole 2ch", 5: "Headphones"}

    status = check_output_routing(_probe(tree, names, default=1))

    assert status.includes_blackhole is True
    assert status.state is RoutingState.OK


def test_a_flat_aggregate_still_works() -> None:
    from src.audio.blackhole_check import RoutingState, check_output_routing

    tree = {1: [2, 3]}
    names = {1: "Otis", 2: "Speakers", 3: "BlackHole 2ch"}
    status = check_output_routing(_probe(tree, names, default=1))
    assert status.state is RoutingState.OK


def test_a_plain_device_with_no_blackhole_is_still_not_routed() -> None:
    from src.audio.blackhole_check import RoutingState, check_output_routing

    status = check_output_routing(_probe({}, {1: "Speakers"}, default=1))
    assert status.state is RoutingState.NOT_ROUTED


def test_an_aggregate_containing_itself_visits_each_device_once() -> None:
    """macOS will not stop you building one.

    Asserted on the number of probe calls rather than the verdict: without the
    guard the walk still answers correctly, because the recursion limit fires
    inside the probe and is caught as "no children" — after a thousand stack
    frames. Correct by accident is not the same as correct.
    """
    from src.audio.blackhole_check import RoutingProbe, check_output_routing

    tree = {1: [2], 2: [1, 3]}
    names = {1: "Loop", 2: "Inner", 3: "BlackHole 2ch"}
    calls: list[int] = []

    probe = RoutingProbe(
        default_output_id=lambda: 1,
        device_name=lambda did: names.get(did, ""),
        device_uid=lambda did: names.get(did, ""),
        aggregate_members=lambda did: (calls.append(did), tree.get(did, []))[1],
        list_devices=lambda: list(names),
    )
    status = check_output_routing(probe)

    assert status.includes_blackhole is True
    assert len(calls) <= len(names) + 1, f"walked {len(calls)} times"


# ---------------------------------------------------------------------------
# Health transitions could be delivered out of order
# ---------------------------------------------------------------------------
def test_transitions_arrive_in_the_order_they_happened() -> None:
    """Callbacks fired outside the lock, so A->B and B->C could land B after C
    and leave the panel showing a state the component had already left.

    Made deterministic rather than raced: the first delivery is held open
    until a second report is provably in flight. Without the ordering lock the
    second report records and dispatches straight past the first.
    """
    from src.health import HealthRegistry, HealthState

    registry = HealthRegistry()
    seen: list[tuple[HealthState, HealthState]] = []
    first_in_callback = threading.Event()
    release_first = threading.Event()

    def gated(report, previous) -> None:
        if not first_in_callback.is_set():
            first_in_callback.set()
            release_first.wait(5.0)
        seen.append((previous, report.state))

    registry.on_change(gated)

    a = threading.Thread(
        target=lambda: registry.report("comp", label="C", state=HealthState.OK),
        daemon=True)
    a.start()
    assert first_in_callback.wait(5.0)

    b = threading.Thread(
        target=lambda: registry.report("comp", label="C", state=HealthState.WARN),
        daemon=True)
    b.start()
    time.sleep(0.1)          # give B every chance to overtake A

    assert seen == [], "a later transition was delivered before an earlier one"

    release_first.set()
    a.join(5.0)
    b.join(5.0)

    assert seen == [(HealthState.UNSET, HealthState.OK),
                    (HealthState.OK, HealthState.WARN)]


def test_a_slow_callback_does_not_lose_a_transition() -> None:
    from src.health import HealthRegistry, HealthState

    registry = HealthRegistry()
    seen: list[HealthState] = []
    registry.on_change(lambda rep, prev: seen.append(rep.state))

    for st in (HealthState.OK, HealthState.WARN, HealthState.FAIL):
        registry.report("comp", label="Component", state=st)

    assert seen == [HealthState.OK, HealthState.WARN, HealthState.FAIL]


def test_a_callback_can_still_read_the_registry_it_was_called_about() -> None:
    """The state lock must be released before dispatch, or this deadlocks."""
    from src.health import HealthRegistry, HealthState

    registry = HealthRegistry()
    read_back: list = []
    registry.on_change(
        lambda rep, prev: read_back.append(registry.get(rep.key))
    )
    registry.report("comp", label="Component", state=HealthState.OK)
    assert read_back and read_back[0].state is HealthState.OK


def test_a_silent_meeting_is_not_recorded_as_zero_minutes(tmp_path: Path) -> None:
    """Through process(), not just the helper.

    A helper test passes perfectly well when nothing calls it, which is the
    shape of the original bug.
    """
    import json
    import wave

    from src.storage.transcript_store import TranscriptStore
    from src.transcription.processor import RecordingSession, TranscriptProcessor

    audio = tmp_path / "audio"
    audio.mkdir()
    mic = audio / "sess-a_mic.wav"
    # 60 seconds of silence at 16 kHz mono 16-bit.
    with wave.open(str(mic), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 60)
    meta_path = audio / "sess-a_metadata.json"
    meta_path.write_text(json.dumps({
        "session_id": "sess-a", "mic_wav": "sess-a_mic.wav",
        "system_wav": None, "sample_rate": 16000,
        "mic_bytes": mic.stat().st_size - 44,
        "start_wall_clock": "2026-08-07T10:00:00+00:00",
    }), encoding="utf-8")

    class NoSegments:
        model_name = "small"

        def transcribe(self, path, **kwargs):
            from src.transcription.whisper_engine import TranscriptionResult

            return TranscriptionResult(segments=[], detected_language="en")

        def shutdown(self) -> None:
            pass

    store = TranscriptStore(tmp_path / "transcripts")
    processor = TranscriptProcessor(engine=NoSegments(), store=store,
                                    audio_dir=audio, archive_format="wav")
    session = RecordingSession.from_recorder_metadata(
        json.loads(meta_path.read_text(encoding="utf-8")), audio_dir=audio
    )
    processor.process(session)

    entry = store.list_transcripts(limit=5)[0]
    assert entry["duration_minutes"] == 1, (
        f"60 s of audio reported as {entry['duration_minutes']} min"
    )
    assert entry["end_time"] != entry["start_time"]
