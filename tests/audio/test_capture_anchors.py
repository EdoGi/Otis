"""P2 item 18 — the alignment anchor must measure capture, not scheduling.

`_compute_offset` shifts every system segment by (system_anchor - mic_anchor),
so that difference is the only thing standing between the two transcripts
lining up. It used to be stamped in the *writer* thread, when it dequeued the
first chunk — behind a queue hand-off, a thread wake and the GIL.

Across four real sessions on identical hardware the resulting skew was
5.1, -7.7, 43.8 and 98.7 ms. A 20x spread is not physics; it is scheduler
noise, and it was being applied as a constant shift to the whole transcript.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.audio.recorder import _capture_timestamp


class FakeTimeInfo(SimpleNamespace):
    """Stands in for PortAudio's CallbackTimeInfo."""


def test_uses_the_hardware_capture_time_when_available() -> None:
    info = FakeTimeInfo(inputBufferAdcTime=1234.5, currentTime=1234.8)
    assert _capture_timestamp(info) == 1234.5


def test_prefers_adc_time_over_the_wall_clock() -> None:
    """The point is to exclude the latency between capture and callback."""
    info = FakeTimeInfo(inputBufferAdcTime=time.monotonic() - 0.28)
    stamped = _capture_timestamp(info)
    assert stamped < time.monotonic() - 0.2


@pytest.mark.parametrize(
    "info",
    [None, FakeTimeInfo(), FakeTimeInfo(inputBufferAdcTime=0.0),
     FakeTimeInfo(inputBufferAdcTime=None), FakeTimeInfo(inputBufferAdcTime="x")],
)
def test_falls_back_when_the_host_supplies_nothing_usable(info) -> None:
    before = time.monotonic()
    stamped = _capture_timestamp(info)
    assert before <= stamped <= time.monotonic()


def test_both_streams_degrade_together() -> None:
    """A mixed fallback would corrupt the offset by the input latency, so the
    fallback has to be deterministic per-host, not per-stream."""
    mic = _capture_timestamp(None)
    system = _capture_timestamp(None)
    assert abs(system - mic) < 0.05


# ---------------------------------------------------------------------------
# End to end through the recorder
# ---------------------------------------------------------------------------
def test_anchor_is_stamped_from_the_callback_not_the_writer(
    tmp_path, fake_sounddevice
) -> None:
    """Regression: the writer thread must no longer set the anchor. If it did,
    a stalled writer would move the anchor and skew the transcript."""
    from src.audio.recorder import DualStreamRecorder

    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)
    rec.start()
    st = rec._mic_state
    deadline = time.monotonic() + 3.0
    while st.start_monotonic is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert st.start_monotonic is not None, "the callback should have stamped it"

    # Let many more blocks flow through the writer: the anchor must not move.
    anchor = st.start_monotonic
    before_bytes = st.bytes_written
    while st.bytes_written < before_bytes + 1000 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert st.start_monotonic == anchor, "the writer thread must not restamp it"
    rec.stop()


def test_offset_between_streams_is_recorded(tmp_path, fake_sounddevice) -> None:
    from src.audio.recorder import DualStreamRecorder

    rec = DualStreamRecorder(audio_dir=tmp_path / "audio", observe_sleep_wake=False)
    rec.start()
    deadline = time.monotonic() + 3.0
    while rec._system_state.start_monotonic is None and time.monotonic() < deadline:
        time.sleep(0.01)
    meta = rec.stop()

    assert meta["mic_start_monotonic"] is not None
    assert meta["system_start_monotonic"] is not None
    skew = abs(meta["system_start_monotonic"] - meta["mic_start_monotonic"])
    assert skew < 1.0, f"implausible cross-stream skew: {skew:.3f}s"
