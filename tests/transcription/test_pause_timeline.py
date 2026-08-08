"""Regression cover for audit finding B1 — pauses corrupt every timestamp.

``recorder.py`` writes a ``pauses`` list into the session metadata, and nothing
in ``src/`` ever read it. Pause *drops* frames rather than writing silence, so
the WAV timeline is compressed by exactly the pause duration while the
transcript is stamped as if it ran continuously.

Verified on the user's real 2026-08-07 session:

    start_wall_clock  12:01 local
    pauses            [{paused_at: 304.6, resumed_at: 1838.8}]   -> 25.6 min
    mic_bytes         85 753 856 / 32 000                        -> 44.7 min audio
    real end (mtime)  13:12                                      -> 71 min elapsed
    frontmatter said  end_time 12:46, duration_minutes 45        -> WRONG

Every segment after 05:04 was mislabelled by ~26 minutes.
"""

from __future__ import annotations

import pytest

from src.transcription.processor import PauseTimeline


class TestPauseTimeline:
    def test_no_pauses_is_the_identity(self) -> None:
        tl = PauseTimeline([])
        assert tl.to_wall(0.0) == 0.0
        assert tl.to_wall(123.4) == 123.4
        assert tl.total_pause_seconds == 0.0

    def test_the_real_2026_08_07_session(self) -> None:
        """The exact numbers from the broken transcript on disk."""
        tl = PauseTimeline([{"paused_at": 304.6, "resumed_at": 1838.8}])

        # Before the pause, audio time == wall time.
        assert tl.to_wall(0.0) == pytest.approx(0.0)
        assert tl.to_wall(300.0) == pytest.approx(300.0)

        # After the pause, everything shifts by the 1534.2 s gap.
        assert tl.to_wall(304.6) == pytest.approx(1838.8)
        assert tl.to_wall(400.0) == pytest.approx(1934.2)

        # The last segment sat at 44:31 of audio (2671 s). In wall-clock terms
        # that is 4205 s = 70.1 min after 12:01, i.e. ~13:11 — matching the
        # WAV's 13:12 mtime instead of the 12:46 the transcript claimed.
        assert tl.to_wall(2671.0) == pytest.approx(4205.2)
        assert tl.total_pause_seconds == pytest.approx(1534.2)

    def test_multiple_pauses_accumulate(self) -> None:
        tl = PauseTimeline([
            {"paused_at": 100.0, "resumed_at": 160.0},   # 60 s gap
            {"paused_at": 300.0, "resumed_at": 330.0},   # 30 s gap
        ])
        # paused_at is wall-elapsed, so the second pause begins at audio
        # position 300 - 60 = 240.
        assert tl.to_wall(50.0) == pytest.approx(50.0)
        assert tl.to_wall(100.0) == pytest.approx(160.0)
        assert tl.to_wall(239.0) == pytest.approx(299.0)
        assert tl.to_wall(240.0) == pytest.approx(330.0)
        assert tl.to_wall(300.0) == pytest.approx(390.0)
        assert tl.total_pause_seconds == pytest.approx(90.0)

    def test_is_monotonic_non_decreasing(self) -> None:
        tl = PauseTimeline([
            {"paused_at": 10.0, "resumed_at": 20.0},
            {"paused_at": 40.0, "resumed_at": 100.0},
        ])
        previous = -1.0
        for t in [x * 0.5 for x in range(200)]:
            wall = tl.to_wall(t)
            assert wall >= previous
            previous = wall

    # ---------------------------------------------------------------- garbage
    def test_ignores_malformed_entries(self) -> None:
        tl = PauseTimeline([
            {"paused_at": 100.0, "resumed_at": 160.0},
            {"paused_at": None, "resumed_at": 200.0},      # never paused
            {"paused_at": 300.0},                          # still paused / crashed
            "not a dict",                                  # junk
            {"paused_at": 500.0, "resumed_at": 400.0},     # negative duration
        ])
        assert tl.total_pause_seconds == pytest.approx(60.0)

    def test_none_metadata_is_safe(self) -> None:
        assert PauseTimeline(None).to_wall(42.0) == 42.0

    def test_unsorted_pauses_are_ordered_first(self) -> None:
        tl = PauseTimeline([
            {"paused_at": 300.0, "resumed_at": 330.0},
            {"paused_at": 100.0, "resumed_at": 160.0},
        ])
        assert tl.total_pause_seconds == pytest.approx(90.0)
        assert tl.to_wall(240.0) == pytest.approx(330.0)


class TestSegmentShifting:
    def test_segments_are_shifted_past_the_pause(self) -> None:
        from src.transcription.whisper_engine import Segment

        tl = PauseTimeline([{"paused_at": 300.0, "resumed_at": 1800.0}])
        segments = [
            Segment(start=10.0, end=15.0, text="before the pause"),
            Segment(start=310.0, end=320.0, text="after the pause"),
        ]
        shifted = tl.shift_segments(segments)

        assert shifted[0].start == pytest.approx(10.0)
        assert shifted[0].end == pytest.approx(15.0)
        assert shifted[0].text == "before the pause"
        # 310 s of audio happened 1500 s of wall-clock later than it looks.
        assert shifted[1].start == pytest.approx(1810.0)
        assert shifted[1].end == pytest.approx(1820.0)

    def test_shifting_without_pauses_returns_equivalent_segments(self) -> None:
        from src.transcription.whisper_engine import Segment

        segments = [Segment(start=1.0, end=2.0, text="hi")]
        out = PauseTimeline([]).shift_segments(segments)
        assert [(s.start, s.end, s.text) for s in out] == [(1.0, 2.0, "hi")]
