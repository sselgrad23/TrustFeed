"""Tests for the extraction pipeline (heuristic backend)."""

from __future__ import annotations

from trustfeed.asr import Segment
from trustfeed.extract.llm import (
    HeuristicBackend,
    _extract_json,
    extract,
    render_segments,
    window_segments,
)


def test_heuristic_extraction_is_well_formed(sample_transcript) -> None:
    ext = extract(sample_transcript, backend=HeuristicBackend())
    assert ext.backend == "heuristic"
    assert ext.status == "pending_review"
    assert ext.schema_valid_first_try is True
    assert ext.language == "en"
    assert 1 <= len(ext.clips) <= 6
    for clip in ext.clips:
        assert 0 <= clip.start < clip.end <= sample_transcript.duration
        assert clip.caption


def test_heuristic_prefers_salient_over_filler(sample_transcript) -> None:
    """The mild-weather filler line should not out-rank the news lines."""
    ext = extract(sample_transcript, backend=HeuristicBackend())
    covered = [
        any(c.start <= seg.start and c.end >= seg.end for c in ext.clips)
        for seg in sample_transcript.segments
    ]
    # The filler segment (index 3) is the least salient; it should be left out
    # while at least one hard-news segment is picked.
    assert covered[3] is False
    assert any(covered)


def test_extract_is_deterministic(sample_transcript) -> None:
    a = extract(sample_transcript, backend=HeuristicBackend())
    b = extract(sample_transcript, backend=HeuristicBackend())
    assert [c.timespan for c in a.clips] == [c.timespan for c in b.clips]


def test_windowing_covers_all_segments_with_overlap() -> None:
    segs = [Segment(i, i * 5.0, i * 5.0 + 5.0, "word " * 20) for i in range(50)]
    windows = window_segments(segs, window_chars=400, overlap_chars=80)
    assert len(windows) > 1
    seen = {s.id for w in windows for s in w}
    assert seen == {s.id for s in segs}  # no segment dropped


def test_render_segments_includes_ids_and_timestamps(sample_transcript) -> None:
    rendered = render_segments(sample_transcript.segments)
    assert "[0 | 00:00]" in rendered
    assert "[4 | 00:24]" in rendered


def test_extract_json_tolerates_fences_and_prose() -> None:
    raw = 'Here you go:\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone.'
    assert _extract_json(raw) == {"a": 1, "b": [2, 3]}
