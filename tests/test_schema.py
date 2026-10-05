"""Unit tests for the structured-output schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustfeed.extract.schema import ClipCandidate, Extraction, format_timestamp


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (5, "00:05"), (65, "01:05"), (3661, "1:01:01"), (-3, "00:00")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_clip_span_is_clamped_when_inverted() -> None:
    clip = ClipCandidate(start=10.0, end=5.0, title="t", caption="c")
    assert clip.end > clip.start
    assert clip.duration > 0


def test_clip_timespan_label() -> None:
    clip = ClipCandidate(start=65.0, end=72.0, title="t", caption="c")
    assert clip.timespan == "01:05-01:12"


def test_empty_caption_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ClipCandidate(start=0.0, end=1.0, title="t", caption="")


def test_approved_clips_filters_on_decision() -> None:
    ext = Extraction(
        language="fr", duration=10.0,
        clips=[
            ClipCandidate(start=0, end=2, title="a", caption="a", approved=True),
            ClipCandidate(start=3, end=4, title="b", caption="b", approved=False),
            ClipCandidate(start=5, end=6, title="c", caption="c"),  # unreviewed
        ],
    )
    assert len(ext.approved_clips) == 1
    assert ext.status == "pending_review"
