"""Shared pytest fixtures.

All tests run against the deterministic **heuristic** backend on synthetic
transcripts, so the suite needs no GPU, no model download and no network - it
finishes in seconds and is safe for CI. The LLM backend is exercised offline via
the evaluation scripts, not the unit tests.
"""

from __future__ import annotations

import os

# Force the no-GPU path before any trustfeed import reads config.
os.environ.setdefault("TRUSTFEED_LLM_BACKEND", "heuristic")

import pytest  # noqa: E402

from trustfeed.asr import Segment, Transcript  # noqa: E402


@pytest.fixture()
def sample_transcript() -> Transcript:
    """A short synthetic news transcript with clear high- and low-value segments."""
    segments = [
        Segment(0, 0.0, 6.0, "The central bank raised interest rates by half a point today."),
        Segment(1, 6.0, 12.0, "Officials said inflation of 7 percent remains their top concern."),
        Segment(2, 12.0, 18.0, "Markets fell sharply within minutes of the surprise announcement."),
        Segment(3, 18.0, 24.0, "In lighter news, the weather stayed mild across the region."),
        Segment(4, 24.0, 31.0, '"We will do whatever it takes," the governor told reporters.'),
    ]
    return Transcript(
        language="en", language_probability=0.99, duration=31.0, segments=segments
    )
