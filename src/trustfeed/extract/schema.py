"""Typed, validated shape of an extraction result.

These Pydantic models are the project's *structured-output contract*. The LLM is
asked to emit JSON that validates against :class:`ClipCandidate` / :class:`Chapter`;
anything malformed is rejected and repaired (see :mod:`trustfeed.extract.llm`)
rather than shipped downstream. Every field the editorial UI and the eval depend
on - clip in/out points, captions, the human-review flag - is declared here once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

ReviewStatus = Literal["pending_review", "approved", "rejected"]


def format_timestamp(seconds: float) -> str:
    """Format a number of seconds as ``H:MM:SS`` (or ``MM:SS`` under an hour)."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ClipCandidate(BaseModel):
    """One proposed social clip: a timestamped span plus its editorial packaging."""

    start: float = Field(..., ge=0, description="Clip start, seconds into the audio.")
    end: float = Field(..., gt=0, description="Clip end, seconds into the audio.")
    title: str = Field(..., min_length=1, description="Short internal headline.")
    caption: str = Field(
        ..., min_length=1,
        description="Attention-catching social caption, in the audio's language.",
    )
    quote: str = Field(
        "", description="Verbatim key line from the transcript for this span."
    )
    rationale: str = Field(
        "", description="Why this segment is high-value for distribution."
    )
    # Human-in-the-loop: None = awaiting review, True/False = editor's decision.
    approved: bool | None = Field(
        default=None, description="Editor decision; None until reviewed."
    )

    @model_validator(mode="after")
    def _check_span(self) -> ClipCandidate:
        """Ensure the clip has positive duration; clamp tiny inversions."""
        if self.end <= self.start:
            self.end = self.start + 1.0
        return self

    @computed_field  # type: ignore[prop-decorator]  # serialised for API clients
    @property
    def duration(self) -> float:
        """Clip length in seconds."""
        return round(self.end - self.start, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timespan(self) -> str:
        """Human-readable ``MM:SS-MM:SS`` label for the UI."""
        return f"{format_timestamp(self.start)}-{format_timestamp(self.end)}"


class Chapter(BaseModel):
    """A coarse thematic section of the broadcast, for navigation."""

    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    title: str = Field(..., min_length=1)
    summary: str = Field("", description="One-line summary of the chapter.")


class Extraction(BaseModel):
    """The full result for one processed item - the API/UI payload."""

    language: str = Field(..., description="Detected language (ISO code).")
    duration: float = Field(..., ge=0, description="Audio duration in seconds.")
    summary: str = Field("", description="Overall summary of the broadcast.")
    chapters: list[Chapter] = Field(default_factory=list)
    clips: list[ClipCandidate] = Field(default_factory=list)
    status: ReviewStatus = Field(
        default="pending_review",
        description="Editorial gate: nothing distributes until 'approved'.",
    )
    # --- observability / provenance ---
    backend: str = Field("", description="Extraction backend used.")
    model: str = Field("", description="Model identifier, when applicable.")
    schema_valid_first_try: bool = Field(
        default=True,
        description="True if the LLM's first JSON output validated (no repair pass).",
    )
    asr_seconds: float = Field(0.0, description="Wall-clock ASR time.")
    extract_seconds: float = Field(0.0, description="Wall-clock extraction time.")

    @property
    def approved_clips(self) -> list[ClipCandidate]:
        """Clips an editor has explicitly approved - the only publishable ones."""
        return [c for c in self.clips if c.approved is True]
