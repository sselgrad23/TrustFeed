"""Prompt templates for the extraction LLM.

Design choices that matter here:

* **Grounded timestamps.** The model references *segment ids*, not raw seconds.
  We map ids back to the real Whisper timestamps ourselves, so a clip can never
  point at a hallucinated time - a small but important reliability win for a
  non-deterministic component.
* **Verbatim quotes.** The trust mission means captions must not fabricate; the
  model is told to copy the key line verbatim and write captions in the audio's
  own language.
* **Strict JSON.** Output is a single JSON object matching
  :class:`~trustfeed.extract.llm.WindowExtraction`; anything else is repaired.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an editorial assistant for a newsroom that fights \
misinformation by getting accurate reporting onto social media quickly. You are \
given one section of a timestamped transcript of a news broadcast. Each line is:

    [<id> | <mm:ss>] text

Your job: select the 3 to %(max_clips)d segments most worth clipping for fast, \
responsible social distribution. Err towards INCLUDING a segment if it carries any \
of: a concrete event, a casualty count or statistic, a direct quote, a sports \
result, or a named person or institution making a claim. Only skip passages that \
are purely descriptive, instructional or encyclopedic. Aim for good coverage - \
missing a newsworthy moment is worse than including a borderline one.

Return ONE JSON object, and nothing else, with exactly these keys:
{
  "chapter_title": "<=8 word title for this whole section",
  "summary": "1-2 sentence neutral summary of this section",
  "clips": [
    {
      "start_id": <integer segment id where the clip starts>,
      "end_id": <integer segment id where the clip ends, >= start_id>,
      "title": "short internal headline (English)",
      "caption": "punchy social caption in the SAME language as the transcript",
      "quote": "the single most important line, copied VERBATIM from the transcript",
      "rationale": "one line (English): why this is worth distributing"
    }
  ]
}

Rules:
- Use only segment ids that appear in the section. end_id >= start_id.
- A clip is usually 1-4 segments long. Return between 3 and %(max_clips)d clips \
when the section has that many newsworthy moments.
- Do not invent facts. The quote must be copied exactly from a transcript line.
- Captions must be accurate, not clickbait - the newsroom's credibility depends on it.
- Output only the JSON object. No markdown, no commentary."""


ONESHOT_USER = """[0 | 00:00] The central bank raised interest rates by half a point today.
[1 | 00:06] Officials said inflation remains their top concern going into the winter.
[2 | 00:12] Markets fell sharply within minutes of the announcement."""

ONESHOT_ASSISTANT = """{"chapter_title": "Central bank raises rates", \
"summary": "The central bank raised rates by half a point, citing inflation, and \
markets fell.", "clips": [{"start_id": 0, "end_id": 1, "title": "Rate hike on \
inflation fears", "caption": "Central bank hikes rates half a point - inflation \
still the top worry.", "quote": "The central bank raised interest rates by half a \
point today.", "rationale": "Concrete, verifiable policy decision."}, {"start_id": \
2, "end_id": 2, "title": "Markets drop on the news", "caption": "Markets fell \
within minutes of the surprise rate decision.", "quote": "Markets fell sharply \
within minutes of the announcement.", "rationale": "Immediate, measurable market \
reaction."}]}"""


def build_user_prompt(section_lines: str, max_clips: int) -> str:
    """Wrap a rendered transcript section into the user turn."""
    return f"Transcript section:\n{section_lines}\n\nReturn the JSON object now."
