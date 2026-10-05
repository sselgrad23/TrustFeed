"""Extract high-value clip candidates from a transcript.

Two interchangeable backends implement the same contract:

* :class:`TransformersBackend` - a local instruct model (default Qwen2.5-3B).
  It emits JSON validated against :class:`WindowExtraction`; malformed output
  triggers one *repair* generation before we fall back. This is the project's
  structured-output story.
* :class:`HeuristicBackend` - a deterministic extractive baseline (no model, no
  GPU). It is the CI/no-GPU fallback *and* the evaluation floor the LLM must beat.

Both work one **window** at a time; :func:`extract` handles the long-context
map/reduce (window the transcript, merge overlapping clips) so a 30-minute
broadcast that overflows the model's context is still processed whole - without
retrieval, because the whole document is in hand.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from trustfeed import config
from trustfeed.asr import Segment, Transcript
from trustfeed.extract import prompts
from trustfeed.extract.schema import Chapter, ClipCandidate, Extraction, format_timestamp

logger = logging.getLogger(__name__)

# --- LLM-facing (raw) schema ------------------------------------------------


class RawClip(BaseModel):
    """A clip as the model expresses it: segment ids, not seconds."""

    start_id: int = Field(..., ge=0)
    end_id: int = Field(..., ge=0)
    title: str = Field(..., min_length=1)
    caption: str = Field(..., min_length=1)
    quote: str = ""
    rationale: str = ""


class WindowExtraction(BaseModel):
    """The strict JSON contract for a single transcript window."""

    chapter_title: str = Field("", max_length=120)
    summary: str = ""
    clips: list[RawClip] = Field(default_factory=list)


# --- helpers ----------------------------------------------------------------


def render_segments(segments: list[Segment]) -> str:
    """Render a window of segments as ``[<id> | <mm:ss>] text`` lines."""
    return "\n".join(
        f"[{s.id} | {format_timestamp(s.start)}] {s.text}" for s in segments
    )


def window_segments(
    segments: list[Segment], window_chars: int, overlap_chars: int
) -> list[list[Segment]]:
    """Split segments into overlapping windows bounded by character budget.

    Overlap re-includes the tail of the previous window so a clip that straddles a
    boundary is seen whole in at least one window.
    """
    if not segments:
        return []
    total = sum(len(s.text) for s in segments)
    if total <= window_chars:
        return [segments]

    windows: list[list[Segment]] = []
    i = 0
    n = len(segments)
    while i < n:
        window: list[Segment] = []
        chars = 0
        j = i
        while j < n and (not window or chars + len(segments[j].text) <= window_chars):
            window.append(segments[j])
            chars += len(segments[j].text)
            j += 1
        windows.append(window)
        if j >= n:
            break
        # Step back to create overlap for the next window.
        back = 0
        k = j - 1
        while k > i and back < overlap_chars:
            back += len(segments[k].text)
            k -= 1
        i = max(k + 1, i + 1)
    return windows


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerantly."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


# --- backends ---------------------------------------------------------------


class Backend(Protocol):
    """Common interface: turn one window of segments into a WindowExtraction."""

    name: str
    model: str

    def run_window(self, segments: list[Segment]) -> tuple[WindowExtraction, bool]:
        """Return ``(result, valid_first_try)`` for one window."""
        ...


class HeuristicBackend:
    """Deterministic extractive baseline - no model, no GPU, language-agnostic.

    Scores each segment by lexical salience: rare-word content (inverse document
    frequency across the window), with bonuses for numbers and quotation marks
    (both strong signals of a concrete, quotable news claim). It relies on no
    language-specific keyword lists, so it works across languages.
    """

    name = "heuristic"
    model = "tfidf-salience"

    _WORD = re.compile(r"\w+", re.UNICODE)

    def _score(self, segments: list[Segment]) -> list[float]:
        docs = [[w.lower() for w in self._WORD.findall(s.text)] for s in segments]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(segments)
        scores = []
        for seg, words in zip(segments, docs, strict=True):
            if not words:
                scores.append(0.0)
                continue
            idf = sum(math.log(1 + n / df[w]) for w in set(words))
            base = idf / math.sqrt(len(words))
            # The French opening guillemet (U+00AB) is a strong quote signal in fr text.
            has_quote = '"' in seg.text or "\u00ab" in seg.text
            bonus = 1.0 + 0.4 * bool(re.search(r"\d", seg.text)) + 0.3 * has_quote
            scores.append(base * bonus)
        return scores

    def run_window(self, segments: list[Segment]) -> tuple[WindowExtraction, bool]:
        scores = self._score(segments)
        order = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)
        picked: list[int] = []
        for i in order:
            if len(picked) >= config.MAX_CLIPS:
                break
            if all(abs(i - p) > 1 for p in picked):  # avoid adjacent duplicates
                picked.append(i)
        picked.sort()
        clips = [
            RawClip(
                start_id=segments[i].id,
                end_id=segments[i].id,
                title=f"Highlight at {format_timestamp(segments[i].start)}",
                caption=segments[i].text[:140].strip(),
                quote=segments[i].text.strip(),
                rationale="High lexical salience (rare terms / numbers / quotes).",
            )
            for i in picked
        ]
        top_terms = [w for w, _ in Counter(
            w.lower() for s in segments for w in self._WORD.findall(s.text) if len(w) > 4
        ).most_common(3)]
        return (
            WindowExtraction(
                chapter_title=" ".join(top_terms).title() or "Section",
                summary=segments[0].text.strip()[:200] if segments else "",
                clips=clips,
            ),
            True,
        )


class TransformersBackend:
    """Local instruct-model backend with JSON validation + one repair pass."""

    def __init__(self, model_name: str | None = None, load_4bit: bool | None = None):
        self.model = model_name or config.LLM_MODEL
        self.name = "transformers"
        self._load_4bit = config.LLM_LOAD_4BIT if load_4bit is None else load_4bit
        self._tok = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        kwargs: dict = {}
        if cuda and self._load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        elif cuda:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        logger.info("Loading LLM '%s' (4bit=%s, cuda=%s)...", self.model, self._load_4bit and cuda, cuda)
        self._tok = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)

    def _generate(self, messages: list[dict]) -> str:
        import torch

        assert self._tok is not None and self._model is not None  # noqa: S101 - post-load invariant for type-narrowing
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                do_sample=False,  # deterministic - reproducible extractions
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_window(self, segments: list[Segment]) -> tuple[WindowExtraction, bool]:
        self._ensure_loaded()
        system = prompts.SYSTEM_PROMPT % {"max_clips": config.MAX_CLIPS}
        user = prompts.build_user_prompt(render_segments(segments), config.MAX_CLIPS)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompts.ONESHOT_USER},
            {"role": "assistant", "content": prompts.ONESHOT_ASSISTANT},
            {"role": "user", "content": user},
        ]
        raw = self._generate(messages)
        try:
            return WindowExtraction.model_validate(_extract_json(raw)), True
        except (ValueError, ValidationError) as err:
            logger.warning("Invalid JSON from LLM, attempting one repair pass: %s", err)
            repair = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That was not valid JSON for the schema ({err}). "
                        "Reply with ONLY the corrected JSON object."
                    ),
                },
            ]
            raw2 = self._generate(repair)
            try:
                return WindowExtraction.model_validate(_extract_json(raw2)), False
            except (ValueError, ValidationError):
                logger.error("Repair pass failed; returning empty window result.")
                return WindowExtraction(), False


_BACKENDS: dict[str, Backend] = {}


def get_backend(name: str | None = None) -> Backend:
    """Resolve and cache a backend by name (``auto``/``transformers``/``heuristic``)."""
    name = name or config.LLM_BACKEND
    if name == "auto":
        name = "transformers" if _cuda_available() else "heuristic"
    if name not in _BACKENDS:
        if name == "transformers":
            _BACKENDS[name] = TransformersBackend()
        else:
            _BACKENDS[name] = HeuristicBackend()
    return _BACKENDS[name]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# --- map / reduce -----------------------------------------------------------


def _overlaps(a: ClipCandidate, b: ClipCandidate, frac: float = 0.5) -> bool:
    """True if two clips overlap by more than ``frac`` of the shorter one."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = min(a.duration, b.duration) or 1.0
    return inter / shorter > frac


def extract(
    transcript: Transcript,
    backend: Backend | str | None = None,
    max_clips: int | None = None,
) -> Extraction:
    """Run windowed extraction over a transcript and assemble an :class:`Extraction`.

    Args:
        transcript: The ASR result to mine for clip candidates.
        backend: A backend instance, a name, or ``None`` for the configured default.
        max_clips: Cap on returned clips. Defaults to :data:`config.MAX_CLIPS`.

    Returns:
        A validated :class:`Extraction` with ``status="pending_review"``.
    """
    if not isinstance(backend, (HeuristicBackend, TransformersBackend)):
        backend = get_backend(backend if isinstance(backend, str) else None)
    max_clips = max_clips or config.MAX_CLIPS
    by_id = {s.id: s for s in transcript.segments}

    started = time.perf_counter()
    windows = window_segments(
        transcript.segments, config.WINDOW_CHARS, config.WINDOW_OVERLAP_CHARS
    )
    logger.info("Extracting with %s over %d window(s)...", backend.name, len(windows))

    chapters: list[Chapter] = []
    summaries: list[str] = []
    clips: list[ClipCandidate] = []
    all_valid = True

    for window in windows:
        result, valid = backend.run_window(window)
        all_valid = all_valid and valid
        if result.summary:
            summaries.append(result.summary.strip())
        chapters.append(
            Chapter(
                start=window[0].start,
                end=window[-1].end,
                title=result.chapter_title or "Section",
                summary=result.summary.strip(),
            )
        )
        for rc in result.clips:
            s_seg, e_seg = by_id.get(rc.start_id), by_id.get(rc.end_id)
            if s_seg is None or e_seg is None:
                continue  # id hallucinated / out of range - drop, never guess a time
            start, end = min(s_seg.start, e_seg.start), max(s_seg.end, e_seg.end)
            cand = ClipCandidate(
                start=start, end=end, title=rc.title, caption=rc.caption,
                quote=rc.quote, rationale=rc.rationale,
            )
            if not any(_overlaps(cand, existing) for existing in clips):
                clips.append(cand)

    clips.sort(key=lambda c: c.start)
    clips = clips[:max_clips]
    overall = " ".join(dict.fromkeys(summaries))[:800].strip()

    return Extraction(
        language=transcript.language,
        duration=transcript.duration,
        summary=overall,
        chapters=chapters,
        clips=clips,
        backend=backend.name,
        model=backend.model,
        schema_valid_first_try=all_valid,
        extract_seconds=round(time.perf_counter() - started, 2),
    )
