"""Transcribe audio/video to timestamped segments with faster-whisper.

`faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ runs the Whisper
model through CTranslate2: 4x faster and far lighter on memory than the reference
implementation, CPU- or GPU-capable, and free. It decodes the audio track of most
media files directly (via PyAV/ffmpeg), so a ``.mp4`` video works as input too.

Whisper detects the spoken language automatically, which is what lets the rest of
the pipeline stay language-agnostic. Segment timestamps are the backbone of the
product: chapters, clip in/out points and the review UI all key off them.

Run as a module::

    python -m trustfeed.asr.transcribe path/to/audio.mp3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trustfeed import config

logger = logging.getLogger(__name__)

# Cache loaded models by (name, device, compute_type). Loading weights is the
# slow part; a warm process should reuse them across requests.
_MODELS: dict[tuple[str, str, str], Any] = {}


@dataclass
class Segment:
    """One timestamped chunk of transcript."""

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the segment."""
        return {"id": self.id, "start": self.start, "end": self.end, "text": self.text}


@dataclass
class Transcript:
    """A full transcription result."""

    language: str
    language_probability: float
    duration: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The whole transcript as a single string."""
        return " ".join(s.text.strip() for s in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the transcript."""
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "duration": round(self.duration, 2),
            "segments": [s.to_dict() for s in self.segments],
        }


def _resolve_device(device: str) -> str:
    """Map ``"auto"`` to ``"cuda"`` when a GPU is present, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick a memory-frugal compute type when the caller left it on ``default``."""
    if compute_type != "default":
        return compute_type
    return "int8_float16" if device == "cuda" else "int8"


def _get_model(name: str, device: str, compute_type: str) -> Any:
    """Load (and cache) a faster-whisper model."""
    key = (name, device, compute_type)
    if key not in _MODELS:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper '%s' on %s (%s)...", name, device, compute_type)
        _MODELS[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _MODELS[key]


def transcribe(
    audio_path: str | Path,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    beam_size: int | None = None,
) -> Transcript:
    """Transcribe an audio or video file into a :class:`Transcript`.

    Args:
        audio_path: Path to an audio or video file (anything ffmpeg can decode).
        model_size: Whisper size (``tiny``/``base``/``small``/``medium``/``large-v3``).
            Defaults to :data:`config.ASR_MODEL`.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``. Defaults to :data:`config.ASR_DEVICE`.
        compute_type: CTranslate2 compute type, or ``"default"`` to auto-pick a
            low-memory one for the device.
        language: ISO code to force a language, or ``None`` to auto-detect (the
            default, and what keeps the pipeline multi-lingual).
        beam_size: Decoding beam width. Defaults to :data:`config.ASR_BEAM_SIZE`.

    Returns:
        A :class:`Transcript` with the detected language and timestamped segments.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    device = _resolve_device(device or config.ASR_DEVICE)
    compute_type = _resolve_compute_type(compute_type or config.ASR_COMPUTE_TYPE, device)
    model = _get_model(model_size or config.ASR_MODEL, device, compute_type)

    started = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size or config.ASR_BEAM_SIZE,
        vad_filter=True,  # drop long silences so timestamps track speech, not gaps
    )
    segments = [
        Segment(id=i, start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
        for i, s in enumerate(segments_iter)
    ]
    elapsed = time.perf_counter() - started
    logger.info(
        "Transcribed %.1fs of audio in %.1fs (%.1fx RT), lang=%s p=%.2f, %d segments",
        info.duration, elapsed, info.duration / max(elapsed, 1e-6),
        info.language, info.language_probability, len(segments),
    )
    return Transcript(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
    )


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m trustfeed.asr.transcribe <audio_or_video>")
    result = transcribe(sys.argv[1])
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
