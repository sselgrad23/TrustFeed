"""End-to-end pipeline: audio/video -> transcript -> review-ready clip candidates.

This is the single entry point the API and the demo call. It stitches ASR and
extraction together, records per-stage timing for observability, and (optionally)
logs the run to MLflow so latency and output shape are tracked over time - the
same experiment-tracking backbone used for the offline evals.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from trustfeed import config
from trustfeed.asr import Transcript, transcribe
from trustfeed.extract.llm import Backend, extract
from trustfeed.extract.schema import Extraction

logger = logging.getLogger(__name__)


def process_transcript(
    transcript: Transcript, backend: Backend | str | None = None
) -> Extraction:
    """Run extraction on an existing transcript (skips ASR)."""
    return extract(transcript, backend=backend)


def process(
    audio_path: str | Path,
    backend: Backend | str | None = None,
    asr_model: str | None = None,
    log_mlflow: bool = False,
) -> Extraction:
    """Transcribe a media file and extract review-ready clip candidates.

    Args:
        audio_path: Path to an audio or video file.
        backend: Extraction backend (name/instance) or ``None`` for the default.
        asr_model: Override the Whisper size for this call.
        log_mlflow: If True, log timing/shape metrics to MLflow.

    Returns:
        An :class:`Extraction` with ``status="pending_review"`` awaiting an editor.
    """
    started = time.perf_counter()
    transcript = transcribe(audio_path, model_size=asr_model)
    asr_seconds = round(time.perf_counter() - started, 2)

    extraction = extract(transcript, backend=backend)
    extraction.asr_seconds = asr_seconds

    logger.info(
        "Processed %s: %.1fs audio, %s, %d clips (asr=%.1fs extract=%.1fs)",
        Path(audio_path).name, transcript.duration, transcript.language,
        len(extraction.clips), asr_seconds, extraction.extract_seconds,
    )

    if log_mlflow:
        _log_run(audio_path, extraction)
    return extraction


def _log_run(audio_path: str | Path, extraction: Extraction) -> None:
    """Log a processing run to MLflow (best-effort; never breaks the pipeline)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"process:{Path(audio_path).name}"):
            mlflow.log_params(
                {"backend": extraction.backend, "model": extraction.model,
                 "language": extraction.language}
            )
            mlflow.log_metrics(
                {
                    "audio_seconds": extraction.duration,
                    "asr_seconds": extraction.asr_seconds,
                    "extract_seconds": extraction.extract_seconds,
                    "n_clips": len(extraction.clips),
                    "n_chapters": len(extraction.chapters),
                    "schema_valid_first_try": float(extraction.schema_valid_first_try),
                }
            )
    except Exception as err:  # noqa: BLE001 - telemetry must never break processing
        logger.warning("MLflow logging skipped: %s", err)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m trustfeed.extract.pipeline <audio_or_video>")
    result = process(sys.argv[1], log_mlflow=True)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
