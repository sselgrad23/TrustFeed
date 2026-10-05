"""Evaluate ASR word error rate on the FLEURS samples, per language.

For each prepared sample (``fleurs_<lang>.wav`` + ``.ref.txt``) this transcribes
the audio and scores it against the gold FLEURS transcript, giving a real,
per-language WER/CER the README can quote. Because Whisper auto-detects language,
the same run demonstrates the pipeline is genuinely multilingual.

Run as a module (samples must exist - see ``trustfeed.media.download_sample``)::

    python -m trustfeed.eval.asr_eval --langs fr_fr de_de
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from trustfeed import config
from trustfeed.asr import transcribe
from trustfeed.eval.wer import character_error_rate, word_error_rate

logger = logging.getLogger(__name__)


def evaluate_language(lang: str, asr_model: str | None = None) -> dict:
    """Transcribe one prepared sample and score WER/CER against its gold text."""
    wav = config.SAMPLE_DIR / f"fleurs_{lang}.wav"
    ref_path = config.SAMPLE_DIR / f"fleurs_{lang}.ref.txt"
    if not wav.exists() or not ref_path.exists():
        raise FileNotFoundError(
            f"Sample for '{lang}' missing. Run: python -m trustfeed.media.download_sample --lang {lang}"
        )
    reference = ref_path.read_text(encoding="utf-8")
    started = time.perf_counter()
    transcript = transcribe(wav, model_size=asr_model)
    elapsed = time.perf_counter() - started
    hypothesis = transcript.text
    return {
        "lang": lang,
        "detected": transcript.language,
        "wer": round(word_error_rate(reference, hypothesis), 3),
        "cer": round(character_error_rate(reference, hypothesis), 3),
        "audio_seconds": round(transcript.duration, 1),
        "asr_seconds": round(elapsed, 1),
        "realtime_factor": round(transcript.duration / max(elapsed, 1e-6), 1),
    }


def _log_mlflow(results: list[dict], asr_model: str) -> None:
    """Log per-language WER to MLflow (best-effort)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        for r in results:
            with mlflow.start_run(run_name=f"asr_eval:{r['lang']}"):
                mlflow.log_params({"asr_model": asr_model, "lang": r["lang"]})
                mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))})
    except Exception as err:  # noqa: BLE001
        logger.warning("MLflow logging skipped: %s", err)


def run(langs: list[str], asr_model: str | None = None) -> list[dict]:
    """Evaluate WER for each language and write JSON + Markdown reports."""
    config.ensure_dirs()
    results = [evaluate_language(lang, asr_model) for lang in langs]
    mean_wer = round(sum(r["wer"] for r in results) / len(results), 3) if results else 0.0

    (config.REPORT_DIR / "asr_eval.json").write_text(
        json.dumps({"asr_model": asr_model or config.ASR_MODEL,
                    "mean_wer": mean_wer, "results": results}, indent=2)
    )
    lines = ["# ASR evaluation (FLEURS)", "",
             f"Model: `{asr_model or config.ASR_MODEL}` | mean WER **{mean_wer}**", "",
             "| Lang | Detected | WER | CER | Audio (s) | ASR (s) | xRT |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['lang']} | {r['detected']} | {r['wer']} | {r['cer']} | "
            f"{r['audio_seconds']} | {r['asr_seconds']} | {r['realtime_factor']} |"
        )
    (config.REPORT_DIR / "asr_eval.md").write_text("\n".join(lines) + "\n")
    _log_mlflow(results, asr_model or config.ASR_MODEL)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="+", default=["fr_fr", "de_de"])
    parser.add_argument("--asr-model", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run(args.langs, args.asr_model)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
