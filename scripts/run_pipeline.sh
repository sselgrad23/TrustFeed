#!/usr/bin/env bash
# End-to-end offline run: fetch samples -> ASR WER eval -> clip-selection eval ->
# drift report. Mirrors `make sample eval drift`; provided for environments
# without make. The transformers backend needs the LLM extras and a GPU is
# recommended; the eval falls back to the heuristic backend without one.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] Fetching FLEURS samples (fr, de)..."
python -m trustfeed.media.download_sample --lang fr_fr
python -m trustfeed.media.download_sample --lang de_de

echo "[2/4] ASR WER evaluation..."
python -m trustfeed.eval.asr_eval --langs fr_fr de_de

echo "[3/4] Clip-selection evaluation (heuristic vs LLM)..."
python -m trustfeed.eval.extraction_eval --backends heuristic transformers || {
  echo "  LLM backend unavailable; scoring heuristic only."
  python -m trustfeed.eval.extraction_eval --backends heuristic
}

echo "[4/4] Drift report..."
python -m trustfeed.monitoring.drift

echo "Done. Reports in ./reports, MLflow in ./mlflow.db"
