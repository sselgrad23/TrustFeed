"""Evaluate clip selection against a hand-labelled gold set.

The hardest question in any LLM project is "does it actually work?". For the
extraction step we answer it with a small, honest, hand-labelled set
(``labels/gold.json``): for each sample, the transcript spans an editor would clip
for distribution, with the encyclopedic/trivia sentences left as distractors.

A predicted clip **matches** a gold span when they overlap by at least
``overlap_fraction`` of the shorter of the two. From the matches we report
precision, recall and F1 per backend, plus the structured-output validity rate.
Running the heuristic and LLM backends side by side gives a like-for-like
comparison of clip-selection F1 between the two.

Extraction is scored on the **gold reference transcript** (clean text, true
timestamps) so this measures the extractor, not ASR errors - WER is evaluated
separately in :mod:`trustfeed.eval.asr_eval`.

Run as a module::

    python -m trustfeed.eval.extraction_eval --backends heuristic transformers
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib import resources

from trustfeed import config
from trustfeed.asr import Segment, Transcript
from trustfeed.extract.llm import extract
from trustfeed.extract.schema import ClipCandidate

logger = logging.getLogger(__name__)


def _load_gold() -> dict:
    """Load the packaged gold-label file."""
    with resources.files("trustfeed.eval.labels").joinpath("gold.json").open(
        encoding="utf-8"
    ) as fh:
        return json.load(fh)


def _transcript_from_reference(sample: str, language: str) -> Transcript:
    """Build a Transcript from a sample's clean reference segments."""
    path = config.SAMPLE_DIR / f"{sample}.segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(id=i, start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for i, s in enumerate(data["segments"])
    ]
    duration = segments[-1].end if segments else 0.0
    return Transcript(language=language, language_probability=1.0,
                      duration=duration, segments=segments)


def _overlap_frac(pred: ClipCandidate, gold: dict) -> float:
    """Intersection over the shorter span."""
    inter = max(0.0, min(pred.end, gold["end"]) - max(pred.start, gold["start"]))
    shorter = min(pred.duration, gold["end"] - gold["start"]) or 1.0
    return inter / shorter


def _score(preds: list[ClipCandidate], golds: list[dict], frac: float) -> tuple[int, int, int]:
    """Greedy match -> (true positives, n_pred, n_gold)."""
    matched = set()
    tp = 0
    for p in preds:
        for gi, g in enumerate(golds):
            if gi in matched:
                continue
            if _overlap_frac(p, g) >= frac:
                matched.add(gi)
                tp += 1
                break
    return tp, len(preds), len(golds)


def evaluate_backend(gold: dict, backend: str) -> dict:
    """Run one backend over every labelled sample and aggregate P/R/F1."""
    frac = gold.get("overlap_fraction", 0.5)
    tp = n_pred = n_gold = 0
    valid_first = 0.0
    n_items = 0
    for item in gold["items"]:
        if not item["gold_clips"]:
            continue  # unlabelled sample - skip in scoring
        n_items += 1
        transcript = _transcript_from_reference(item["sample"], item["language"])
        extraction = extract(transcript, backend=backend)
        a, b, c = _score(extraction.clips, item["gold_clips"], frac)
        tp += a
        n_pred += b
        n_gold += c
        valid_first += float(extraction.schema_valid_first_try)

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "backend": backend,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "schema_valid_first_try_rate": round(valid_first / n_items, 3) if n_items else None,
        "n_items": n_items,
    }


def _log_mlflow(results: list[dict]) -> None:
    """Log each backend's scores as an MLflow run (best-effort)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        for r in results:
            with mlflow.start_run(run_name=f"extraction_eval:{r['backend']}"):
                mlflow.log_params({"backend": r["backend"], "task": "clip_selection"})
                mlflow.log_metrics(
                    {k: v for k, v in r.items()
                     if isinstance(v, (int, float)) and v is not None}
                )
    except Exception as err:  # noqa: BLE001
        logger.warning("MLflow logging skipped: %s", err)


def run(backends: list[str]) -> list[dict]:
    """Evaluate the given backends and write JSON + Markdown reports."""
    config.ensure_dirs()
    gold = _load_gold()
    results = [evaluate_backend(gold, b) for b in backends]

    (config.REPORT_DIR / "extraction_eval.json").write_text(
        json.dumps({"overlap_fraction": gold.get("overlap_fraction", 0.5),
                    "results": results}, indent=2)
    )
    lines = ["# Clip-selection evaluation", "",
             f"Hand-labelled gold set | overlap >= {gold.get('overlap_fraction', 0.5)} of shorter span",
             "", "| Backend | Precision | Recall | F1 | Valid JSON (1st try) |",
             "|---|---|---|---|---|"]
    for r in results:
        v = r["schema_valid_first_try_rate"]
        lines.append(
            f"| {r['backend']} | {r['precision']} | {r['recall']} | {r['f1']} | "
            f"{'n/a' if v is None else v} |"
        )
    (config.REPORT_DIR / "extraction_eval.md").write_text("\n".join(lines) + "\n")
    _log_mlflow(results)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["heuristic"],
                        help="Backends to evaluate, e.g. heuristic transformers")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run(args.backends)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
