"""Drift monitoring for the incoming media stream.

An ASR + extraction pipeline tuned on one kind of input (say, French radio news)
quietly degrades when the stream it is fed changes - a new language mix, much
longer files, or clip density collapsing (a signal the extractor has stopped
finding anything). This module quantifies that shift with the standard
**Population Stability Index (PSI)** between a reference batch (what the pipeline
was validated on) and a current batch of processed items, across three signals an
editor can reason about:

  * language distribution (categorical),
  * audio duration (numeric), and
  * clips-per-minute (numeric) - the extractor's yield.

It writes a machine-readable ``drift_report.json`` and a self-contained
``drift_report.html``. PSI is implemented directly so the maths is transparent and
the dependency surface stays small.

Interpretation (industry convention): <0.10 stable | 0.10-0.25 moderate | >0.25 significant.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trustfeed import config

logger = logging.getLogger(__name__)

STABLE, MODERATE, SIGNIFICANT = "stable", "moderate", "significant"


def _band(psi: float) -> str:
    """Map a PSI value onto its severity band."""
    if psi < 0.10:
        return STABLE
    if psi < 0.25:
        return MODERATE
    return SIGNIFICANT


def numeric_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for a continuous feature.

    Bin edges are fixed on the reference distribution (quantile-based) so the
    comparison is stable; a small epsilon avoids division by zero in empty bins.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Population Stability Index for a categorical feature."""
    categories = sorted(set(reference) | set(current))
    eps = 1e-6
    ref_pct = reference.value_counts(normalize=True).reindex(categories).fillna(0)
    cur_pct = current.value_counts(normalize=True).reindex(categories).fillna(0)
    ref_arr = np.clip(ref_pct.to_numpy(), eps, None)
    cur_arr = np.clip(cur_pct.to_numpy(), eps, None)
    return float(np.sum((cur_arr - ref_arr) * np.log(cur_arr / ref_arr)))


@dataclass
class DriftMetric:
    """One monitored signal and its drift verdict."""

    name: str
    psi: float
    band: str


@dataclass
class DriftReport:
    """Aggregate drift result across all monitored signals."""

    metrics: list[DriftMetric]
    verdict: str
    n_reference: int
    n_current: int


def _frame(records: list[dict]) -> pd.DataFrame:
    """Normalise processed-item records into a monitoring frame."""
    df = pd.DataFrame(records)
    df["duration"] = df["duration"].astype(float).clip(lower=1e-6)
    df["n_clips"] = df.get("n_clips", 0)
    df["clips_per_min"] = df["n_clips"] / (df["duration"] / 60.0)
    return df


def compute_drift(reference: list[dict], current: list[dict]) -> DriftReport:
    """Compare two batches of processed items and produce a drift report.

    Each record needs ``language``, ``duration`` (seconds) and ``n_clips``.
    """
    ref, cur = _frame(reference), _frame(current)
    metrics = [
        DriftMetric("language_distribution",
                    (a := categorical_psi(ref["language"], cur["language"])), _band(a)),
        DriftMetric("audio_duration",
                    (b := numeric_psi(ref["duration"].to_numpy(), cur["duration"].to_numpy())),
                    _band(b)),
        DriftMetric("clips_per_minute",
                    (c := numeric_psi(ref["clips_per_min"].to_numpy(), cur["clips_per_min"].to_numpy())),
                    _band(c)),
    ]
    order = {STABLE: 0, MODERATE: 1, SIGNIFICANT: 2}
    verdict = max((m.band for m in metrics), key=lambda x: order[x])
    return DriftReport(metrics, verdict, len(reference), len(current))


def _render_html(report: DriftReport) -> str:
    """Render a self-contained HTML summary of the drift report."""
    colours = {STABLE: "#1a7f37", MODERATE: "#9a6700", SIGNIFICANT: "#cf222e"}
    rows = "".join(
        f"<tr><td>{m.name}</td><td>{m.psi:.3f}</td>"
        f"<td style='color:{colours[m.band]};font-weight:600'>{m.band}</td></tr>"
        for m in report.metrics
    )
    return f"""<!doctype html><meta charset="utf-8">
<title>TrustFeed - Drift Report</title>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;color:#1c1c1c">
<h1>Input-Stream Drift Report</h1>
<p>Reference batch: {report.n_reference} items &nbsp;|&nbsp; Current batch: {report.n_current} items</p>
<p style="font-size:1.2rem">Overall verdict:
  <strong style="color:{colours[report.verdict]}">{report.verdict.upper()}</strong></p>
<table cellpadding="8" style="border-collapse:collapse;width:100%">
  <thead><tr style="text-align:left;border-bottom:2px solid #ddd">
    <th>Signal</th><th>PSI</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p style="color:#666;font-size:.85rem;margin-top:1.5rem">
  PSI &lt; 0.10 stable | 0.10-0.25 moderate | &gt; 0.25 significant.
  Significant drift is the signal to re-validate the pipeline on the new stream.</p>
</body>"""


def run(reference: list[dict], current: list[dict]) -> DriftReport:
    """Compute drift and write JSON + HTML reports to the report directory."""
    config.ensure_dirs()
    report = compute_drift(reference, current)
    payload = {
        "verdict": report.verdict,
        "n_reference": report.n_reference,
        "n_current": report.n_current,
        "metrics": [asdict(m) for m in report.metrics],
    }
    (config.REPORT_DIR / "drift_report.json").write_text(json.dumps(payload, indent=2))
    (config.REPORT_DIR / "drift_report.html").write_text(_render_html(report))
    logger.info("Drift verdict: %s", report.verdict)
    return report


if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Demo: a French-news reference stream vs. a shifted current stream (more
    # languages, longer files, lower clip yield) - the kind of shift to catch.
    rng = random.Random(42)  # noqa: S311 - demo data only, not security-sensitive
    ref = [{"language": "fr", "duration": rng.uniform(120, 360), "n_clips": rng.randint(4, 6)}
           for _ in range(80)]
    cur = [{"language": rng.choice(["fr", "de", "es", "ar"]),
            "duration": rng.uniform(300, 900), "n_clips": rng.randint(1, 4)}
           for _ in range(80)]
    result = run(ref, cur)
    print(json.dumps({"verdict": result.verdict,
                      "metrics": [asdict(m) for m in result.metrics]}, indent=2))
