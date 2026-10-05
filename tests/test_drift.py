"""Tests for the PSI drift monitor."""

from __future__ import annotations

from trustfeed.monitoring.drift import compute_drift, numeric_psi


def test_identical_batches_are_stable() -> None:
    records = [{"language": "fr", "duration": 200.0, "n_clips": 5} for _ in range(40)]
    report = compute_drift(records, records)
    assert report.verdict == "stable"
    assert all(m.psi < 0.10 for m in report.metrics)


def test_shifted_stream_is_flagged() -> None:
    ref = [{"language": "fr", "duration": 200.0, "n_clips": 5} for _ in range(40)]
    cur = [{"language": "ar", "duration": 800.0, "n_clips": 1} for _ in range(40)]
    report = compute_drift(ref, cur)
    assert report.verdict == "significant"


def test_numeric_psi_zero_for_same_distribution() -> None:
    import numpy as np

    x = np.linspace(0, 100, 500)
    assert numeric_psi(x, x) < 1e-6
