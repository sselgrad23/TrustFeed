"""End-to-end tests for the FastAPI service (heuristic backend, in-process)."""

from __future__ import annotations

from fastapi.testclient import TestClient

import trustfeed.api.main as api_main

client = TestClient(api_main.app)

_DOC = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 6.0, "text": "The central bank raised interest rates by half a point."},
        {"start": 6.0, "end": 12.0, "text": "Officials said inflation of 7 percent remains a concern."},
        {"start": 12.0, "end": 18.0, "text": "Markets fell sharply after the surprise announcement."},
    ],
}


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_config_lists_channels() -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    assert "distribution_channels" in resp.json()


def test_extract_returns_pending_job() -> None:
    resp = client.post("/extract", json=_DOC)
    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction"]["status"] == "pending_review"
    assert body["extraction"]["clips"]
    # Fetchable by id.
    got = client.get(f"/jobs/{body['job_id']}")
    assert got.status_code == 200


def test_review_gate_approves_and_publishes() -> None:
    job = client.post("/extract", json=_DOC).json()
    job_id = job["job_id"]
    resp = client.post(
        f"/jobs/{job_id}/review",
        json={"decisions": [{"index": 0, "approved": True}], "publish": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction"]["status"] == "approved"
    assert len(body["published"]) == 1
    assert body["published"][0]["channels"]


def test_review_rejects_when_nothing_approved() -> None:
    job = client.post("/extract", json=_DOC).json()
    resp = client.post(
        f"/jobs/{job['job_id']}/review",
        json={"decisions": [{"index": 0, "approved": False}], "publish": True},
    )
    assert resp.json()["extraction"]["status"] == "rejected"


def test_unknown_job_is_404() -> None:
    assert client.get("/jobs/deadbeef").status_code == 404


def test_empty_document_is_rejected() -> None:
    assert client.post("/extract", json={"language": "en", "segments": []}).status_code == 422
