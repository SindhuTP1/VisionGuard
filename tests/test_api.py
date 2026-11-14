import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_components(client):
    j = client.get("/health").json()
    assert set(j["components"]) == {"yolov8_person", "mediapipe_hands", "mediapipe_face", "mobilenetv3_classifier"}
    assert j["confidence_threshold"] == settings.arbitration.confidence_threshold


def test_rejects_wrong_content_type(client):
    assert client.post("/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/analyze", files={"file": ("x.jpg", b"\xff\xd8garbage", "image/jpeg")})
    assert r.status_code == 422


def test_rejects_oversized_upload(client):
    big = b"\0" * (settings.runtime.max_upload_bytes + 1)
    assert client.post("/analyze", files={"file": ("x.png", big, "image/png")}).status_code == 413


def test_metrics_without_runs_is_honest(client):
    j = client.get("/metrics").json()
    assert j["available"] is False and j["metrics"] is None


def test_metrics_serves_run_file_verbatim(client):
    run = settings.paths.runs / "unit-run"
    run.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": "unit-run", "test": {"accuracy": 0.123}}
    (run / "metrics.json").write_text(json.dumps(payload))
    j = client.get("/metrics", params={"run_id": "unit-run"}).json()
    assert j["available"] and j["metrics"] == payload
    assert any(r["run_id"] == "unit-run" for r in client.get("/metrics/runs").json())


@pytest.mark.parametrize("bad", ["../../etc", "a/b", "..\\x"])
def test_metrics_rejects_path_traversal(client, bad):
    assert client.get("/metrics", params={"run_id": bad}).status_code == 400
