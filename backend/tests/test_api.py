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
    assert j["thresholds"]["face_asymmetry"] == settings.face.asymmetry_threshold
    assert j["thresholds"]["hand_fail_fraction"] == settings.hand.anomaly_fail_fraction


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


@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:3000", True),
    ("http://localhost:3001", True),      # next dev falls back to 3001 when 3000 is busy
    ("http://127.0.0.1:5173", True),
    ("https://evil.example.com", False),
    ("http://localhost.evil.com", False),
])
def test_cors_allows_any_local_port_only(client, origin, allowed):
    r = client.options("/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed


def test_anatomy_metrics_served_verbatim(client):
    assert client.get("/metrics/anatomy").json()["available"] is False
    d = settings.paths.runs / "anatomy-20990101"
    d.mkdir(parents=True, exist_ok=True)
    (d / "finger_count_test.json").write_text(json.dumps({"hands_counted": 7}))
    (d / "hadm_eval_records.json").write_text("[]")          # raw records are not served
    j = client.get("/metrics/anatomy").json()
    assert j["available"] and j["run_id"] == "anatomy-20990101"
    assert j["files"] == {"finger_count_test": {"hands_counted": 7}}


@pytest.mark.parametrize("bad", ["../../etc", "a/b", "..\\x"])
def test_metrics_rejects_path_traversal(client, bad):
    assert client.get("/metrics", params={"run_id": bad}).status_code == 400
