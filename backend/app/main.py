"""VisionGuard FastAPI service.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .config import settings
from .model_loader import get_registry, resolve_device
from .pipeline import VisionGuardPipeline
from .schemas import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    RunSummary,
)
from .vision_utils import ImageDecodeError

__version__ = "0.1.0"

logging.basicConfig(
    level=os.getenv("VG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("visionguard.api")

# Not trained in this install (see training/train_artifacts.py); its absence is not a fault.
OPTIONAL_COMPONENTS = {"artifact_detector"}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

# Result cache: same image bytes + same settings -> same report (the pipeline is
# deterministic). Keeps the last 64 results in memory.
_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 64


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.paths.ensure()
    registry = get_registry()
    app.state.pipeline = VisionGuardPipeline(registry, settings)
    if os.getenv("VG_SKIP_WARMUP", "0") != "1":
        # Load (and download, on first boot) every model before serving traffic.
        # Failures are recorded and shown by /health; the API still starts.
        status = await run_in_threadpool(registry.warmup)
        for name, st in status.items():
            log.info("component %-24s loaded=%s %s", name, st["loaded"], st["error"] or "")
    yield


app = FastAPI(
    title="VisionGuard",
    version=__version__,
    description="Multi-layer AI image hallucination detector (research prototype).",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.runtime.cors_origins),
    allow_origin_regex=settings.runtime.cors_origin_regex,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    comps = get_registry().status()
    for name, c in comps.items():
        c["optional"] = name in OPTIONAL_COMPONENTS
    return HealthResponse(
        status="ok" if all(c["loaded"] for n, c in comps.items() if n not in OPTIONAL_COMPONENTS) else "degraded",
        version=__version__,
        device=str(resolve_device(settings.runtime.device)),
        components=comps,
        confidence_threshold=settings.arbitration.confidence_threshold,
        thresholds={
            "confidence": settings.arbitration.confidence_threshold,
            "face_asymmetry": settings.face.asymmetry_threshold,
            "hand_fail_fraction": settings.hand.anomaly_fail_fraction,
        },
    )


# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #
@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze(
    file: UploadFile = File(..., description="JPEG, PNG or WebP image"),
    sensitivity: float = Form(50.0, ge=0, le=100, description="0-100; 50 = calibrated defaults"),
    engine: str = Form("full", pattern="^(full|geometry|classifier)$"),
) -> AnalysisResponse:
    rt = settings.runtime
    if file.content_type not in rt.allowed_mime:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; allowed: {list(rt.allowed_mime)}")

    data = await file.read(rt.max_upload_bytes + 1)
    if len(data) > rt.max_upload_bytes:
        raise HTTPException(413, f"File exceeds {rt.max_upload_bytes // (1024 * 1024)} MB limit")

    pipeline: VisionGuardPipeline = app.state.pipeline
    sha = hashlib.sha256(data).hexdigest()
    key = (sha, round(sensitivity, 1), engine)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
    if cached is not None:
        body, cache = cached, "HIT"
    else:
        try:
            # CPU-bound: keep it off the event loop.
            report = await run_in_threadpool(pipeline.analyze_bytes, data, sensitivity, engine)
        except ImageDecodeError as exc:
            raise HTTPException(422, str(exc)) from exc
        body, cache = report.to_dict(), "MISS"
        with _CACHE_LOCK:
            _CACHE[key] = body
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)

    request_id = uuid.uuid4().hex[:12]
    arb = body["arbitration"]
    log.info("analyze id=%s cache=%s verdict=%s class=%s conf=%s total_ms=%s", request_id, cache,
             arb["verdict"], arb["anomaly_class"], arb["confidence"], body["timings_ms"].get("total"))
    return AnalysisResponse.model_validate({
        "request_id": request_id, "filename": file.filename, "cache": cache, "image_sha256": sha,
        "executed_at": datetime.now(timezone.utc).isoformat(), **body,
    })


# --------------------------------------------------------------------------- #
# Metrics (read from real training runs, never hardcoded)
# --------------------------------------------------------------------------- #
def _runs() -> list[Path]:
    root = settings.paths.runs
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)


@app.get("/metrics/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return [
        RunSummary(
            run_id=p.name,
            created_at=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            has_metrics=(p / "metrics.json").is_file(),
        )
        for p in _runs()
    ]


@app.get("/metrics/anatomy")
def anatomy_metrics() -> dict:
    """Latest anatomy evaluation (runs/anatomy-<date>/*.json), served verbatim.

    Written by training/eval_fingers.py (real hands: false alarms) and
    training/eval_hadm.py (labelled AI images: detection). Nothing is hardcoded.
    """
    root = settings.paths.runs
    dirs = sorted((p for p in root.glob("anatomy-*") if p.is_dir()), key=lambda p: p.name, reverse=True) if root.is_dir() else []
    if not dirs:
        return {"available": False, "message": "No anatomy evaluation found. Run training/eval_fingers.py and training/eval_hadm.py."}
    d = dirs[0]
    files = {}
    for f in sorted(d.glob("*.json")):
        if f.stem.endswith("_records"):
            continue
        try:
            files[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            files[f.stem] = None
    return {"available": True, "run_id": d.name, "files": files}


@app.get("/metrics", response_model=MetricsResponse)
def metrics(run_id: str | None = Query(None, description="Defaults to the run behind the loaded classifier, else the latest run")) -> MetricsResponse:
    if run_id is not None and not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "Invalid run_id")

    if run_id is None:
        clf = get_registry().classifier.try_get()
        if clf is not None and (settings.paths.runs / clf.run_id / "metrics.json").is_file():
            run_id = clf.run_id
        else:
            run_id = next((p.name for p in _runs() if (p / "metrics.json").is_file()), None)

    if run_id is None:
        return MetricsResponse(available=False, message="No training run found. Run `python -m training.train` first.")

    path = settings.paths.runs / run_id / "metrics.json"
    if not path.is_file():
        raise HTTPException(404, f"No metrics.json for run {run_id!r}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Corrupt metrics file for run {run_id!r}: {exc}") from exc
    # Report a stable logical path, never the host filesystem layout.
    return MetricsResponse(available=True, run_id=run_id, source=f"runs/{run_id}/metrics.json", metrics=payload)
