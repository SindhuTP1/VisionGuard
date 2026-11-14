"""VisionGuard FastAPI service.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


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
    return HealthResponse(
        status="ok" if all(c["loaded"] for c in comps.values()) else "degraded",
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
async def analyze(file: UploadFile = File(..., description="JPEG, PNG or WebP image")) -> AnalysisResponse:
    rt = settings.runtime
    if file.content_type not in rt.allowed_mime:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; allowed: {list(rt.allowed_mime)}")

    data = await file.read(rt.max_upload_bytes + 1)
    if len(data) > rt.max_upload_bytes:
        raise HTTPException(413, f"File exceeds {rt.max_upload_bytes // (1024 * 1024)} MB limit")

    pipeline: VisionGuardPipeline = app.state.pipeline
    try:
        # CPU-bound: keep it off the event loop.
        report = await run_in_threadpool(pipeline.analyze_bytes, data)
    except ImageDecodeError as exc:
        raise HTTPException(422, str(exc)) from exc

    request_id = uuid.uuid4().hex[:12]
    arb = report.arbitration
    log.info("analyze id=%s verdict=%s class=%s conf=%s total_ms=%s",
             request_id, arb.verdict.value, getattr(arb.anomaly_class, "value", None),
             arb.confidence, report.timings_ms.get("total"))
    return AnalysisResponse.model_validate(
        {"request_id": request_id, "filename": file.filename, **report.to_dict()}
    )


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
