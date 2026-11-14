"""API response models.

These mirror the pipeline dataclasses. ``main.py`` validates every report
through them, so any drift between pipeline output and the API contract fails
loudly instead of silently changing the JSON the frontend receives.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import AnomalyClass, Verdict

LayerStatusT = Literal["ok", "skipped", "unavailable", "error"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class PersonOut(_Strict):
    status: LayerStatusT
    bbox: list[int] = Field(description="Region analysed by later layers: x1, y1, x2, y2")
    used_fallback: bool
    num_persons: int
    person_score: float | None = None
    note: str | None = None
    person_boxes: list[list[int]] = Field(default_factory=list, description="All person detections, largest first")


class FingerOut(_Strict):
    length_ratio: float | None
    length_ok: bool
    phalanx_ratio: float | None
    phalanx_ok: bool
    max_hyperextension_deg: float | None
    hyperextension_ok: bool
    extended: bool
    ok: bool


class HandOut(_Strict):
    index: int
    handedness: str | None
    handedness_score: float | None
    palm_width_px: float | None
    extended_fingers: int | None
    fingers: dict[str, FingerOut]
    failed_fingers: list[str]
    fail_fraction: float
    anomalous: bool
    skipped_reason: str | None


class HandsOut(_Strict):
    status: LayerStatusT
    num_hands: int
    hands: list[HandOut]
    anomalous: bool
    max_fail_fraction: float | None
    note: str | None


class PairScore(_Strict):
    pair: list[int]
    score: float | None


class FaceOut(_Strict):
    index: int
    iod_px: float | None
    roll_deg: float | None
    yaw_ratio: float | None
    asymmetry_score: float | None
    lateral_mean: float | None
    vertical_mean: float | None
    midline_deviation: float | None
    worst_pairs: list[PairScore]
    anomalous: bool
    skipped_reason: str | None


class FacesOut(_Strict):
    status: LayerStatusT
    num_faces: int
    faces: list[FaceOut]
    anomalous: bool
    max_score: float | None
    note: str | None


class ClassifierOut(_Strict):
    status: LayerStatusT
    label: Literal["real", "fake"] | None
    p_fake: float | None
    p_real: float | None
    confidence: float | None
    run_id: str | None
    input_region: str
    note: str | None


class ArbitrationOut(_Strict):
    verdict: Verdict
    anomaly_class: AnomalyClass | None
    attribution: Literal["face_geometry", "hand_geometry", "classifier_only", "classifier", "none"]
    confidence: float | None
    reasons: list[str]
    signals: dict[str, float | None]


class ImageInfo(_Strict):
    width: int
    height: int


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class AnalysisResponse(_Strict):
    request_id: str
    filename: str | None
    image: ImageInfo
    person: PersonOut
    hands: HandsOut
    face: FacesOut
    classifier: ClassifierOut
    arbitration: ArbitrationOut
    timings_ms: dict[str, float]


class ComponentStatus(_Strict):
    loaded: bool
    error: str | None
    load_seconds: float | None


class HealthResponse(_Strict):
    status: Literal["ok", "degraded"]
    version: str
    device: str
    components: dict[str, ComponentStatus]
    confidence_threshold: float
    thresholds: dict[str, float] = Field(description="Decision thresholds from config.py, so clients never hardcode them")


class RunSummary(_Strict):
    run_id: str
    created_at: str | None = None
    has_metrics: bool


class MetricsResponse(BaseModel):
    """Metrics are copied verbatim from ``runs/<run_id>/metrics.json``."""

    available: bool
    run_id: str | None = None
    source: str | None = None
    metrics: dict[str, Any] | None = None
    message: str | None = None


class ErrorResponse(_Strict):
    detail: str
