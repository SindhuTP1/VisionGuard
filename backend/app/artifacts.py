"""Layer 2d: learned anatomy-artefact detector.

A YOLOv8n fine-tuned on HADM (AI images where people drew boxes around
malformed hands, faces and limbs, and tagged people with extra or missing
limbs). Unlike the hand-written geometry checks, it learned what generator
mistakes look like from labelled examples. Its confidence threshold was chosen
on HADM's validation half and ships with the weights. Its measured accuracy is
in runs/artifacts-*/artifact_metrics.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnomalyClass
from .detectors import LayerStatus, _r
from .model_loader import ModelRegistry

ANOMALY_OF = {
    "malformed_hand": AnomalyClass.ANATOMICAL_EXTRA,
    "extra_limb": AnomalyClass.ANATOMICAL_EXTRA,
    "malformed_face": AnomalyClass.MELTING_GEOMETRY,
    "malformed_body": AnomalyClass.MELTING_GEOMETRY,
    "missing_limb": AnomalyClass.MELTING_GEOMETRY,
}
LABEL = {
    "malformed_hand": "malformed hand",
    "malformed_face": "distorted face",
    "malformed_body": "distorted arm/leg/body",
    "extra_limb": "person with an extra limb",
    "missing_limb": "person with a missing limb",
}


@dataclass
class ArtifactBox:
    cls: str
    score: float
    box: list[float]          # [x1, y1, x2, y2] full-image pixels


@dataclass
class ArtifactResult:
    status: LayerStatus
    boxes: list[ArtifactBox] = field(default_factory=list)   # at or above threshold
    threshold: float | None = None
    run_id: str | None = None
    note: str | None = None


def detect_artifacts(rgb: np.ndarray, registry: ModelRegistry, imgsz: int = 512) -> ArtifactResult:
    comp = registry.artifacts
    det = comp.try_get()
    if det is None:
        return ArtifactResult(LayerStatus.UNAVAILABLE, note=comp.status()["error"])
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with comp.lock:
        r = det.model.predict(bgr, conf=det.threshold, imgsz=imgsz, verbose=False, device="cpu")[0]
    boxes = [
        ArtifactBox(det.classes[int(c)], _r(float(s)), [round(float(v), 1) for v in xy])
        for c, s, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    boxes.sort(key=lambda b: b.score, reverse=True)
    return ArtifactResult(LayerStatus.OK, boxes, det.threshold, det.run_id,
                          note=None if boxes else "No artefact detected above the threshold")
