"""Layer 1 (part 2): general object detection with YOLOv8n (COCO, 80 classes).

This shows *what is in the image* (people, phones, cups...) for context and for
the Layer 1 bounding-box view. A standard object detector can't tell whether an
image is AI-generated. It is not used to decide the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detectors import LayerStatus, _r
from .model_loader import ModelRegistry


@dataclass
class DetectedObject:
    label: str
    score: float
    box: list[float]           # [x1, y1, x2, y2] full-image pixels


@dataclass
class ObjectsResult:
    status: LayerStatus
    objects: list[DetectedObject] = field(default_factory=list)
    note: str | None = None


def detect_objects(rgb: np.ndarray, registry: ModelRegistry, conf: float = 0.35) -> ObjectsResult:
    comp = registry.objects
    model = comp.try_get()
    if model is None:
        return ObjectsResult(LayerStatus.UNAVAILABLE, note=comp.status()["error"])
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with comp.lock:
        r = model.predict(bgr, conf=conf, verbose=False, device="cpu")[0]
    objs = [
        DetectedObject(model.names[int(c)], _r(float(s)), [round(float(v), 1) for v in xy])
        for c, s, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    objs.sort(key=lambda o: o.score, reverse=True)
    return ObjectsResult(LayerStatus.OK, objs, note=None if objs else "No objects above 35% confidence")
