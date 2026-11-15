"""End-to-end orchestration: person + pose → hands (+ finger count) + face → body → classifier → arbitration.

A failure in one layer is recorded in that layer's status and never crashes the
whole request. The report always says which layers actually ran.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

import numpy as np

from .arbitration import ArbitrationResult, arbitrate
from .body import BodyResult, analyze_body
from .classifier import ClassifierResult, classify
from .config import Settings, settings as default_settings
from .detectors import (
    FaceResult,
    HandResult,
    LayerStatus,
    PersonResult,
    analyze_face,
    analyze_hands,
    detect_person,
    head_crops,
)
from .model_loader import ModelRegistry, get_registry
from .vision_utils import BBox, load_image_bytes

log = logging.getLogger("visionguard.pipeline")
R = TypeVar("R")


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass
class AnalysisReport:
    image: dict[str, int]
    person: PersonResult
    hands: HandResult
    face: FaceResult
    body: BodyResult
    classifier: ClassifierResult
    arbitration: ArbitrationResult
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


class VisionGuardPipeline:
    def __init__(self, registry: ModelRegistry | None = None, cfg: Settings = default_settings) -> None:
        self.registry = registry or get_registry()
        self.cfg = cfg

    def _timed(self, name: str, timings: dict[str, float], fn: Callable[[], R], on_error: Callable[[str], R]) -> R:
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one layer must not take down the request
            log.exception("Layer %s failed", name)
            return on_error(f"{type(exc).__name__}: {exc}")
        finally:
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    def analyze(self, rgb: np.ndarray) -> AnalysisReport:
        cfg, reg = self.cfg, self.registry
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        h, w = rgb.shape[:2]

        def person_fallback(err: str) -> tuple[PersonResult, np.ndarray]:
            return PersonResult(LayerStatus.ERROR, BBox.full(rgb).as_list(), True, 0,
                                note=f"Person detection failed ({err}); using full image"), rgb

        person, region = self._timed("person", timings, lambda: detect_person(rgb, reg, cfg.yolo), person_fallback)

        offset = (float(person.bbox[0]), float(person.bbox[1]))   # region origin in the full image
        hands = self._timed("hands", timings, lambda: analyze_hands(region, reg, cfg.hand, cfg.count, offset),
                            lambda e: HandResult(LayerStatus.ERROR, 0, note=e))
        face = self._timed("face", timings,
                           lambda: analyze_face(region, reg, cfg.face, head_crops(rgb, person.person_boxes)),
                           lambda e: FaceResult(LayerStatus.ERROR, 0, note=e))
        body = self._timed("body", timings, lambda: analyze_body(person, hands, cfg.body),
                           lambda e: BodyResult(LayerStatus.ERROR, note=e))
        clf = self._timed("classifier", timings, lambda: classify(rgb, reg),
                          lambda e: ClassifierResult(LayerStatus.ERROR, note=e))

        arb = arbitrate(person, hands, face, body, clf, cfg)
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        return AnalysisReport({"width": w, "height": h}, person, hands, face, body, clf, arb, timings)

    def analyze_bytes(self, data: bytes) -> AnalysisReport:
        rt = self.cfg.runtime
        return self.analyze(load_image_bytes(data, rt.max_image_side, rt.max_image_pixels))
