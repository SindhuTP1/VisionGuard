"""Layers 1-2: person localisation, hand plausibility and face symmetry.

Every result is a plain dataclass whose fields are JSON-safe Python types
(no numpy scalars, no NaN), so it can be returned from the API directly.

Known limits (by design, documented rather than hidden):
* MediaPipe fits a fixed 21-landmark, 5-finger skeleton. It cannot report a 6th
  finger. Extra or fused fingers are inferred indirectly, when the fitted skeleton
  is physically implausible.
* Face symmetry assumes a neutral-ish expression. Smirks, winks and strong
  side lighting can raise the score. Strongly turned heads are skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .anatomy import FingerCount, count_fingers
from .config import FaceConfig, FingerCountConfig, HandConfig, YoloConfig
from .model_loader import ModelRegistry
from .vision_utils import (
    BBox,
    distance_3d,
    joint_angle_deg,
    landmarks_to_array,
    polyline_length,
    roll_compensate,
    select_primary_box,
    signed_distance_to_plane,
)

log = logging.getLogger("visionguard.detectors")


class LayerStatus(str, Enum):
    OK = "ok"                     # ran and produced a result
    SKIPPED = "skipped"           # ran, but nothing to judge (e.g. 0 hands)
    UNAVAILABLE = "unavailable"   # model could not be loaded
    ERROR = "error"               # crashed during inference


def _r(x: float | None, nd: int = 4) -> float | None:
    """Round and make JSON-safe (None for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _to_mp_image(rgb: np.ndarray):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# =========================================================================== #
# Layer 1: person crop (YOLOv8)
# =========================================================================== #
@dataclass
class PersonResult:
    status: LayerStatus
    bbox: list[int]                 # region passed to the next layers (x1, y1, x2, y2)
    used_fallback: bool             # True → full image was used
    num_persons: int
    person_score: float | None = None
    note: str | None = None
    person_boxes: list[list[int]] = field(default_factory=list)  # all detections, largest first
    # 17 COCO keypoints per person ([x, y, conf], full-image pixels), same order as person_boxes.
    keypoints: list[list[list[float]]] = field(default_factory=list)


def head_crops(rgb: np.ndarray, person_boxes: list[list[int]], limit: int = 3) -> list[np.ndarray]:
    """Square crop at the top of each person box, where the head should be.

    The FaceLandmarker detector is short-range (selfie framing), so faces that are
    small relative to the frame are missed. Cropping to head-sized regions fixes that.
    """
    h, w = rgb.shape[:2]
    crops = []
    for x1, y1, x2, y2 in person_boxes[:limit]:
        side = min(x2 - x1, y2 - y1)
        if side < 32:
            continue
        cx = (x1 + x2) // 2
        box = BBox(max(0, cx - side // 2), y1, min(w, cx + side // 2), min(h, y1 + side))
        crops.append(box.padded(0.1, w, h).crop(rgb))
    return crops


def detect_person(rgb: np.ndarray, registry: ModelRegistry, cfg: YoloConfig) -> tuple[PersonResult, np.ndarray]:
    """Return the crop around the dominant person, or the full image as fallback."""
    full = BBox.full(rgb)
    h, w = rgb.shape[:2]

    comp = registry.yolo
    model = comp.try_get()
    if model is None:
        return PersonResult(LayerStatus.UNAVAILABLE, full.as_list(), True, 0,
                            note="YOLO unavailable; using full image"), rgb

    # Ultralytics treats numpy input as BGR (OpenCV convention).
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with comp.lock:  # one shared predictor; not safe for concurrent calls
        res = model.predict(bgr, classes=[cfg.person_class_id], conf=cfg.conf_threshold,
                            iou=cfg.iou_threshold, verbose=False)[0]
    boxes: list[BBox] = []
    kps: list[list[list[float]]] = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kp_data = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else None
        for j, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, confs)):
            boxes.append(BBox(int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2)), float(c)))
            kps.append([[round(float(v), 1) if k < 2 else round(float(v), 3) for k, v in enumerate(p)]
                        for p in kp_data[j]] if kp_data is not None and len(kp_data) > j else [])

    order = sorted(range(len(boxes)), key=lambda i: boxes[i].area, reverse=True)
    all_boxes = [boxes[i].as_list() for i in order]
    all_kps = [kps[i] for i in order]
    primary = select_primary_box(boxes, min_area=cfg.min_box_area_frac * w * h)
    if primary is None:
        note = "No person detected; using full image" if not boxes else "Persons too small; using full image"
        return PersonResult(LayerStatus.OK, full.as_list(), True, len(boxes), note=note,
                            person_boxes=all_boxes, keypoints=all_kps), rgb

    crop_box = primary.padded(cfg.crop_padding, w, h)
    return (
        PersonResult(LayerStatus.OK, crop_box.as_list(), False, len(boxes), _r(primary.score),
                     person_boxes=all_boxes, keypoints=all_kps),
        crop_box.crop(rgb),
    )


# =========================================================================== #
# Layer 2a: hands
# =========================================================================== #
# MediaPipe hand landmark indices: (base, joint, joint, tip) per finger.
FINGERS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17


@dataclass
class FingerCheck:
    length_ratio: float | None
    length_ok: bool
    phalanx_ratio: float | None          # None for the thumb
    phalanx_ok: bool
    max_hyperextension_deg: float | None  # None for the thumb
    hyperextension_ok: bool
    extended: bool

    @property
    def ok(self) -> bool:
        return self.length_ok and self.phalanx_ok and self.hyperextension_ok


@dataclass
class HandFinding:
    index: int
    handedness: str | None
    handedness_score: float | None
    palm_width_px: float | None
    extended_fingers: int | None        # informational only; NOT an anomaly signal
    fingers: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_fingers: list[str] = field(default_factory=list)
    fail_fraction: float = 0.0
    anomalous: bool = False             # skeleton implausible (lengths / bends)
    skipped_reason: str | None = None
    # 21 MediaPipe landmarks as [x, y] in full-image pixels (for drawing)
    landmarks: list[list[float]] = field(default_factory=list)
    # Independent fingertip count from the hand's outline (see anatomy.py)
    finger_count: FingerCount | None = None


@dataclass
class HandResult:
    status: LayerStatus
    num_hands: int
    hands: list[HandFinding] = field(default_factory=list)
    anomalous: bool = False             # any judged hand has an implausible skeleton
    max_fail_fraction: float | None = None
    note: str | None = None
    extra_finger: bool = False          # any hand counted 6+ fingertips


def _signed_bends(pts: np.ndarray) -> dict[str, list[float]]:
    """Signed PIP/DIP bends (deg) per non-thumb finger.

    The sign is the bend's rotation about the knuckle axis (index MCP → pinky MCP).
    Fingers flex the same way, so the dominant sign across joints is "flexion" and
    a large bend the other way is hyperextension. Using a consensus avoids relying
    on MediaPipe's left/right label, which flips with mirroring.
    """
    lateral = pts[PINKY_MCP] - pts[INDEX_MCP]
    out: dict[str, list[float]] = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        if name == "thumb":
            continue
        bends = []
        for a, b, c in ((mcp, pip, dip), (pip, dip, tip)):
            ang = joint_angle_deg(pts[a], pts[b], pts[c])
            if not math.isfinite(ang):
                bends.append(0.0)
                continue
            sign = np.sign(np.dot(np.cross(pts[b] - pts[a], pts[c] - pts[b]), lateral)) or 1.0
            bends.append(float((180.0 - ang) * sign))
        out[name] = bends
    return out


def _is_extended(pts: np.ndarray, name: str) -> bool:
    a, b, c, d = FINGERS[name]
    if name == "thumb":
        straight = joint_angle_deg(pts[b], pts[c], pts[d]) > 150
        return bool(straight and distance_3d(pts[d], pts[PINKY_MCP]) > distance_3d(pts[c], pts[PINKY_MCP]))
    straight = joint_angle_deg(pts[a], pts[b], pts[c]) > 160
    return bool(straight and distance_3d(pts[d], pts[WRIST]) > distance_3d(pts[b], pts[WRIST]))


def evaluate_hand(pts: np.ndarray, cfg: HandConfig, index: int = 0,
                  handedness: str | None = None, handedness_score: float | None = None) -> HandFinding:
    """Plausibility checks on a 21x3 landmark array (pixel units). Pure function."""
    finding = HandFinding(index, handedness, _r(handedness_score), None, None)
    palm = distance_3d(pts[INDEX_MCP], pts[PINKY_MCP])
    finding.palm_width_px = _r(palm, 2)
    if palm < cfg.min_palm_px:
        finding.skipped_reason = f"palm width {palm:.1f}px < {cfg.min_palm_px}px"
        return finding

    lo, hi = cfg.finger_length_ratio_bounds
    bends = _signed_bends(pts)
    all_bends = [b for v in bends.values() for b in v]
    # Majority vote over clearly bent joints (|bend| > 10°), so one grossly
    # hyperextended joint cannot outweigh several normal flexions. Ties fall back
    # to the magnitude sum.
    votes = sum(np.sign(b) for b in all_bends if abs(b) > 10.0)
    flex_sign = float(np.sign(votes)) or (1.0 if sum(all_bends) >= 0 else -1.0)

    extended = 0
    for name, chain in FINGERS.items():
        seg_pts = pts[list(chain)] if name != "thumb" else pts[[1, 2, 3, 4]]
        length_ratio = polyline_length(seg_pts) / palm
        length_ok = lo <= length_ratio <= hi

        if name == "thumb":
            phalanx_ratio, phalanx_ok, hyper, hyper_ok = None, True, None, True
        else:
            segs = np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)
            phalanx_ratio = float(segs.max() / max(segs.min(), 1e-6))
            phalanx_ok = phalanx_ratio <= cfg.max_phalanx_ratio
            # Bend against the dominant flexion direction = hyperextension.
            hyper = max(0.0, *(-flex_sign * b for b in bends[name]))
            hyper_ok = hyper <= cfg.max_joint_hyperextension_deg

        ext = _is_extended(pts, name)
        extended += int(ext)
        check = FingerCheck(_r(length_ratio), bool(length_ok), _r(phalanx_ratio), bool(phalanx_ok),
                            _r(hyper, 2), bool(hyper_ok), ext)
        finding.fingers[name] = {**check.__dict__, "ok": check.ok}
        if not check.ok:
            finding.failed_fingers.append(name)

    finding.extended_fingers = extended
    finding.fail_fraction = _r(len(finding.failed_fingers) / len(FINGERS))
    finding.anomalous = bool(finding.fail_fraction >= cfg.anomaly_fail_fraction)
    return finding


def analyze_hands(rgb: np.ndarray, registry: ModelRegistry, cfg: HandConfig,
                  count_cfg: FingerCountConfig | None = None,
                  offset: tuple[float, float] = (0.0, 0.0)) -> HandResult:
    """``rgb`` is the analysed region, and ``offset`` its top-left in the full image.
    All coordinates in the result are full-image pixels."""
    comp = registry.hands
    landmarker = comp.try_get()
    if landmarker is None:
        return HandResult(LayerStatus.UNAVAILABLE, 0, note=comp.status()["error"])

    h, w = rgb.shape[:2]
    dx, dy = offset
    with comp.lock:  # MediaPipe tasks are not thread-safe
        res = landmarker.detect(_to_mp_image(rgb))

    if not res.hand_landmarks:
        return HandResult(LayerStatus.SKIPPED, 0, note="0 hands detected; finger check skipped")

    hands: list[HandFinding] = []
    for i, lms in enumerate(res.hand_landmarks):
        cat = res.handedness[i][0] if i < len(res.handedness) and res.handedness[i] else None
        pts = landmarks_to_array(lms, w, h)
        hf = evaluate_hand(pts, cfg, i, cat.category_name if cat else None, cat.score if cat else None)
        hf.landmarks = [[round(float(x + dx), 1), round(float(y + dy), 1)] for x, y in pts[:, :2]]
        if count_cfg is not None:
            try:
                hf.finger_count = count_fingers(rgb, pts, [n for n, f in hf.fingers.items() if f['extended']], count_cfg).shifted(dx, dy)
            except Exception as exc:  # noqa: BLE001 - counting is best-effort per hand
                log.exception("finger counting failed")
                hf.finger_count = FingerCount("cant_count", reason=f"counting error: {type(exc).__name__}")
        hands.append(hf)

    extra = any(hf.finger_count is not None and hf.finger_count.extra_finger for hf in hands)
    judged = [hf for hf in hands if hf.skipped_reason is None]
    if not judged:
        return HandResult(LayerStatus.SKIPPED, len(hands), hands, note="All hands too small to judge",
                          extra_finger=extra)
    return HandResult(
        LayerStatus.OK, len(hands), hands,
        extra_finger=extra,
        anomalous=any(hf.anomalous for hf in judged),
        max_fail_fraction=max(hf.fail_fraction for hf in judged),
    )


# =========================================================================== #
# Layer 2b: face symmetry
# =========================================================================== #
@dataclass
class FaceFinding:
    index: int
    iod_px: float | None
    roll_deg: float | None
    yaw_ratio: float | None
    asymmetry_score: float | None = None    # mean per-pair asymmetry / IOD
    lateral_mean: float | None = None       # 3D distance-to-midline mismatch
    vertical_mean: float | None = None      # height mismatch after roll compensation
    midline_deviation: float | None = None  # informational: bend in the nose/lip/chin line
    worst_pairs: list[dict[str, Any]] = field(default_factory=list)
    anomalous: bool = False
    skipped_reason: str | None = None


@dataclass
class FaceResult:
    status: LayerStatus
    num_faces: int
    faces: list[FaceFinding] = field(default_factory=list)
    anomalous: bool = False
    max_score: float | None = None
    note: str | None = None


def evaluate_face(pts: np.ndarray, cfg: FaceConfig, index: int = 0) -> FaceFinding:
    """Symmetry score on a (478|468)x3 landmark array (pixel units). Pure function."""
    L, R = cfg.left_eye_outer, cfg.right_eye_outer
    iod = distance_3d(pts[L], pts[R])
    yaw = abs(pts[L, 2] - pts[R, 2]) / iod if iod > 1e-6 else float("inf")
    rp, roll = roll_compensate(pts, L, R)
    f = FaceFinding(index, _r(iod, 2), _r(math.degrees(roll), 2), _r(yaw))

    if iod < cfg.min_iod_px:
        f.skipped_reason = f"inter-ocular distance {iod:.1f}px < {cfg.min_iod_px}px"
        return f
    if yaw > cfg.max_yaw_ratio:
        f.skipped_reason = f"head turned too far (yaw ratio {yaw:.2f} > {cfg.max_yaw_ratio})"
        return f

    left = np.array([p[0] for p in cfg.symmetry_pairs])
    right = np.array([p[1] for p in cfg.symmetry_pairs])

    # Mid-sagittal plane: through the midline centroid, normal = mean left→right
    # direction across all mirrored pairs (more robust than the eye axis alone).
    normal = (rp[right] - rp[left]).mean(axis=0)
    centroid = rp[list(cfg.midline)].mean(axis=0)
    dl = signed_distance_to_plane(rp[left], centroid, normal)
    dr = signed_distance_to_plane(rp[right], centroid, normal)

    lateral = np.abs(dl + dr) / iod                      # mirrored points should be equidistant
    vertical = np.abs(rp[left, 1] - rp[right, 1]) / iod  # only meaningful after roll compensation
    per_pair = np.hypot(lateral, vertical)
    mid_dev = signed_distance_to_plane(rp[list(cfg.midline)], centroid, normal).std() / iod

    score = float(per_pair.mean())
    f.asymmetry_score = _r(score)
    f.lateral_mean = _r(lateral.mean())
    f.vertical_mean = _r(vertical.mean())
    f.midline_deviation = _r(mid_dev)
    f.worst_pairs = [
        {"pair": [int(left[i]), int(right[i])], "score": _r(per_pair[i])}
        for i in np.argsort(per_pair)[::-1][:3]
    ]
    f.anomalous = bool(score > cfg.asymmetry_threshold)
    return f


def analyze_face(rgb: np.ndarray, registry: ModelRegistry, cfg: FaceConfig,
                 fallback_crops: list[np.ndarray] | None = None) -> FaceResult:
    """Run on the person region; if no face is found, search ``fallback_crops`` (head crops)."""
    comp = registry.face
    landmarker = comp.try_get()
    if landmarker is None:
        return FaceResult(LayerStatus.UNAVAILABLE, 0, note=comp.status()["error"])

    def detect(img: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        with comp.lock:
            res = landmarker.detect(_to_mp_image(img))
        return [landmarks_to_array(lms, w, h) for lms in res.face_landmarks]

    note = None
    point_sets = detect(rgb)
    if not point_sets and fallback_crops:
        for crop in fallback_crops:
            point_sets.extend(detect(crop))
            if len(point_sets) >= cfg.max_num_faces:
                break
        if point_sets:
            note = "Faces found via head-crop search"

    if not point_sets:
        return FaceResult(LayerStatus.SKIPPED, 0, note="0 faces detected; symmetry check skipped")

    faces = [evaluate_face(pts, cfg, i) for i, pts in enumerate(point_sets[: cfg.max_num_faces])]
    judged = [f for f in faces if f.skipped_reason is None]
    if not judged:
        return FaceResult(LayerStatus.SKIPPED, len(faces), faces, note="No face suitable for symmetry analysis")
    return FaceResult(
        LayerStatus.OK, len(faces), faces,
        anomalous=any(f.anomalous for f in judged),
        max_score=max(f.asymmetry_score for f in judged),
        note=note,
    )


__all__ = [
    "LayerStatus", "PersonResult", "HandResult", "HandFinding", "FaceResult", "FaceFinding",
    "detect_person", "analyze_hands", "analyze_face", "evaluate_hand", "evaluate_face",
]
