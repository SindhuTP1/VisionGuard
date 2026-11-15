"""Central configuration for VisionGuard.

Every tunable lives here. Values can be overridden with environment variables
prefixed ``VG_`` (e.g. ``VG_CONFIDENCE_THRESHOLD=0.7``).

NOTE: The geometric thresholds below (face asymmetry, hand plausibility) are
*uncalibrated starting points*, not measured results. They are hyperparameters
to be tuned against real data; no performance metric is stored in this file.
All reported metrics come from ``runs/<run_id>/metrics.json`` produced by the
training/evaluation scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    return os.getenv(f"VG_{name}", default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(f"VG_{name}")
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(f"VG_{name}")
    return int(raw) if raw is not None else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(f"VG_{name}")
    return Path(raw).expanduser().resolve() if raw else default


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class AnomalyClass(str, Enum):
    NORMAL = "normal"
    ANATOMICAL_EXTRA = "anatomical_extra"
    FLOATING_ANOMALY = "floating_anomaly"
    MELTING_GEOMETRY = "melting_geometry"
    TEXT_DISTORTION = "text_distortion"


class Verdict(str, Enum):
    VERIFIED_SAFE = "Verified Safe"
    NEEDS_REVIEW = "Needs Review"
    ANOMALY_DETECTED = "Anomaly Detected"


# Binary classifier label order. Index i of the model output == CLASSIFIER_LABELS[i].
# Must match the label order used in training/prepare_dataset.py.
CLASSIFIER_LABELS: tuple[str, ...] = ("real", "fake")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    backend: Path = BACKEND_DIR
    weights: Path = _env_path("WEIGHTS_DIR", BACKEND_DIR / "weights")
    data: Path = _env_path("DATA_DIR", BACKEND_DIR / "data")
    runs: Path = _env_path("RUNS_DIR", BACKEND_DIR / "runs")

    @property
    def classifier_checkpoint(self) -> Path:
        # Written by training/train.py; overridable to point at a specific run.
        return _env_path("CLASSIFIER_CKPT", self.weights / "mobilenetv3_small_aivsreal.pt")

    def ensure(self) -> None:
        for p in (self.weights, self.data, self.runs):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Layer 1: person detection (YOLOv8)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YoloConfig:
    # The pose variant returns person boxes AND 17 COCO body keypoints, so one model
    # drives both the person crop and the body checks (hands per person, arm shape).
    weights_name: str = _env_str("YOLO_WEIGHTS", "yolov8n-pose.pt")
    weights_url: str = _env_str(
        "YOLO_URL",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
    )
    person_class_id: int = 0          # COCO "person"
    conf_threshold: float = _env_float("YOLO_CONF", 0.35)
    iou_threshold: float = 0.45
    crop_padding: float = 0.10        # fractional padding around the chosen box
    min_box_area_frac: float = 0.02   # ignore persons smaller than 2% of the image


# --------------------------------------------------------------------------- #
# Layer 2a: hands (MediaPipe Tasks HandLandmarker)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HandConfig:
    model_name: str = "hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    max_num_hands: int = 4
    min_detection_conf: float = 0.5
    min_presence_conf: float = 0.5
    min_tracking_conf: float = 0.5

    expected_fingers: int = 5
    # MediaPipe always fits a 21-point, 5-finger skeleton, so it cannot "see" a
    # sixth finger directly. Anomalies are inferred from skeleton implausibility:
    # bone-length ratios vs palm width and joint angles beyond anatomical limits.
    # Ratio bounds are (min, max) of finger_length / palm_width, where palm width is
    # the index-MCP to pinky-MCP distance (~5.5-6 cm in adults; a middle finger from
    # MCP to tip is ~1.6-1.7x that). Uncalibrated.
    finger_length_ratio_bounds: tuple[float, float] = (0.5, 2.4)
    # Longest / shortest phalanx within one finger (normal is around 2-2.5).
    max_phalanx_ratio: float = 4.0
    # PIP/DIP bend *against* the hand's dominant flexion direction.
    max_joint_hyperextension_deg: float = 50.0
    # Hands smaller than this (palm width in px) are too small to judge.
    min_palm_px: float = 15.0
    # Fraction of failed plausibility checks on a hand that flags it anomalous.
    anomaly_fail_fraction: float = _env_float("HAND_FAIL_FRAC", 0.34)


# --------------------------------------------------------------------------- #
# Layer 2c: body pose (hands per person, arm proportions)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BodyConfig:
    kp_min_conf: float = 0.5          # keypoints below this are ignored
    # A detected hand belongs to a person if its wrist is within this many shoulder
    # widths of one of that person's pose wrists.
    hand_assoc_shoulders: float = 0.6
    extra_hand_min: int = 3           # this many hands attached to one person → extra hand
    # Upper-arm / forearm length in 2D. Foreshortening makes this noisy, so only
    # extreme values are flagged, and only as a weak signal. Uncalibrated.
    arm_ratio_bounds: tuple[float, float] = (0.33, 3.0)
    min_segment_frac: float = 0.06    # segments shorter than this x box height are ignored


# --------------------------------------------------------------------------- #
# Layer 2a+: finger counting from the hand silhouette
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FingerCountConfig:
    """MediaPipe always fits 5 fingers, so we count fingertips independently:
    GrabCut cuts out the hand (seeded by the landmarks), then bumps along its
    outline are counted. Thresholds are in palm widths (index-MCP to pinky-MCP).
    Tuned on real hands (HaGRID sample), where any count >= 6 is a false alarm."""
    min_palm_px: float = _env_float("COUNT_MIN_PALM_PX", 30.0)   # smaller hands can't be counted
    min_extended: int = 3            # fists / grips can't be counted from an outline
    work_palm_px: float = 110.0      # crops are rescaled so the palm is about this wide
    max_work_side: int = 380         # cap on the rescaled crop's longer side (keeps GrabCut fast)
    crop_pad: float = 0.9            # padding around the landmarks, in palm widths
    grabcut_iters: int = 5
    # Outline peaks = fingertip candidates.
    tip_min_dist: float = 0.75       # distance from palm centre
    tip_max_dist: float = 3.0
    tip_min_prominence: float = 0.22
    tip_min_axis_proj: float = 0.25  # must lie on the finger side of the wrist (fraction of wrist→middle-MCP)
    tip_match_dist: float = 0.4      # candidate within this of a MediaPipe tip = that finger
    # Segmentation sanity checks: if these fail, the result is "can't count".
    max_area_ratio: float = 7.0      # silhouette area / palm polygon area (leaked into face/background)
    max_border_fraction: float = 0.12  # silhouette pixels touching the crop border (except wrist side)
    extra_finger_min_count: int = 6
    # Scanline method (works on flat palms where fingers touch): sample lines
    # across the fingers just above the knuckles and count the dark seams between them.
    scan_offsets: tuple[float, ...] = (0.3, 0.45, 0.6)   # beyond the knuckle line, in palm widths
    scan_min_digit_width: float = 0.16                    # a finger is ~0.25-0.3 palm widths, so its halves can't both pass
    scan_max_digit_width: float = 0.45                    # wider single pieces are suspect (merged/leak)
    scan_valley_prominence: float = 0.12                  # fraction of the run's brightness range
    scan_valley_min_abs: float = 6.0                      # grey levels


# --------------------------------------------------------------------------- #
# Layer 2b: face symmetry (MediaPipe Tasks FaceLandmarker, 478 landmarks)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FaceConfig:
    model_name: str = "face_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    max_num_faces: int = 3
    min_detection_conf: float = 0.5
    min_presence_conf: float = 0.5

    # Outer eye corners: define the eye axis for roll compensation and the
    # inter-ocular distance used to normalise every measurement.
    left_eye_outer: int = 33
    right_eye_outer: int = 263

    # Mirrored landmark pairs (canonical MediaPipe face mesh indices).
    symmetry_pairs: tuple[tuple[int, int], ...] = (
        (33, 263),    # eye outer corners
        (133, 362),   # eye inner corners
        (159, 386),   # upper eyelids
        (145, 374),   # lower eyelids
        (70, 300),    # brow outer
        (107, 336),   # brow inner
        (50, 280),    # cheeks
        (116, 345),   # cheekbones
        (234, 454),   # face edge (near tragion)
        (132, 361),   # upper jaw
        (172, 397),   # jaw
        (136, 365),   # lower jaw
        (58, 288),    # jaw near mouth
        (129, 358),   # nose alae
        (98, 327),    # nostrils
        (61, 291),    # mouth corners
    )
    # Midline landmarks: forehead, nose bridge, nose tip, lips, chin.
    midline: tuple[int, ...] = (10, 168, 6, 4, 1, 0, 17, 152)

    # Head yaw beyond this (estimated from the depth difference between eye corners,
    # normalised by inter-ocular distance) makes symmetry unreliable, so the check is skipped.
    max_yaw_ratio: float = 0.35
    # Faces whose inter-ocular distance is below this (px) are too small to judge.
    min_iod_px: float = 20.0
    # Normalised asymmetry score above which a face is flagged. Uncalibrated.
    asymmetry_threshold: float = _env_float("FACE_ASYM_THRESHOLD", 0.12)


# --------------------------------------------------------------------------- #
# Layer 3: classifier (MobileNetV3-Small, real vs fake)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassifierConfig:
    arch: str = "mobilenet_v3_small"
    img_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)   # ImageNet
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    num_outputs: int = len(CLASSIFIER_LABELS)
    # Shortcut control. In AI-vs-Real, image size/aspect (fakes 1024x1024, reals ~640x442)
    # and JPEG quality (all fakes q~75, 20% of reals q~95) reveal the label. So every input
    # is (1) re-encoded as JPEG at a fixed quality, then (2) cropped at native resolution
    # with no resizing. Checkpoints record this tag and are refused if it differs.
    preprocessing: str = "jpeg75_native_crop_v1"
    jpeg_normalize_quality: int = 75
    eval_crops: int = 5     # FiveCrop averaged at eval/inference


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    hf_dataset: str = "Parveshiiii/AI-vs-Real"
    # From the dataset card: 0 = AI-generated, 1 = real. NOTE: this is the reverse of
    # CLASSIFIER_LABELS index order, so it must always be mapped through this table.
    hf_label_map: tuple[tuple[int, str], ...] = ((0, "fake"), (1, "real"))
    n_real: int = 1000
    n_fake: int = 1000
    # The parquet shards are sorted by source (100-row row groups). Sampling whole
    # row groups and splitting *by group* keeps neighbouring images from the same
    # source out of both train and test. 25 groups x 40 rows = 1,000 per class.
    groups_per_class: int = 25
    # Stratified (per class) split by row group, close to 70/15/15.
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = _env_int("SEED", 42)

    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "CrossEntropyLoss"
    epochs: int = 12
    early_stopping_patience: int = 3
    early_stopping_metric: str = "val_loss"
    batch_size: int = 32
    num_workers: int = _env_int("NUM_WORKERS", 2)


# --------------------------------------------------------------------------- #
# Arbitration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArbitrationConfig:
    # Final confidence below this → "Needs Review".
    confidence_threshold: float = _env_float("CONFIDENCE_THRESHOLD", 0.65)
    # Fixed geometric confidences for count-based findings (heuristic, documented).
    # Alone, 0.75 clears the review bar. Averaged with a classifier that says
    # "real", it drops below it, so disagreement goes to a human.
    extra_finger_conf: float = 0.75           # exactly 6 fingertips counted
    extra_finger_conf_many: float = 0.65      # 7+: more often a segmentation artefact
    extra_hand_conf: float = 0.8              # 3+ hands on one person
    extra_hand_same_wrist_conf: float = 0.7   # two separate hands at one wrist
    # Arm ratios are shown but NOT used as a finding: on real photos, an arm pointing
    # at the camera foreshortens past any usable bound (e.g. 3.7 on a real photo).
    limb_findings_enabled: bool = False
    distorted_limb_conf: float = 0.5


# --------------------------------------------------------------------------- #
# Runtime / API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeConfig:
    device: str = _env_str("DEVICE", "auto")          # auto | cpu | cuda | mps
    max_upload_bytes: int = _env_int("MAX_UPLOAD_MB", 15) * 1024 * 1024
    max_image_side: int = _env_int("MAX_IMAGE_SIDE", 2048)
    max_image_pixels: int = 50_000_000                  # decompression-bomb guard
    allowed_mime: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in _env_str("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
    )
    # Also allow any localhost/127.0.0.1 port, because `next dev` moves to 3001+ when
    # 3000 is busy. Set VG_CORS_ORIGIN_REGEX="" in production to use only the list above.
    cors_origin_regex: str | None = (
        _env_str("CORS_ORIGIN_REGEX", r"^http://(localhost|127\.0\.0\.1)(:\d+)?$") or None
    )


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    count: FingerCountConfig = field(default_factory=FingerCountConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arbitration: ArbitrationConfig = field(default_factory=ArbitrationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


settings = Settings()
