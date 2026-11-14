"""Image I/O, bounding-box handling and 3D landmark geometry helpers.

Everything here is a pure function over numpy / PIL, with no model state, so it can be
unit-tested in isolation and shared by the inference pipeline and training.
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ClassifierConfig, settings


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded into a safe, usable RGB image."""


def _decode(data: bytes, max_side: int, max_pixels: int) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # verify() catches truncated/corrupt files; it invalidates the handle,
            # so the image is reopened afterwards.
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
            img = Image.open(io.BytesIO(data))
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageDecodeError(f"Unreadable image: {exc}") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageDecodeError("Image exceeds the maximum pixel budget") from exc

    img = ImageOps.exif_transpose(img)          # honour camera orientation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) < 16:
        raise ImageDecodeError(f"Unsupported image shape {arr.shape}")
    return np.ascontiguousarray(arr)


def load_image_bytes(
    data: bytes,
    max_side: int | None = None,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Decode uploaded bytes into an RGB ``uint8`` array of shape (H, W, 3)."""
    if not data:
        raise ImageDecodeError("Empty upload")
    rt = settings.runtime
    return _decode(data, max_side or rt.max_image_side, max_pixels or rt.max_image_pixels)


def load_image_path(path: str | Path, max_side: int | None = None) -> np.ndarray:
    return load_image_bytes(Path(path).read_bytes(), max_side=max_side)


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB")


# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def padded(self, frac: float, img_w: int, img_h: int) -> "BBox":
        """Expand by ``frac`` of the box size on each side, then clip to the image."""
        px, py = int(round(self.width * frac)), int(round(self.height * frac))
        return BBox(
            max(0, self.x1 - px),
            max(0, self.y1 - py),
            min(img_w, self.x2 + px),
            min(img_h, self.y2 + py),
            self.score,
        )

    def crop(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(img[self.y1 : self.y2, self.x1 : self.x2])

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def full(cls, img: np.ndarray) -> "BBox":
        h, w = img.shape[:2]
        return cls(0, 0, w, h, 1.0)


def select_primary_box(boxes: Sequence[BBox], min_area: float = 0.0) -> BBox | None:
    """Pick the dominant subject: largest area weighted by detector confidence."""
    candidates = [b for b in boxes if b.area > 0 and b.area >= min_area]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.area * b.score)


# --------------------------------------------------------------------------- #
# Landmark geometry (3D)
# --------------------------------------------------------------------------- #
class _Landmark(Protocol):
    x: float
    y: float
    z: float


def landmarks_to_array(landmarks: Iterable[_Landmark], width: int, height: int) -> np.ndarray:
    """Convert MediaPipe normalised landmarks to an (N, 3) array in pixel units.

    MediaPipe gives x in [0,1] of width, y in [0,1] of height, and z on roughly the
    same scale as x. Scaling x and z by width and y by height gives an isotropic
    space, so 3D distances are not distorted by the image aspect ratio.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width
    return pts


def distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def eye_axis_roll(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """In-plane head roll (radians): the angle of the eye axis from horizontal."""
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    return math.atan2(dy, dx)


def rotate_about_z(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """Rotate points by ``angle`` radians in the image plane around ``center``."""
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (points - center) @ rot.T + center


def roll_compensate(points: np.ndarray, left_idx: int, right_idx: int) -> tuple[np.ndarray, float]:
    """Level the eye axis so axis-aligned comparisons (e.g. eye height) are fair.

    Returns (rotated_points, roll_radians). Euclidean 3D distances do not change
    under this rotation. It only matters for per-axis measures such as the
    vertical offset between mirrored landmarks.
    """
    l, r = points[left_idx], points[right_idx]
    roll = eye_axis_roll(l, r)
    center = (l + r) / 2.0
    return rotate_about_z(points, -roll, center), roll


def signed_distance_to_plane(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, float)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        raise ValueError("Degenerate plane normal")
    return (np.atleast_2d(points) - plane_point) @ (n / norm)


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex ``b`` formed by segments b→a and b→c, in degrees [0, 180]."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


# --------------------------------------------------------------------------- #
# Classifier preprocessing (training, evaluation and inference share this code)
# --------------------------------------------------------------------------- #
# Plain classes (not lambdas) so DataLoader workers can pickle them on Windows.
class JpegNormalize:
    """Re-encode as JPEG at a fixed quality (4:2:0) so source compression can't leak the label."""

    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=self.quality, subsampling=2)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out


class PadToMin:
    """Reflect-pad images smaller than ``size`` so a native-resolution crop always fits."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w >= self.size and h >= self.size:
            return img
        arr = np.asarray(img)
        ph, pw = max(0, self.size - h), max(0, self.size - w)
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), mode="reflect")
        return Image.fromarray(arr)


class StackCrops:
    """FiveCrop output → tensor of shape (n_crops, 3, H, W)."""

    def __init__(self, to_tensor, normalize) -> None:
        self.to_tensor, self.normalize = to_tensor, normalize

    def __call__(self, crops):
        import torch

        return torch.stack([self.normalize(self.to_tensor(c)) for c in crops])


def build_classifier_transform(train: bool, cfg: ClassifierConfig | None = None):
    """Train: one random native-resolution crop → (3, H, W).
    Eval/inference: five native-resolution crops → (5, 3, H, W), averaged by the caller.

    No resizing anywhere. Resizing would hand the model the image's original
    resolution (via resampling traces), which is a label shortcut in the training data.
    """
    from torchvision import transforms as T  # local import keeps geometry utils torch-free

    cfg = cfg or settings.classifier
    normalize = T.Normalize(mean=cfg.mean, std=cfg.std)
    pre = [JpegNormalize(cfg.jpeg_normalize_quality), PadToMin(cfg.img_size)]
    if train:
        # Mild augmentation only: colour/blur jitter would erase the high-frequency
        # generation artefacts the classifier needs to learn.
        return T.Compose([*pre, T.RandomCrop(cfg.img_size), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    return T.Compose([*pre, T.FiveCrop(cfg.img_size), StackCrops(T.ToTensor(), normalize)])
