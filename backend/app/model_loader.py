"""Model download, construction and lazy, thread-safe loading.

Each pipeline layer is a ``Component`` that loads on first use and records its
status, so ``/health`` can report exactly what is available. A missing
classifier checkpoint is reported as missing and never replaced by a dummy
model: the pipeline must say "classifier unavailable" rather than output fake
scores.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import torch
from torch import nn

from .config import CLASSIFIER_LABELS, Settings, settings as default_settings
from .vision_utils import build_classifier_transform

log = logging.getLogger("visionguard.models")

T = TypeVar("T")

CHECKPOINT_FORMAT_VERSION = 1


class ModelUnavailableError(RuntimeError):
    """A component could not be loaded (missing weights, download failure, ...)."""


# --------------------------------------------------------------------------- #
# Device + downloads
# --------------------------------------------------------------------------- #
def resolve_device(pref: str = "auto") -> torch.device:
    pref = pref.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(pref)


def download_file(url: str, dest: Path, timeout: float = 60.0, retries: int = 3) -> Path:
    """Download to ``dest`` atomically (temp file + rename) with simple retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "visionguard-model-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            if os.path.getsize(tmp_name) == 0:
                raise IOError("Downloaded file is empty")
            os.replace(tmp_name, dest)
            log.info("Downloaded %s -> %s", url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - surfaced after retries
            last_exc = exc
            log.warning("Download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(min(2 ** attempt, 8))
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
    raise ModelUnavailableError(f"Failed to download {url}: {last_exc}")


def ensure_weight(name: str, url: str, cfg: Settings = default_settings) -> Path:
    path = cfg.paths.weights / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    return download_file(url, path)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_yolo(cfg: Settings = default_settings):
    # A server must never pip-install packages at runtime (Ultralytics does by default).
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO

    path = ensure_weight(cfg.yolo.weights_name, cfg.yolo.weights_url, cfg)
    return YOLO(str(path))


def load_hand_landmarker(cfg: Settings = default_settings):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    path = ensure_weight(cfg.hand.model_name, cfg.hand.model_url, cfg)
    opts = mp_vision.HandLandmarkerOptions(
        # Pass a buffer instead of a path: avoids MediaPipe path issues on Windows (non-ASCII dirs).
        base_options=mp_python.BaseOptions(model_asset_buffer=path.read_bytes()),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=cfg.hand.max_num_hands,
        min_hand_detection_confidence=cfg.hand.min_detection_conf,
        min_hand_presence_confidence=cfg.hand.min_presence_conf,
        min_tracking_confidence=cfg.hand.min_tracking_conf,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def load_face_landmarker(cfg: Settings = default_settings):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    path = ensure_weight(cfg.face.model_name, cfg.face.model_url, cfg)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=path.read_bytes()),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=cfg.face.max_num_faces,
        min_face_detection_confidence=cfg.face.min_detection_conf,
        min_face_presence_confidence=cfg.face.min_presence_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def build_classifier(num_outputs: int = len(CLASSIFIER_LABELS), pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with its final Linear layer replaced for ``num_outputs`` classes."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    head = model.classifier[-1]
    assert isinstance(head, nn.Linear)
    model.classifier[-1] = nn.Linear(head.in_features, num_outputs)
    return model


def save_classifier_checkpoint(model: nn.Module, path: Path, run_id: str, extra: dict[str, Any] | None = None) -> None:
    cfg = default_settings.classifier
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "arch": cfg.arch,
        "labels": list(CLASSIFIER_LABELS),
        "img_size": cfg.img_size,
        "mean": list(cfg.mean),
        "std": list(cfg.std),
        "preprocessing": cfg.preprocessing,
        "run_id": run_id,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@dataclass
class LoadedClassifier:
    model: nn.Module
    labels: tuple[str, ...]
    run_id: str
    device: torch.device
    transform: Callable


def load_classifier(cfg: Settings = default_settings, path: Path | None = None) -> LoadedClassifier:
    path = path or cfg.paths.classifier_checkpoint
    if not path.is_file():
        raise ModelUnavailableError(
            f"Classifier checkpoint not found at {path}. Train it with `python -m training.train`."
        )
    # weights_only=True: refuse arbitrary pickled code inside checkpoints.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ModelUnavailableError(f"Unsupported checkpoint format: {ckpt.get('format_version')}")
    if ckpt.get("arch") != cfg.classifier.arch:
        raise ModelUnavailableError(f"Checkpoint arch {ckpt.get('arch')!r} != {cfg.classifier.arch!r}")
    prep = ckpt.get("preprocessing", "resize_center_crop_v0")   # pre-tag checkpoints resized inputs
    if prep != cfg.classifier.preprocessing:
        raise ModelUnavailableError(
            f"Checkpoint was trained with preprocessing {prep!r} but the server uses "
            f"{cfg.classifier.preprocessing!r}; retrain with `python -m training.train`."
        )
    labels = tuple(ckpt["labels"])
    if labels != CLASSIFIER_LABELS:
        raise ModelUnavailableError(f"Checkpoint labels {labels} != expected {CLASSIFIER_LABELS}")

    device = resolve_device(cfg.runtime.device)
    model = build_classifier(len(labels), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return LoadedClassifier(
        model=model,
        labels=labels,
        run_id=str(ckpt.get("run_id", "unknown")),
        device=device,
        transform=build_classifier_transform(train=False, cfg=cfg.classifier),
    )


# --------------------------------------------------------------------------- #
# Lazy, thread-safe registry
# --------------------------------------------------------------------------- #
@dataclass
class Component(Generic[T]):
    name: str
    loader: Callable[[], T]
    # Serialises *inference* too: MediaPipe task objects are not thread-safe.
    lock: threading.Lock = field(default_factory=threading.Lock)
    _value: T | None = None
    _error: str | None = None
    _load_seconds: float | None = None

    def get(self) -> T:
        if self._value is not None:
            return self._value
        with self.lock:
            if self._value is None:
                t0 = time.perf_counter()
                try:
                    self._value = self.loader()
                    self._error = None
                    self._load_seconds = time.perf_counter() - t0
                    log.info("Loaded %s in %.2fs", self.name, self._load_seconds)
                except Exception as exc:  # noqa: BLE001 - recorded for /health
                    err = f"{type(exc).__name__}: {exc}"
                    # Retried on every call (so a newly trained checkpoint is picked up
                    # without a restart), but logged loudly only when the error changes.
                    (log.debug if err == self._error else log.error)("Failed to load %s: %s", self.name, err)
                    self._error = err
                    raise ModelUnavailableError(err) from exc
        return self._value

    def try_get(self) -> T | None:
        try:
            return self.get()
        except ModelUnavailableError:
            return None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._value is not None,
            "error": self._error,
            "load_seconds": round(self._load_seconds, 3) if self._load_seconds else None,
        }


class ModelRegistry:
    def __init__(self, cfg: Settings = default_settings) -> None:
        cfg.paths.ensure()
        self.cfg = cfg
        self.yolo: Component = Component("yolov8_person", lambda: load_yolo(cfg))
        self.hands: Component = Component("mediapipe_hands", lambda: load_hand_landmarker(cfg))
        self.face: Component = Component("mediapipe_face", lambda: load_face_landmarker(cfg))
        self.classifier: Component[LoadedClassifier] = Component("mobilenetv3_classifier", lambda: load_classifier(cfg))

    def components(self) -> dict[str, Component]:
        return {c.name: c for c in (self.yolo, self.hands, self.face, self.classifier)}

    def warmup(self) -> dict[str, dict[str, Any]]:
        """Try to load everything up front; failures are recorded, not raised."""
        for comp in self.components().values():
            comp.try_get()
        yolo = self.yolo.try_get()
        if yolo is not None:
            # The first YOLO call pays a one-off graph/fuse cost (~2 s on CPU); pay it here.
            import numpy as np

            with self.yolo.lock:
                yolo.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
        return self.status()

    def status(self) -> dict[str, dict[str, Any]]:
        return {name: comp.status() for name, comp in self.components().items()}


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
