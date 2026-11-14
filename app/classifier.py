"""Layer 3: MobileNetV3-Small real-vs-fake classifier (inference only).

The classifier runs on the *full* image, not the person crop, because it was
trained on full images from Parveshiiii/AI-vs-Real. Preprocessing matches
training exactly: JPEG re-encode at q75, then five 224px crops at native
resolution (no resizing), with probabilities averaged over the crops.
Uploads larger than RuntimeConfig.max_image_side are downscaled on decode,
which is the one step that training never sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .detectors import LayerStatus, _r
from .model_loader import ModelRegistry
from .vision_utils import to_pil


@dataclass
class ClassifierResult:
    status: LayerStatus
    label: str | None = None          # "real" | "fake"
    p_fake: float | None = None
    p_real: float | None = None
    confidence: float | None = None   # max(p_real, p_fake)
    run_id: str | None = None         # training run that produced the checkpoint
    input_region: str = "full_image"
    note: str | None = None


def classify(rgb: np.ndarray, registry: ModelRegistry) -> ClassifierResult:
    comp = registry.classifier
    clf = comp.try_get()
    if clf is None:
        return ClassifierResult(LayerStatus.UNAVAILABLE, note=comp.status()["error"])

    x = clf.transform(to_pil(rgb))            # (n_crops, 3, H, W)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    with torch.inference_mode():
        # Average the class probabilities over the native-resolution crops.
        probs = torch.softmax(clf.model(x.to(clf.device)), dim=1).mean(0).float().cpu().tolist()

    by_label = dict(zip(clf.labels, probs))
    p_fake, p_real = by_label["fake"], by_label["real"]
    label = "fake" if p_fake >= p_real else "real"
    return ClassifierResult(
        LayerStatus.OK, label, _r(p_fake), _r(p_real), _r(max(p_fake, p_real)), clf.run_id,
    )
