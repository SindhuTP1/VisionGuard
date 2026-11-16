"""Layer 3: MobileNetV3-Small real-vs-fake classifier (inference only).

The classifier runs on the *full* image, not the person crop, because it was
trained on full images from Parveshiiii/AI-vs-Real. Preprocessing matches
training exactly: JPEG re-encode at q75, then five 224px crops at native
resolution (no resizing), with probabilities averaged over the crops.
Uploads larger than RuntimeConfig.max_image_side are downscaled on decode,
which is the one step that training never sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import settings
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


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
@dataclass
class GradCamCrop:
    box: list[float]            # [x1, y1, x2, y2] in full-image pixels (may extend past the edge if padded)
    grid: list[list[float]]     # CAM on the model's 7x7 feature grid, 0..1 (normalised across all crops)
    p_target: float             # probability of the target class for this crop alone


@dataclass
class GradCamResult:
    status: LayerStatus
    target: str | None = None   # class the heatmap explains (the predicted one)
    crops: list[GradCamCrop] = field(default_factory=list)
    note: str | None = None


def _five_crop_boxes(w: int, h: int, size: int) -> list[list[float]]:
    """Where torchvision's PadToMin + FiveCrop take their crops, in original-image pixels.
    Order matches FiveCrop: top-left, top-right, bottom-left, bottom-right, centre."""
    ph, pw = max(0, size - h), max(0, size - w)
    oy, ox = ph // 2, pw // 2            # PadToMin pads symmetrically (top/left get the floor half)
    W, H = w + pw, h + ph
    cx, cy = int(round((W - size) / 2.0)), int(round((H - size) / 2.0))
    corners = [(0, 0), (W - size, 0), (0, H - size), (W - size, H - size), (cx, cy)]
    return [[float(x - ox), float(y - oy), float(x - ox + size), float(y - oy + size)] for x, y in corners]


def grad_cam(rgb: np.ndarray, registry: ModelRegistry, target: str | None = None) -> GradCamResult:
    """Grad-CAM (Selvaraju et al. 2017) on the classifier's last conv block, per crop.

    The classifier only ever sees its five 224px native-resolution crops, so the
    heatmap exists only where those crops are. Everywhere else, the model didn't look.
    """
    comp = registry.classifier
    clf = comp.try_get()
    if clf is None:
        return GradCamResult(LayerStatus.UNAVAILABLE, note=comp.status()["error"])

    x = clf.transform(to_pil(rgb))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(clf.device)
    layer = clf.model.features[-1]
    store: dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        with comp.lock, torch.enable_grad():
            clf.model.zero_grad(set_to_none=True)
            logits = clf.model(x)
            probs = torch.softmax(logits, 1)
            if target is None:
                target = clf.labels[int(probs.mean(0).argmax())]
            t = clf.labels.index(target)
            logits[:, t].sum().backward()
            act, grad = store["act"].detach(), store["grad"].detach()      # (n, C, 7, 7)
    finally:
        h1.remove()
        h2.remove()
        clf.model.zero_grad(set_to_none=True)

    weights = grad.mean(dim=(2, 3), keepdim=True)                         # per-channel importance
    cam = torch.relu((weights * act).sum(1)).cpu().numpy()                  # (n, 7, 7)
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    h, w = rgb.shape[:2]
    boxes = _five_crop_boxes(w, h, settings.classifier.img_size)
    crops = [GradCamCrop(b, [[round(float(v), 3) for v in row] for row in cam[i]], _r(float(probs[i, t].detach())))
             for i, b in enumerate(boxes[: len(cam)])]
    return GradCamResult(LayerStatus.OK, target, crops,
                         note="Heatmap covers only the five 224px patches the classifier examines")
