"""Evaluate a classifier checkpoint on a manifest split and compute metrics.

Usage:
  python -m training.evaluate --run-id <run_id> [--split test]
  python -m training.evaluate --checkpoint path/to/model.pt [--split val]

Every number written here is computed from model predictions on the stored
split. Nothing is hardcoded. Also used by train.py at the end of each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CLASSIFIER_LABELS, settings  # noqa: E402
from app.model_loader import load_classifier  # noqa: E402
from app.vision_utils import load_image_path, to_pil  # noqa: E402

DATA_DIR = settings.paths.data / "aivsreal"
POS = CLASSIFIER_LABELS.index("fake")   # "positive" class for precision/recall/AUC


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_manifest(split: str | None = None, root: Path = DATA_DIR) -> list[dict]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run `python -m training.prepare_dataset` first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


class ManifestDataset(Dataset):
    """Loads images through the *same* decode path the API uses (load_image_path → PIL).

    ``rescale_short_side`` is only used by the scale stress test: it resizes every
    image so its short side equals that value before the normal transform runs.
    """

    def __init__(self, rows: list[dict], transform, root: Path = DATA_DIR,
                 rescale_short_side: int | None = None) -> None:
        self.rows, self.transform, self.root = rows, transform, root
        self.rescale_short_side = rescale_short_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = to_pil(load_image_path(self.root / r["path"]))
        if self.rescale_short_side:
            s = self.rescale_short_side / min(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
        return self.transform(img), CLASSIFIER_LABELS.index(r["label"])


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (probs [N, C], labels [N], mean cross-entropy loss)."""
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.dim() == 5:                          # (B, n_crops, 3, H, W): average over crops
            b, n = x.shape[:2]
            p = torch.softmax(model(x.flatten(0, 1)), 1).view(b, n, -1).mean(1)
        else:
            p = torch.softmax(model(x), 1)
        # Cross-entropy of the (crop-averaged) predicted distribution.
        loss_sum += torch.nn.functional.nll_loss(torch.log(p.clamp_min(1e-12)), y, reduction="sum").item()
        probs.append(p.float().cpu().numpy())
        labels.append(y.cpu().numpy())
    p, l = np.concatenate(probs), np.concatenate(labels)
    return p, l, loss_sum / max(len(l), 1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.5, 1.0, bins + 1)   # binary: max-prob is always >= 0.5
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss, classification_report,
        confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
    )

    pred = probs.argmax(1)
    p_pos = probs[:, POS]
    y_pos = (labels == POS).astype(int)
    conf = probs.max(1)
    correct = (pred == labels).astype(float)
    covered = conf >= threshold
    both_classes = len(np.unique(labels)) == 2

    cm = confusion_matrix(labels, pred, labels=list(range(len(CLASSIFIER_LABELS))))
    tn, fp, fn, tp = cm[1 - POS, 1 - POS], cm[1 - POS, POS], cm[POS, 1 - POS], cm[POS, POS]
    report = classification_report(labels, pred, labels=list(range(len(CLASSIFIER_LABELS))),
                                   target_names=list(CLASSIFIER_LABELS), output_dict=True, zero_division=0)
    return {
        "n": int(len(labels)),
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(CLASSIFIER_LABELS)},
        "positive_class": CLASSIFIER_LABELS[POS],
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(y_pos, pred == POS, zero_division=0)),
        "recall": float(recall_score(y_pos, pred == POS, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_pos, pred == POS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_pos, p_pos)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_pos, p_pos)) if both_classes else None,
        "log_loss": float(log_loss(labels, probs, labels=list(range(len(CLASSIFIER_LABELS))))),
        "brier": float(brier_score_loss(y_pos, p_pos)),
        "ece": _ece(conf, correct),
        "confusion_matrix": {"labels": list(CLASSIFIER_LABELS), "rows_true_cols_pred": cm.tolist(),
                             "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSIFIER_LABELS},
        # How the 0.65 "Needs Review" gate behaves on this split (classifier alone).
        "selective": {
            "threshold": threshold,
            "coverage": float(covered.mean()),
            "review_rate": float(1 - covered.mean()),
            "accuracy_on_covered": float(correct[covered].mean()) if covered.any() else None,
            "accuracy_on_review": float(correct[~covered].mean()) if (~covered).any() else None,
        },
    }


def bootstrap_ci(labels: np.ndarray, probs: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95% percentile bootstrap CIs. With only ~300 test images these intervals matter."""
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    acc, auc, f1 = [], [], []
    n = len(labels)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        y, p = labels[i], pred[i]
        acc.append((y == p).mean())
        f1.append(f1_score(y == POS, p == POS, zero_division=0))
        if len(np.unique(y)) == 2:
            auc.append(roc_auc_score(y == POS, probs[i, POS]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    return {"n_boot": n_boot, "accuracy": ci(acc), "f1": ci(f1), "roc_auc": ci(auc)}


def metadata_baseline(train_split: str = "train", eval_split: str = "test") -> dict[str, Any]:
    """Shortcut probe: how well can real vs fake be told apart *without pixels*?

    Logistic regression on width, height, aspect ratio and file size only. If this
    rivals the CNN, the dataset leaks the label through metadata and the CNN's
    accuracy cannot be read as "detects generation artefacts".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def feats(rows):
        X = np.array([[float(r["width"]), float(r["height"]), float(r["width"]) / float(r["height"]),
                       np.log(float(r["bytes"]))] for r in rows])
        y = np.array([CLASSIFIER_LABELS.index(r["label"]) for r in rows])
        return X, y

    Xtr, ytr = feats(read_manifest(train_split))
    Xte, yte = feats(read_manifest(eval_split))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, POS]
    return {
        "features": ["width", "height", "aspect_ratio", "log_file_bytes"],
        "model": "logistic_regression",
        "split": eval_split,
        "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(yte == POS, p)),
    }


def evaluate_checkpoint(ckpt: Path, split: str = "test", batch_size: int | None = None,
                        num_workers: int | None = None, rescale_short_side: int | None = None) -> dict[str, Any]:
    clf = load_classifier(settings, ckpt)
    rows = read_manifest(split)
    if not rows:
        raise SystemExit(f"No rows for split {split!r}")
    loader = DataLoader(ManifestDataset(rows, clf.transform, rescale_short_side=rescale_short_side),
                        batch_size=batch_size or settings.train.batch_size, shuffle=False,
                        num_workers=settings.train.num_workers if num_workers is None else num_workers)
    probs, labels, loss = predict(clf.model, loader, clf.device)
    m = compute_metrics(labels, probs, settings.arbitration.confidence_threshold)
    if rescale_short_side:
        m["rescale_short_side"] = rescale_short_side
    m["mean_ce_loss"] = loss
    m["ci95"] = bootstrap_ci(labels, probs, seed=settings.train.seed)
    m["split"] = split
    m["checkpoint_run_id"] = clf.run_id
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--checkpoint", type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    ckpt = args.checkpoint or settings.paths.runs / args.run_id / "best.pt"
    m = evaluate_checkpoint(ckpt, args.split)
    out = ckpt.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps({k: m[k] for k in ("n", "accuracy", "f1", "roc_auc", "selective", "ci95")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
