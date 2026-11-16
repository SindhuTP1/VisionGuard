"""Fine-tune YOLOv8n to detect human anatomy artefacts (HADM labels), then evaluate.

Usage:
  python -m training.train_artifacts [--epochs 20] [--imgsz 512] [--fraction 1.0] [--batch 16]
  python -m training.train_artifacts --eval-only runs/artifacts-<id>/weights/best.pt

Evaluation (all computed, written to runs/artifacts-<id>/artifact_metrics.json):
  * Ultralytics mAP on the HADM *test* half (never used for training or tuning).
  * Image-level detection per class: an image counts as "caught" if a box of that
    class with conf >= threshold overlaps a labelled box of the same class (IoU >= 0.1).
  * The confidence threshold is chosen on the HADM *val* half (best image-level
    F1 for "any artefact"), then frozen for test and real photos.
  * False alarms on REAL photos never used in training: HaGRID held-out + fresh
    test sets and the AI-vs-Real real test split. Any box there is a false alarm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from app.config import settings  # noqa: E402
from app.model_loader import ensure_weight  # noqa: E402
from training.eval_fingers import wilson  # noqa: E402
from training.prepare_hadm_yolo import CLASSES, OUT as YOLO_DIR  # noqa: E402

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _labels(split: str) -> dict[str, list[tuple[int, list[float]]]]:
    """Ground truth per image: [(cls, [x1, y1, x2, y2] normalised)]."""
    out = {}
    for f in (YOLO_DIR / "labels" / split).glob("*.txt"):
        boxes = []
        for line in f.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            boxes.append((int(c), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        out[f.stem] = boxes
    return out


def _predict(model, paths: list[Path], conf: float = 0.05, imgsz: int = 512, batch: int = 16) -> dict[str, list]:
    """All boxes with conf >= ``conf``: {stem: [(cls, conf, [x1,y1,x2,y2] normalised)]}."""
    preds = {}
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        for p, r in zip(chunk, model.predict(chunk, conf=conf, imgsz=imgsz, verbose=False, device="cpu")):
            b = r.boxes
            preds[Path(p).stem] = [(int(c), float(s), list(map(float, xy)))
                                   for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
    return preds


def _image_level(preds: dict, gts: dict, thr: float) -> dict:
    """Per class + 'any': caught = a predicted box of that class (conf>=thr) overlaps a labelled one."""
    res = {}
    for ci, name in [(i, c) for i, c in enumerate(CLASSES)] + [(-1, "any_artefact")]:
        tp = fn = fp = tn = 0
        for stem, gt in gts.items():
            # labelled and predicted boxes for this class (all classes for "any")
            g = [(c, b) for c, b in gt if ci == -1 or c == ci]
            p = [(c, b) for c, s, b in preds.get(stem, []) if s >= thr and (ci == -1 or c == ci)]
            # caught = some prediction overlaps a labelled box OF THE SAME CLASS
            hit = any(pc == gc and _iou(pb, gb) >= 0.1 for pc, pb in p for gc, gb in g)
            if g:
                tp += hit
                fn += not hit
            else:
                fp += bool(p)
                tn += not p
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        res[name] = {"images_with_label": tp + fn, "caught": tp, "recall": rec, "recall_ci95": wilson(tp, tp + fn),
                     "images_without_label": fp + tn, "flagged_without_label": fp,
                     "false_alarm_rate": fp / (fp + tn) if fp + tn else None, "precision": prec, "f1": f1}
    return res


def _real_photo_paths() -> dict[str, list[Path]]:
    d = settings.paths.data
    sets = {
        "hagrid_holdout": sorted((d / "hagrid_holdout").glob("*.jpg")),
        "hagrid_fresh_test": sorted((d / "hagrid_test2").glob("*.jpg")),
    }
    man = d / "aivsreal" / "manifest.csv"
    if man.is_file():
        with open(man, newline="", encoding="utf-8") as f:
            sets["aivsreal_real_test"] = [d / "aivsreal" / r["path"] for r in csv.DictReader(f)
                                          if r["label"] == "real" and r["split"] == "test"]
    return {k: v for k, v in sets.items() if v}


def evaluate(weights: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict = {"weights": str(weights), "classes": CLASSES}

    m = model.val(data=str(YOLO_DIR / "data.yaml"), split="test", imgsz=imgsz, device="cpu", verbose=False, plots=False)
    out["test_map"] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                       "per_class_mAP50": {CLASSES[i]: float(v) for i, v in zip(m.box.ap_class_index, m.box.ap50)}}

    # threshold from VAL only
    val_gt = _labels("val")
    val_pred = _predict(model, [p for p in (YOLO_DIR / "images" / "val").iterdir()], imgsz=imgsz)
    grid = [round(x, 2) for x in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)]
    scores = {t: _image_level(val_pred, val_gt, t)["any_artefact"]["f1"] for t in grid}
    thr = max(scores, key=scores.get)
    out["threshold"] = {"value": thr, "chosen_on": "HADM val half, best image-level F1 (any artefact)", "val_f1_by_threshold": scores}

    test_gt = _labels("test")
    test_pred = _predict(model, [p for p in (YOLO_DIR / "images" / "test").iterdir()], imgsz=imgsz)
    out["test_image_level"] = _image_level(test_pred, test_gt, thr)

    out["real_photos"] = {}
    for name, paths in _real_photo_paths().items():
        preds = _predict(model, paths, imgsz=imgsz)
        flagged = [Path(p).stem for p in paths if any(s >= thr for _, s, _ in preds.get(Path(p).stem, []))]
        per_cls = {c: sum(any(pc == i and s >= thr for pc, s, _ in preds.get(Path(p).stem, [])) for p in paths)
                   for i, c in enumerate(CLASSES)}
        out["real_photos"][name] = {"images": len(paths), "flagged": len(flagged),
                                    "false_alarm_rate": len(flagged) / len(paths), "ci95": wilson(len(flagged), len(paths)),
                                    "flagged_by_class": per_cls, "examples": flagged[:10]}
    return out


def promote(run_dir: Path, threshold: float, run_id: str) -> None:
    """Install best.pt where the API loads it, with its val-chosen threshold beside it."""
    import shutil

    dest = settings.paths.weights / settings.artifacts.weights_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "weights" / "best.pt", dest)
    dest.with_suffix(".json").write_text(json.dumps({"threshold": threshold, "run_id": run_id}, indent=2))
    print(f"promoted -> {dest} (threshold {threshold})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train split to use")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-only", type=Path, default=None)
    args = ap.parse_args()

    if args.eval_only:
        res = evaluate(args.eval_only, args.imgsz)
        dest = args.eval_only.parent.parent / "artifact_metrics.json"
        dest.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    from ultralytics import YOLO

    run_id = "artifacts-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    base = ensure_weight("yolov8n.pt", BASE_URL)
    t0 = time.time()
    model = YOLO(str(base))
    model.train(
        data=str(YOLO_DIR / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="cpu", workers=args.workers, patience=args.patience, fraction=args.fraction,
        project=str(settings.paths.runs), name=run_id, exist_ok=False, seed=settings.train.seed,
        deterministic=True, plots=True, verbose=True,
    )
    run_dir = settings.paths.runs / run_id
    res = evaluate(run_dir / "weights" / "best.pt", args.imgsz)
    res["training"] = {"run_id": run_id, "epochs_requested": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                       "fraction": args.fraction, "seconds": round(time.time() - t0), "base": "yolov8n.pt (COCO)",
                       "train_stats": json.loads((YOLO_DIR / "stats.json").read_text())}
    (run_dir / "artifact_metrics.json").write_text(json.dumps(res, indent=2))
    promote(run_dir, res["threshold"]["value"], run_id)
    print(json.dumps({k: res[k] for k in ("test_map", "threshold", "real_photos")}, indent=2))
    print("ANY ARTEFACT (test):", json.dumps(res["test_image_level"]["any_artefact"]))


if __name__ == "__main__":
    main()
