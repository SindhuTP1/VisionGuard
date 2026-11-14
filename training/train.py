"""Fine-tune MobileNetV3-Small (ImageNet weights) for real vs fake.

Usage:  python -m training.train [--epochs N] [--no-promote]

Spec: AdamW (lr 1e-3, wd 1e-4), CrossEntropyLoss, batch 32, up to 12 epochs,
early stopping with patience 3 on val loss. The best-val-loss weights are kept.

Writes runs/<run_id>/:
  best.pt         checkpoint (best val loss)
  history.json    per-epoch train/val loss and accuracy
  metrics.json    everything the API's /metrics serves, computed from this run
Then copies best.pt to the serving path (weights/mobilenetv3_small_aivsreal.pt)
unless --no-promote is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CLASSIFIER_LABELS, settings  # noqa: E402
from app.model_loader import build_classifier, resolve_device, save_classifier_checkpoint  # noqa: E402
from app.vision_utils import build_classifier_transform  # noqa: E402
from training.evaluate import (  # noqa: E402
    DATA_DIR, ManifestDataset, compute_metrics, evaluate_checkpoint, metadata_baseline, predict, read_manifest,
)

log = logging.getLogger("visionguard.train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=settings.train.epochs)
    ap.add_argument("--no-promote", action="store_true", help="don't copy the checkpoint to the serving path")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = settings.train
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = settings.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")])

    seed_everything(cfg.seed)
    device = resolve_device(settings.runtime.device)
    log.info("run %s on %s (torch %s)", run_id, device, torch.__version__)

    train_rows, val_rows = read_manifest("train"), read_manifest("val")
    log.info("train=%d val=%d", len(train_rows), len(val_rows))
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0, worker_init_fn=_worker_init)
    train_dl = DataLoader(ManifestDataset(train_rows, build_classifier_transform(True)), shuffle=True,
                          generator=g, drop_last=False, **common)
    val_dl = DataLoader(ManifestDataset(val_rows, build_classifier_transform(False)), shuffle=False, **common)

    model = build_classifier(len(CLASSIFIER_LABELS), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    history: list[dict] = []
    t_run = time.time()
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        val_probs, val_labels, val_loss = predict(model, val_dl, device)
        val_m = compute_metrics(val_labels, val_probs, settings.arbitration.confidence_threshold)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "train_accuracy": correct / n,
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_roc_auc": val_m["roc_auc"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)

        improved = val_loss < best_loss - 1e-6
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            save_classifier_checkpoint(model, run_dir / "best.pt", run_id, extra={"epoch": epoch})
        else:
            bad_epochs += 1
        log.info("epoch %2d | train loss %.4f acc %.4f | val loss %.4f acc %.4f auc %.4f | %.0fs%s",
                 epoch, rec["train_loss"], rec["train_accuracy"], val_loss, rec["val_accuracy"],
                 rec["val_roc_auc"] or float("nan"), rec["seconds"], "  *best*" if improved else "")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if bad_epochs >= cfg.early_stopping_patience:
            stopped_early = True
            log.info("early stopping: no val-loss improvement for %d epochs", bad_epochs)
            break

    train_seconds = time.time() - t_run
    log.info("best epoch %d (val loss %.4f); evaluating best checkpoint on val + test", best_epoch, best_loss)
    val_best = evaluate_checkpoint(run_dir / "best.pt", "val")
    test = evaluate_checkpoint(run_dir / "best.pt", "test")
    # Scale stress test: put every test image at the same short side (512px, so fakes
    # shrink and reals grow), then evaluate. If accuracy collapses, the model is
    # still keying on native scale rather than generation artefacts.
    stress = evaluate_checkpoint(run_dir / "best.pt", "test", rescale_short_side=512)

    stats_path = DATA_DIR / "dataset_stats.json"
    metrics = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"arch": settings.classifier.arch, "pretrained": "ImageNet1K_V1",
                  "labels": list(CLASSIFIER_LABELS), "img_size": settings.classifier.img_size,
                  "params": sum(p.numel() for p in model.parameters())},
        "preprocessing": {
            "tag": settings.classifier.preprocessing,
            "jpeg_normalize_quality": settings.classifier.jpeg_normalize_quality,
            "train": "JPEG re-encode -> random native-resolution crop (no resize) -> hflip",
            "eval": f"JPEG re-encode -> {settings.classifier.eval_crops} native-resolution crops, probabilities averaged",
        },
        "train_config": {**dataclasses.asdict(cfg), "epochs_requested": args.epochs},
        "dataset": json.loads(stats_path.read_text()) if stats_path.is_file() else None,
        "training": {"epochs_run": len(history), "best_epoch": best_epoch, "best_val_loss": best_loss,
                     "stopped_early": stopped_early, "seconds": round(train_seconds, 1), "device": str(device)},
        "history": history,
        "val": val_best,
        "test": test,
        # Describes the *raw dataset*: metadata alone predicts the label. The
        # preprocessing above removes size/aspect/JPEG-quality from the model's input.
        "shortcut_probe": metadata_baseline("train", "test"),
        "scale_stress_test": stress,
        "env": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sp = metrics["shortcut_probe"]
    log.info("SHORTCUT PROBE (raw-dataset metadata only, no pixels): test acc=%.4f auc=%.4f", sp["accuracy"], sp["roc_auc"])
    log.info("SCALE STRESS (all test images at short side 512): acc=%.4f auc=%.4f", stress["accuracy"], stress["roc_auc"])
    log.info("TEST n=%d acc=%.4f (95%% CI %s) f1=%.4f auc=%.4f | coverage@%.2f=%.3f acc_on_covered=%s",
             test["n"], test["accuracy"], [round(v, 4) for v in test["ci95"]["accuracy"]], test["f1"],
             test["roc_auc"], test["selective"]["threshold"], test["selective"]["coverage"],
             test["selective"]["accuracy_on_covered"])

    if not args.no_promote:
        dest = settings.paths.classifier_checkpoint
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "best.pt", dest)
        log.info("promoted checkpoint -> %s", dest)
    log.info("wrote %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
