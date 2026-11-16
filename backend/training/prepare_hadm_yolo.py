"""Build a YOLO dataset from HADM (AI images with human-labelled anatomy errors).

Usage:  python -m training.prepare_hadm_yolo [--max-train N]

Classes (local = part boxes, global = whole-person boxes with a tag):
  0 malformed_hand   local "hand"
  1 malformed_face   local face / eye / mouth / nose / teeth / ear
  2 malformed_body   local arm / leg / feet / torso
  3 extra_limb       global "human with extra <part>"
  4 missing_limb     global "human missing <part>"

Splits:
  train   HADM train_ALL (the part downloaded) + real photos as label-free
          backgrounds (AI-vs-Real real train/val, HaGRID tuning sample), so the
          detector also learns what correct anatomy looks like.
  val     half of HADM val_ALL (deterministic by file name): early stopping and
          threshold choice.
  test    the other half of HADM val_ALL: final numbers only. Never used to
          choose anything.
The real-photo sets used for false-alarm testing (HaGRID held-out/fresh test,
AI-vs-Real real test split) are NOT used here.

Images are hard-linked (no extra disk space on NTFS), with a copy as fallback.
HADM has no stated licence: everything here stays local and is not redistributed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

DATA = settings.paths.data
HADM = DATA / "hadm"
OUT = DATA / "hadm_yolo"
CLASSES = ["malformed_hand", "malformed_face", "malformed_body", "extra_limb", "missing_limb"]
LOCAL = {"hand": 0, "face": 1, "eye": 1, "mouth": 1, "nose": 1, "teeth": 1, "ear": 1,
         "arm": 2, "leg": 2, "feet": 2, "torso": 2}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _yolo_lines(ann: dict) -> tuple[list[str], Counter]:
    W, H = ann["image"]["width"], ann["image"]["height"]
    lines, counts = [], Counter()

    def add(cls: int, box: list[float]) -> None:
        x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(box, (W, H, W, H)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        counts[CLASSES[cls]] += 1

    for a in ann.get("annotation", []):
        cls = LOCAL.get(str(a.get("body_parts")))
        if cls is not None:
            add(cls, a["bbox"])
    for h in ann.get("human", []):
        tags = h.get("tag", [])
        if any(t.startswith("human with extra") for t in tags):
            add(3, h["bbox"])
        if any(t.startswith("human missing") for t in tags):
            add(4, h["bbox"])
    return lines, counts


def _hadm_split(split_name: str, names: list[str], ann_dir: Path, img_dir: Path, stats: Counter) -> int:
    n = 0
    for name in names:
        ann = json.loads((ann_dir / name).read_text())
        img = img_dir / ann["image"]["file_name"]
        if not img.is_file():
            continue
        lines, counts = _yolo_lines(ann)
        _link(img, OUT / "images" / split_name / img.name)
        (OUT / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split_name / (img.stem + ".txt")).write_text("\n".join(lines))
        stats.update({f"{split_name}:{k}": v for k, v in counts.items()})
        stats[f"{split_name}:images"] += 1
        stats[f"{split_name}:images_without_boxes"] += int(not lines)
        n += 1
    return n


def _add_backgrounds(paths: list[Path], stats: Counter) -> None:
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
        dst = OUT / "images" / "train" / f"real_{p.parent.name}_{p.name}"
        _link(p, dst)
        (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / "train" / (dst.stem + ".txt")).write_text("")
        stats["train:real_backgrounds"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0, help="cap on HADM training images (0 = all downloaded)")
    ap.add_argument("--seed", type=int, default=settings.train.seed)
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    stats: Counter = Counter()

    # train: only images that were actually downloaded
    tr_ann = HADM / "annotations" / "train_ALL"
    tr_img = HADM / "images" / "train_ALL"
    have = {p.name for p in tr_img.iterdir()}
    train_names = [n for n in sorted(os.listdir(tr_ann)) if n.endswith(".json")
                   and json.loads((tr_ann / n).read_text())["image"]["file_name"] in have]
    random.Random(args.seed).shuffle(train_names)
    if args.max_train:
        train_names = train_names[: args.max_train]
    _hadm_split("train", train_names, tr_ann, tr_img, stats)

    # val / test: deterministic halves of val_ALL
    va_ann = HADM / "annotations" / "val_ALL"
    va_img = HADM / "images" / "val_ALL"
    val_names = sorted(n for n in os.listdir(va_ann) if n.endswith(".json"))
    half = lambda n: int(hashlib.md5(n.encode()).hexdigest(), 16) % 2
    _hadm_split("val", [n for n in val_names if half(n) == 0], va_ann, va_img, stats)
    _hadm_split("test", [n for n in val_names if half(n) == 1], va_ann, va_img, stats)

    # real backgrounds (label-free) for training only
    real = []
    manifest = DATA / "aivsreal" / "manifest.csv"
    if manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as f:
            real += [DATA / "aivsreal" / r["path"] for r in csv.DictReader(f) if r["label"] == "real" and r["split"] != "test"]
    real += sorted((DATA / "hagrid_sample").glob("*.jpg"))
    _add_backgrounds(real, stats)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (OUT / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
