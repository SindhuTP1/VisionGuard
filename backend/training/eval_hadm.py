"""Evaluate the anatomy checks against HADM's human labels (AI images only).

HADM ("Detecting Human Artifacts from Text-to-Image Models", wangkaihong/HADM)
labels AI images with boxes on malformed body parts ("hand", severity
mild/severe) and per-person tags such as "human with extra hand". The dataset
is used here only locally, for evaluation, and is never redistributed (no
licence is stated).

Measures:
1. Extra hand/arm, per image. Positives are images tagged "human with extra
   hand" or "human with extra arm"; negatives are the remaining images. Signal:
   the body check's extra_hand finding.
2. Malformed hand, per detected hand. A hand MediaPipe detects is a positive
   if it overlaps a labelled hand box, a negative otherwise. Unlabelled is not
   proof of correct, so the "false alarm" rate here is an upper bound. Signals:
   extra finger (6+ counted), implausible skeleton, either.

Usage:  python -m training.eval_hadm [--limit N] [--workers 6] [--out runs/.../hadm_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.body import analyze_body  # noqa: E402
from app.config import settings  # noqa: E402
from app.detectors import analyze_hands, detect_person  # noqa: E402
from app.model_loader import get_registry  # noqa: E402
from app.vision_utils import load_image_path  # noqa: E402
from training.eval_fingers import wilson  # noqa: E402

HADM = Path(__file__).resolve().parent.parent / "data" / "hadm"
EXTRA_TAGS = {"human with extra hand", "human with extra arm"}


def _overlaps(hand_lm: list[list[float]], box: list[float], grow: float = 0.2) -> bool:
    xs = [p[0] for p in hand_lm]
    ys = [p[1] for p in hand_lm]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x1, y1, x2, y2 = box
    gw, gh = (x2 - x1) * grow, (y2 - y1) * grow
    if x1 - gw <= cx <= x2 + gw and y1 - gh <= cy <= y2 + gh:
        return True
    # or any landmark inside the box
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in hand_lm)


def _process(names: list[str]) -> list[dict]:
    reg = get_registry()
    reg.yolo.get(), reg.hands.get()
    out = []
    for name in names:
        ann = json.loads((HADM / "annotations" / "val_ALL" / name).read_text())
        img_name = ann["image"]["file_name"]
        path = HADM / "images" / "val_ALL" / img_name
        if not path.is_file():
            continue
        rgb = load_image_path(path)
        person, region = detect_person(rgb, reg, settings.yolo)
        hands = analyze_hands(region, reg, settings.hand, settings.count, (float(person.bbox[0]), float(person.bbox[1])))
        body = analyze_body(person, hands, settings.body)
        tags = {t for h in ann.get("human", []) for t in h.get("tag", [])}
        hand_boxes = [a["bbox"] for a in ann.get("annotation", []) if a.get("body_parts") == "hand"]
        rec_hands = []
        for hf in hands.hands:
            if len(hf.landmarks) < 21:
                continue
            fc = hf.finger_count
            rec_hands.append({
                "labelled_malformed": any(_overlaps(hf.landmarks, b) for b in hand_boxes),
                "extra_finger": bool(fc and fc.extra_finger),
                "counted": bool(fc and fc.status == "counted"),
                "count": fc.count if fc else None,
                "skeleton_flag": bool(hf.skipped_reason is None and hf.anomalous),
            })
        out.append({
            "image": img_name,
            "extra_tag": bool(tags & EXTRA_TAGS),
            "tags": sorted(tags),
            "n_hand_boxes": len(hand_boxes),
            "persons": person.num_persons,
            "extra_hand_flag": bool(body.extra_hand),
            "hands": rec_hands,
        })
    return out


def _rates(pos_hits: int, pos: int, neg_hits: int, neg: int) -> dict:
    prec = pos_hits / (pos_hits + neg_hits) if (pos_hits + neg_hits) else None
    return {
        "positives": pos, "negatives": neg,
        "detected_positives": pos_hits, "recall": round(pos_hits / pos, 4) if pos else None,
        "recall_ci95": wilson(pos_hits, pos),
        "flagged_negatives": neg_hits, "false_alarm_rate": round(neg_hits / neg, 4) if neg else None,
        "false_alarm_ci95": wilson(neg_hits, neg),
        "precision": round(prec, 4) if prec is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = sorted(p.name for p in (HADM / "annotations" / "val_ALL").glob("*.json"))
    if args.limit:
        names = names[: args.limit]
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [names[i:: args.workers] for i in range(args.workers)]
        with ProcessPoolExecutor(args.workers) as ex:
            recs = [r for part in ex.map(_process, chunks) for r in part]
    else:
        recs = _process(names)

    # 1. extra hand / arm (image level)
    pos = [r for r in recs if r["extra_tag"]]
    neg = [r for r in recs if not r["extra_tag"]]
    extra_hand = _rates(sum(r["extra_hand_flag"] for r in pos), len(pos),
                        sum(r["extra_hand_flag"] for r in neg), len(neg))

    # 2. malformed hand (hand level)
    hands = [h for r in recs for h in r["hands"]]
    hp = [h for h in hands if h["labelled_malformed"]]
    hn = [h for h in hands if not h["labelled_malformed"]]
    def by(key):
        return _rates(sum(h[key] for h in hp), len(hp), sum(h[key] for h in hn), len(hn))
    for h in hands:
        h["either"] = h["extra_finger"] or h["skeleton_flag"]

    summary = {
        "dataset": "HADM val_ALL (AI images from SDXL, DALL-E 2/3, Midjourney; human-labelled artefacts)",
        "images": len(recs),
        "images_with_person": sum(r["persons"] > 0 for r in recs),
        "extra_hand_or_arm": extra_hand,
        "malformed_hand": {
            "hands_detected": len(hands),
            "hands_overlapping_a_labelled_malformed_hand": len(hp),
            "labelled_malformed_hand_boxes_total": sum(r["n_hand_boxes"] for r in recs),
            "hands_countable": sum(h["counted"] for h in hands),
            "by_extra_finger": by("extra_finger"),
            "by_skeleton": by("skeleton_flag"),
            "by_either": by("either"),
            "note": "negatives are hands without a malformed-hand label; unlabelled is not proof of correct, so false-alarm rates are upper bounds",
        },
        "config": {"count": {k: getattr(settings.count, k) for k in settings.count.__dataclass_fields__},
                   "body": {k: getattr(settings.body, k) for k in settings.body.__dataclass_fields__}},
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        (args.out.parent / (args.out.stem + "_records.json")).write_text(json.dumps(recs))


if __name__ == "__main__":
    main()
