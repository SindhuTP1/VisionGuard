"""Run the finger counter over a folder of images and summarise the counts.

Usage:
  python -m training.eval_fingers <image_dir> [--limit N] [--dump DIR] [--tag NAME]

On a folder of REAL hands, every "6+ fingertips" result is a false alarm. That
is the number this script exists to measure. Writes a JSON summary next to
--dump (or prints it).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.anatomy import count_fingers  # noqa: E402
from app.config import settings  # noqa: E402
from app.detectors import _to_mp_image, evaluate_hand  # noqa: E402
from app.model_loader import get_registry  # noqa: E402
from app.vision_utils import landmarks_to_array, load_image_path  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def draw(img, pts, fc, path):
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fc.outline:
        cv2.polylines(vis, [np.array(fc.outline, np.int32)], True, (0, 255, 255), 1)
    for (x, y), name in zip(fc.tips, fc.tip_fingers):
        cv2.circle(vis, (int(x), int(y)), 5, (0, 200, 0) if name else (0, 0, 255), 2)
    for (ax, ay), (bx, by) in fc.scan_lines:
        cv2.line(vis, (int(ax), int(ay)), (int(bx), int(by)), (255, 0, 255), 1)
    for i in (4, 8, 12, 16, 20):
        cv2.drawMarker(vis, (int(pts[i, 0]), int(pts[i, 1])), (255, 0, 0), cv2.MARKER_CROSS, 7, 1)
    label = (f"{fc.count} ({fc.method}: outline {fc.outline_count}, scan {fc.scan_count} {fc.scan_row_counts})"
             if fc.status == "counted" else "can't count")
    cv2.putText(vis, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if fc.extra_finger else (255, 255, 255), 2)
    cv2.imwrite(str(path), vis)


def _process(files: list[Path], dump: Path | None) -> list[dict]:
    """Worker: count fingers on every hand in ``files``. One landmarker per process."""
    landmarker = get_registry().hands.get()
    out = []
    for f in files:
        img = load_image_path(f)
        h, w = img.shape[:2]
        res = landmarker.detect(_to_mp_image(img))
        for i, lms in enumerate(res.hand_landmarks):
            pts = landmarks_to_array(lms, w, h)
            hf = evaluate_hand(pts, settings.hand, i)
            ext = [n for n, fd in hf.fingers.items() if fd["extended"]] if hf.fingers else None
            fc = count_fingers(img, pts, ext, settings.count)
            if dump and fc.status == "counted":
                draw(img, pts, fc, dump / f"{f.stem}_h{i}_{fc.count}.jpg")
            out.append({"file": f.name, "hand": i, "status": fc.status, "reason": fc.reason, "count": fc.count,
                        "outline": fc.outline_count, "scan": fc.scan_count, "extra": fc.extra_finger})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", type=Path, default=None, help="save visualisations of counted hands here")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    args = ap.parse_args()

    files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.limit:
        files = files[: args.limit]
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [files[i:: args.workers] for i in range(args.workers)]
        with ProcessPoolExecutor(args.workers) as ex:
            records = [r for part in ex.map(_process, chunks, [args.dump] * len(chunks)) for r in part]
    else:
        records = _process(files, args.dump)
    records.sort(key=lambda r: (r["file"], r["hand"]))

    hands = len(records)
    status = collections.Counter(r["status"] for r in records)
    reasons = collections.Counter((r["reason"] or "").split(" (")[0] for r in records if r["status"] != "counted")
    counted_recs = [r for r in records if r["status"] == "counted"]
    counts = collections.Counter(r["count"] for r in counted_recs)
    outline_counts = collections.Counter(r["outline"] for r in counted_recs)
    scan_counts = collections.Counter(str(r["scan"]) for r in counted_recs)
    flagged = [f"{r['file']}#{r['hand']}" for r in counted_recs if r["extra"]]
    images_flagged = {r["file"] for r in counted_recs if r["extra"]}

    counted = status["counted"]
    summary = {
        "tag": args.tag or args.image_dir.name,
        "images": len(files),
        "hands_detected": hands,
        "hands_counted": counted,
        "hands_cant_count": status["cant_count"],
        "cant_count_reasons": dict(reasons.most_common()),
        "count_distribution": {str(k): v for k, v in sorted(counts.items())},
        "outline_count_distribution": {str(k): v for k, v in sorted(outline_counts.items())},
        "scan_count_distribution": dict(sorted(scan_counts.items())),
        "hands_flagged_extra_finger": len(flagged),
        "flag_rate_per_counted_hand": round(len(flagged) / counted, 4) if counted else None,
        "flag_rate_ci95": wilson(len(flagged), counted),
        "images_flagged": len(images_flagged),
        "flagged_examples": flagged[:20],
        "config": {k: getattr(settings.count, k) for k in settings.count.__dataclass_fields__},
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    for path in filter(None, [args.out, args.dump / "summary.json" if args.dump else None]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
