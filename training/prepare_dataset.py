"""Download a balanced 1,000 real / 1,000 fake subset of Parveshiiii/AI-vs-Real.

Usage:  python -m training.prepare_dataset [--force]

Facts about the source data (verified from the parquet files, not taken from the card):
* Labels: 0 = AI-generated, 1 = real (mapped via TrainConfig.hf_label_map).
* The shards are sorted by source: whole shards are one class, stored in
  100-row row groups. Neighbouring rows probably share a source and generator.

So we sample whole row groups, take ``n / groups_per_class`` random rows from
each, and assign train/val/test **by row group**. That stops near-duplicate
neighbours from leaking between splits and inflating test metrics. Only the
chosen row groups are downloaded (HTTP range reads), not the full ~2 GB.

Original image bytes are written unchanged. Re-encoding would alter the
compression artefacts a real-vs-fake classifier learns from.

Outputs (under data/aivsreal/):
  real/*.jpg|png, fake/*.jpg|png
  manifest.csv          one row per image, including its split
  dataset_stats.json    counts, formats and sizes per class/split, plus the source revision
  label_index.json      cached row-level labels (avoids re-scanning shards)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

log = logging.getLogger("visionguard.prepare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _noisy in ("httpx", "huggingface_hub", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

OUT_DIR = settings.paths.data / "aivsreal"
MANIFEST_FIELDS = ["path", "label", "hf_label", "shard", "row_group", "row", "split",
                   "sha1", "width", "height", "format", "bytes"]


@dataclass(frozen=True)
class GroupRef:
    shard: str
    row_group: int
    rows: tuple[int, ...]   # row indices *within the row group* for one class


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def resolve_revision(repo: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def build_label_index(repo: str, revision: str, cache: Path) -> dict:
    """Row-level labels for every row group: {shard: [[labels of rg0], [labels of rg1], ...]}."""
    if cache.is_file():
        idx = json.loads(cache.read_text())
        if idx.get("revision") == revision:
            return idx
    import pyarrow.parquet as pq

    fs = _fs()
    shards = sorted(p for p in fs.ls(f"datasets/{repo}@{revision}/data", detail=False) if p.endswith(".parquet"))
    idx = {"revision": revision, "shards": {}}
    for shard in shards:
        t0 = time.time()
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            groups = [pf.read_row_group(g, columns=["binary_label"]).column(0).to_pylist()
                      for g in range(pf.metadata.num_row_groups)]
        idx["shards"][shard] = groups
        log.info("indexed %s: %d row groups (%.1fs)", shard.rsplit("/", 1)[-1], len(groups), time.time() - t0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def candidate_groups(idx: dict, hf_label: int, min_rows: int) -> list[GroupRef]:
    out = []
    for shard, groups in idx["shards"].items():
        for g, labels in enumerate(groups):
            rows = tuple(i for i, lab in enumerate(labels) if lab == hf_label)
            if len(rows) >= min_rows:
                out.append(GroupRef(shard, g, rows))
    return out


def _split_groups(groups: list[GroupRef], val_frac: float, test_frac: float) -> dict[GroupRef, str]:
    n = len(groups)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    assign = {}
    for i, g in enumerate(groups):  # groups are already shuffled
        assign[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def _inspect(data: bytes) -> tuple[int, int, str] | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.width, im.height, (im.format or "UNKNOWN").upper()
    except Exception:  # noqa: BLE001 - any decode failure means skip
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete and rebuild the output directory")
    args = ap.parse_args()

    cfg = settings.train
    label_map = dict(cfg.hf_label_map)
    targets = {"real": cfg.n_real, "fake": cfg.n_fake}

    if (OUT_DIR / "manifest.csv").is_file() and not args.force:
        log.info("Dataset already prepared at %s (use --force to rebuild)", OUT_DIR)
        return
    if OUT_DIR.exists() and args.force:
        cache = OUT_DIR / "label_index.json"
        keep = cache.read_bytes() if cache.is_file() else None
        shutil.rmtree(OUT_DIR)
        if keep:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(keep)

    revision = resolve_revision(cfg.hf_dataset)
    log.info("dataset %s @ %s", cfg.hf_dataset, revision)
    idx = build_label_index(cfg.hf_dataset, revision, OUT_DIR / "label_index.json")

    all_counts = Counter(lab for groups in idx["shards"].values() for g in groups for lab in g)
    log.info("source label counts: %s", {label_map[k]: v for k, v in sorted(all_counts.items())})

    rng = random.Random(cfg.seed)
    plan: dict[str, list[GroupRef]] = {}
    for hf_label, name in label_map.items():
        per_group = math.ceil(targets[name] / cfg.groups_per_class)
        cands = candidate_groups(idx, hf_label, per_group)
        if len(cands) < cfg.groups_per_class:
            raise SystemExit(f"Only {len(cands)} row groups with >= {per_group} {name} rows; need {cfg.groups_per_class}")
        rng.shuffle(cands)
        plan[name] = cands[: cfg.groups_per_class]
        log.info("%s: %d candidate groups, sampling %d x %d rows", name, len(cands), cfg.groups_per_class, per_group)

    import pyarrow.parquet as pq

    fs = _fs()
    seen_sha: dict[str, str] = {}
    rows_out: list[dict] = []
    skipped = Counter()

    for name, groups in plan.items():
        hf_label = next(k for k, v in label_map.items() if v == name)
        split_of = _split_groups(groups, cfg.val_frac, cfg.test_frac)
        per_group = math.ceil(targets[name] / len(groups))
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
        by_shard: dict[str, list[GroupRef]] = defaultdict(list)
        for g in groups:
            by_shard[g.shard].append(g)

        got = 0
        for shard, gs in by_shard.items():
            with fs.open(shard, "rb", block_size=8 << 20) as f:
                pf = pq.ParquetFile(f)
                for g in sorted(gs, key=lambda x: x.row_group):
                    t0 = time.time()
                    images = pf.read_row_group(g.row_group, columns=["image"]).column("image").to_pylist()
                    order = list(g.rows)
                    rng.shuffle(order)
                    taken = 0
                    for r in order:   # extra rows in the group top up any rejected images
                        if taken >= per_group or got >= targets[name]:
                            break
                        data = images[r]["bytes"]
                        sha = hashlib.sha1(data).hexdigest()
                        if sha in seen_sha:
                            skipped["duplicate" if seen_sha[sha] == name else "cross_class_duplicate"] += 1
                            continue
                        info = _inspect(data)
                        if info is None:
                            skipped["undecodable"] += 1
                            continue
                        w, h, fmt = info
                        ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(fmt, fmt.lower())
                        shard_id = Path(shard).stem.split("-")[1]
                        rel = Path(name) / f"{shard_id}_{g.row_group:03d}_{r:03d}.{ext}"
                        (OUT_DIR / rel).write_bytes(data)
                        seen_sha[sha] = name
                        rows_out.append({
                            "path": rel.as_posix(), "label": name, "hf_label": hf_label,
                            "shard": Path(shard).name, "row_group": g.row_group, "row": r,
                            "split": split_of[g], "sha1": sha, "width": w, "height": h,
                            "format": fmt, "bytes": len(data),
                        })
                        taken += 1
                        got += 1
                    log.info("%s %s rg%03d -> %d images [%s] (%.1fs)", name, Path(shard).stem, g.row_group,
                             taken, split_of[g], time.time() - t0)
        if got < targets[name]:
            raise SystemExit(f"Collected only {got}/{targets[name]} {name} images (skipped: {dict(skipped)})")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        wr.writeheader()
        wr.writerows(rows_out)

    stats = build_stats(rows_out, revision, all_counts, label_map, dict(skipped))
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("wrote %d images; split counts %s", len(rows_out), stats["split_counts"])
    for name, s in stats["per_class"].items():
        log.info("%s formats=%s median_size=%sx%s", name, s["formats"], s["median_width"], s["median_height"])


def build_stats(rows: list[dict], revision: str, source_counts: Counter, label_map: dict, skipped: dict) -> dict:
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    groups: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        split_counts[r["split"]][r["label"]] += 1
        groups[r["split"]][r["label"]].add((r["shard"], r["row_group"]))

    per_class = {}
    for name in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == name]
        per_class[name] = {
            "n": len(rs),
            "formats": dict(Counter(r["format"] for r in rs)),
            "median_width": statistics.median(r["width"] for r in rs),
            "median_height": statistics.median(r["height"] for r in rs),
            "median_bytes": statistics.median(r["bytes"] for r in rs),
        }
    return {
        "dataset": settings.train.hf_dataset,
        "revision": revision,
        "hf_label_map": {str(k): v for k, v in label_map.items()},
        "source_label_counts": {label_map[k]: v for k, v in sorted(source_counts.items())},
        "sampling": "whole row groups, split by row group (no group shared across splits)",
        "seed": settings.train.seed,
        "split_counts": {s: dict(c) for s, c in split_counts.items()},
        "row_groups_per_split": {s: {k: len(v) for k, v in d.items()} for s, d in groups.items()},
        "per_class": per_class,
        "skipped": skipped,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


if __name__ == "__main__":
    main()
