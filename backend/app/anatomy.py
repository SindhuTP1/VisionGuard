"""Finger counting from the hand silhouette.

MediaPipe fits a fixed 5-finger skeleton, so it can never report a sixth finger.
This module counts fingertips *independently* of that skeleton:

1. Crop around the hand and rescale so the palm is about ``work_palm_px`` wide.
2. GrabCut segments the hand. Seeds: the landmark skeleton is "sure hand", a
   thicker band around it is "probably hand", and the crop border is "sure
   background". Colour models then pull in any skin-coloured finger, including
   one MediaPipe didn't fit.
3. The forearm is cut off below the wrist, and only the connected blob holding
   the palm is kept.
4. Distance from the palm centre is traced along the outline. Each prominent
   peak on the finger side of the wrist is a fingertip candidate.
5. Candidates are matched to MediaPipe's 5 fingertips. 6 or more candidates
   means "possible extra finger", and unmatched ones are marked.

A count is refused ("can't count") rather than guessed when the hand is too
small, closed (fist or grip), or when the cut-out leaks into the face,
background or sleeve. Undercounting is common (fingers pressed together look
like one) and harmless for this purpose. Only overcounting causes false alarms.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import FingerCountConfig

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
TIP_IDS = (4, 8, 12, 16, 20)
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17))
PALM = (0, 1, 5, 9, 13, 17)
N_SAMPLES = 400


@dataclass
class FingerCount:
    status: str                                    # "counted" | "cant_count"
    reason: str | None = None
    count: int | None = None
    tips: list[list[float]] = field(default_factory=list)            # [x, y] in input-image pixels
    tip_fingers: list[str | None] = field(default_factory=list)      # matched MediaPipe finger, or None
    outline: list[list[float]] = field(default_factory=list)         # simplified silhouette
    extra_finger: bool = False
    area_ratio: float | None = None
    # Scanline method: fingers crossing the knuckle-parallel lines, plus the thumb
    outline_count: int | None = None        # fingertip peaks on the outline
    scan_count: int | None = None           # digits in the finger row + thumb
    scan_row_counts: list[int] = field(default_factory=list)
    scan_lines: list[list[list[float]]] = field(default_factory=list)   # [[x0,y0],[x1,y1]] per line
    method: str | None = None               # which method produced ``count``

    def shifted(self, dx: float, dy: float) -> "FingerCount":
        """Same result with all coordinates moved by (dx, dy)."""
        mv = lambda pts: [[round(x + dx, 1), round(y + dy, 1)] for x, y in pts]
        return FingerCount(self.status, self.reason, self.count, mv(self.tips), list(self.tip_fingers),
                           mv(self.outline), self.extra_finger, self.area_ratio, self.outline_count,
                           self.scan_count, list(self.scan_row_counts), [mv(l) for l in self.scan_lines],
                           self.method)


def _cant(reason: str, area_ratio: float | None = None) -> FingerCount:
    return FingerCount("cant_count", reason=reason, area_ratio=area_ratio)


def count_fingers(img: np.ndarray, pts: np.ndarray, extended: Collection[str] | None, cfg: FingerCountConfig) -> FingerCount:
    """``img`` is RGB uint8, ``pts`` the 21x3 hand landmarks in that image's pixel space, and
    ``extended`` the names of the fingers MediaPipe sees as extended (None = unknown)."""
    xy = pts[:, :2].astype(np.float64)
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    if palm < cfg.min_palm_px:
        return _cant(f"hand too small to count (palm {palm:.0f}px < {cfg.min_palm_px:.0f}px)")
    if extended is not None and len(extended) < cfg.min_extended:
        return _cant(f"hand is closed or gripping ({len(extended)} fingers extended) - fingers overlap in the outline")
    open_row = extended is None or {"index", "middle", "ring", "pinky"} <= set(extended)

    # ---- 1. crop + rescale ------------------------------------------------ #
    H, W = img.shape[:2]
    pad = cfg.crop_pad * palm
    x0, y0 = np.floor(xy.min(0) - pad).astype(int)
    x1, y1 = np.ceil(xy.max(0) + pad).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    s = cfg.work_palm_px / palm
    s = min(s, cfg.max_work_side / max(x1 - x0, y1 - y0, 1))   # big crops: keep GrabCut fast
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    loc = (xy - (x0, y0)) * s
    P = palm * s                     # palm width in the working crop
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return _cant("hand crop too small")

    # ---- 2. GrabCut seeded by the skeleton -------------------------------- #
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    b = 3
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    pt = lambda i: (int(round(loc[i][0])), int(round(loc[i][1])))
    for ch in CHAINS:
        for a, c in zip(ch, ch[1:]):
            cv2.line(mask, pt(a), pt(c), cv2.GC_PR_FGD, max(3, int(0.24 * P)))
    for ch in CHAINS:
        for a, c in zip(ch, ch[1:]):
            cv2.line(mask, pt(a), pt(c), cv2.GC_FGD, max(2, int(0.06 * P)))
    palm_poly = loc[list(PALM)]
    centre = palm_poly.mean(0)
    cv2.fillPoly(mask, [np.round(centre + 0.75 * (palm_poly - centre)).astype(np.int32)], cv2.GC_FGD)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.setRNGSeed(0)   # GrabCut's k-means init is random: fix it so the same image always gives the same count
    try:
        cv2.grabCut(np.ascontiguousarray(crop[..., ::-1]), mask, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _cant("segmentation failed")
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)

    # ---- 3. cut the forearm, keep the palm blob ---------------------------- #
    axis = loc[MIDDLE_MCP] - loc[WRIST]
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-6:
        return _cant("degenerate landmarks")
    u = axis / axis_len
    yy, xx = np.mgrid[0:h, 0:w]
    proj = (xx - loc[WRIST][0]) * u[0] + (yy - loc[WRIST][1]) * u[1]
    fg[proj < -0.05 * axis_len] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(fg)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    cid = lab[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0
    if cid == 0:
        return _cant("palm not found in segmentation")
    fg = (lab == cid).astype(np.uint8)

    # ---- sanity checks: refuse to count leaky segmentations ---------------- #
    area_ratio = float(fg.sum() / max(cv2.contourArea(palm_poly.astype(np.float32)), 1.0))
    if area_ratio > cfg.max_area_ratio:
        return _cant("hand outline merged with surroundings (face, sleeve or background)", round(area_ratio, 2))
    border = np.zeros_like(fg)
    border[:b + 1], border[-b - 1:], border[:, :b + 1], border[:, -b - 1:] = 1, 1, 1, 1
    border[proj < 0.1 * axis_len] = 0          # the wrist side is allowed to touch the border
    touching = float((fg & border).sum() / max(border.sum(), 1))
    if touching > cfg.max_border_fraction:
        return _cant("hand outline runs off the crop (merged with surroundings)", round(area_ratio, 2))

    # ---- 4. outline peaks = fingertip candidates --------------------------- #
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(cont) < 20:
        return _cant("outline too small")
    closed = np.vstack([cont, cont[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], N_SAMPLES, endpoint=False)
    Pts = np.stack([np.interp(t, cum, closed[:, k]) for k in (0, 1)], 1)
    c = loc[[WRIST, INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP]].mean(0)
    d = np.linalg.norm(Pts - c, axis=1) / P
    k = 7
    d = np.convolve(np.concatenate([d[-(k // 2):], d, d[: k // 2]]), np.ones(k) / k, "valid")

    from scipy.signal import find_peaks

    tiled = np.concatenate([d, d, d])
    peaks, _ = find_peaks(tiled, height=cfg.tip_min_dist, prominence=cfg.tip_min_prominence, distance=10)
    idx = sorted({int(p - N_SAMPLES) for p in peaks if N_SAMPLES <= p < 2 * N_SAMPLES})
    tips = []
    for i in idx:
        p = Pts[i]
        along = float(np.dot(p - loc[WRIST], u)) / axis_len
        if along < cfg.tip_min_axis_proj or d[i] > cfg.tip_max_dist:
            continue
        tips.append(p)

    # ---- 5. match to MediaPipe fingertips ----------------------------------- #
    mp_tips = loc[list(TIP_IDS)]
    names: list[str | None] = []
    used: set[int] = set()
    for p in tips:
        dist = np.linalg.norm(mp_tips - p, axis=1) / P
        j = int(np.argmin(dist))
        if dist[j] < cfg.tip_match_dist and j not in used:
            names.append(TIP_NAMES[j])
            used.add(j)
        else:
            names.append(None)

    # ---- 6. scanline count (handles touching fingers) ---------------------- #
    # Only on open hands: with curled fingers the row is ambiguous.
    scan_count, rows, lines = _scanline_count(crop, fg, loc, P, u, cfg) if open_row else (None, [], [])

    # back to input-image coordinates
    to_img = lambda q: [round(float(q[0] / s + x0), 1), round(float(q[1] / s + y0), 1)]
    approx = cv2.approxPolyDP(cont.astype(np.float32).reshape(-1, 1, 2), 1.5, True)[:, 0, :]
    if len(approx) > 160:
        approx = approx[:: int(np.ceil(len(approx) / 160))]
    outline_count = len(tips)
    count, method = outline_count, "outline"
    if scan_count is not None and scan_count > count:
        count, method = scan_count, "scanline"
    return FingerCount(
        status="counted",
        count=count,
        tips=[to_img(p) for p in tips],
        tip_fingers=names,
        outline=[to_img(q) for q in approx],
        extra_finger=count >= cfg.extra_finger_min_count,
        area_ratio=round(area_ratio, 2),
        outline_count=outline_count,
        scan_count=scan_count,
        scan_row_counts=rows,
        scan_lines=[[to_img(a), to_img(b)] for a, b in lines],
        method=method,
    )


def _chain_crossing(loc: np.ndarray, chain: tuple[int, ...], c0: np.ndarray, u: np.ndarray, k: np.ndarray) -> float | None:
    """Lateral position (along k) where a landmark polyline crosses the scan line through c0."""
    proj = (loc[list(chain)] - c0) @ u
    for i in range(len(chain) - 1):
        a, b = proj[i], proj[i + 1]
        if a == b or (a > 0) == (b > 0):
            continue
        f = a / (a - b)
        p = loc[chain[i]] + f * (loc[chain[i + 1]] - loc[chain[i]])
        return float((p - c0) @ k)
    return None


def _scanline_count(crop: np.ndarray, fg: np.ndarray, loc: np.ndarray, P: float, u: np.ndarray,
                    cfg: FingerCountConfig) -> tuple[int | None, list[int], list[tuple[np.ndarray, np.ndarray]]]:
    """Count the fingers in the index→pinky row, crossing lines parallel to the knuckles.

    Along each line, the hand mask gives runs of hand pixels. Inside each run,
    dark valleys are seams between touching fingers. A seam is kept only if a
    seam sits at the same lateral position on another line, because real seams
    run along the fingers while creases, rings and shadows don't. Pieces are
    counted only inside the finger row, from just outside the index finger to
    one finger-width past the pinky, located via MediaPipe's finger chains, so the
    thumb is never counted here. A normal open hand has 4 in the row. Returns
    (row + 1 thumb, per-line row counts, line endpoints).
    """
    from scipy.ndimage import map_coordinates
    from scipy.signal import find_peaks

    h, w = fg.shape
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    base = loc[[INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP]].mean(0)
    k = np.array([-u[1], u[0]])
    half = 1.6 * P
    svals = np.arange(-half, half, 1.0)
    min_w = cfg.scan_min_digit_width * P

    per_line = []   # (runs [(a, b)], valley s-positions, row bounds) for each line
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for t in cfg.scan_offsets:
        c0 = base + t * P * u
        xs, ys = c0[0] + svals * k[0], c0[1] + svals * k[1]
        inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        m = (map_coordinates(fg.astype(np.float32), [ys, xs], order=0, mode="constant") > 0.5) & inside
        g = map_coordinates(gray, [ys, xs], order=1, mode="nearest")
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(np.int8), [0]])))
        runs, valleys_s = [], []
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < min_w:
                continue
            run = g[a:b]
            spread = float(np.percentile(run, 95) - np.percentile(run, 5))
            prom = max(cfg.scan_valley_min_abs, cfg.scan_valley_prominence * spread)
            vs, _ = find_peaks(-run, prominence=prom, distance=max(2, int(min_w)))
            valleys_s += [float(svals[a + v]) for v in vs if min_w <= v <= (b - a) - min_w]
            runs.append((a, b))
        s_index = _chain_crossing(loc, (5, 6, 7, 8), c0, u, k)
        s_pinky = _chain_crossing(loc, (17, 18, 19, 20), c0, u, k)
        s_thumb = _chain_crossing(loc, (1, 2, 3, 4), c0, u, k)   # a raised thumb can reach the lines
        per_line.append((runs, valleys_s, s_index, s_pinky, s_thumb))
        lines.append((c0 - half * k, c0 + half * k))

    tol = 0.1 * P
    rows: list[int] = []
    for i, (runs, valleys_s, s_index, s_pinky, s_thumb) in enumerate(per_line):
        if s_index is None or s_pinky is None:
            continue
        # finger-row bounds: just outside the index finger → one finger past the pinky
        direction = 1.0 if s_pinky > s_index else -1.0
        lo, hi = sorted((s_index - direction * 0.18 * P, s_pinky + direction * 0.4 * P))
        others = [v for j, l in enumerate(per_line) if j != i for v in l[1]]
        seams = [v for v in valleys_s if any(abs(v - o) <= tol for o in others)]
        count = 0
        for a, b in runs:
            cuts = sorted([float(svals[a])] + [v for v in seams if svals[a] < v < svals[b - 1]] + [float(svals[b - 1])])
            for p, q in zip(cuts, cuts[1:]):
                mid = (p + q) / 2
                is_thumb = s_thumb is not None and abs(mid - s_thumb) <= 0.25 * P
                if q - p >= min_w and lo <= mid <= hi and not is_thumb:
                    count += 1
        rows.append(count)

    if len(rows) < 2:
        return None, rows, lines
    # Lowest line count: undercounting is harmless here, overcounting is a false alarm.
    row = int(min(rows))
    if row == 0:
        return None, rows, lines
    return row + 1, rows, lines
