"""Body-level anatomy: hands per person and arm proportions.

YOLOv8-pose gives each person 17 COCO keypoints. Like MediaPipe, a pose model
fits a fixed skeleton (2 arms), so it can't show a third arm itself. What it
does give is where each person's wrists are. Hands found by MediaPipe are then
attached to the nearest pose wrist:

* 3+ hands attached to one person → extra hand/arm.
* 2 clearly separate hands at the same wrist → extra hand.
* Upper-arm vs forearm length far outside human range → distorted limb. This
  is a weak signal, because 2D foreshortening can shrink either segment.

Hands that attach to no wrist are listed but not flagged: pose models often
miss wrists that are occluded or out of frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BodyConfig
from .detectors import HandResult, LayerStatus, PersonResult

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
ARMS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}


@dataclass
class ArmCheck:
    side: str
    upper_px: float
    fore_px: float
    ratio: float
    ok: bool


@dataclass
class BodyPerson:
    index: int
    box: list[int]
    hands: list[int] = field(default_factory=list)                    # indices into HandResult.hands
    hands_by_side: dict[str, list[int]] = field(default_factory=dict)
    extra_hand: bool = False
    extra_hand_reason: str | None = None
    arms: list[ArmCheck] = field(default_factory=list)
    distorted_limb: bool = False


@dataclass
class BodyResult:
    status: LayerStatus
    persons: list[BodyPerson] = field(default_factory=list)
    unattached_hands: list[int] = field(default_factory=list)
    extra_hand: bool = False
    distorted_limb: bool = False
    note: str | None = None


def _kp(kps: list[list[float]], i: int, min_conf: float) -> np.ndarray | None:
    if i >= len(kps) or kps[i][2] < min_conf:
        return None
    return np.array(kps[i][:2], dtype=float)


def analyze_body(person: PersonResult, hands: HandResult, cfg: BodyConfig) -> BodyResult:
    if person.status not in (LayerStatus.OK,) or not person.keypoints or not any(person.keypoints):
        return BodyResult(LayerStatus.SKIPPED, note="No body pose available")

    persons: list[BodyPerson] = []
    wrists: list[tuple[int, str, np.ndarray, float]] = []       # (person, side, xy, attach radius)
    for j, (box, kps) in enumerate(zip(person.person_boxes, person.keypoints)):
        if not kps:
            continue
        bp = BodyPerson(j, box, hands_by_side={"left": [], "right": []})
        box_h = max(box[3] - box[1], 1)
        ls, rs = _kp(kps, L_SHOULDER, cfg.kp_min_conf), _kp(kps, R_SHOULDER, cfg.kp_min_conf)
        shoulder = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else 0.25 * (box[2] - box[0])
        radius = max(cfg.hand_assoc_shoulders * shoulder, 0.08 * box_h)
        for side, (s_i, e_i, w_i) in ARMS.items():
            wr = _kp(kps, w_i, cfg.kp_min_conf)
            if wr is not None:
                wrists.append((j, side, wr, radius))
            s, e = _kp(kps, s_i, cfg.kp_min_conf), _kp(kps, e_i, cfg.kp_min_conf)
            if s is not None and e is not None and wr is not None:
                upper, fore = float(np.linalg.norm(s - e)), float(np.linalg.norm(e - wr))
                if min(upper, fore) >= cfg.min_segment_frac * box_h:
                    ratio = upper / fore
                    lo, hi = cfg.arm_ratio_bounds
                    bp.arms.append(ArmCheck(side, round(upper, 1), round(fore, 1), round(ratio, 3), lo <= ratio <= hi))
        bp.distorted_limb = any(not a.ok for a in bp.arms)
        persons.append(bp)

    # ---- attach hands to the nearest pose wrist ---------------------------- #
    unattached: list[int] = []
    by_index = {p.index: p for p in persons}
    centres: dict[int, tuple[np.ndarray, float]] = {}
    for k, hf in enumerate(hands.hands):
        if len(hf.landmarks) < 21:
            continue
        lm = np.array(hf.landmarks, dtype=float)
        palm = float(np.linalg.norm(lm[5] - lm[17])) or 1.0
        centres[k] = (lm[[0, 5, 9, 13, 17]].mean(0), palm)
        best = None
        for j, side, wr, radius in wrists:
            dist = float(np.linalg.norm(lm[0] - wr))
            if dist <= radius and (best is None or dist < best[0]):
                best = (dist, j, side)
        if best is None:
            unattached.append(k)
        else:
            by_index[best[1]].hands.append(k)
            by_index[best[1]].hands_by_side[best[2]].append(k)

    for bp in persons:
        if len(bp.hands) >= cfg.extra_hand_min:
            bp.extra_hand = True
            bp.extra_hand_reason = f"{len(bp.hands)} hands attached to one person"
            continue
        for side, ks in bp.hands_by_side.items():
            if len(ks) >= 2:
                (c1, p1), (c2, p2) = centres[ks[0]], centres[ks[1]]
                if np.linalg.norm(c1 - c2) > 0.8 * max(p1, p2):        # distinct hands, not a duplicate detection
                    bp.extra_hand = True
                    bp.extra_hand_reason = f"two separate hands at the {side} wrist"

    return BodyResult(
        LayerStatus.OK if persons else LayerStatus.SKIPPED,
        persons,
        unattached,
        extra_hand=any(p.extra_hand for p in persons),
        distorted_limb=any(p.distorted_limb for p in persons),
        note=None if persons else "No pose keypoints",
    )
