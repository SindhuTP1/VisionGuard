"""Synthetic landmark generators shared by the tests."""
import math

import numpy as np

from app.config import settings
from app.vision_utils import rotate_about_z


def symmetric_face(roll_deg: float = 0.0, seed: int = 0) -> np.ndarray:
    fc = settings.face
    rng = np.random.default_rng(seed)
    pts = np.zeros((478, 3))
    for l, r in fc.symmetry_pairs:
        dx, y, z = rng.uniform(10, 80), rng.uniform(50, 250), rng.uniform(-20, 20)
        pts[l] = (200 - dx, y, z)
        pts[r] = (200 + dx, y, z)
    pts[fc.left_eye_outer] = (130, 120, 0)
    pts[fc.right_eye_outer] = (270, 120, 0)
    for i, m in enumerate(fc.midline):
        pts[m] = (200, 40 + 30 * i, rng.uniform(-30, 0))
    return rotate_about_z(pts, math.radians(roll_deg), np.array([200, 150, 0.0]))


def melt(face: np.ndarray) -> np.ndarray:
    out = face.copy()
    for l, _ in settings.face.symmetry_pairs[5:]:
        out[l] += (8, 25, 0)
    return out


def normal_hand() -> np.ndarray:
    """Palm faces +z, fingers point -y, with mild natural flexion."""
    H = np.zeros((21, 3))
    for base, x in {5: -25, 9: -8, 13: 8, 17: 25}.items():
        H[base] = (x, -60, 0)
        for k, (seg, flex) in enumerate([(25, .25), (15, .45), (12, .45)], 1):
            ang = flex * k
            H[base + k] = H[base + k - 1] + (0, -seg * math.cos(ang), seg * math.sin(ang))
    H[1:5] = [(-30, -15, 0), (-50, -30, 3), (-65, -45, 6), (-75, -60, 9)]
    return H


def broken_hand() -> np.ndarray:
    H = normal_hand()
    for base in (5, 9):                       # distal joints bent backwards
        H[base + 3] = H[base + 2] + (0, -6, -10)
    H[20] = H[19] + (0, -90, 0)               # absurdly long pinky
    return H
