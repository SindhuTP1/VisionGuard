"""Finger counting and body checks on synthetic, fully-known inputs.

The synthetic hand is drawn: a skin-coloured palm plus N finger bars on a plain
background, with MediaPipe-style landmarks placed on the first five fingers,
just as MediaPipe fits only 5 even when 6 are visible.
"""

import numpy as np
import cv2
import pytest

from app.anatomy import count_fingers
from app.arbitration import arbitrate
from app.body import analyze_body
from app.classifier import ClassifierResult
from app.config import AnomalyClass, Verdict, settings
from app.detectors import FaceResult, HandFinding, HandResult, LayerStatus, PersonResult

SKIN = (205, 160, 130)
BG = (40, 70, 110)
SEAM = (120, 80, 60)
ALL_EXTENDED = ["thumb", "index", "middle", "ring", "pinky"]


def draw_hand(n_fingers: int, spread: bool, palm_w: int = 120):
    """Return (rgb image, 21x3 landmarks). Fingers point up; the thumb sticks out to the left."""
    img = np.full((520, 520, 3), BG, np.uint8)
    cx, mcp_y = 260, 300
    gap = 14 if spread else 0
    fw = 30                                   # finger width (px)
    row_w = n_fingers_row = n_fingers - 1     # thumb is drawn separately
    total = row_w * fw + (row_w - 1) * gap
    x_start = cx - total // 2
    # palm
    cv2.rectangle(img, (x_start - 4, mcp_y - 10), (x_start + total + 4, mcp_y + 120), SKIN, -1)
    centres = []
    for i in range(n_fingers_row):
        x0 = x_start + i * (fw + gap)
        cv2.rectangle(img, (x0, mcp_y - 150), (x0 + fw - 1, mcp_y), SKIN, -1)
        cv2.ellipse(img, (x0 + fw // 2, mcp_y - 150), (fw // 2, 10), 0, 180, 360, SKIN, -1)
        centres.append(x0 + fw // 2)
        if not spread and i > 0:              # touching fingers: a darker seam between them
            cv2.line(img, (x0, mcp_y - 150), (x0, mcp_y - 5), SEAM, 2)
    # thumb, angled out to the left
    tx, ty = x_start - 4, mcp_y + 60
    cv2.line(img, (tx + 10, ty), (tx - 80, ty - 90), SKIN, 30)

    pts = np.zeros((21, 3))
    pts[0] = (cx, mcp_y + 130, 0)                                  # wrist
    thumb = [(tx + 5, ty), (tx - 25, ty - 30), (tx - 50, ty - 60), (tx - 78, ty - 92)]
    for j, p in enumerate(thumb):
        pts[1 + j] = (*p, 0)
    # MediaPipe fits only 4 long fingers: the first four of the row
    for f in range(4):
        x = centres[f]
        for j, y in enumerate((mcp_y, mcp_y - 50, mcp_y - 100, mcp_y - 150)):
            pts[5 + 4 * f + j] = (x, y, 0)
    return img, pts


@pytest.mark.parametrize("spread", [True, False])
def test_normal_hand_counts_five(spread):
    img, pts = draw_hand(5, spread)
    fc = count_fingers(img, pts, ALL_EXTENDED, settings.count)
    assert fc.status == "counted"
    assert fc.count == 5, (fc.outline_count, fc.scan_count, fc.scan_row_counts)
    assert not fc.extra_finger


@pytest.mark.parametrize("spread", [True, False])
def test_six_fingered_hand_is_flagged(spread):
    img, pts = draw_hand(6, spread)
    fc = count_fingers(img, pts, ALL_EXTENDED, settings.count)
    assert fc.status == "counted"
    assert fc.count == 6, (fc.method, fc.outline_count, fc.scan_count, fc.scan_row_counts)
    assert fc.extra_finger


def test_closed_hand_is_not_counted():
    img, pts = draw_hand(5, True)
    fc = count_fingers(img, pts, ["index"], settings.count)
    assert fc.status == "cant_count" and "closed" in fc.reason


def test_tiny_hand_is_not_counted():
    img, pts = draw_hand(5, True)
    small = cv2.resize(img, None, fx=0.15, fy=0.15)
    fc = count_fingers(small, pts * 0.15, ALL_EXTENDED, settings.count)
    assert fc.status == "cant_count" and "small" in fc.reason


def test_coordinates_shift():
    img, pts = draw_hand(5, True)
    fc = count_fingers(img, pts, ALL_EXTENDED, settings.count)
    moved = fc.shifted(100, 50)
    assert moved.tips[0] == [round(fc.tips[0][0] + 100, 1), round(fc.tips[0][1] + 50, 1)]


# --------------------------------------------------------------------------- #
# Body: hands per person
# --------------------------------------------------------------------------- #
def _person_with_wrists(lw, rw):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[5], kps[6] = [200, 200, 0.9], [300, 200, 0.9]        # shoulders
    kps[7], kps[8] = [180, 280, 0.9], [320, 280, 0.9]        # elbows
    kps[9], kps[10] = [*lw, 0.9], [*rw, 0.9]                 # wrists
    return PersonResult(LayerStatus.OK, [100, 100, 400, 600], False, 1, 0.9,
                        person_boxes=[[100, 100, 400, 600]], keypoints=[kps])


def _hand_at(i, wrist):
    hf = HandFinding(i, None, None, 40.0, 5)
    lm = np.zeros((21, 2))
    lm[:] = wrist
    lm[5] = (wrist[0] - 20, wrist[1] - 40)
    lm[17] = (wrist[0] + 20, wrist[1] - 40)
    lm[9] = (wrist[0], wrist[1] - 42)
    lm[13] = (wrist[0] + 10, wrist[1] - 41)
    hf.landmarks = lm.tolist()
    return hf


def test_two_hands_two_wrists_is_normal():
    person = _person_with_wrists((170, 360), (330, 360))
    hands = HandResult(LayerStatus.OK, 2, [_hand_at(0, (172, 362)), _hand_at(1, (328, 358))])
    body = analyze_body(person, hands, settings.body)
    assert body.status == LayerStatus.OK and not body.extra_hand
    assert sorted(body.persons[0].hands) == [0, 1]


def test_three_hands_on_one_person_is_extra_hand():
    person = _person_with_wrists((170, 360), (330, 360))
    hands = HandResult(LayerStatus.OK, 3, [_hand_at(0, (172, 362)), _hand_at(1, (328, 358)), _hand_at(2, (300, 380))])
    body = analyze_body(person, hands, settings.body)
    assert body.extra_hand and "3 hands" in body.persons[0].extra_hand_reason


def test_extra_hand_finding_decides_class():
    person = _person_with_wrists((170, 360), (330, 360))
    hands = HandResult(LayerStatus.OK, 3, [_hand_at(0, (172, 362)), _hand_at(1, (328, 358)), _hand_at(2, (300, 380))])
    body = analyze_body(person, hands, settings.body)
    face = FaceResult(LayerStatus.SKIPPED, 0, note="none")
    clf = ClassifierResult(LayerStatus.OK, "fake", 0.9, 0.1, 0.9, "t")
    r = arbitrate(person, hands, face, body, clf, settings)
    assert r.anomaly_class is AnomalyClass.ANATOMICAL_EXTRA and r.attribution == "body_geometry"
    assert r.verdict is Verdict.ANOMALY_DETECTED
    assert r.findings[0].kind == "extra_hand" and len(r.findings[0].points) == 3


def test_extra_finger_finding_names_the_hand():
    img, pts = draw_hand(6, True)
    hf = HandFinding(0, None, None, 120.0, 5)
    hf.finger_count = count_fingers(img, pts, ALL_EXTENDED, settings.count)
    hands = HandResult(LayerStatus.OK, 1, [hf], extra_finger=True)
    person = PersonResult(LayerStatus.OK, [0, 0, 520, 520], True, 0)
    clf = ClassifierResult(LayerStatus.UNAVAILABLE, note="none")
    r = arbitrate(person, hands, FaceResult(LayerStatus.SKIPPED, 0, note="x"), None, clf, settings)
    assert r.findings[0].kind == "extra_finger"
    assert "6 fingertips" in r.findings[0].message
    assert r.anomaly_class is AnomalyClass.ANATOMICAL_EXTRA
    assert r.verdict is Verdict.ANOMALY_DETECTED     # 0.75 geometric confidence, no classifier
