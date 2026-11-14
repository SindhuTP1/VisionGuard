import pytest

from app.config import settings
from app.detectors import evaluate_face, evaluate_hand
from tests.synthetic import broken_hand, melt, normal_hand, symmetric_face


@pytest.mark.parametrize("roll", [0, 20, -35])
def test_symmetric_face_scores_zero_at_any_roll(roll):
    f = evaluate_face(symmetric_face(roll), settings.face)
    assert f.roll_deg == pytest.approx(roll, abs=0.01)
    assert f.asymmetry_score == pytest.approx(0, abs=1e-6)
    assert not f.anomalous


def test_melted_face_is_flagged():
    f = evaluate_face(melt(symmetric_face(20)), settings.face)
    assert f.anomalous and f.asymmetry_score > settings.face.asymmetry_threshold
    assert len(f.worst_pairs) == 3


def test_turned_head_is_skipped():
    face = symmetric_face()
    face[settings.face.right_eye_outer, 2] += 100   # strong yaw
    f = evaluate_face(face, settings.face)
    assert f.skipped_reason and "turned" in f.skipped_reason and not f.anomalous


def test_tiny_face_is_skipped():
    f = evaluate_face(symmetric_face() * 0.05, settings.face)
    assert f.skipped_reason and "inter-ocular" in f.skipped_reason


def test_normal_hand_passes():
    h = evaluate_hand(normal_hand(), settings.hand)
    assert h.failed_fingers == [] and not h.anomalous


def test_broken_hand_is_flagged():
    h = evaluate_hand(broken_hand(), settings.hand)
    assert set(h.failed_fingers) == {"index", "middle", "pinky"}
    assert h.anomalous
    assert not h.fingers["index"]["hyperextension_ok"]
    assert not h.fingers["pinky"]["length_ok"]


def test_tiny_hand_is_skipped():
    h = evaluate_hand(normal_hand() * 0.1, settings.hand)
    assert h.skipped_reason and not h.anomalous
