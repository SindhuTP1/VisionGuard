import pytest

from app.arbitration import arbitrate, geometric_confidence
from app.classifier import ClassifierResult
from app.config import AnomalyClass, Verdict, settings
from app.detectors import FaceResult, HandResult, LayerStatus, PersonResult, evaluate_face, evaluate_hand
from tests.synthetic import broken_hand, melt, symmetric_face

P = PersonResult(LayerStatus.OK, [0, 0, 10, 10], False, 1, 0.9)
NO_HANDS = HandResult(LayerStatus.SKIPPED, 0, note="0 hands detected")
_f_ok = evaluate_face(symmetric_face(), settings.face)
_f_bad = evaluate_face(melt(symmetric_face()), settings.face)
_h_bad = evaluate_hand(broken_hand(), settings.hand)
OK_FACE = FaceResult(LayerStatus.OK, 1, [_f_ok], False, _f_ok.asymmetry_score)
BAD_FACE = FaceResult(LayerStatus.OK, 1, [_f_bad], True, _f_bad.asymmetry_score)
BAD_HANDS = HandResult(LayerStatus.OK, 1, [_h_bad], True, _h_bad.fail_fraction)
NO_CLF = ClassifierResult(LayerStatus.UNAVAILABLE, note="missing")


def clf(p_fake):
    return ClassifierResult(LayerStatus.OK, "fake" if p_fake >= 0.5 else "real",
                            p_fake, 1 - p_fake, max(p_fake, 1 - p_fake), "test")


def run(hands, face, c):
    return arbitrate(P, hands, face, None, c, settings)


def test_geometric_confidence_maps_threshold_to_half():
    assert geometric_confidence(0.12, 0.12) == pytest.approx(0.5)
    assert geometric_confidence(1.0, 0.12) == 1.0
    assert geometric_confidence(None, 0.12) is None


def test_zero_hands_falls_back_to_classifier():
    r = run(NO_HANDS, OK_FACE, clf(0.08))
    assert (r.verdict, r.anomaly_class, r.attribution) == (Verdict.VERIFIED_SAFE, AnomalyClass.NORMAL, "classifier")
    assert any("Finger check not applied" in s for s in r.reasons)


@pytest.mark.parametrize("p_fake,verdict", [(0.40, Verdict.NEEDS_REVIEW), (0.30, Verdict.VERIFIED_SAFE)])
def test_low_confidence_needs_review(p_fake, verdict):
    assert run(NO_HANDS, OK_FACE, clf(p_fake)).verdict is verdict


def test_unexplained_fake_is_classifier_only():
    r = run(NO_HANDS, OK_FACE, clf(0.9))
    assert (r.verdict, r.anomaly_class, r.attribution) == (Verdict.ANOMALY_DETECTED, None, "classifier_only")


def test_face_asymmetry_maps_to_melting_geometry():
    r = run(NO_HANDS, BAD_FACE, clf(0.9))
    assert (r.verdict, r.anomaly_class) == (Verdict.ANOMALY_DETECTED, AnomalyClass.MELTING_GEOMETRY)


def test_classifier_disagreement_sends_to_review():
    r = run(NO_HANDS, BAD_FACE, clf(0.08))
    assert (r.verdict, r.anomaly_class) == (Verdict.NEEDS_REVIEW, AnomalyClass.MELTING_GEOMETRY)


def test_skeleton_check_is_not_used_to_decide_by_default():
    r = run(BAD_HANDS, OK_FACE, clf(0.9))
    assert r.attribution == "classifier_only" and not r.findings


def test_hand_anomaly_maps_to_anatomical_extra_when_enabled():
    import dataclasses
    cfg = dataclasses.replace(settings, arbitration=dataclasses.replace(settings.arbitration, skeleton_findings_enabled=True))
    r = arbitrate(P, BAD_HANDS, OK_FACE, None, clf(0.9), cfg)
    assert (r.verdict, r.anomaly_class, r.attribution) == (
        Verdict.ANOMALY_DETECTED, AnomalyClass.ANATOMICAL_EXTRA, "hand_geometry")


def test_no_classifier_never_verifies():
    r = run(NO_HANDS, OK_FACE, NO_CLF)
    assert r.verdict is Verdict.NEEDS_REVIEW and r.confidence is None
