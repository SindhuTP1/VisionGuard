"""Arbitration: combine layer outputs into one class + verdict.

Rules (from the spec):
1. 0 hands → the finger check is skipped and the decision falls to the classifier.
2. Face asymmetry → melting_geometry; hand anomaly → anatomical_extra.
3. Final confidence < threshold (0.65) → "Needs Review".

How confidence is computed (heuristic, documented):
* Geometric confidence = clip(score / threshold / 2, 0, 1). A score exactly at the
  flag threshold gives 0.5, so a borderline geometric hit can never clear the
  0.65 bar alone. Twice the threshold or more gives 1.0.
* If the classifier is available, the anomaly confidence is the mean of the
  geometric confidence and p_fake. When the geometry says "anomaly" but the
  classifier says "real", the mean falls and the case goes to review.
* With no geometric anomaly, the classifier decides. A confident "fake" verdict
  that no geometric check explains is reported with anomaly_class=None and
  attribution="classifier_only". The binary classifier cannot tell which kind of
  artefact it found, and floating_anomaly / text_distortion have no detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import ClassifierResult
from .config import AnomalyClass, ArbitrationConfig, FaceConfig, HandConfig, Verdict
from .detectors import FaceResult, HandResult, LayerStatus, PersonResult, _r


@dataclass
class ArbitrationResult:
    verdict: Verdict
    anomaly_class: AnomalyClass | None
    attribution: str        # face_geometry | hand_geometry | classifier_only | classifier | none
    confidence: float | None
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float | None] = field(default_factory=dict)


def geometric_confidence(score: float | None, threshold: float) -> float | None:
    if score is None or threshold <= 0:
        return None
    return min(1.0, max(0.0, score / threshold / 2.0))


def arbitrate(
    person: PersonResult,
    hands: HandResult,
    face: FaceResult,
    clf: ClassifierResult,
    arb_cfg: ArbitrationConfig,
    hand_cfg: HandConfig,
    face_cfg: FaceConfig,
) -> ArbitrationResult:
    thr = arb_cfg.confidence_threshold
    reasons: list[str] = []
    clf_ok = clf.status == LayerStatus.OK

    if person.used_fallback:
        reasons.append(person.note or "Full image used (no person crop)")

    # --- Rule 1: hands --------------------------------------------------- #
    hand_conf = None
    if hands.status == LayerStatus.OK:
        hand_conf = geometric_confidence(hands.max_fail_fraction, hand_cfg.anomaly_fail_fraction)
        if hands.anomalous:
            worst = max((h for h in hands.hands if h.skipped_reason is None), key=lambda h: h.fail_fraction)
            reasons.append(f"Hand {worst.index}: implausible fingers {worst.failed_fingers}")
    else:
        reasons.append(f"Finger check not applied ({hands.status.value}): {hands.note}")

    # --- Face ------------------------------------------------------------ #
    face_conf = None
    if face.status == LayerStatus.OK:
        face_conf = geometric_confidence(face.max_score, face_cfg.asymmetry_threshold)
        if face.anomalous:
            reasons.append(f"Face asymmetry {face.max_score} > threshold {face_cfg.asymmetry_threshold}")
    else:
        reasons.append(f"Face check not applied ({face.status.value}): {face.note}")

    if not clf_ok:
        reasons.append(f"Classifier unavailable: {clf.note}")

    signals = {
        "hand_geometric_confidence": _r(hand_conf),
        "face_geometric_confidence": _r(face_conf),
        "p_fake": clf.p_fake,
    }

    # --- Rule 2: geometric anomalies ------------------------------------- #
    candidates: list[tuple[float, AnomalyClass, str]] = []
    if hands.anomalous and hand_conf is not None:
        candidates.append((hand_conf, AnomalyClass.ANATOMICAL_EXTRA, "hand_geometry"))
    if face.anomalous and face_conf is not None:
        candidates.append((face_conf, AnomalyClass.MELTING_GEOMETRY, "face_geometry"))

    if candidates:
        geo_conf, cls, source = max(candidates, key=lambda c: c[0])
        if clf_ok:
            conf = (geo_conf + clf.p_fake) / 2.0
            if clf.label == "real":
                reasons.append(f"Classifier disagrees (p_real={clf.p_real})")
        else:
            conf = geo_conf
        # Rule 3
        verdict = Verdict.ANOMALY_DETECTED if conf >= thr else Verdict.NEEDS_REVIEW
        return ArbitrationResult(verdict, cls, source, _r(conf), reasons, signals)

    # --- No geometric anomaly: classifier decides ------------------------ #
    if not clf_ok:
        reasons.append("No geometric anomaly found, but there is no classifier to confirm the image is authentic")
        return ArbitrationResult(Verdict.NEEDS_REVIEW, None, "none", None, reasons, signals)

    if clf.label == "fake":
        reasons.append(f"Classifier flags image as AI-generated (p_fake={clf.p_fake}); no geometric cue localises it")
        verdict = Verdict.ANOMALY_DETECTED if clf.p_fake >= thr else Verdict.NEEDS_REVIEW
        return ArbitrationResult(verdict, None, "classifier_only", clf.p_fake, reasons, signals)

    verdict = Verdict.VERIFIED_SAFE if clf.p_real >= thr else Verdict.NEEDS_REVIEW
    if verdict is Verdict.NEEDS_REVIEW:
        reasons.append(f"Classifier leans real but p_real={clf.p_real} < {thr}")
    return ArbitrationResult(verdict, AnomalyClass.NORMAL, "classifier", clf.p_real, reasons, signals)
