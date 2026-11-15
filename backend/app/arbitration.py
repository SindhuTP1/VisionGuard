"""Arbitration: turn layer outputs into findings, a class and a verdict.

Rules (from the spec, extended for Anatomy v2):
1. 0 hands → the finger checks are skipped and the decision falls to the classifier.
2. Face asymmetry or a distorted limb → melting_geometry. Extra finger, extra
   hand, or an implausible finger skeleton → anatomical_extra.
3. Final confidence < threshold (0.65) → "Needs Review".

Every check that fires becomes a ``Finding``: what was found, where (points to
mark on the image), and a geometric confidence. The strongest finding decides
the class.

How confidence is computed (heuristic, documented):
* Score-based checks (face asymmetry, finger skeleton): clip(score / threshold / 2, 0, 1).
  At the threshold this is 0.5, so a borderline hit can't clear 0.65 alone.
* Count-based checks (extra finger / extra hand / limb) use fixed values from
  ArbitrationConfig.
* If the classifier is available, the final confidence is the mean of the
  strongest geometric confidence and p_fake. Disagreement with a "real"
  classifier pulls the case down to review.
* With no geometric finding, the classifier decides. A confident "fake" verdict
  with no geometric explanation has anomaly_class=None ("classifier_only").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .body import BodyResult
from .classifier import ClassifierResult
from .config import AnomalyClass, Settings, Verdict
from .detectors import FaceResult, HandResult, LayerStatus, PersonResult, _r


@dataclass
class Finding:
    kind: str                     # extra_finger | extra_hand | implausible_fingers | face_asymmetry | distorted_limb
    anomaly_class: AnomalyClass
    confidence: float             # geometric confidence (heuristic, see module docstring)
    message: str
    layer: str                    # hands | body | face
    hand_index: int | None = None
    person_index: int | None = None
    face_index: int | None = None
    points: list[list[float]] = field(default_factory=list)   # full-image [x, y] to highlight


@dataclass
class ArbitrationResult:
    verdict: Verdict
    anomaly_class: AnomalyClass | None
    attribution: str        # face_geometry | hand_geometry | body_geometry | classifier_only | classifier | none
    confidence: float | None
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float | None] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def geometric_confidence(score: float | None, threshold: float) -> float | None:
    if score is None or threshold <= 0:
        return None
    return min(1.0, max(0.0, score / threshold / 2.0))


ATTRIBUTION = {"hands": "hand_geometry", "face": "face_geometry", "body": "body_geometry"}


def collect_findings(hands: HandResult, face: FaceResult, body: BodyResult | None, cfg: Settings) -> list[Finding]:
    a = cfg.arbitration
    out: list[Finding] = []

    for hf in hands.hands:
        n = hf.index + 1
        fc = hf.finger_count
        if fc is not None and fc.extra_finger:
            extra_pts = [p for p, name in zip(fc.tips, fc.tip_fingers) if name is None] or fc.tips
            out.append(Finding(
                "extra_finger", AnomalyClass.ANATOMICAL_EXTRA,
                a.extra_finger_conf if fc.count == 6 else a.extra_finger_conf_many,
                f"Hand {n}: {fc.count} fingertips counted in its outline (a hand has at most 5)",
                "hands", hand_index=hf.index, points=extra_pts,
            ))
        if hf.skipped_reason is None and hf.anomalous:
            conf = geometric_confidence(hf.fail_fraction, cfg.hand.anomaly_fail_fraction) or 0.0
            tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
            pts = [hf.landmarks[tips[f]] for f in hf.failed_fingers if hf.landmarks]
            out.append(Finding(
                "implausible_fingers", AnomalyClass.ANATOMICAL_EXTRA, conf,
                f"Hand {n}: implausible {', '.join(hf.failed_fingers)} (impossible length or backward bend)",
                "hands", hand_index=hf.index, points=pts,
            ))

    if body is not None and body.status == LayerStatus.OK:
        for bp in body.persons:
            if bp.extra_hand:
                same_wrist = "same" in (bp.extra_hand_reason or "") or "wrist" in (bp.extra_hand_reason or "")
                pts = [hands.hands[k].landmarks[0] for k in bp.hands if hands.hands[k].landmarks]
                out.append(Finding(
                    "extra_hand", AnomalyClass.ANATOMICAL_EXTRA,
                    a.extra_hand_same_wrist_conf if same_wrist else a.extra_hand_conf,
                    f"Person {bp.index + 1}: {bp.extra_hand_reason}",
                    "body", person_index=bp.index, points=pts,
                ))
            for arm in bp.arms if a.limb_findings_enabled else []:
                if not arm.ok:
                    out.append(Finding(
                        "distorted_limb", AnomalyClass.MELTING_GEOMETRY, a.distorted_limb_conf,
                        f"Person {bp.index + 1}: {arm.side} arm proportions implausible "
                        f"(upper arm / forearm = {arm.ratio:.2f})",
                        "body", person_index=bp.index,
                    ))

    if face.status == LayerStatus.OK:
        for f in face.faces:
            if f.skipped_reason is None and f.anomalous:
                out.append(Finding(
                    "face_asymmetry", AnomalyClass.MELTING_GEOMETRY,
                    geometric_confidence(f.asymmetry_score, cfg.face.asymmetry_threshold) or 0.0,
                    f"Face {f.index + 1}: asymmetry {f.asymmetry_score} > threshold {cfg.face.asymmetry_threshold}",
                    "face", face_index=f.index,
                ))

    for f in out:
        f.confidence = round(f.confidence, 4)
    return sorted(out, key=lambda f: f.confidence, reverse=True)


def arbitrate(
    person: PersonResult,
    hands: HandResult,
    face: FaceResult,
    body: BodyResult | None,
    clf: ClassifierResult,
    cfg: Settings,
) -> ArbitrationResult:
    thr = cfg.arbitration.confidence_threshold
    reasons: list[str] = []
    clf_ok = clf.status == LayerStatus.OK

    if person.used_fallback:
        reasons.append(person.note or "Full image used (no person crop)")
    if hands.status != LayerStatus.OK and not hands.extra_finger:
        reasons.append(f"Finger check not applied ({hands.status.value}): {hands.note}")
    if face.status != LayerStatus.OK:
        reasons.append(f"Face check not applied ({face.status.value}): {face.note}")
    if not clf_ok:
        reasons.append(f"Classifier unavailable: {clf.note}")

    findings = collect_findings(hands, face, body, cfg)
    for f in findings:
        reasons.append(f.message)

    hand_conf = max((f.confidence for f in findings if f.layer == "hands"), default=None)
    face_conf = max((f.confidence for f in findings if f.layer == "face"), default=None)
    body_conf = max((f.confidence for f in findings if f.layer == "body"), default=None)
    signals = {
        "hand_geometric_confidence": _r(hand_conf),
        "face_geometric_confidence": _r(face_conf),
        "body_geometric_confidence": _r(body_conf),
        "p_fake": clf.p_fake,
    }

    # ---- a geometric finding decides the class ----------------------------- #
    if findings:
        top = findings[0]
        if clf_ok:
            conf = (top.confidence + clf.p_fake) / 2.0
            if clf.label == "real":
                reasons.append(f"Classifier disagrees (p_real={clf.p_real})")
        else:
            conf = top.confidence
        verdict = Verdict.ANOMALY_DETECTED if conf >= thr else Verdict.NEEDS_REVIEW
        return ArbitrationResult(verdict, top.anomaly_class, ATTRIBUTION[top.layer], _r(conf),
                                 reasons, signals, findings)

    # ---- no geometric finding: the classifier decides ---------------------- #
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
