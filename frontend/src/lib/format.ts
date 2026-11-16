export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const ms = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v)} ms`);

export const CLASS_LABEL: Record<string, string> = {
  normal: "Normal",
  anatomical_extra: "Anatomical extra",
  floating_anomaly: "Floating anomaly",
  melting_geometry: "Melting geometry",
  text_distortion: "Text distortion",
};

export const ATTRIBUTION_LABEL: Record<string, string> = {
  face_geometry: "Face geometry",
  body_geometry: "Body geometry (hands per person / limbs)",
  artifact_detector: "Learned artefact detector",
  hand_geometry: "Hand geometry",
  classifier_only: "Classifier only (no geometric cue)",
  classifier: "Classifier",
  none: "No decision layer available",
};

export const FINDING_LABEL: Record<string, string> = {
  extra_finger: "Extra finger",
  extra_hand: "Extra hand",
  implausible_fingers: "Impossible finger shape",
  face_asymmetry: "Distorted face",
  distorted_limb: "Distorted limb",
  artifact: "Anatomy error (learned detector)",
};

export const ARTIFACT_LABEL: Record<string, string> = {
  malformed_hand: "malformed hand",
  malformed_face: "distorted face",
  malformed_body: "distorted arm/leg/body",
  extra_limb: "extra limb",
  missing_limb: "missing limb",
};
