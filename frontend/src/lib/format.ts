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
  hand_geometry: "Hand geometry",
  classifier_only: "Classifier only (no geometric cue)",
  classifier: "Classifier",
  none: "No decision layer available",
};
