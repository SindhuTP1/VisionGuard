// Typed client for the VisionGuard FastAPI backend.
// Types mirror backend/app/schemas.py. If you change one, change the other.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type LayerStatus = "ok" | "skipped" | "unavailable" | "error";
export type Verdict = "Verified Safe" | "Needs Review" | "Anomaly Detected";
export type AnomalyClass =
  | "normal"
  | "anatomical_extra"
  | "floating_anomaly"
  | "melting_geometry"
  | "text_distortion";
export type Attribution = "face_geometry" | "hand_geometry" | "classifier_only" | "classifier" | "none";

export interface PersonOut {
  status: LayerStatus;
  bbox: number[];
  used_fallback: boolean;
  num_persons: number;
  person_score: number | null;
  note: string | null;
  person_boxes: number[][];
}

export interface FingerOut {
  length_ratio: number | null;
  length_ok: boolean;
  phalanx_ratio: number | null;
  phalanx_ok: boolean;
  max_hyperextension_deg: number | null;
  hyperextension_ok: boolean;
  extended: boolean;
  ok: boolean;
}

export interface HandOut {
  index: number;
  handedness: string | null;
  handedness_score: number | null;
  palm_width_px: number | null;
  extended_fingers: number | null;
  fingers: Record<string, FingerOut>;
  failed_fingers: string[];
  fail_fraction: number;
  anomalous: boolean;
  skipped_reason: string | null;
}

export interface HandsOut {
  status: LayerStatus;
  num_hands: number;
  hands: HandOut[];
  anomalous: boolean;
  max_fail_fraction: number | null;
  note: string | null;
}

export interface FaceOut {
  index: number;
  iod_px: number | null;
  roll_deg: number | null;
  yaw_ratio: number | null;
  asymmetry_score: number | null;
  lateral_mean: number | null;
  vertical_mean: number | null;
  midline_deviation: number | null;
  worst_pairs: { pair: number[]; score: number | null }[];
  anomalous: boolean;
  skipped_reason: string | null;
}

export interface FacesOut {
  status: LayerStatus;
  num_faces: number;
  faces: FaceOut[];
  anomalous: boolean;
  max_score: number | null;
  note: string | null;
}

export interface ClassifierOut {
  status: LayerStatus;
  label: "real" | "fake" | null;
  p_fake: number | null;
  p_real: number | null;
  confidence: number | null;
  run_id: string | null;
  input_region: string;
  note: string | null;
}

export interface ArbitrationOut {
  verdict: Verdict;
  anomaly_class: AnomalyClass | null;
  attribution: Attribution;
  confidence: number | null;
  reasons: string[];
  signals: Record<string, number | null>;
}

export interface AnalysisResponse {
  request_id: string;
  filename: string | null;
  image: { width: number; height: number };
  person: PersonOut;
  hands: HandsOut;
  face: FacesOut;
  classifier: ClassifierOut;
  arbitration: ArbitrationOut;
  timings_ms: Record<string, number>;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  device: string;
  components: Record<string, { loaded: boolean; error: string | null; load_seconds: number | null }>;
  confidence_threshold: number;
  thresholds: { confidence: number; face_asymmetry: number; hand_fail_fraction: number };
}

export interface RunSummary {
  run_id: string;
  created_at: string | null;
  has_metrics: boolean;
}

// metrics.json is written by training/train.py. Older runs may lack newer
// sections (e.g. scale_stress_test), so everything past the basics is optional.
export interface SplitMetrics {
  n: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number | null;
  f1: number;
  roc_auc: number | null;
  ece?: number;
  confusion_matrix: { labels: string[]; rows_true_cols_pred: number[][] };
  selective?: {
    threshold: number;
    coverage: number;
    review_rate: number;
    accuracy_on_covered: number | null;
    accuracy_on_review: number | null;
  };
  ci95?: { n_boot: number; accuracy: [number, number] | null; f1: [number, number] | null; roc_auc: [number, number] | null };
  rescale_short_side?: number;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
  val_roc_auc: number | null;
  seconds: number;
}

export interface RunMetrics {
  run_id: string;
  created_at: string;
  model?: { arch: string; params: number; img_size: number };
  preprocessing?: { tag: string; train: string; eval: string; jpeg_normalize_quality: number };
  train_config?: Record<string, unknown>;
  dataset?: {
    dataset: string;
    revision: string;
    source_label_counts: Record<string, number>;
    split_counts: Record<string, Record<string, number>>;
    per_class: Record<string, { n: number; formats: Record<string, number>; median_width: number; median_height: number; median_bytes: number }>;
  } | null;
  training?: { epochs_run: number; best_epoch: number; best_val_loss: number; stopped_early: boolean; seconds: number; device: string };
  history?: EpochRecord[];
  val?: SplitMetrics;
  test?: SplitMetrics;
  shortcut_probe?: { features: string[]; accuracy: number; roc_auc: number };
  scale_stress_test?: SplitMetrics;
}

export interface MetricsResponse {
  available: boolean;
  run_id: string | null;
  source: string | null;
  metrics: RunMetrics | null;
  message: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(
      0,
      `Can't reach the VisionGuard API at ${API_URL}. Check that the backend is running, and that it allows ` +
        `requests from this page's address (${typeof window !== "undefined" ? window.location.origin : "this site"}).`,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  runs: () => request<RunSummary[]>("/metrics/runs"),
  metrics: (runId?: string) =>
    request<MetricsResponse>(runId ? `/metrics?run_id=${encodeURIComponent(runId)}` : "/metrics"),
  analyze: (file: File, signal?: AbortSignal) => {
    const body = new FormData();
    body.append("file", file);
    return request<AnalysisResponse>("/analyze", { method: "POST", body, signal });
  },
};

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MAX_UPLOAD_MB = 15;
