"use client";

import { useEffect, useState, type ReactNode } from "react";
import type { AnalysisResponse, LayerStatus, MatrixRow } from "@/lib/api";
import { ARTIFACT_LABEL, CLASS_LABEL, FINDING_LABEL, ms, num, pct } from "@/lib/format";

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------
export function Panel({ title, kicker, status, children }: {
  title: string;
  kicker?: string;
  status?: LayerStatus;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-line bg-surface-1 p-4 shadow-[var(--shadow)]">
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          {kicker && <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{kicker}</p>}
          <h3 className="text-sm font-semibold text-ink-1">{title}</h3>
        </div>
        {status && <StatusDot status={status} />}
      </header>
      {children}
    </section>
  );
}

const STATUS: Record<LayerStatus, { label: string; dot: string; text: string }> = {
  ok: { label: "Online", dot: "bg-good", text: "text-good-ink" },
  skipped: { label: "Nothing to check", dot: "bg-line-strong", text: "text-ink-3" },
  unavailable: { label: "Unavailable", dot: "bg-warning", text: "text-warning-ink" },
  error: { label: "Error", dot: "bg-critical", text: "text-critical-ink" },
};

export function StatusDot({ status }: { status: LayerStatus }) {
  const s = STATUS[status];
  return (
    <span className={`flex shrink-0 items-center gap-1.5 text-[11px] font-medium ${s.text}`}>
      <span className={`h-2 w-2 rounded-full ${s.dot}`} aria-hidden="true" />
      {s.label}
    </span>
  );
}

function Row({ k, v, tone }: { k: ReactNode; v: ReactNode; tone?: "bad" | "good" }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-line py-1.5 text-sm first:border-t-0">
      <span className="text-ink-2">{k}</span>
      <span className={`tnum text-right font-medium ${tone === "bad" ? "text-critical-ink" : tone === "good" ? "text-good-ink" : "text-ink-1"}`}>{v}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Layer cards
// ---------------------------------------------------------------------------
export function Layer1Card({ r }: { r: AnalysisResponse }) {
  const counts = r.objects.objects.reduce<Record<string, number>>((m, o) => ({ ...m, [o.label]: (m[o.label] ?? 0) + 1 }), {});
  const top = r.objects.objects[0];
  return (
    <Panel kicker="Layer 1" title="YOLOv8 Position Override" status={r.person.status === "ok" ? r.objects.status : r.person.status}>
      <Row k="Objects detected" v={r.objects.objects.length} />
      <Row k="Human figures" v={r.person.num_persons} />
      <Row k="Faces (MediaPipe)" v={r.face.num_faces} />
      {top && <Row k="Top detection" v={`${top.label} · ${pct(top.score, 0)}`} />}
      {Object.keys(counts).length > 0 && (
        <p className="mt-2 text-xs text-ink-3">
          {Object.entries(counts).map(([k, n]) => `${n}× ${k}`).join(" · ")}
        </p>
      )}
      <p className="mt-2 text-xs text-ink-3">
        Detection only. An object detector can&apos;t tell whether an image is AI-generated; this layer locates the
        people and objects the other layers inspect.
      </p>
    </Panel>
  );
}

export function Layer2Card({ r }: { r: AnalysisResponse }) {
  const faces = r.face.faces;
  const judged = faces.filter((f) => !f.skipped_reason);
  const hands = r.hands.hands;
  const counted = hands.filter((h) => h.finger_count?.status === "counted");
  const structural = r.arbitration.findings.filter((f) => f.layer !== "artifacts");
  const status: LayerStatus = r.face.status === "ok" || r.hands.status === "ok" ? "ok" : r.hands.status === "skipped" && r.face.status === "skipped" ? "skipped" : r.face.status;
  return (
    <Panel kicker="Layer 2" title="MediaPipe Biometric Integrity" status={status}>
      <Row k="Face landmarks" v={faces.length ? `${faces.length} face${faces.length > 1 ? "s" : ""} · ${faces[0].landmarks.length} pts` : "no face"} />
      <Row k="Facial symmetry"
        v={judged.length ? `${num(r.face.max_score)} / ${num(r.settings_used.face_asymmetry_threshold, 3)}` : faces[0]?.skipped_reason ? "skipped (pose)" : "—"}
        tone={r.face.anomalous ? "bad" : judged.length ? "good" : undefined} />
      <Row k="Hands detected" v={r.hands.num_hands} />
      <Row k="Finger count"
        v={counted.length ? counted.map((h) => `hand ${h.index + 1}: ${h.finger_count!.count}`).join(", ") : hands.length ? "not countable" : "—"}
        tone={r.hands.extra_finger ? "bad" : counted.length ? "good" : undefined} />
      <Row k="Hands per person"
        v={r.body.persons.length ? r.body.persons.map((p) => p.hands.length).join(" / ") : "—"}
        tone={r.body.extra_hand ? "bad" : undefined} />
      <div className="mt-2">
        {structural.length === 0 ? (
          <p className="text-xs text-good-ink">✓ No structural inconsistency found by the checks that could run</p>
        ) : (
          <ul className="space-y-1">
            {structural.map((f, i) => (
              <li key={i} className="text-xs text-critical-ink">✗ {FINDING_LABEL[f.kind] ?? f.kind}: {f.message}</li>
            ))}
          </ul>
        )}
      </div>
      {hands.some((h) => h.finger_count?.status === "cant_count") && (
        <p className="mt-1 text-xs text-ink-3">
          Some hands can&apos;t be counted (fist, grip or too small); the counter refuses rather than guesses.
        </p>
      )}
    </Panel>
  );
}

export function Layer3Card({ r }: { r: AnalysisResponse }) {
  const c = r.classifier;
  const fake = c.label === "fake";
  return (
    <Panel kicker="Layer 3" title="Neural Classification Model" status={c.status}>
      {c.status === "ok" ? (
        <>
          <Row k="Predicted category" v={fake ? "AI-generated" : "Normal (real photo)"} tone={fake ? "bad" : "good"} />
          <Row k="Model confidence" v={pct(c.confidence)} />
          <Row k="P(AI-generated) / P(real)" v={`${pct(c.p_fake)} / ${pct(c.p_real)}`} />
          <Row k="Grad-CAM target" v={r.gradcam.status === "ok" ? `${r.gradcam.target} · 5 patches` : "—"} />
          <p className="mt-2 text-xs text-ink-3">
            MobileNetV3-Small trained on AI-vs-Real (run <span className="font-mono">{c.run_id}</span>). It outputs real vs
            AI-generated only, not the anomaly categories. Expect mistakes on images unlike its training data.
          </p>
        </>
      ) : (
        <p className="text-sm text-ink-2">{c.note ?? "Classifier did not run."}</p>
      )}
      {r.artifacts.status === "ok" && r.artifacts.boxes.length > 0 && (
        <p className="mt-1 text-xs text-critical-ink">
          Learned detector: {r.artifacts.boxes.map((b) => ARTIFACT_LABEL[b.cls]).join(", ")}
        </p>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Probability matrix
// ---------------------------------------------------------------------------
const KIND_LABEL: Record<MatrixRow["kind"], string> = {
  model_probability: "model probability",
  check_score: "check score",
  not_measured: "not measured",
};

export function MatrixCard({ r }: { r: AnalysisResponse }) {
  // The parent keys this card by request_id, so each new scan remounts it and the
  // bars animate from 0 to their real values.
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(t);
  }, []);
  const rows = [...r.telemetry.matrix].sort((a, b) => CLASS_LABEL[a.category].localeCompare(CLASS_LABEL[b.category]));
  return (
    <Panel title="Anomaly Category Matrix Probabilities">
      <ul className="space-y-2.5">
        {rows.map((row) => (
          <li key={row.category} title={row.source}>
            <div className="flex items-baseline justify-between text-sm">
              <span className="font-medium text-ink-1">{CLASS_LABEL[row.category]}</span>
              <span className="tnum text-ink-1">
                {row.value == null ? <span className="text-xs text-ink-3">not measured</span> : pct(row.value)}
              </span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-surface-2">
              {row.value != null && (
                <div className={`h-full rounded-full transition-[width] duration-700 ease-out ${row.category === "normal" ? "bg-good" : row.value >= 0.5 ? "bg-critical" : "bg-series-1"}`}
                  style={{ width: shown ? `${row.value * 100}%` : "0%" }} />
              )}
            </div>
            <p className="mt-0.5 text-[11px] text-ink-3">{KIND_LABEL[row.kind]} · {row.source}</p>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-ink-3">
        No model in this system was trained on these five categories, so each bar shows its real source. Categories
        without a detector say &ldquo;not measured&rdquo; instead of showing an invented number.
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------
export const VERDICT = {
  "Verified Safe": { title: "VERIFIED SECURE", tone: "border-good bg-good-soft", ink: "text-good-ink", icon: "✓", bar: "bg-good" },
  "Needs Review": { title: "HUMAN REVIEW", tone: "border-warning bg-warning-soft", ink: "text-warning-ink", icon: "!", bar: "bg-warning" },
  "Anomaly Detected": { title: "ANOMALY CONFIRMED", tone: "border-critical bg-critical-soft", ink: "text-critical-ink", icon: "✗", bar: "bg-critical" },
} as const;

export function primaryReason(r: AnalysisResponse): string {
  const a = r.arbitration;
  if (a.findings.length && a.attribution !== "classifier_only" && a.attribution !== "classifier") return a.findings[0].message;
  return a.reasons[a.reasons.length - 1] ?? "—";
}

export function VerdictCard({ r }: { r: AnalysisResponse }) {
  const a = r.arbitration;
  const v = VERDICT[a.verdict];
  const t = r.telemetry;
  return (
    <section className={`rounded-xl border-2 p-4 shadow-[var(--shadow)] ${v.tone}`} aria-live="polite">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Arbitrated Intelligence Verdict</p>
      <div className="mt-1 flex items-center gap-3">
        <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border-2 text-lg font-bold ${v.ink} border-current`} aria-hidden="true">{v.icon}</span>
        <div className="min-w-0 flex-1">
          <h3 className={`text-xl font-bold tracking-wide ${v.ink}`}>{v.title}</h3>
          <p className="text-sm text-ink-2">
            {a.anomaly_class && a.anomaly_class !== "normal" ? <>Category: <b className="text-ink-1">{CLASS_LABEL[a.anomaly_class]}</b></> :
              a.anomaly_class === "normal" ? "No significant anomaly detected" : a.verdict === "Verified Safe" ? "" : "Category unattributed (no localising check fired)"}
          </p>
        </div>
        <div className="text-right">
          <div className="tnum text-3xl font-bold text-ink-1">{pct(a.confidence)}</div>
          <div className="text-xs text-ink-3">confidence</div>
        </div>
      </div>
      <div className="relative mt-3 h-2 rounded-full bg-white/70">
        <div className={`h-full rounded-full ${v.bar}`} style={{ width: `${(a.confidence ?? 0) * 100}%` }} />
        <div className="absolute -top-1 h-4 w-0.5 bg-ink-1" style={{ left: `${r.settings_used.review_threshold * 100}%` }} title="review threshold" />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
        <Metric k="Matrix boost" v={t.confidence_boost == null ? "—" : `${t.confidence_boost >= 0 ? "+" : ""}${(t.confidence_boost * 100).toFixed(1)} pts`}
          hint="Final confidence minus the classifier's own support for this verdict" />
        <Metric k="Entropy integrity" v={t.entropy_level ? `${t.entropy_level} · ${num(t.entropy_bits, 2)} bits` : "—"}
          hint="Uncertainty of the classifier's output: 0 bits = certain, 1 bit = coin flip" />
        <Metric k="Cache" v={r.cache} hint="HIT = this exact image and settings were analysed before" />
        <Metric k="Latency" v={ms(r.timings_ms.total)} hint="Server-side pipeline time" />
      </dl>
      <div className="mt-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-ink-3">System rationale</p>
        <ul className="mt-1 space-y-0.5 text-sm text-ink-2">
          {a.reasons.map((x, i) => <li key={i} className="flex gap-2"><span className="text-ink-3">•</span><span className="min-w-0 break-words">{x}</span></li>)}
        </ul>
      </div>
    </section>
  );
}

function Metric({ k, v, hint }: { k: string; v: ReactNode; hint: string }) {
  return (
    <div title={hint}>
      <dt className="text-[11px] uppercase tracking-wide text-ink-3">{k}</dt>
      <dd className="tnum font-semibold text-ink-1">{v}</dd>
    </div>
  );
}
