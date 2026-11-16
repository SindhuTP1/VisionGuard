import type { AnalysisResponse } from "@/lib/api";
import { CLASS_LABEL, ms, num, pct } from "@/lib/format";
import { primaryReason, VERDICT } from "./Telemetry";
import { Visualizer } from "./Visualizer";

/** Printable forensic report. Hidden on screen and shown only when printing
 *  (globals.css @media print), so "Print Diagnostic PDF" = the browser's Save as PDF. */
export function ForensicReport({ r, src, clientLatency }: { r: AnalysisResponse; src: string; clientLatency: number | null }) {
  const a = r.arbitration;
  const when = r.executed_at ? new Date(r.executed_at) : new Date();
  return (
    <article id="forensic-report" className="hidden text-[12px] text-black">
      <header className="mb-4 border-b-2 border-black pb-2">
        <h1 className="text-2xl font-bold">VisionGuard Forensics Report</h1>
        <p>A Multi-Layer AI Hallucination Detection Prototype with Explainable Visual Validation</p>
      </header>

      <table className="mb-4 w-full border-collapse">
        <tbody>
          {[
            ["Image file", r.filename ?? "—"],
            ["SHA-256", r.image_sha256 ?? "—"],
            ["Executed", when.toLocaleString()],
            ["System latency", `${ms(r.timings_ms.total)} server${clientLatency != null ? ` · ${Math.round(clientLatency)} ms round-trip` : ""}`],
            ["Entropy index", r.telemetry.entropy_bits == null ? "—" : `${num(r.telemetry.entropy_bits, 3)} bits (${r.telemetry.entropy_level})`],
            ["Engine / sensitivity", `${r.settings_used.engine} · ${r.settings_used.sensitivity}%`],
            ["Cache", r.cache],
          ].map(([k, v]) => (
            <tr key={k} className="border-b border-gray-300"><td className="w-40 py-1 font-semibold">{k}</td><td className="break-all">{v}</td></tr>
          ))}
        </tbody>
      </table>

      <section className="mb-4 border-2 border-black p-3">
        <h2 className="text-lg font-bold">Final verdict: {VERDICT[a.verdict].title}</h2>
        <p>Confidence: <b>{pct(a.confidence)}</b>{a.anomaly_class && <> · Category: <b>{CLASS_LABEL[a.anomaly_class]}</b></>}
          {r.telemetry.confidence_boost != null && <> · Matrix boost: {(r.telemetry.confidence_boost * 100).toFixed(1)} pts</>}</p>
        <p>Primary reason: {primaryReason(r)}</p>
      </section>

      <h2 className="mb-1 text-base font-bold">Layer 1: YOLOv8 detection</h2>
      <p className="mb-3">
        {r.objects.objects.length} objects ({r.objects.objects.map((o) => `${o.label} ${pct(o.score, 0)}`).join(", ") || "none"}); {r.person.num_persons} human figure(s); {r.face.num_faces} face(s).
      </p>

      <h2 className="mb-1 text-base font-bold">Layer 2: MediaPipe biometric integrity</h2>
      <ul className="mb-3 list-disc pl-5">
        <li>Faces: {r.face.faces.map((f) => f.skipped_reason ? `face ${f.index + 1} skipped (${f.skipped_reason})` : `face ${f.index + 1} asymmetry ${num(f.asymmetry_score)} vs threshold ${num(r.settings_used.face_asymmetry_threshold)}`).join("; ") || "none"}</li>
        <li>Hands: {r.hands.hands.map((h) => `hand ${h.index + 1}: ${h.finger_count?.status === "counted" ? `${h.finger_count.count} fingers` : `not countable (${h.finger_count?.reason ?? "—"})`}`).join("; ") || "none"}</li>
        <li>Hands per person: {r.body.persons.map((p) => `person ${p.index + 1}: ${p.hands.length}${p.extra_hand ? " (EXTRA)" : ""}`).join("; ") || "—"}</li>
        <li>Structural findings: {a.findings.map((f) => f.message).join("; ") || "none"}</li>
      </ul>

      <h2 className="mb-1 text-base font-bold">Layer 3: neural classification</h2>
      <p className="mb-3">
        {r.classifier.status === "ok"
          ? `MobileNetV3-Small (run ${r.classifier.run_id}): ${r.classifier.label === "fake" ? "AI-generated" : "real"} · P(AI) ${pct(r.classifier.p_fake)} · P(real) ${pct(r.classifier.p_real)}`
          : `Unavailable: ${r.classifier.note}`}
      </p>

      <h2 className="mb-1 text-base font-bold">Anomaly probability matrix</h2>
      <table className="mb-4 w-full border-collapse">
        <thead><tr className="border-b-2 border-black text-left"><th className="py-1">Category</th><th>Value</th><th>Source</th></tr></thead>
        <tbody>
          {r.telemetry.matrix.map((m) => (
            <tr key={m.category} className="border-b border-gray-300">
              <td className="py-1">{CLASS_LABEL[m.category]}</td>
              <td>{m.value == null ? "not measured" : pct(m.value)}</td>
              <td>{m.source}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <section className="break-inside-avoid">
      <h2 className="mb-2 text-base font-bold">Visual evidence</h2>
      <div className="grid grid-cols-2 gap-3">
        {([["original", "Original source"], ["boxes", "Layer 1 bounding boxes"], ["mesh", "Layer 2 mesh overlay"], ["gradcam", "Layer 3 Grad-CAM"]] as const).map(([tab, label]) => (
          <figure key={tab} className="break-inside-avoid">
            <Visualizer src={src} report={r} tab={tab} fixedWidth={330} />
            <figcaption className="mt-1 text-center font-semibold">{label}</figcaption>
          </figure>
        ))}
      </div>
      </section>

      <h2 className="mb-1 mt-4 text-base font-bold">System rationale</h2>
      <ul className="list-disc pl-5">{a.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>

      <footer className="mt-4 border-t border-gray-400 pt-2 text-[10px]">
        Research prototype. Verdicts are signals for a human reviewer, not proof. Request {r.request_id}.
      </footer>
    </article>
  );
}
