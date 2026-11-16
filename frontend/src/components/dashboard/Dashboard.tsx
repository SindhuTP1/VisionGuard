"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ACCEPTED_TYPES, api, MAX_UPLOAD_MB, type AnalysisResponse, type Engine } from "@/lib/api";
import { ms } from "@/lib/format";
import { ForensicReport } from "./ForensicReport";
import { Layer1Card, Layer2Card, Layer3Card, MatrixCard, Panel, VerdictCard } from "./Telemetry";
import { Visualizer, VIZ_TABS, type VizTab } from "./Visualizer";

const ENGINES: { id: Engine; label: string; help: string }[] = [
  { id: "full", label: "Multi-Layer Arbitration (all layers)", help: "Geometry checks and the classifier both feed the verdict." },
  { id: "geometry", label: "Biometric Geometry Only", help: "Finger count, hands per person, face symmetry. The classifier is shown but ignored." },
  { id: "classifier", label: "Neural Classifier Only", help: "Real-vs-AI classifier decides. Geometry checks are shown but ignored." },
];

type Phase = "empty" | "ready" | "scanning" | "done" | "error";

export function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("empty");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisResponse | null>(null);
  const [tab, setTab] = useState<VizTab>("original");
  const [engine, setEngine] = useState<Engine>("full");
  const [sensitivity, setSensitivity] = useState(50);
  const [elapsed, setElapsed] = useState(0);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => { if (src) URL.revokeObjectURL(src); }, [src]);

  // live elapsed timer while the pipeline runs (real wall-clock time)
  useEffect(() => {
    if (phase !== "scanning") return;
    const t0 = performance.now();
    const id = setInterval(() => setElapsed(performance.now() - t0), 100);
    return () => clearInterval(id);
  }, [phase]);

  const choose = useCallback((f: File | undefined) => {
    if (!f) return;
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError(`Unsupported file type "${f.type || "unknown"}". Use JPG, JPEG, PNG or WEBP.`);
      setPhase(file ? "ready" : "empty");
      return;
    }
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`File is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_UPLOAD_MB} MB.`);
      return;
    }
    abortRef.current?.abort();
    setFile(f);
    setSrc(URL.createObjectURL(f));
    setReport(null);
    setError(null);
    setTab("original");
    setPhase("ready");
  }, [file]);

  const scan = useCallback(async () => {
    if (!file) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setPhase("scanning");
    setElapsed(0);
    setError(null);
    const t0 = performance.now();
    try {
      const r = await api.analyze(file, { sensitivity, engine }, ctrl.signal);
      if (ctrl.signal.aborted) return;
      setRoundTrip(performance.now() - t0);
      setReport(r);
      setPhase("done");
      setTab((t) => (t === "original" ? "boxes" : t));
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError((e as Error).message);
      setPhase("error");
    }
  }, [file, sensitivity, engine]);

  const reset = () => {
    abortRef.current?.abort();
    setFile(null);
    setSrc(null);
    setReport(null);
    setError(null);
    setTab("original");
    setPhase("empty");
    setRoundTrip(null);
    setSensitivity(50);
    setEngine("full");
  };

  const exportJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `visionguard_${(report.filename ?? "scan").replace(/\.[^.]+$/, "")}_${report.request_id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const settingsChanged = report != null && (report.settings_used.sensitivity !== sensitivity || report.settings_used.engine !== engine);

  return (
    <>
      <div className="print-hide grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        {/* ------------------------- LEFT: Scan Control Center ------------------------- */}
        <section className="space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-[0.18em] text-ink-3">Scan Control Center</h2>

          <div className="rounded-xl border border-line bg-surface-1 p-4 shadow-[var(--shadow)]">
            <label className="block text-sm font-semibold text-ink-1" htmlFor="engine">Forensic Detection Engine</label>
            <select id="engine" value={engine} onChange={(e) => setEngine(e.target.value as Engine)} disabled={phase === "scanning"}
              className="mt-1.5 w-full rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm text-ink-1 focus:border-accent focus:outline-none">
              {ENGINES.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
            </select>
            <p className="mt-1 text-xs text-ink-3">{ENGINES.find((e) => e.id === engine)?.help}</p>

            <div className="mt-4 flex items-baseline justify-between">
              <label className="text-sm font-semibold text-ink-1" htmlFor="sens">Detection Sensitivity</label>
              <span className="tnum text-sm font-semibold text-ink-1">{sensitivity}%</span>
            </div>
            <input id="sens" type="range" min={0} max={100} step={5} value={sensitivity} disabled={phase === "scanning"}
              onChange={(e) => setSensitivity(Number(e.target.value))} className="mt-2 w-full accent-[var(--accent)]" />
            <p className="mt-1 text-xs text-ink-3">
              50% = calibrated defaults. Higher values scale geometric evidence up (×{(2 ** ((sensitivity - 50) / 50)).toFixed(2)}) and lower
              the face-symmetry threshold. The review bar stays at 65%.
            </p>
          </div>

          <div className="rounded-xl border border-line bg-surface-1 p-4 shadow-[var(--shadow)]">
            {!src ? (
              <label
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}
                className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-14 text-center ${dragging ? "border-accent bg-surface-2" : "border-line-strong hover:bg-surface-2"}`}>
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" className="text-ink-3" aria-hidden="true">
                  <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className="text-sm font-semibold text-ink-1">Upload target image</span>
                <span className="text-xs text-ink-3">Drag & drop or click · JPG, JPEG, PNG, WEBP · up to {MAX_UPLOAD_MB} MB</span>
                <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(",")} className="sr-only"
                  onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
              </label>
            ) : (
              <>
                <div role="tablist" aria-label="Visualization" className="mb-3 grid grid-cols-2 gap-1 rounded-lg bg-surface-2 p-1 sm:grid-cols-4">
                  {VIZ_TABS.map((t) => {
                    const disabled = t.id !== "original" && !report;
                    return (
                      <button key={t.id} role="tab" aria-selected={tab === t.id} disabled={disabled} onClick={() => setTab(t.id)}
                        className={`rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${tab === t.id ? "bg-surface-1 text-ink-1 shadow-sm" : "text-ink-2 hover:text-ink-1"} disabled:cursor-not-allowed disabled:opacity-40`}>
                        {t.label}
                      </button>
                    );
                  })}
                </div>
                <div className="relative">
                  <Visualizer src={src} report={report} tab={tab} />
                  {phase === "scanning" && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-lg bg-white/70 backdrop-blur-[1px]">
                      <span className="h-8 w-8 animate-spin rounded-full border-4 border-line border-t-accent" aria-hidden="true" />
                      <span className="tnum text-sm font-semibold text-ink-1">Running pipeline… {(elapsed / 1000).toFixed(1)} s</span>
                    </div>
                  )}
                </div>
                <div className="mt-2 flex items-center justify-between gap-2 text-xs text-ink-3">
                  <span className="truncate">{file?.name}</span>
                  <button onClick={() => inputRef.current?.click()} className="shrink-0 text-accent hover:underline" disabled={phase === "scanning"}>Replace image</button>
                  <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(",")} className="sr-only"
                    onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
                </div>
                {tab === "gradcam" && report?.gradcam.status === "ok" && (
                  <p className="mt-1 text-xs text-ink-3">
                    Heatmap of what pushed the classifier toward <b>{report.gradcam.target === "fake" ? "AI-generated" : "real"}</b>.
                    The model only examines the five 224px patches outlined; the rest of the image isn&apos;t seen.
                  </p>
                )}
              </>
            )}

            {/* processing status */}
            <div className="mt-3 rounded-lg bg-surface-2 px-3 py-2 text-xs" aria-live="polite">
              {phase === "empty" && <span className="text-ink-3">Status: awaiting image</span>}
              {phase === "ready" && <span className="text-ink-2">Status: image loaded. Press <b>Initialize Target Scan</b>.</span>}
              {phase === "scanning" && <span className="text-ink-1">Status: scanning, running Layers 1–3 and arbitration…</span>}
              {phase === "error" && <span className="text-critical-ink">Status: scan failed. {error}</span>}
              {phase === "done" && report && (
                <div className="tnum flex flex-wrap gap-x-3 gap-y-0.5 text-ink-2">
                  <span className="font-semibold text-good-ink">Status: complete</span>
                  {Object.entries(report.timings_ms).filter(([k]) => k !== "total").map(([k, v]) => <span key={k}>{k} {ms(v)}</span>)}
                  <span className="font-semibold">total {ms(report.timings_ms.total)}</span>
                  {roundTrip != null && <span>round-trip {ms(roundTrip)}</span>}
                </div>
              )}
              {error && phase !== "error" && <p className="mt-1 text-critical-ink">{error}</p>}
            </div>

            <div className="mt-3 flex gap-2">
              <button onClick={scan} disabled={!file || phase === "scanning"}
                className="flex-1 rounded-lg bg-ink-1 px-4 py-2.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">
                {phase === "scanning" ? "Scanning…" : settingsChanged ? "Rescan with New Settings" : report ? "Rescan Target" : "Initialize Target Scan"}
              </button>
              <button onClick={reset} disabled={phase === "scanning" && !file}
                className="rounded-lg border border-line-strong px-4 py-2.5 text-sm font-semibold text-ink-1 hover:bg-surface-2">
                Reset
              </button>
            </div>
          </div>
        </section>

        {/* --------------------------- RIGHT: System Telemetry --------------------------- */}
        <section className="space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-[0.18em] text-ink-3">System Telemetry</h2>
          {!report ? (
            <div className="rounded-xl border border-dashed border-line-strong bg-surface-1 p-8 text-center text-sm text-ink-3">
              {phase === "scanning" ? "Analysis in progress…" : "Telemetry appears here after a scan: three layer readouts, the anomaly matrix and the arbitrated verdict."}
            </div>
          ) : (
            <>
              <VerdictCard r={report} />
              <div className="grid gap-4 xl:grid-cols-3">
                <Layer1Card r={report} />
                <Layer2Card r={report} />
                <Layer3Card r={report} />
              </div>
              <MatrixCard key={report.request_id} r={report} />
              <Panel title="Generate Forensic Archive">
                <p className="mb-3 text-sm text-ink-2">Export this scan&apos;s complete results.</p>
                <div className="flex flex-wrap gap-2">
                  <button onClick={exportJson} className="rounded-lg border border-line-strong px-4 py-2 text-sm font-semibold text-ink-1 hover:bg-surface-2">
                    Export Raw JSON
                  </button>
                  <button onClick={() => window.print()} className="rounded-lg bg-ink-1 px-4 py-2 text-sm font-semibold text-white hover:opacity-90">
                    Print Diagnostic PDF
                  </button>
                </div>
                <p className="mt-2 text-xs text-ink-3">For the PDF, choose &ldquo;Save as PDF&rdquo; as the printer in the print dialog.</p>
              </Panel>
            </>
          )}
        </section>
      </div>

      {report && src && <ForensicReport r={report} src={src} clientLatency={roundTrip} />}
    </>
  );
}
