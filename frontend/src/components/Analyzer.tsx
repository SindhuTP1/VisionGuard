"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ACCEPTED_TYPES, api, MAX_UPLOAD_MB, type AnalysisResponse, type HealthResponse } from "@/lib/api";
import { ms } from "@/lib/format";
import { ImagePreview } from "./ImagePreview";
import { ClassifierCard, FaceCard, HandsCard, PersonCard } from "./LayerCards";
import { VerdictCard } from "./VerdictCard";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; report: AnalysisResponse }
  | { kind: "error"; message: string };

export function Analyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [state, setState] = useState<State>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const [thresholds, setThresholds] = useState<HealthResponse["thresholds"] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.health().then((h) => setThresholds(h.thresholds)).catch(() => {});
  }, []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const run = useCallback(async (f: File) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setState({ kind: "loading" });
    try {
      const report = await api.analyze(f, ctrl.signal);
      if (!ctrl.signal.aborted) setState({ kind: "done", report });
    } catch (e) {
      if (!ctrl.signal.aborted) setState({ kind: "error", message: (e as Error).message });
    }
  }, []);

  const accept = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      if (!ACCEPTED_TYPES.includes(f.type)) {
        setState({ kind: "error", message: `Unsupported file type "${f.type || "unknown"}". Use JPEG, PNG or WebP.` });
        return;
      }
      if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
        setState({ kind: "error", message: `File is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_UPLOAD_MB} MB.` });
        return;
      }
      setFile(f);
      setPreview(URL.createObjectURL(f));
      void run(f);
    },
    [run],
  );

  const report = state.kind === "done" ? state.report : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Analyze an image</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          The image goes through four layers: the person is cropped, hands and the face are checked for
          impossible geometry, a classifier estimates whether it&apos;s AI-generated, and the results are
          combined into one verdict. Anything below 65% confidence is sent for human review.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-3">
          <label
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]); }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
              dragging ? "border-accent bg-surface-2" : "border-line-strong bg-surface-1 hover:bg-surface-2"
            }`}
          >
            <UploadIcon />
            <span className="text-sm font-medium">Drop an image here or click to choose</span>
            <span className="text-xs text-ink-3">JPEG, PNG or WebP · up to {MAX_UPLOAD_MB} MB</span>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES.join(",")}
              className="sr-only"
              onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }}
            />
          </label>

          {preview && (
            <ImagePreview src={preview} name={file?.name ?? ""} report={report} loading={state.kind === "loading"} />
          )}
        </div>

        <div className="space-y-4" aria-live="polite">
          {state.kind === "idle" && (
            <div className="flex h-full min-h-48 items-center justify-center rounded-xl border border-line bg-surface-1 p-6 text-sm text-ink-3">
              Results will appear here.
            </div>
          )}
          {state.kind === "loading" && <LoadingCard />}
          {state.kind === "error" && (
            <div role="alert" className="rounded-xl border border-critical/40 bg-critical-soft p-4 text-sm text-critical-ink">
              <p className="font-medium">Analysis failed</p>
              <p className="mt-1">{state.message}</p>
              {file && (
                <button onClick={() => run(file)} className="mt-3 rounded-md border border-critical/40 px-3 py-1 text-xs hover:bg-surface-1">
                  Try again
                </button>
              )}
            </div>
          )}
          {report && (
            <>
              <VerdictCard arb={report.arbitration} threshold={thresholds?.confidence ?? null} />
              <div className="grid gap-4 md:grid-cols-2">
                <PersonCard person={report.person} />
                <ClassifierCard clf={report.classifier} />
                <FaceCard face={report.face} threshold={thresholds?.face_asymmetry ?? null} />
                <HandsCard hands={report.hands} />
              </div>
              <p className="tnum text-xs text-ink-3">
                Request {report.request_id} · {report.image.width}×{report.image.height}px ·{" "}
                {Object.entries(report.timings_ms)
                  .filter(([k]) => k !== "total")
                  .map(([k, v]) => `${k} ${ms(v)}`)
                  .join(" · ")}{" "}
                · total {ms(report.timings_ms.total)}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function LoadingCard() {
  return (
    <div className="space-y-3 rounded-xl border border-line bg-surface-1 p-4" aria-busy="true">
      <p className="text-sm text-ink-2">Running the detection layers…</p>
      {[70, 90, 55].map((w) => (
        <div key={w} className="h-3 animate-pulse rounded bg-surface-2" style={{ width: `${w}%` }} />
      ))}
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-ink-3" aria-hidden="true">
      <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
