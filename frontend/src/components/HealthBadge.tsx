"use client";

import { useEffect, useRef, useState } from "react";
import { api, type HealthResponse } from "@/lib/api";

const NAMES: Record<string, string> = {
  yolov8_person: "Person + pose detector (YOLOv8n-pose)",
  yolov8_objects: "Object detector (YOLOv8n, COCO)",
  mediapipe_hands: "Hand landmarks (MediaPipe)",
  mediapipe_face: "Face mesh (MediaPipe)",
  mobilenetv3_classifier: "Real/fake classifier (MobileNetV3)",
  artifact_detector: "Learned anatomy-error detector (optional, not trained)",
};

export function HealthBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    // Real-time latency = measured round trip of a /health request, every 5 s.
    const load = () => {
      const t0 = performance.now();
      return api
        .health()
        .then((h) => { if (alive) { setHealth(h); setError(null); setLatency(performance.now() - t0); } })
        .catch((e: Error) => { if (alive) { setError(e.message); setLatency(null); } });
    };
    load();
    const t = setInterval(load, 5_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const state = error ? "down" : !health ? "loading" : health.status;
  const label = { down: "API offline", loading: "Checking API…", ok: "All layers ready", degraded: "Degraded" }[state];
  const dot = { down: "bg-critical", loading: "bg-line-strong", ok: "bg-good", degraded: "bg-warning" }[state];

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:bg-surface-2"
        aria-expanded={open}
      >
        <span className={`h-2 w-2 rounded-full ${dot} ${state === "ok" ? "animate-pulse" : ""}`} aria-hidden="true" />
        {label}
        {latency != null && (
          <span className="tnum border-l border-line pl-2 font-semibold text-ink-1" title="Round-trip time to the API, measured every 5 s">
            {Math.round(latency)} ms
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-2 w-80 rounded-lg border border-line bg-surface-1 p-3 text-xs shadow-lg">
          {error && <p className="text-critical-ink">{error}</p>}
          {health && (
            <>
              <p className="mb-2 text-ink-3">
                API v{health.version} · {health.device}
              </p>
              <ul className="space-y-1.5">
                {Object.entries(health.components).map(([k, c]) => (
                  <li key={k} className="flex items-start gap-2">
                    <span className={`mt-0.5 font-semibold ${c.loaded ? "text-good-ink" : "text-critical-ink"}`}>
                      {c.loaded ? "✓" : "✗"}
                    </span>
                    <span>
                      <span className="text-ink-1">{NAMES[k] ?? k}{c.optional && !c.loaded && <span className="text-ink-3"> · optional</span>}</span>
                      {c.error && <span className="block break-words text-ink-3">{c.error}</span>}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
