"use client";

import { useEffect, useRef, useState } from "react";
import { api, type HealthResponse } from "@/lib/api";

const NAMES: Record<string, string> = {
  yolov8_person: "Person detector (YOLOv8)",
  mediapipe_hands: "Hand landmarks (MediaPipe)",
  mediapipe_face: "Face mesh (MediaPipe)",
  mobilenetv3_classifier: "Real/fake classifier",
};

export function HealthBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setError(null)))
        .catch((e: Error) => alive && setError(e.message));
    load();
    const t = setInterval(load, 30_000);
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
        <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
        {label}
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
                      <span className="text-ink-1">{NAMES[k] ?? k}</span>
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
