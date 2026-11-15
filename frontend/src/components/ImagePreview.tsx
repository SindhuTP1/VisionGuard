"use client";

import { useState } from "react";
import type { AnalysisResponse } from "@/lib/api";
import { AnatomyOverlay, type OverlayLayers } from "./AnatomyOverlay";
import { useWidth } from "./charts/useWidth";

const LAYER_LABELS: [keyof OverlayLayers, string][] = [
  ["boxes", "People"],
  ["body", "Body pose"],
  ["hands", "Hand skeleton"],
  ["count", "Finger count"],
];

/** Uploaded image with every detection drawn over it. The SVG uses the analysed
 *  image's pixel space (after EXIF rotation / downscale) as its viewBox. */
export function ImagePreview({ src, name, report, loading }: {
  src: string;
  name: string;
  report: AnalysisResponse | null;
  loading: boolean;
}) {
  const [layers, setLayers] = useState<OverlayLayers>({ boxes: true, body: true, hands: true, count: true });
  const [ref, width] = useWidth<HTMLDivElement>();
  const W = report?.image.width ?? 1;
  const H = report?.image.height ?? 1;

  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-surface-1">
      <div ref={ref} className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element -- local blob URL, nothing to optimise */}
        <img src={src} alt={`Uploaded image ${name}`} className={`block w-full ${loading ? "opacity-60" : ""}`} />
        {report && (
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full"
            role="img" aria-label="Detections drawn over the image">
            <AnatomyOverlay report={report} layers={layers} scale={W / Math.max(width, 1)} />
          </svg>
        )}
      </div>
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 text-xs text-ink-3">
        <span className="mr-auto truncate">{name}</span>
        {report && LAYER_LABELS.map(([k, label]) => (
          <label key={k} className="flex shrink-0 items-center gap-1">
            <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers((l) => ({ ...l, [k]: e.target.checked }))} />
            {label}
          </label>
        ))}
      </figcaption>
    </figure>
  );
}
