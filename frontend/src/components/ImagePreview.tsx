"use client";

import { useState } from "react";
import type { AnalysisResponse } from "@/lib/api";

/** Uploaded image with the detected person boxes drawn over it.
 *  Boxes are in the analysed image's pixel space (after EXIF rotation and any
 *  downscale), so they are placed by percentage and survive any display size. */
export function ImagePreview({ src, name, report, loading }: {
  src: string;
  name: string;
  report: AnalysisResponse | null;
  loading: boolean;
}) {
  const [showBoxes, setShowBoxes] = useState(true);
  const W = report?.image.width ?? 1;
  const H = report?.image.height ?? 1;
  const box = (b: number[]) => ({
    left: `${(b[0] / W) * 100}%`,
    top: `${(b[1] / H) * 100}%`,
    width: `${((b[2] - b[0]) / W) * 100}%`,
    height: `${((b[3] - b[1]) / H) * 100}%`,
  });
  const person = report?.person;

  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-surface-1">
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element -- local blob URL, nothing to optimise */}
        <img src={src} alt={`Uploaded image ${name}`} className={`block w-full ${loading ? "opacity-60" : ""}`} />
        {person && showBoxes && (
          <>
            {person.person_boxes.map((b, i) => (
              <div key={i} className="pointer-events-none absolute rounded-sm border border-white/80 shadow-[0_0_0_1px_rgba(0,0,0,0.5)]" style={box(b)} />
            ))}
            {!person.used_fallback && (
              <div className="pointer-events-none absolute rounded-sm border-2 border-dashed border-accent" style={box(person.bbox)}>
                <span className="absolute -top-0.5 left-0 -translate-y-full rounded-t bg-accent px-1.5 py-0.5 text-[10px] font-medium text-accent-ink">
                  analysed region
                </span>
              </div>
            )}
          </>
        )}
      </div>
      <figcaption className="flex items-center justify-between gap-2 px-3 py-2 text-xs text-ink-3">
        <span className="truncate">{name}</span>
        {person && (
          <label className="flex shrink-0 items-center gap-1.5">
            <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
            Show boxes
          </label>
        )}
      </figcaption>
    </figure>
  );
}
