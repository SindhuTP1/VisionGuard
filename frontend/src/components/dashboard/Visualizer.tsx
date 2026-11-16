"use client";

import { useId } from "react";
import type { AnalysisResponse } from "@/lib/api";
import { AnatomyOverlay } from "../AnatomyOverlay";
import { useWidth } from "../charts/useWidth";

export type VizTab = "original" | "boxes" | "mesh" | "gradcam";
export const VIZ_TABS: { id: VizTab; label: string }[] = [
  { id: "original", label: "Original Source" },
  { id: "boxes", label: "Layer 1 Bounding Boxes" },
  { id: "mesh", label: "Layer 2 Mesh Overlay" },
  { id: "gradcam", label: "Layer 3 Grad-CAM Heatmap" },
];

/** Image + the selected layer's overlay. The SVG viewBox is the analysed image's
 *  pixel space, so every overlay lines up at any display size. */
export function Visualizer({ src, report, tab, fixedWidth }: {
  src: string;
  report: AnalysisResponse | null;
  tab: VizTab;
  fixedWidth?: number;                 // print/report rendering
}) {
  const [ref, measured] = useWidth<HTMLDivElement>();
  const width = fixedWidth ?? measured;
  const W = report?.image.width ?? 1;
  const H = report?.image.height ?? 1;
  const scale = W / Math.max(width, 1);

  return (
    <div ref={ref} className="relative w-full overflow-hidden rounded-lg bg-surface-2">
      {/* eslint-disable-next-line @next/next/no-img-element -- local blob URL */}
      <img src={src} alt="Image under analysis" className="block w-full" />
      {report && tab !== "original" && (
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full"
          role="img" aria-label={VIZ_TABS.find((t) => t.id === tab)?.label}>
          {tab === "boxes" && <BoxesOverlay report={report} scale={scale} />}
          {tab === "mesh" && <MeshOverlay report={report} scale={scale} />}
          {tab === "gradcam" && <GradCamOverlay report={report} scale={scale} />}
        </svg>
      )}
    </div>
  );
}

function Label({ x, y, text, color, scale }: { x: number; y: number; text: string; color: string; scale: number }) {
  const fs = 11 * scale;
  const w = text.length * fs * 0.58 + 8 * scale;
  return (
    <g>
      <rect x={x} y={y - fs - 6 * scale} width={w} height={fs + 6 * scale} fill={color} rx={2 * scale} />
      <text x={x + 4 * scale} y={y - 4 * scale} fontSize={fs} fontWeight={600} fill="#ffffff">{text}</text>
    </g>
  );
}

/** Layer 1: YOLOv8 object boxes (COCO), MediaPipe face boxes, and the analysed region. */
function BoxesOverlay({ report, scale }: { report: AnalysisResponse; scale: number }) {
  const sw = 2 * scale;
  const { person, objects, face } = report;
  return (
    <g>
      {!person.used_fallback && (
        <rect x={person.bbox[0]} y={person.bbox[1]} width={person.bbox[2] - person.bbox[0]} height={person.bbox[3] - person.bbox[1]}
          fill="none" stroke="#ffffff" strokeOpacity={0.9} strokeWidth={sw * 0.75} strokeDasharray={`${6 * scale} ${4 * scale}`} />
      )}
      {/* People found by the pose detector that the COCO detector missed (e.g. close-ups) */}
      {person.person_boxes
        .filter((b) => !objects.objects.some((o) => o.label === "person" && iou(o.box, b) > 0.4))
        .map((b, i) => (
          <g key={`p${i}`}>
            <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" stroke="var(--accent)" strokeWidth={sw} />
            <Label x={b[0]} y={b[1] + 18 * scale} text="person (pose)" color="var(--accent)" scale={scale} />
          </g>
        ))}
      {objects.objects.map((o, i) => (
        <g key={`o${i}`}>
          <rect x={o.box[0]} y={o.box[1]} width={o.box[2] - o.box[0]} height={o.box[3] - o.box[1]} fill="none"
            stroke="var(--series-1)" strokeWidth={sw} />
          <Label x={o.box[0]} y={o.box[1]} text={`${o.label} ${Math.round(o.score * 100)}%`} color="var(--series-1)" scale={scale} />
        </g>
      ))}
      {face.faces.map((f) => f.box && (
        <g key={`f${f.index}`}>
          <rect x={f.box[0]} y={f.box[1]} width={f.box[2] - f.box[0]} height={f.box[3] - f.box[1]} fill="none"
            stroke="var(--series-2)" strokeWidth={sw} />
          <Label x={f.box[0]} y={f.box[3] + 18 * scale} text={`face ${f.index + 1}`} color="var(--series-2)" scale={scale} />
        </g>
      ))}
    </g>
  );
}

/** Layer 2: MediaPipe face mesh + hand skeletons / finger count + body pose. */
function MeshOverlay({ report, scale }: { report: AnalysisResponse; scale: number }) {
  return (
    <g>
      {report.face.faces.map((f) => (
        <g key={f.index}>
          {f.landmarks.map((p, i) => (
            <circle key={i} cx={p[0]} cy={p[1]} r={1.1 * scale} fill={f.anomalous ? "var(--critical)" : "#22d3ee"} fillOpacity={0.9} />
          ))}
        </g>
      ))}
      <AnatomyOverlay report={report} scale={scale} layers={{ artifacts: false, boxes: false, body: true, hands: true, count: true }} />
    </g>
  );
}

/** Layer 3: Grad-CAM of the classifier's prediction on the 5 patches it examined. */
function GradCamOverlay({ report, scale }: { report: AnalysisResponse; scale: number }) {
  const uid = useId().replace(/:/g, "");
  const { gradcam } = report;
  if (gradcam.status !== "ok") return null;
  return (
    <g>
      <defs>
        {gradcam.crops.map((c, i) => (
          <g key={i}>
            <clipPath id={`clip-${uid}-${i}`}>
              <rect x={c.box[0]} y={c.box[1]} width={c.box[2] - c.box[0]} height={c.box[3] - c.box[1]} />
            </clipPath>
            {/* blur relative to the patch, so the 7x7 grid looks smooth at any image size */}
            <filter id={`blur-${uid}-${i}`} x="-10%" y="-10%" width="120%" height="120%">
              <feGaussianBlur stdDeviation={(c.box[2] - c.box[0]) / 18} />
            </filter>
          </g>
        ))}
      </defs>
      {gradcam.crops.map((c, i) => {
        const n = c.grid.length;
        const cw = (c.box[2] - c.box[0]) / n;
        const ch = (c.box[3] - c.box[1]) / n;
        return (
          <g key={i} clipPath={`url(#clip-${uid}-${i})`}>
            <g filter={`url(#blur-${uid}-${i})`}>
              {c.grid.map((row, r) => row.map((v, k) => (
                <rect key={`${r}-${k}`} x={c.box[0] + k * cw} y={c.box[1] + r * ch} width={cw + 1} height={ch + 1}
                  fill={heat(v)} />
              )))}
            </g>
          </g>
        );
      })}
      {gradcam.crops.map((c, i) => (
        <g key={`o${i}`}>
          <rect x={c.box[0]} y={c.box[1]} width={c.box[2] - c.box[0]} height={c.box[3] - c.box[1]} fill="none"
            stroke="#ffffff" strokeWidth={1.5 * scale} strokeDasharray={`${4 * scale} ${3 * scale}`} />
          <Label x={c.box[0] + 2 * scale} y={c.box[1] + 20 * scale} text={`patch ${i + 1} · ${Math.round((c.p_target ?? 0) * 100)}%`}
            color="rgba(15,27,51,0.75)" scale={scale} />
        </g>
      ))}
    </g>
  );
}

function iou(a: number[], b: number[]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** 0..1 → transparent blue → yellow → red. */
export function heat(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.5) {
    const u = t / 0.5;
    return `rgba(${Math.round(40 + 215 * u)}, ${Math.round(90 + 140 * u)}, ${Math.round(220 - 200 * u)}, ${0.25 + 0.3 * u})`;
  }
  const u = (t - 0.5) / 0.5;
  return `rgba(255, ${Math.round(230 - 200 * u)}, 20, ${0.55 + 0.2 * u})`;
}
