import type { AnalysisResponse, HandOut } from "@/lib/api";
import { AnatomyOverlay } from "./AnatomyOverlay";

/** Enlarged view of one hand: the image cropped (via the SVG viewBox) to the
 *  hand, with its skeleton, outline and numbered fingertips. */
export function HandZoom({ src, report, hand, size = 180 }: {
  src: string;
  report: AnalysisResponse;
  hand: HandOut;
  size?: number;
}) {
  const pts = [...hand.landmarks, ...(hand.finger_count?.outline ?? [])];
  if (pts.length === 0) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const pad = 0.15 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 20);
  const x0 = Math.min(...xs) - pad;
  const y0 = Math.min(...ys) - pad;
  const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 2 * pad;
  const { width: W, height: H } = report.image;

  return (
    <svg viewBox={`${x0} ${y0} ${side} ${side}`} width={size} height={size}
      className="shrink-0 rounded-lg border border-line bg-surface-2" role="img"
      aria-label={`Zoomed view of hand ${hand.index + 1}`}>
      <image href={src} x={0} y={0} width={W} height={H} preserveAspectRatio="none" />
      <AnatomyOverlay report={report} only={hand.index} layers={{ artifacts: false, boxes: false, body: false, hands: true, count: true }}
        scale={side / size} />
    </svg>
  );
}
