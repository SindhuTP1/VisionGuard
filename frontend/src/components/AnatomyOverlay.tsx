import type { AnalysisResponse } from "@/lib/api";
import { ARTIFACT_LABEL } from "@/lib/format";

// MediaPipe hand skeleton (21 landmarks) and the COCO body skeleton (17 keypoints).
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const FINGER_OF_BONE: Record<string, string> = {
  "1-2": "thumb", "2-3": "thumb", "3-4": "thumb",
  "5-6": "index", "6-7": "index", "7-8": "index",
  "9-10": "middle", "10-11": "middle", "11-12": "middle",
  "13-14": "ring", "14-15": "ring", "15-16": "ring",
  "17-18": "pinky", "18-19": "pinky", "19-20": "pinky",
};
const BODY_BONES: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

export interface OverlayLayers {
  artifacts: boolean;
  boxes: boolean;
  body: boolean;
  hands: boolean;
  count: boolean;
}

/** SVG drawn in the analysed image's pixel space (viewBox = image size), so it
 *  lines up at any display size. ``scale`` makes strokes/labels constant on screen. */
export function AnatomyOverlay({ report, layers, scale = 1, only }: {
  report: AnalysisResponse;
  layers: OverlayLayers;
  scale?: number;            // image pixels per screen pixel
  only?: number;             // draw only this hand (zoom view)
}) {
  const { person, hands, body, arbitration, artifacts } = report;
  const sw = 2 * scale;
  const flaggedHands = new Set(arbitration.findings.filter((f) => f.hand_index != null).map((f) => f.hand_index));
  const extraPersons = new Set(body.persons.filter((p) => p.extra_hand).map((p) => p.index));

  return (
    <g>
      {layers.boxes && only == null && (
        <>
          {person.person_boxes.map((b, i) => (
            <rect key={i} x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none"
              stroke={extraPersons.has(i) ? "var(--critical)" : "#ffffff"} strokeOpacity={0.85} strokeWidth={sw * 0.75} />
          ))}
          {!person.used_fallback && (
            <rect x={person.bbox[0]} y={person.bbox[1]} width={person.bbox[2] - person.bbox[0]} height={person.bbox[3] - person.bbox[1]}
              fill="none" stroke="var(--accent)" strokeWidth={sw} strokeDasharray={`${6 * scale} ${4 * scale}`} />
          )}
        </>
      )}

      {layers.body && only == null && person.keypoints.map((kps, i) =>
        kps.length === 17 ? (
          <g key={i}>
            {BODY_BONES.map(([a, b]) =>
              kps[a][2] >= 0.5 && kps[b][2] >= 0.5 ? (
                <line key={`${a}-${b}`} x1={kps[a][0]} y1={kps[a][1]} x2={kps[b][0]} y2={kps[b][1]}
                  stroke="#ffffff" strokeOpacity={0.9} strokeWidth={sw * 1.25} strokeLinecap="round"
                  style={{ filter: "drop-shadow(0 0 1px rgba(0,0,0,0.8))" }} />
              ) : null,
            )}
            {[9, 10].map((w) => kps[w][2] >= 0.5 ? (
              <circle key={w} cx={kps[w][0]} cy={kps[w][1]} r={4 * scale} fill="#ffffff" stroke="#000" strokeWidth={scale} />
            ) : null)}
          </g>
        ) : null,
      )}

      {layers.artifacts && only == null && artifacts.boxes.map((b, i) => (
        <g key={`art-${i}`}>
          <rect x={b.box[0]} y={b.box[1]} width={b.box[2] - b.box[0]} height={b.box[3] - b.box[1]} fill="none"
            stroke="var(--critical)" strokeWidth={sw * 1.25} rx={3 * scale} />
          <text x={b.box[0]} y={b.box[1] - 4 * scale} fontSize={11 * scale} fontWeight={700} fill="#ffffff"
            stroke="var(--critical)" strokeWidth={3 * scale} paintOrder="stroke">
            {ARTIFACT_LABEL[b.cls] ?? b.cls} {Math.round(b.score * 100)}%
          </text>
        </g>
      ))}

      {hands.hands.map((h) => {
        if (only != null && h.index !== only) return null;
        if (h.landmarks.length < 21) return null;
        const bad = new Set(h.failed_fingers);
        const fc = h.finger_count;
        const handFlagged = flaggedHands.has(h.index);
        return (
          <g key={h.index}>
            {layers.count && only != null && fc?.status === "counted" && fc.scan_lines.map((l, i) => (
              <line key={`scan-${i}`} x1={l[0][0]} y1={l[0][1]} x2={l[1][0]} y2={l[1][1]}
                stroke="#ff4fd8" strokeOpacity={0.8} strokeWidth={sw * 0.5} />
            ))}
            {layers.count && fc?.status === "counted" && fc.outline.length > 2 && (
              <polygon points={fc.outline.map((p) => p.join(",")).join(" ")} fill="none"
                stroke={fc.extra_finger ? "var(--critical)" : "#ffd84d"} strokeWidth={sw * 0.75}
                strokeDasharray={`${4 * scale} ${3 * scale}`} />
            )}
            {layers.hands && HAND_BONES.map(([a, b]) => {
              const finger = FINGER_OF_BONE[`${a}-${b}`];
              const red = finger && bad.has(finger);
              return (
                <line key={`${a}-${b}`} x1={h.landmarks[a][0]} y1={h.landmarks[a][1]} x2={h.landmarks[b][0]} y2={h.landmarks[b][1]}
                  stroke={red ? "var(--critical)" : "var(--accent)"} strokeWidth={sw * (red ? 1.4 : 1)} strokeLinecap="round" />
              );
            })}
            {layers.hands && h.landmarks.map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r={1.8 * scale} fill="#ffffff" stroke="var(--accent)" strokeWidth={scale * 0.8} />
            ))}
            {layers.count && fc?.status === "counted" && fc.tips.map((p, i) => {
              const extra = fc.extra_finger && fc.tip_fingers[i] == null;
              return (
                <g key={i}>
                  <circle cx={p[0]} cy={p[1]} r={7 * scale} fill={extra ? "var(--critical)" : "#ffffff"}
                    stroke={extra ? "#ffffff" : "#000000"} strokeWidth={scale} />
                  <text x={p[0]} y={p[1]} dy="0.35em" textAnchor="middle" fontSize={9 * scale} fontWeight={700}
                    fill={extra ? "#ffffff" : "#000000"}>{i + 1}</text>
                </g>
              );
            })}
            {layers.hands && only == null && (
              <text x={h.landmarks[0][0]} y={h.landmarks[0][1] + 14 * scale} textAnchor="middle" fontSize={11 * scale} fontWeight={700}
                fill={handFlagged ? "var(--critical)" : "#ffffff"} stroke="#000" strokeWidth={2.5 * scale} paintOrder="stroke">
                Hand {h.index + 1}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}
