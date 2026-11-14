import type { ClassifierOut, FacesOut, HandsOut, PersonOut } from "@/lib/api";
import { num, pct } from "@/lib/format";
import { Card, KV, LayerBadge } from "./ui";

export function PersonCard({ person }: { person: PersonOut }) {
  return (
    <Card title="1 · Person crop" aside={<LayerBadge status={person.status} />}>
      <dl>
        <KV k="People detected" v={person.num_persons} />
        <KV k="Region analysed" v={person.used_fallback ? "Full image" : "Largest person"} />
        {person.person_score != null && <KV k="Detector score" v={pct(person.person_score)} />}
      </dl>
      {person.note && <Note>{person.note}</Note>}
    </Card>
  );
}

export function ClassifierCard({ clf }: { clf: ClassifierOut }) {
  return (
    <Card title="4 · Real-vs-fake classifier" aside={<LayerBadge status={clf.status} />}>
      {clf.status === "ok" && clf.p_fake != null ? (
        <>
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-sm text-ink-2">Leans</span>
            <span className="text-sm font-semibold text-ink-1">{clf.label === "fake" ? "AI-generated" : "Real"}</span>
          </div>
          <ProbabilityBar pFake={clf.p_fake} />
          <dl className="mt-2">
            <KV k="Model run" v={<span className="font-mono text-xs">{clf.run_id}</span>} />
          </dl>
          <Note>
            Trained on one small dataset (AI-vs-Real). Expect mistakes on images that look different from
            it, such as video frames or heavily edited photos.
          </Note>
        </>
      ) : (
        <Note>{clf.note ?? "Classifier did not run."}</Note>
      )}
    </Card>
  );
}

/** Two-part bar: share of probability for "real" vs "AI-generated". Both parts
 *  are labelled with text, so color is never the only cue. */
function ProbabilityBar({ pFake }: { pFake: number }) {
  return (
    <div>
      <div className="flex h-2.5 gap-0.5 overflow-hidden rounded-full" aria-hidden="true">
        <div className="rounded-l-full bg-series-1" style={{ width: `${(1 - pFake) * 100}%` }} />
        <div className="rounded-r-full bg-series-2" style={{ width: `${pFake * 100}%` }} />
      </div>
      <div className="tnum mt-1 flex justify-between text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-series-1" aria-hidden="true" />
          Real {pct(1 - pFake)}
        </span>
        <span className="flex items-center gap-1.5">
          AI-generated {pct(pFake)}
          <span className="h-2 w-2 rounded-full bg-series-2" aria-hidden="true" />
        </span>
      </div>
    </div>
  );
}

export function FaceCard({ face, threshold }: { face: FacesOut; threshold: number | null }) {
  return (
    <Card title="2 · Face symmetry" aside={<LayerBadge status={face.status} />}>
      {face.faces.length === 0 && <Note>{face.note ?? "No face found."}</Note>}
      <ul className="space-y-3">
        {face.faces.map((f) => (
          <li key={f.index} className="text-sm">
            <div className="flex items-baseline justify-between">
              <span className="text-ink-2">Face {f.index + 1}</span>
              {f.skipped_reason ? (
                <span className="text-xs text-ink-3">skipped</span>
              ) : (
                <span className={`text-xs font-medium ${f.anomalous ? "text-critical-ink" : "text-good-ink"}`}>
                  {f.anomalous ? "✗ Asymmetric" : "✓ Within range"}
                </span>
              )}
            </div>
            {f.skipped_reason ? (
              <p className="text-xs text-ink-3">{f.skipped_reason}</p>
            ) : (
              <>
                <ScoreVsThreshold score={f.asymmetry_score} threshold={threshold} anomalous={f.anomalous} />
                <p className="tnum mt-1 text-xs text-ink-3">
                  Head tilt {num(f.roll_deg, 1)}° corrected · turn ratio {num(f.yaw_ratio, 2)}
                </p>
              </>
            )}
          </li>
        ))}
      </ul>
      {face.note && face.faces.length > 0 && <Note>{face.note}</Note>}
    </Card>
  );
}

/** Score on a 0 → 2×threshold scale, with the threshold marked. */
function ScoreVsThreshold({ score, threshold, anomalous }: { score: number | null; threshold: number | null; anomalous: boolean }) {
  if (score == null) return null;
  const max = threshold ? threshold * 2 : Math.max(score, 1);
  const w = Math.min(score / max, 1) * 100;
  return (
    <div className="mt-1.5">
      <div className="relative h-1.5 rounded-full bg-surface-2">
        <div className={`h-full rounded-full ${anomalous ? "bg-critical" : "bg-ink-3"}`} style={{ width: `${w}%` }} />
        {threshold != null && <div className="absolute -top-1 left-1/2 h-3.5 w-0.5 rounded bg-ink-1" aria-hidden="true" />}
      </div>
      <p className="tnum mt-1 text-xs text-ink-2">
        Asymmetry {num(score)} {threshold != null && <>· threshold {num(threshold, 2)}</>}
      </p>
    </div>
  );
}

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"];

export function HandsCard({ hands }: { hands: HandsOut }) {
  return (
    <Card title="3 · Hand anatomy" aside={<LayerBadge status={hands.status} />}>
      {hands.hands.length === 0 && <Note>{hands.note ?? "No hands found."}</Note>}
      <ul className="space-y-3">
        {hands.hands.map((h) => (
          <li key={h.index} className="text-sm">
            <div className="flex items-baseline justify-between">
              <span className="text-ink-2">
                Hand {h.index + 1}
                {h.handedness && <span className="text-ink-3"> · {h.handedness}</span>}
              </span>
              {h.skipped_reason ? (
                <span className="text-xs text-ink-3">skipped</span>
              ) : (
                <span className={`text-xs font-medium ${h.anomalous ? "text-critical-ink" : "text-good-ink"}`}>
                  {h.anomalous ? `✗ ${h.failed_fingers.length} implausible fingers` : "✓ Plausible"}
                </span>
              )}
            </div>
            {h.skipped_reason ? (
              <p className="text-xs text-ink-3">{h.skipped_reason}</p>
            ) : (
              <table className="tnum mt-1.5 w-full text-xs">
                <thead>
                  <tr className="text-left text-ink-3">
                    <th className="font-normal">Finger</th>
                    <th className="font-normal">Length</th>
                    <th className="font-normal">Bend back</th>
                    <th className="text-right font-normal">Check</th>
                  </tr>
                </thead>
                <tbody>
                  {FINGERS.map((name) => {
                    const f = h.fingers[name];
                    if (!f) return null;
                    return (
                      <tr key={name} className="border-t border-line">
                        <td className="py-0.5 capitalize text-ink-2">{name}</td>
                        <td className={f.length_ok ? "text-ink-1" : "text-critical-ink"}>{num(f.length_ratio, 2)}×</td>
                        <td className={f.hyperextension_ok ? "text-ink-1" : "text-critical-ink"}>
                          {f.max_hyperextension_deg == null ? "—" : `${num(f.max_hyperextension_deg, 0)}°`}
                        </td>
                        <td className={`text-right font-medium ${f.ok ? "text-good-ink" : "text-critical-ink"}`}>
                          {f.ok ? "✓" : "✗"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </li>
        ))}
      </ul>
      {hands.hands.length > 0 && (
        <Note>
          MediaPipe always fits a 5-finger skeleton, so extra fingers show up only as implausible lengths or
          bends. Length is relative to palm width.
        </Note>
      )}
    </Card>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mt-2 text-xs text-ink-3">{children}</p>;
}
