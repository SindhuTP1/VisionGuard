import type { AnalysisResponse, ArtifactsOut, BodyOut, ClassifierOut, FacesOut, Finding, HandOut, HandsOut, PersonOut } from "@/lib/api";
import { ARTIFACT_LABEL, FINDING_LABEL, num, pct } from "@/lib/format";
import { HandZoom } from "./HandZoom";
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

export function HandsCard({ hands, report, src }: { hands: HandsOut; report: AnalysisResponse; src: string }) {
  return (
    <Card title="3 · Hand anatomy" aside={<LayerBadge status={hands.status} />}>
      {hands.hands.length === 0 && <Note>{hands.note ?? "No hands found."}</Note>}
      <ul className="space-y-4">
        {hands.hands.map((h) => (
          <li key={h.index} className="text-sm">
            <div className="flex items-baseline justify-between">
              <span className="text-ink-2">
                Hand {h.index + 1}
                {h.handedness && <span className="text-ink-3"> · {h.handedness}</span>}
              </span>
              {h.skipped_reason ? (
                <span className="text-xs text-ink-3">skeleton skipped</span>
              ) : (
                <span className={`text-xs font-medium ${h.anomalous ? "text-critical-ink" : "text-good-ink"}`}>
                  {h.anomalous ? `✗ ${h.failed_fingers.length} implausible fingers` : "✓ Skeleton plausible"}
                </span>
              )}
            </div>
            <div className="mt-2 flex flex-wrap items-start gap-3">
              <HandZoom src={src} report={report} hand={h} size={150} />
              <FingerCountSummary hand={h} />
            </div>
            {h.skipped_reason ? (
              <p className="mt-1 text-xs text-ink-3">{h.skipped_reason}</p>
            ) : (
              <details className="mt-2">
              <summary className="cursor-pointer text-xs text-ink-2">Per-finger skeleton checks</summary>
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
              </details>
            )}
          </li>
        ))}
      </ul>
      {hands.hands.length > 0 && (
        <Note>
          MediaPipe always fits a 5-finger skeleton, so fingers are also counted independently from the hand&apos;s
          outline (yellow) and from the seams between touching fingers along the pink lines in the zoom.
          Numbered dots are fingertips found on the outline, and red ones aren&apos;t explained by the skeleton.
        </Note>
      )}
    </Card>
  );
}

function FingerCountSummary({ hand }: { hand: HandOut }) {
  const fc = hand.finger_count;
  if (!fc) return null;
  if (fc.status !== "counted") {
    return (
      <div className="min-w-0 flex-1 text-xs">
        <p className="font-medium text-ink-2">Fingers: can&apos;t count</p>
        <p className="mt-0.5 text-ink-3">{fc.reason}</p>
      </div>
    );
  }
  return (
    <div className="min-w-0 flex-1 text-xs">
      <p className={`text-2xl font-semibold ${fc.extra_finger ? "text-critical-ink" : "text-ink-1"}`}>
        {fc.count} <span className="text-sm font-normal text-ink-2">finger{fc.count === 1 ? "" : "s"} counted</span>
      </p>
      {fc.extra_finger ? (
        <p className="mt-1 font-medium text-critical-ink">✗ More than 5: possible extra finger</p>
      ) : (
        <p className="mt-1 text-good-ink">✓ No extra finger seen</p>
      )}
      <dl className="tnum mt-2 space-y-0.5 text-ink-3">
        <div>Outline fingertips: {fc.outline_count ?? "—"}</div>
        <div>Finger row + thumb: {fc.scan_count ?? "— (hand not fully open)"}</div>
      </dl>
      <p className="mt-1 text-ink-3">Fewer than 5 is normal when fingers are bent or hidden.</p>
    </div>
  );
}

export function BodyCard({ body, hands }: { body: BodyOut; hands: HandsOut }) {
  return (
    <Card title="5 · Body (hands per person)" aside={<LayerBadge status={body.status} />}>
      {body.persons.length === 0 && <Note>{body.note ?? "No body pose found."}</Note>}
      <ul className="space-y-2 text-sm">
        {body.persons.map((p) => (
          <li key={p.index}>
            <div className="flex items-baseline justify-between">
              <span className="text-ink-2">Person {p.index + 1}</span>
              <span className={`text-xs font-medium ${p.extra_hand || p.distorted_limb ? "text-critical-ink" : "text-good-ink"}`}>
                {p.extra_hand ? `✗ ${p.extra_hand_reason}` : p.distorted_limb ? "✗ Arm proportions implausible" : "✓ Plausible"}
              </span>
            </div>
            <p className="tnum text-xs text-ink-3">
              {p.hands.length} hand{p.hands.length === 1 ? "" : "s"} attached
              {p.hands.length > 0 && ` (hand ${p.hands.map((k) => k + 1).join(", ")})`}
              {p.arms.length > 0 && ` · arm ratio ${p.arms.map((a) => `${a.side} ${a.ratio.toFixed(2)}`).join(", ")}`}
            </p>
          </li>
        ))}
      </ul>
      {body.unattached_hands.length > 0 && (
        <Note>
          {body.unattached_hands.length} of {hands.num_hands} hand(s) couldn&apos;t be matched to a body (wrist hidden or
          out of frame), so they aren&apos;t counted.
        </Note>
      )}
    </Card>
  );
}

export function FindingsCard({ findings }: { findings: Finding[] }) {
  return (
    <Card title="What looks wrong" aside={`${findings.length} finding${findings.length === 1 ? "" : "s"}`}>
      {findings.length === 0 ? (
        <p className="text-sm text-ink-2">
          No anatomical problem found in the checks that could run (finger count, hand skeleton, hands per person,
          arm proportions, face symmetry).
        </p>
      ) : (
        <ul className="space-y-2">
          {findings.map((f, i) => (
            <li key={i} className="flex gap-3 rounded-lg border border-critical/30 bg-critical-soft px-3 py-2 text-sm">
              <span className="font-semibold text-critical-ink" aria-hidden="true">✗</span>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-ink-1">{FINDING_LABEL[f.kind] ?? f.kind}</p>
                <p className="text-ink-2">{f.message}</p>
              </div>
              <span className="tnum shrink-0 text-xs text-ink-3" title="Heuristic confidence of this geometric check">
                {pct(f.confidence, 0)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mt-2 text-xs text-ink-3">{children}</p>;
}

export function ArtifactsCard({ artifacts }: { artifacts: ArtifactsOut }) {
  return (
    <Card title="6 · Learned anatomy-error detector" aside={<LayerBadge status={artifacts.status} />}>
      {artifacts.status !== "ok" ? (
        <Note>{artifacts.note ?? "Detector did not run."}</Note>
      ) : artifacts.boxes.length === 0 ? (
        <p className="text-sm text-good-ink">✓ No malformed hands, faces or limbs detected</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {artifacts.boxes.map((b, i) => (
            <li key={i} className="flex justify-between">
              <span className="text-critical-ink">✗ {ARTIFACT_LABEL[b.cls] ?? b.cls}</span>
              <span className="tnum text-ink-2">{pct(b.score, 0)}</span>
            </li>
          ))}
        </ul>
      )}
      {artifacts.status === "ok" && (
        <Note>
          YOLOv8n trained on HADM (AI images where people boxed anatomy errors). It only reports boxes scoring at
          least {pct(artifacts.threshold, 0)}, a cutoff chosen on held-out data. Its measured accuracy is on the Metrics page.
        </Note>
      )}
    </Card>
  );
}
