"use client";

import { useEffect, useState } from "react";
import { api, type AnatomyMetricsResponse, type FingerCountEval, type RateBlock } from "@/lib/api";
import { pct } from "@/lib/format";
import { Card, Stat } from "./ui";

const ci = (c: [number, number] | null | undefined) => (c ? `95% CI ${pct(c[0])}–${pct(c[1])}` : undefined);

/** Results of the anatomy checks, read from runs/anatomy-<date>/*.json. */
export function AnatomyMetrics() {
  const [data, setData] = useState<AnatomyMetricsResponse | null>(null);
  useEffect(() => {
    api.anatomy().then(setData).catch(() => setData({ available: false, message: "Couldn't load anatomy metrics." }));
  }, []);
  if (!data) return null;
  if (!data.available || !data.files) {
    return <Card title="Anatomy checks"><p className="text-sm text-ink-2">{data.message}</p></Card>;
  }
  const { finger_count_tuning: tune, finger_count_holdout: hold, finger_count_test: test, hadm_eval: hadm } = data.files;

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Anatomy checks</h2>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          How well the extra-finger and extra-hand checks work, measured two ways: on <strong>real</strong> hand photos,
          where every alarm is false, and on <strong>AI images labelled by people</strong> (HADM), where we can see what
          gets caught. Read from <code className="text-xs">runs/{data.run_id}/</code>.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {test && (
          <Stat label="False &ldquo;extra finger&rdquo; on real hands" value={pct(test.flag_rate_per_counted_hand, 2)}
            sub={`${test.hands_flagged_extra_finger} of ${test.hands_counted} countable hands · ${ci(test.flag_rate_ci95)}`} />
        )}
        {test && (
          <Stat label="Real hands the counter can count" value={pct(test.hands_counted / Math.max(test.hands_detected, 1), 0)}
            sub={`${test.hands_counted} of ${test.hands_detected}; the rest are fists, grips or too small`} />
        )}
        {hadm && (
          <Stat label="Extra hand/arm images caught" value={pct(hadm.extra_hand_or_arm.recall)}
            sub={`${hadm.extra_hand_or_arm.detected_positives} of ${hadm.extra_hand_or_arm.positives} · ${ci(hadm.extra_hand_or_arm.recall_ci95)}`} />
        )}
        {hadm && (
          <Stat label="Malformed AI hands flagged" value={pct(hadm.malformed_hand.by_either.recall)}
            sub={`${hadm.malformed_hand.by_either.detected_positives} of ${hadm.malformed_hand.by_either.positives} detected malformed hands`} />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Real hands (HaGRID): false alarms" aside="any 6+ count is wrong">
          <table className="tnum w-full text-left text-sm">
            <thead className="text-xs text-ink-3">
              <tr><th className="py-1 font-normal">Set</th><th className="font-normal">Hands counted</th><th className="font-normal">Said 6+</th><th className="font-normal">Rate</th></tr>
            </thead>
            <tbody>
              {([["Tuning (thresholds set here)", tune], ["Held-out (1st check)", hold], ["Fresh test (final)", test]] as [string, FingerCountEval | undefined][])
                .filter(([, e]) => e)
                .map(([name, e]) => (
                  <tr key={name} className={`border-t border-line ${name.startsWith("Fresh") ? "font-semibold" : ""}`}>
                    <td className="py-1">{name}</td>
                    <td>{e!.hands_counted}</td>
                    <td>{e!.hands_flagged_extra_finger}</td>
                    <td>{pct(e!.flag_rate_per_counted_hand, 2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-ink-3">
            The held-out set exposed one bug (a raised thumb counted twice), which was then fixed. That&apos;s why the final
            number comes from a third, untouched set.
          </p>
        </Card>

        {hadm && (
          <Card title="AI images labelled by people (HADM)" aside={`${hadm.images.toLocaleString()} images`}>
            <table className="tnum w-full text-left text-sm">
              <thead className="text-xs text-ink-3">
                <tr><th className="py-1 font-normal">Check</th><th className="font-normal">Caught</th><th className="font-normal">False alarms*</th><th className="font-normal">Precision</th></tr>
              </thead>
              <tbody>
                {([
                  ["Extra hand / arm (per image)", hadm.extra_hand_or_arm],
                  ["6+ fingers (per hand)", hadm.malformed_hand.by_extra_finger],
                  ["Impossible finger shape (per hand)", hadm.malformed_hand.by_skeleton],
                  ["Either hand check (per hand)", hadm.malformed_hand.by_either],
                ] as [string, RateBlock][]).map(([name, r]) => (
                  <tr key={name} className="border-t border-line">
                    <td className="py-1">{name}</td>
                    <td>{pct(r.recall)} <span className="text-xs text-ink-3">({r.detected_positives}/{r.positives})</span></td>
                    <td>{pct(r.false_alarm_rate)}</td>
                    <td>{pct(r.precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-ink-3">
              * &ldquo;False alarm&rdquo; here means a flag on something people didn&apos;t label. Unlabelled AI hands can still be
              wrong, so this is an upper bound. Hand rates cover only the {hadm.malformed_hand.hands_detected} hands MediaPipe
              found, out of {hadm.malformed_hand.labelled_malformed_hand_boxes_total} labelled malformed-hand boxes.
            </p>
          </Card>
        )}
      </div>
    </section>
  );
}
