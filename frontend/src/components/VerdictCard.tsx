import type { ArbitrationOut } from "@/lib/api";
import { ATTRIBUTION_LABEL, CLASS_LABEL, pct } from "@/lib/format";
import { VERDICT_STYLE } from "./ui";

export function VerdictCard({ arb, threshold }: { arb: ArbitrationOut; threshold: number | null }) {
  const s = VERDICT_STYLE[arb.verdict];
  const cls = arb.anomaly_class ? CLASS_LABEL[arb.anomaly_class] : "Unattributed";

  return (
    <section className={`rounded-xl border border-line p-4 ${s.soft}`}>
      <div className="flex flex-wrap items-start gap-3">
        <span className={s.ink}>{s.icon}</span>
        <div className="min-w-0 flex-1">
          <h2 className={`text-lg font-semibold ${s.ink}`}>{arb.verdict}</h2>
          <p className="text-sm text-ink-2">
            Class: <span className="font-medium text-ink-1">{cls}</span> · Decided by{" "}
            <span className="font-medium text-ink-1">{ATTRIBUTION_LABEL[arb.attribution] ?? arb.attribution}</span>
          </p>
        </div>
        <div className="text-right">
          <div className="tnum text-2xl font-semibold text-ink-1">{pct(arb.confidence)}</div>
          <div className="text-xs text-ink-3">confidence</div>
        </div>
      </div>

      <ConfidenceMeter value={arb.confidence} threshold={threshold} barClass={s.bar} />

      {arb.reasons.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm text-ink-2">
          {arb.reasons.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span aria-hidden="true" className="text-ink-3">•</span>
              <span className="min-w-0 break-words">{r}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** 0–100% track with the review threshold marked. The bar uses the verdict's
 *  status color, but the meaning is carried by the label and icon above it. */
function ConfidenceMeter({ value, threshold, barClass }: { value: number | null; threshold: number | null; barClass: string }) {
  if (value == null) {
    return <p className="mt-3 text-xs text-ink-3">No confidence available: no decision layer could run.</p>;
  }
  return (
    <div className="mt-4">
      <div
        className="relative h-2 rounded-full bg-surface-2"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        aria-label="Verdict confidence"
      >
        <div className={`h-full rounded-full ${barClass}`} style={{ width: `${value * 100}%` }} />
        {threshold != null && (
          <div className="absolute -top-1 h-4 w-0.5 rounded bg-ink-1" style={{ left: `${threshold * 100}%` }} aria-hidden="true" />
        )}
      </div>
      {threshold != null && (
        <div className="relative mt-1 h-4 text-[11px] text-ink-3">
          <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: `${threshold * 100}%` }}>
            {pct(threshold, 0)} review threshold
          </span>
        </div>
      )}
    </div>
  );
}
