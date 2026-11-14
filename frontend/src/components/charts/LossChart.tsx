"use client";

import { useState } from "react";
import type { EpochRecord } from "@/lib/api";
import { num, pct } from "@/lib/format";
import { Tooltip, TipRow } from "./Tooltip";
import { useWidth } from "./useWidth";

const SERIES = [
  { key: "train_loss", label: "Train loss", color: "var(--series-1)" },
  { key: "val_loss", label: "Validation loss", color: "var(--series-2)" },
] as const;

const H = 240;
const M = { top: 16, right: 112, bottom: 32, left: 44 };

/** Train vs validation loss per epoch. One y-axis (both are cross-entropy).
 *  The kept checkpoint (best validation loss) is marked. */
export function LossChart({ history, bestEpoch }: { history: EpochRecord[]; bestEpoch: number | null }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  if (history.length === 0) return <p className="text-sm text-ink-3">No training history recorded.</p>;

  const iw = width - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const epochs = history.map((h) => h.epoch);
  const e0 = Math.min(...epochs);
  const e1 = Math.max(...epochs);
  const { max: yMax, step } = niceScale(Math.max(...history.flatMap((h) => [h.train_loss, h.val_loss])));
  const x = (e: number) => M.left + (e1 === e0 ? iw / 2 : ((e - e0) / (e1 - e0)) * iw);
  const y = (v: number) => M.top + ih - (v / yMax) * ih;
  const ticks = Array.from({ length: Math.round(yMax / step) + 1 }, (_, i) => i * step);
  const tickDigits = step < 0.1 ? 2 : 1;
  const path = (k: (typeof SERIES)[number]["key"]) =>
    history.map((h, i) => `${i ? "L" : "M"}${x(h.epoch).toFixed(1)},${y(h[k]).toFixed(1)}`).join("");

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left + M.left;
    let best = 0;
    history.forEach((h, i) => { if (Math.abs(x(h.epoch) - px) < Math.abs(x(history[best].epoch) - px)) best = i; });
    setHover(best);
  };

  const last = history[history.length - 1];
  // Direct end labels: nudge apart if the two lines end close together.
  let ly = SERIES.map((s) => y(last[s.key]));
  if (Math.abs(ly[0] - ly[1]) < 14) {
    const mid = (ly[0] + ly[1]) / 2;
    ly = ly[0] <= ly[1] ? [mid - 7, mid + 7] : [mid + 7, mid - 7];
  }
  const h = hover != null ? history[hover] : null;

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-4 text-xs text-ink-2" aria-hidden="true">
        {SERIES.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        {bestEpoch != null && (
          <span className="flex items-center gap-1.5">
            <span className="h-3 border-l-2 border-dashed border-ink-3" />
            Kept checkpoint
          </span>
        )}
      </div>
      <div ref={ref} className="relative">
        <svg width={width} height={H} role="img" aria-label="Training and validation loss by epoch" className="block">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={M.left + iw} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
              <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--ink-3)" className="tnum">
                {t.toFixed(tickDigits)}
              </text>
            </g>
          ))}
          {epochs.map((e) => (
            <text key={e} x={x(e)} y={H - M.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--ink-3)" className="tnum">
              {e}
            </text>
          ))}
          <text x={M.left + iw / 2} y={H - 2} textAnchor="middle" fontSize={11} fill="var(--ink-3)">epoch</text>

          {bestEpoch != null && (
            <g>
              <line x1={x(bestEpoch)} x2={x(bestEpoch)} y1={M.top} y2={M.top + ih} stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="4 4" />
              <text x={x(bestEpoch) + 6} y={M.top + 10} fontSize={11} fill="var(--ink-2)">kept (epoch {bestEpoch})</text>
            </g>
          )}

          {hover != null && (
            <line x1={x(history[hover].epoch)} x2={x(history[hover].epoch)} y1={M.top} y2={M.top + ih} stroke="var(--line-strong)" strokeWidth={1} />
          )}

          {SERIES.map((s, si) => (
            <g key={s.key}>
              <path d={path(s.key)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {history.map((h, i) => (
                <circle
                  key={h.epoch}
                  cx={x(h.epoch)}
                  cy={y(h[s.key])}
                  r={hover === i ? 5 : 4}
                  fill={s.color}
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
              ))}
              <text x={x(last.epoch) + 10} y={ly[si]} dy="0.32em" fontSize={11} fill="var(--ink-2)">
                {s.label.replace(" loss", "")} {num(last[s.key], 3)}
              </text>
            </g>
          ))}

          <rect
            x={M.left}
            y={M.top}
            width={iw}
            height={ih}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {h && (
          <Tooltip x={x(h.epoch)} y={M.top + ih / 2} containerWidth={width}>
            <div className="mb-1 font-medium text-ink-1">
              Epoch {h.epoch}
              {h.epoch === bestEpoch && <span className="font-normal text-ink-3"> · kept</span>}
            </div>
            <TipRow swatch="var(--series-1)" label="Train loss" value={num(h.train_loss)} />
            <TipRow swatch="var(--series-2)" label="Val loss" value={num(h.val_loss)} />
            <TipRow label="Val accuracy" value={pct(h.val_accuracy)} />
            <TipRow label="Time" value={`${Math.round(h.seconds)} s`} />
          </Tooltip>
        )}
      </div>
      <details className="mt-2 text-xs">
        <summary className="cursor-pointer text-ink-2">Show as table</summary>
        <table className="tnum mt-2 w-full text-left">
          <thead className="text-ink-3">
            <tr>
              <th className="py-1 font-normal">Epoch</th>
              <th className="font-normal">Train loss</th>
              <th className="font-normal">Train acc</th>
              <th className="font-normal">Val loss</th>
              <th className="font-normal">Val acc</th>
              <th className="font-normal">Val AUC</th>
            </tr>
          </thead>
          <tbody>
            {history.map((r) => (
              <tr key={r.epoch} className={`border-t border-line ${r.epoch === bestEpoch ? "font-semibold" : ""}`}>
                <td className="py-1">{r.epoch}{r.epoch === bestEpoch && " (kept)"}</td>
                <td>{num(r.train_loss)}</td>
                <td>{pct(r.train_accuracy)}</td>
                <td>{num(r.val_loss)}</td>
                <td>{pct(r.val_accuracy)}</td>
                <td>{num(r.val_roc_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

/** Round tick step (1, 2 or 5 × 10^k) giving 3–6 ticks, and an axis max that is a whole number of steps. */
function niceScale(v: number) {
  if (!(v > 0)) return { max: 1, step: 0.25 };
  const raw = (v * 1.05) / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * p >= raw) ?? 10) * p;
  return { max: Math.ceil((v * 1.05) / step) * step, step };
}
