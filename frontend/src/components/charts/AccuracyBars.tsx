"use client";

import { useState } from "react";
import { pct } from "@/lib/format";
import { Tooltip, TipRow } from "./Tooltip";
import { useWidth } from "./useWidth";

export interface AccuracyRow {
  key: string;
  label: string;
  detail: string;
  accuracy: number;
  ci?: [number, number] | null;
  auc?: number | null;
  n?: number;
  reference?: boolean;   // a baseline, not the model: drawn in neutral gray
}

const ROW = 44;
const M = { top: 8, right: 64, bottom: 24, left: 8 };

/** Horizontal bars on a shared 0–100% axis with a 50% chance line.
 *  Every bar is directly labelled, and the CI shows as a whisker. */
export function AccuracyBars({ rows }: { rows: AccuracyRow[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const iw = width - M.left - M.right;
  const H = M.top + rows.length * ROW + M.bottom;
  const x = (v: number) => M.left + v * iw;
  const barH = 14;

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={H} role="img" aria-label="Accuracy by evaluation" className="block">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={M.top} y2={H - M.bottom} stroke={t === 0.5 ? "var(--line-strong)" : "var(--line)"} strokeDasharray={t === 0.5 ? "3 3" : undefined} />
            <text x={x(t)} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--ink-3)" className="tnum">
              {t === 0.5 ? "50% (chance)" : `${t * 100}%`}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const y0 = M.top + i * ROW;
          const by = y0 + 20;
          const fill = r.reference ? "var(--ink-3)" : "var(--series-1)";
          return (
            <g key={r.key} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={y0} width={width} height={ROW} fill="transparent" />
              <text x={M.left} y={y0 + 12} fontSize={12} fill="var(--ink-1)">
                {r.label}
              </text>
              <rect x={x(0)} y={by} width={Math.max(0, x(r.accuracy) - x(0))} height={barH} rx={4} fill={fill} opacity={hover == null || hover === i ? 1 : 0.55} />
              {r.ci && (
                <g stroke="var(--ink-1)" strokeWidth={1.5}>
                  <line x1={x(r.ci[0])} x2={x(r.ci[1])} y1={by + barH / 2} y2={by + barH / 2} />
                  <line x1={x(r.ci[0])} x2={x(r.ci[0])} y1={by + 3} y2={by + barH - 3} />
                  <line x1={x(r.ci[1])} x2={x(r.ci[1])} y1={by + 3} y2={by + barH - 3} />
                </g>
              )}
              <text x={x(1) + 8} y={by + barH / 2} dy="0.32em" fontSize={12} fontWeight={600} fill="var(--ink-1)" className="tnum">
                {pct(r.accuracy)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <Tooltip x={Math.min(x(rows[hover].accuracy), width - 40)} y={M.top + hover * ROW + 27} containerWidth={width}>
          <div className="mb-1 max-w-56 font-medium text-ink-1">{rows[hover].label}</div>
          <p className="mb-1 max-w-56 text-ink-3">{rows[hover].detail}</p>
          <TipRow label="Accuracy" value={pct(rows[hover].accuracy)} />
          {rows[hover].ci && <TipRow label="95% CI" value={`${pct(rows[hover].ci![0])} – ${pct(rows[hover].ci![1])}`} />}
          {rows[hover].auc != null && <TipRow label="ROC-AUC" value={rows[hover].auc!.toFixed(3)} />}
          {rows[hover].n != null && <TipRow label="Images" value={rows[hover].n} />}
        </Tooltip>
      )}
    </div>
  );
}
