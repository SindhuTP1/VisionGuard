"use client";

import { useState } from "react";
import { pct } from "@/lib/format";

const NAMES: Record<string, string> = { real: "Real", fake: "AI-generated" };
// Sequential blue: light → dark by share of the true class (row-normalised).
const RAMP = ["var(--seq-100)", "var(--seq-200)", "var(--seq-300)", "var(--seq-400)", "var(--seq-500)", "var(--seq-600)", "var(--seq-700)"];

/** 2×2 confusion matrix as a real <table>: rows = true class, columns = predicted.
 *  Cells are shaded by row share; every cell shows its count and percentage. */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="overflow-x-auto">
      <table className="tnum mx-auto border-separate border-spacing-0.5 text-sm">
        <caption className="caption-bottom pt-2 text-xs text-ink-3">
          Rows: true class · Columns: predicted class · Shade: share of the true class
        </caption>
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} scope="col" className="px-2 pb-1 text-xs font-normal text-ink-3">
                Predicted {NAMES[l] ?? l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th scope="row" className="pr-2 text-right text-xs font-normal text-ink-3">
                  True {NAMES[labels[i]] ?? labels[i]}
                </th>
                {row.map((v, j) => {
                  const share = v / total;
                  const step = Math.min(RAMP.length - 1, Math.floor(share * RAMP.length));
                  const dark = step >= 4;
                  const id = `${i}-${j}`;
                  const correct = i === j;
                  return (
                    <td
                      key={j}
                      onPointerEnter={() => setHover(id)}
                      onPointerLeave={() => setHover(null)}
                      title={`True ${NAMES[labels[i]]}, predicted ${NAMES[labels[j]]}: ${v} of ${total} (${pct(share)})`}
                      className={`h-20 w-32 rounded-md text-center align-middle transition-shadow ${hover === id ? "ring-2 ring-ink-1" : ""}`}
                      style={{ background: RAMP[step], color: dark ? "#ffffff" : "#0b0b0b" }}
                    >
                      <div className="text-xl font-semibold">{v}</div>
                      <div className="text-xs opacity-80">
                        {pct(share)} · {correct ? "correct" : "wrong"}
                      </div>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
