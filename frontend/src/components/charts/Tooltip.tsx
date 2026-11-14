import type { ReactNode } from "react";

/** Floating tooltip positioned inside a `relative` chart container.
 *  Flips to the left of the pointer near the right edge. */
export function Tooltip({ x, y, containerWidth, children }: {
  x: number;
  y: number;
  containerWidth: number;
  children: ReactNode;
}) {
  const flip = x > containerWidth - 180;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-10 w-max min-w-44 max-w-64 rounded-lg border border-line bg-surface-1 px-3 py-2 text-xs shadow-lg"
      style={{ left: x, top: y, transform: `translate(${flip ? "calc(-100% - 12px)" : "12px"}, -50%)` }}
    >
      {children}
    </div>
  );
}

export function TipRow({ swatch, label, value }: { swatch?: string; label: ReactNode; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-0.5">
      <span className="flex items-center gap-1.5 text-ink-2">
        {swatch && <span className="h-2 w-2 rounded-full" style={{ background: swatch }} aria-hidden="true" />}
        {label}
      </span>
      <span className="tnum whitespace-nowrap font-medium text-ink-1">{value}</span>
    </div>
  );
}
