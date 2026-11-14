import type { ReactNode } from "react";
import type { LayerStatus, Verdict } from "@/lib/api";

export function Card({ title, aside, children, className = "" }: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`min-w-0 rounded-xl border border-line bg-surface-1 p-4 ${className}`}>
      {(title || aside) && (
        <header className="mb-3 flex items-baseline justify-between gap-3">
          {title && <h2 className="text-sm font-semibold text-ink-1">{title}</h2>}
          {aside && <div className="text-xs text-ink-3">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

const LAYER_STYLE: Record<LayerStatus, { cls: string; label: string }> = {
  ok: { cls: "border-good/40 text-good-ink", label: "Ran" },
  skipped: { cls: "border-line-strong text-ink-2", label: "Skipped" },
  unavailable: { cls: "border-warning/60 text-warning-ink", label: "Unavailable" },
  error: { cls: "border-critical/50 text-critical-ink", label: "Error" },
};

export function LayerBadge({ status }: { status: LayerStatus }) {
  const s = LAYER_STYLE[status];
  return <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${s.cls}`}>{s.label}</span>;
}

export const VERDICT_STYLE: Record<Verdict, { soft: string; ink: string; bar: string; icon: ReactNode }> = {
  "Verified Safe": { soft: "bg-good-soft", ink: "text-good-ink", bar: "bg-good", icon: <CheckIcon /> },
  "Needs Review": { soft: "bg-warning-soft", ink: "text-warning-ink", bar: "bg-warning", icon: <AlertIcon /> },
  "Anomaly Detected": { soft: "bg-critical-soft", ink: "text-critical-ink", bar: "bg-critical", icon: <XIcon /> },
};

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
      <path d="M7.5 12.5l3 3 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function AlertIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
      <path d="M12 3l10 18H2L12 3z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M12 10v5M12 18v.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
function XIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
      <path d="M9 9l6 6M15 9l-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface-1 p-4">
      <div className="text-xs text-ink-3">{label}</div>
      <div className="tnum mt-1 text-2xl font-semibold tracking-tight text-ink-1">{value}</div>
      {sub && <div className="tnum mt-1 text-xs text-ink-2">{sub}</div>}
    </div>
  );
}

export function KV({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-4 py-1 text-sm">
      <dt className="text-ink-2">{k}</dt>
      <dd className="tnum text-right text-ink-1">{v}</dd>
    </div>
  );
}
