"use client";

import { useEffect, useState } from "react";
import { api, type MetricsResponse, type RunMetrics, type RunSummary } from "@/lib/api";
import { num, pct } from "@/lib/format";
import { AccuracyBars, type AccuracyRow } from "./charts/AccuracyBars";
import { ConfusionMatrix } from "./charts/ConfusionMatrix";
import { LossChart } from "./charts/LossChart";
import { Card, KV, Stat } from "./ui";

export function MetricsView() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | undefined>(undefined);
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.runs().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .metrics(runId)
      .then((d) => { if (alive) { setData(d); setError(null); } })
      .catch((e: Error) => alive && setError(e.message));
    return () => { alive = false; };
  }, [runId]);

  const m = data?.metrics ?? null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Model metrics</h1>
          <p className="mt-1 max-w-3xl text-sm text-ink-2">
            Every number on this page is read from <code className="text-xs">{data?.source ?? "runs/<id>/metrics.json"}</code>,
            written by the training script from a real run. Nothing here is typed in by hand.
          </p>
        </div>
        {runs.filter((r) => r.has_metrics).length > 1 && (
          <label className="text-sm text-ink-2">
            Training run{" "}
            <select
              value={runId ?? data?.run_id ?? ""}
              onChange={(e) => setRunId(e.target.value)}
              className="ml-1 rounded-md border border-line bg-surface-1 px-2 py-1 font-mono text-xs text-ink-1"
            >
              {runs.filter((r) => r.has_metrics).map((r) => (
                <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
              ))}
            </select>
          </label>
        )}
      </div>

      {error && <div role="alert" className="rounded-xl border border-critical/40 bg-critical-soft p-4 text-sm text-critical-ink">{error}</div>}
      {data && !data.available && (
        <Card><p className="text-sm text-ink-2">{data.message}</p></Card>
      )}
      {!data && !error && <p className="text-sm text-ink-3">Loading metrics…</p>}
      {m && <RunReport m={m} />}
    </div>
  );
}

function RunReport({ m }: { m: RunMetrics }) {
  const t = m.test;
  const controlled = Boolean(m.preprocessing);

  const rows: AccuracyRow[] = [];
  if (t) rows.push({ key: "test", label: "Classifier on test set", detail: "Held-out images, never used in training.", accuracy: t.accuracy, ci: t.ci95?.accuracy, auc: t.roc_auc, n: t.n });
  if (m.scale_stress_test) rows.push({ key: "stress", label: "Classifier, all images resized to one scale", detail: "Every test image resized to a 512px short side. A big drop would mean the model relies on image scale.", accuracy: m.scale_stress_test.accuracy, ci: m.scale_stress_test.ci95?.accuracy, auc: m.scale_stress_test.roc_auc, n: m.scale_stress_test.n });
  if (m.shortcut_probe) rows.push({ key: "probe", label: "Baseline: image size & file size only", detail: `A logistic regression that never sees pixels (${m.shortcut_probe.features.join(", ")}).`, accuracy: m.shortcut_probe.accuracy, auc: m.shortcut_probe.roc_auc, reference: true });

  return (
    <div className="space-y-6">
      <ShortcutNotice m={m} controlled={controlled} />

      {t && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Test accuracy" value={pct(t.accuracy)} sub={t.ci95?.accuracy ? `95% CI ${pct(t.ci95.accuracy[0])}–${pct(t.ci95.accuracy[1])}` : `${t.n} images`} />
          <Stat label="ROC-AUC" value={num(t.roc_auc)} sub={t.ci95?.roc_auc ? `95% CI ${num(t.ci95.roc_auc[0])}–${num(t.ci95.roc_auc[1])}` : undefined} />
          <Stat label="F1 (AI-generated = positive)" value={num(t.f1)} sub={`precision ${num(t.precision)} · recall ${num(t.recall)}`} />
          {t.selective && (
            <Stat
              label={`Sent to review (< ${pct(t.selective.threshold, 0)} conf.)`}
              value={pct(t.selective.review_rate)}
              sub={`accuracy on the rest ${pct(t.selective.accuracy_on_covered)}`}
            />
          )}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {rows.length > 0 && (
          <Card title="How much of the accuracy is real?" aside="hover a bar for detail">
            <AccuracyBars rows={rows} />
          </Card>
        )}
        {t && (
          <Card title="Confusion matrix · test set" aside={`${t.n} images`}>
            <ConfusionMatrix labels={t.confusion_matrix.labels} matrix={t.confusion_matrix.rows_true_cols_pred} />
          </Card>
        )}
      </div>

      {m.history && (
        <Card
          title="Training curve"
          aside={m.training ? `${m.training.epochs_run} epochs · ${m.training.stopped_early ? "stopped early" : "ran to the limit"} · ${Math.round(m.training.seconds)} s on ${m.training.device}` : undefined}
        >
          <LossChart history={m.history} bestEpoch={m.training?.best_epoch ?? null} />
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {m.dataset && <DatasetCard d={m.dataset} />}
        <Card title="Run configuration">
          <dl>
            <KV k="Run" v={<span className="font-mono text-xs">{m.run_id}</span>} />
            {m.model && <KV k="Model" v={`${m.model.arch} · ${(m.model.params / 1e6).toFixed(2)}M params`} />}
            {m.train_config && (
              <>
                <KV k="Optimizer" v={`${String(m.train_config.optimizer)} · lr ${String(m.train_config.lr)} · wd ${String(m.train_config.weight_decay)}`} />
                <KV k="Batch / max epochs / patience" v={`${String(m.train_config.batch_size)} / ${String(m.train_config.epochs)} / ${String(m.train_config.early_stopping_patience)}`} />
              </>
            )}
            {m.training && <KV k="Kept checkpoint" v={`epoch ${m.training.best_epoch} (val loss ${num(m.training.best_val_loss)})`} />}
            <KV k="Preprocessing" v={m.preprocessing ? <span className="font-mono text-xs">{m.preprocessing.tag}</span> : "resize + center crop (original)"} />
          </dl>
          {m.preprocessing && (
            <p className="mt-2 text-xs text-ink-3">
              Train: {m.preprocessing.train}. Evaluate: {m.preprocessing.eval}.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}

function ShortcutNotice({ m, controlled }: { m: RunMetrics; controlled: boolean }) {
  if (!m.shortcut_probe) return null;
  const leak = m.shortcut_probe.accuracy >= 0.9;
  if (!leak) return null;
  return (
    <div className={`rounded-xl border p-4 text-sm ${controlled ? "border-warning/50 bg-warning-soft" : "border-critical/40 bg-critical-soft"}`}>
      <p className={`font-semibold ${controlled ? "text-warning-ink" : "text-critical-ink"}`}>
        {controlled ? "⚠ Read these numbers with the dataset's shortcut in mind" : "✗ This run can cheat: it saw image size"}
      </p>
      <p className="mt-1 text-ink-2">
        In this dataset, image size and file size alone separate real from AI-generated with{" "}
        <strong className="tnum text-ink-1">{pct(m.shortcut_probe.accuracy)}</strong> accuracy (AI images are all
        1024×1024, real ones are mostly ~640×440).{" "}
        {controlled ? (
          <>
            This run hides that from the model: every image is re-saved at the same JPEG quality and cropped
            at its original resolution with no resizing. The scale stress test below checks what&apos;s left.
            Accuracy on a different dataset is still unknown.
          </>
        ) : (
          <>
            This run resized whole images, so its accuracy may reflect image size rather than generation
            artefacts. A newer run with shortcut controls replaces it.
          </>
        )}
      </p>
    </div>
  );
}

function DatasetCard({ d }: { d: NonNullable<RunMetrics["dataset"]> }) {
  const classes = Object.keys(d.per_class);
  return (
    <Card title="Dataset" aside={<a className="underline" href={`https://huggingface.co/datasets/${d.dataset}`} target="_blank" rel="noreferrer">{d.dataset}</a>}>
      <table className="tnum w-full text-left text-sm">
        <thead className="text-xs text-ink-3">
          <tr>
            <th className="py-1 font-normal">Class</th>
            <th className="font-normal">Train</th>
            <th className="font-normal">Val</th>
            <th className="font-normal">Test</th>
            <th className="font-normal">Typical size</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c} className="border-t border-line">
              <td className="py-1">{c === "fake" ? "AI-generated" : "Real"}</td>
              <td>{d.split_counts.train?.[c] ?? 0}</td>
              <td>{d.split_counts.val?.[c] ?? 0}</td>
              <td>{d.split_counts.test?.[c] ?? 0}</td>
              <td>{Math.round(d.per_class[c].median_width)}×{Math.round(d.per_class[c].median_height)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">
        Sampled from {Object.entries(d.source_label_counts).map(([k, v]) => `${v.toLocaleString()} ${k === "fake" ? "AI-generated" : "real"}`).join(" and ")}{" "}
        images (the dataset card says it&apos;s balanced; it isn&apos;t). Images are split by storage block, so near-duplicate
        neighbours never span train and test. Revision <span className="font-mono">{d.revision.slice(0, 8)}</span>.
      </p>
    </Card>
  );
}
