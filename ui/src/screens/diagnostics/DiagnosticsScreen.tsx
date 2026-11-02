import { useState } from "react";

import { dispositionView, finalRecordFor, ruleReceiptFor, shortCommit } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import { clockTime } from "../overview/format";
import "./diagnostics.css";

export interface DiagnosticsScreenProps { readonly model: InspectionModel }

const unavailable = <span className="unavailable">unavailable</span>;
const val = (v: unknown) => (v == null || v === "" ? unavailable : String(v));

function Sparkline({ values }: { values: readonly number[] }) {
  if (values.length < 2) return <p className="unavailable">Not enough rate samples for a chart.</p>;
  const max = Math.max(...values, 1e-9);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 300},${40 - (v / max) * 36}`).join(" ");
  return <svg viewBox="0 0 300 44" width="300" height="44" role="img" aria-label={`Processing rate, frames per second, max ${max.toFixed(1)}`}><polyline points={pts} fill="none" stroke="var(--neutral)" strokeWidth="2" /></svg>;
}

function downloadJson(name: string, data: unknown) {
  if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  URL.revokeObjectURL?.(url);
}

/** L4 diagnostics (engineer only): service identity, models, event log, raw JSON and operator records. */
export function DiagnosticsScreen({ model }: DiagnosticsScreenProps) {
  const { health, runner } = model;
  const [typeFilter, setTypeFilter] = useState("all");
  const vialIds = runner?.trayOrder ?? [];
  const [vialPick, setVialPick] = useState<string | null>(null);
  const vialId = vialPick ?? vialIds[0] ?? null;
  const log = [...(runner?.eventLog ?? [])].reverse().filter(e => typeFilter === "all" || e.type === typeFilter);
  const types = [...new Set((runner?.eventLog ?? []).map(e => e.type))];
  const finals = model.decisions?.finals ?? [];
  const rate = runner?.rate;
  const fp = health?.frozen_profile;
  const raw = vialId && runner ? { vial: runner.vials[vialId], rule_receipt: ruleReceiptFor(model.decisions, vialId), final_record: finalRecordFor(model.decisions, vialId) } : null;
  const rawText = JSON.stringify(raw, null, 2);
  const bundle = model.bundle === undefined ? "not fetched" : model.bundle === null ? "not exposed by this runner" : "exposed";
  return (
    <section className="screen diag" aria-label="Diagnostics">
      <h2>Diagnostics</h2>
      <div className="panel diag-panel" role="region" aria-label="Service">
        <h3>Service</h3>
        {health ? (
          <dl className="diag-dl">
            <dt>Service</dt><dd>{health.service}</dd>
            <dt>API version</dt><dd>{health.api_version}</dd>
            <dt>Status</dt><dd>{health.status}</dd>
            <dt>Pipeline commit</dt><dd><code>{health.pipeline_commit}</code></dd>
            <dt>Expected pin</dt><dd><code>{model.expectedCommit}</code> ({model.pinStatus})</dd>
            <dt>Test only</dt><dd>{String(health.test_only)}</dd>
            <dt>Inference</dt><dd>{String(health.inference)}</dd>
            <dt>Limits</dt><dd>{health.limits.length ? health.limits.join("; ") : "none reported"}</dd>
            <dt>Frozen profile</dt>
            <dd>{fp ? <>id {val(fp.profile_id)} · fps {val(fp.fps)} · vials {val(fp.vial_count)} · frames/vial {val(fp.frames_per_vial)} · resolution {fp.resolution ? fp.resolution.join("×") : unavailable} · recipe {val(fp.recipe_id)}</> : unavailable}</dd>
            {health.replay && <><dt>Replay disclosure</dt><dd>Recorded replay of session {health.replay.recorded_session_id} (commit {shortCommit(health.replay.recorded_service_commit)}, {health.replay.recorded_at}, {health.replay.source_label}). {health.replay.note}</dd></>}
          </dl>
        ) : <p className="unavailable">Health unavailable{model.healthError ? `: ${model.healthError}` : ""}.</p>}
      </div>
      <div className="panel diag-panel" role="region" aria-label="Models">
        <h3>Models</h3>
        {model.decisionsError && <p className="diag-error">Decisions fetch error: {model.decisionsError}</p>}
        <p>Evidence bundle: {bundle}</p>
        {finals.length === 0 ? <p className="unavailable">No decision receipts fetched.</p> : (
          <table><thead><tr><th>Vial</th><th>Model</th><th>Repository</th><th>Revision</th><th>Guard</th><th>Elapsed (s)</th></tr></thead><tbody>
            {finals.flatMap(f => f.models.map(m => (
              <tr key={`${f.vial_id}-${m.source_id}`}><td>{f.vial_id}</td><td>{m.source_id}</td><td>{val(m.repository)}</td><td>{m.revision ? shortCommit(m.revision) : unavailable}</td><td>{val(m.guard_status)}</td><td>{m.elapsed_seconds == null ? unavailable : m.elapsed_seconds.toFixed(1)}</td></tr>
            )))}
          </tbody></table>
        )}
        <dl className="diag-dl">
          <dt>Active model</dt><dd>{val(rate?.active_model_id)}</dd>
          <dt>GPU status</dt><dd>{val(rate?.gpu_status)}</dd>
          <dt>GPU free (MiB)</dt><dd>{rate?.gpu_free_mib == null ? "unmeasured" : rate.gpu_free_mib}</dd>
          <dt>Queue depth</dt><dd>{val(rate?.queue_depth)}</dd>
          <dt>Rate status</dt><dd>{val(rate?.status)}</dd>
          <dt>Processing rate</dt><dd><Sparkline values={(runner?.rateHistory ?? []).map(r => r.processing_fps)} /></dd>
        </dl>
      </div>
      <div className="panel diag-panel" role="region" aria-label="Event log">
        <h3>Event log</h3>
        <label>Type <select className="hmi-control" value={typeFilter} onChange={e => setTypeFilter(e.target.value)}>
          <option value="all">all</option>{types.map(t => <option key={t} value={t}>{t}</option>)}</select></label>
        <div className="diag-scroll">
          <table><thead><tr><th>Seq</th><th>Time</th><th>Type</th><th>Vial</th><th>Summary</th></tr></thead><tbody>
            {log.map(e => <tr key={e.sequence}><td>{e.sequence}</td><td>{clockTime(e.emitted_at)}</td><td>{e.type}</td><td>{e.vialId ?? "—"}</td><td>{e.summary}</td></tr>)}
          </tbody></table>
        </div>
      </div>
      <div className="panel diag-panel" role="region" aria-label="Raw JSON">
        <h3>Raw JSON per vial</h3>
        {vialId ? <>
          <label>Vial <select className="hmi-control" value={vialId} onChange={e => setVialPick(e.target.value)}>{vialIds.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
          <button type="button" className="hmi-button" onClick={() => { void navigator.clipboard?.writeText(rawText); }}>Copy</button>
          <pre className="diag-pre" tabIndex={0}>{rawText}</pre>
        </> : <p className="unavailable">No vials yet.</p>}
      </div>
      <div className="panel diag-panel" role="region" aria-label="Operator records">
        <h3>Access ledger / operator records</h3>
        <p>Stored in this browser only (localStorage). Not sent to the runner. Does not change decisions.</p>
        {model.operatorLog.length === 0 ? <p className="unavailable">No operator records.</p> : (
          <table><thead><tr><th>Time</th><th>Vial</th><th>Action</th><th>System</th><th>Operator</th><th>Reason</th><th>By</th></tr></thead><tbody>
            {model.operatorLog.map(r => <tr key={r.record_id}><td>{clockTime(r.recorded_at)}</td><td>{r.vial_id}</td><td>{r.action}</td><td>{dispositionView(r.system_disposition).label}</td><td>{r.operator_disposition ? dispositionView(r.operator_disposition).label : "—"}</td><td>{r.reason}</td><td>{r.operator}</td></tr>)}
          </tbody></table>
        )}
        <button type="button" className="hmi-button" onClick={() => downloadJson("operator-records.json", model.operatorLog)}>Download JSON</button>
      </div>
    </section>
  );
}
