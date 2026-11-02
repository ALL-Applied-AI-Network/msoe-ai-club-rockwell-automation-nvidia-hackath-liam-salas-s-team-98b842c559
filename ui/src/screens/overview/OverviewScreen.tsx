import { counts, dispositionView, finalDisposition, formatPercent, vialDeltaE, vialLabel } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import { DeltaETrend } from "./DeltaETrend";
import { clockTime } from "./format";
import "./overview.css";

export interface OverviewScreenProps { readonly model: InspectionModel; readonly onOpenDetail: (vialId: string) => void }

function Tile({ label, value, tone, unit }: { label: string; value: string; tone?: "discard" | "amber"; unit?: string }) {
  return (
    <div className="overview-tile panel" role="group" aria-label={label}>
      <p className="kicker">{label}</p>
      <p className={`overview-numeral${tone ? ` overview-numeral--${tone}` : ""}`}>{value}{unit && <span className="overview-unit"> {unit}</span>}</p>
    </div>
  );
}

/** L1 line overview: session KPIs, ΔE trend and the latest results. */
export function OverviewScreen({ model, onOpenDetail }: OverviewScreenProps) {
  const runner = model.runner;
  const c = counts(runner);
  const fps = runner && runner.rateHistory.length ? runner.rateHistory.reduce((s, r) => s + r.processing_fps, 0) / runner.rateHistory.length : null;
  const recent = (runner?.trayOrder ?? []).map(id => runner!.vials[id]).filter(v => finalDisposition(v))
    .sort((a, b) => (b.decidedAt ?? b.firstSeenAt).localeCompare(a.decidedAt ?? a.firstSeenAt)).slice(0, 10);
  return (
    <section className="screen overview" aria-label="Line overview">
      <h2>Line overview</h2>
      <p className="overview-scope">This session · judged vials only (references not counted)</p>
      <div className="overview-tiles" role="region" aria-label="Session KPIs">
        <Tile label="Inspected" value={String(c.inspected)} />
        <Tile label="Discard rate" value={formatPercent(c.discardRate)} tone={c.discard > 0 ? "discard" : undefined} />
        <Tile label="Can't-assess rate" value={formatPercent(c.cantAssessRate)} tone={c.cantAssess > 0 ? "amber" : undefined} />
        <Tile label="Average processing rate" value={fps == null ? "—" : fps.toFixed(1)} unit={fps == null ? undefined : "frames/s"} />
        <Tile label="Keep" value={String(c.keep)} />
        <Tile label="Discard" value={String(c.discard)} tone={c.discard > 0 ? "discard" : undefined} />
      </div>
      <div className="panel overview-chart" role="region" aria-label="ΔE trend">
        <h3>ΔE trend</h3>
        <DeltaETrend model={model} onOpenDetail={onOpenDetail} />
      </div>
      <div className="panel overview-recent" role="region" aria-label="Recent results">
        <h3>Recent results</h3>
        {recent.length === 0 ? <p className="unavailable">No judged vials yet.</p> : (
          <table>
            <thead><tr><th>Time</th><th>Vial</th><th>State</th><th>ΔE</th><th><span className="overview-sr">Open</span></th></tr></thead>
            <tbody>
              {recent.map(v => {
                const view = dispositionView(finalDisposition(v)); const de = vialDeltaE(v, model.decisions).value;
                return (
                  <tr key={v.vialId}>
                    <td>{clockTime(v.decidedAt ?? v.firstSeenAt)}</td>
                    <td>{vialLabel(v)}</td>
                    <td><span className={`state state--${view.tone}`}><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}</span></td>
                    <td>{de == null ? <span className="unavailable">unavailable</span> : de.toFixed(1)}</td>
                    <td><button type="button" className="hmi-button" onClick={() => onOpenDetail(v.vialId)} aria-label={`Open ${vialLabel(v)}`}>Open</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
