import type { DemoData, VialViewModel } from "../data/adapter";

export function SidePanel({ data, vial, tab, onTab, onClose }: { readonly data: DemoData; readonly vial: VialViewModel; readonly tab: "history" | "model"; readonly onTab: (tab: "history" | "model") => void; readonly onClose: () => void }) {
  return <aside className="side-panel" role="dialog" aria-label="Evidence panel" aria-modal="false">
    <div className="section-heading"><div><span className="eyebrow">Evidence drawer</span><h2>Recipe {vial.recipe.recipe_id}</h2></div><button type="button" className="icon-button" aria-label="Close evidence panel" onClick={onClose}>×</button></div>
    <div role="tablist" aria-label="Evidence views" className="tabs">
      <button type="button" role="tab" aria-selected={tab === "history"} onClick={() => onTab("history")}>History</button>
      <button type="button" role="tab" aria-selected={tab === "model"} onClick={() => onTab("model")}>Model</button>
    </div>
    {tab === "history" ? <div role="tabpanel" className="drawer-list">
      <p className="drawer-note">Reference records for selected recipe · run / vial / relative-time provenance</p>
      {vial.history.map(record => <article key={record.history_id}><span className={`decision-chip decision-chip--${record.decision}`}>{record.decision.replace("_", " ")}</span><h3>{record.run_id} · {record.vial_id} · {record.time_seconds.value.toFixed(1)} s</h3><p>{record.explanation.text}</p><small title={record.provenance.note}>{record.provenance.kind} · {record.provenance.source_id}</small></article>)}
    </div> : <div role="tabpanel" className="drawer-list"><p className="warning-copy">{data.wp6.disclaimer}</p>{data.wp6.rows.map(row => <article key={row.method}><h3>{row.label} <b>{row.passed} / {row.total}</b></h3><p>{row.method_note}</p>{row.failure_note && <small>{row.failure_note}</small>}<small title={row.provenance.note}>{row.provenance.kind} · {row.provenance.source_id}</small></article>)}<article><h3>{data.wp6.inn67.label}</h3><p>{data.wp6.inn67.note}</p></article></div>}
  </aside>;
}
