import { useEffect, useRef, type KeyboardEvent } from "react";
import type { DemoData, VialViewModel } from "../data/adapter";

export function SidePanel({ data, vial, tab, onTab, onClose }: { readonly data: DemoData; readonly vial: VialViewModel; readonly tab: "history" | "model"; readonly onTab: (tab: "history" | "model") => void; readonly onClose: () => void }) {
  const historyRef = useRef<HTMLButtonElement>(null);
  const modelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => { (tab === "history" ? historyRef : modelRef).current?.focus(); }, []);
  const selectTab = (next: "history" | "model") => {
    onTab(next);
    (next === "history" ? historyRef : modelRef).current?.focus();
  };
  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const next = event.key === "Home" ? "history" : event.key === "End" ? "model" : event.key === "ArrowRight" || event.key === "ArrowLeft" ? (tab === "history" ? "model" : "history") : null;
    if (next) { event.preventDefault(); selectTab(next); }
  };
  return <aside className="side-panel" role="dialog" aria-label="Evidence panel" aria-modal="false" onKeyDown={event => { if (event.key === "Escape") { event.preventDefault(); onClose(); } }}>
    <div className="section-heading"><div><span className="eyebrow">Evidence drawer</span><h2>Recipe {vial.recipe.recipe_id}</h2></div><button type="button" className="icon-button" aria-label="Close evidence panel" onClick={onClose}>×</button></div>
    <div role="tablist" aria-label="Evidence views" className="tabs">
      <button ref={historyRef} id="evidence-history-tab" type="button" role="tab" aria-selected={tab === "history"} aria-controls="evidence-history-panel" tabIndex={tab === "history" ? 0 : -1} onClick={() => selectTab("history")} onKeyDown={handleTabKey}>History</button>
      <button ref={modelRef} id="evidence-model-tab" type="button" role="tab" aria-selected={tab === "model"} aria-controls="evidence-model-panel" tabIndex={tab === "model" ? 0 : -1} onClick={() => selectTab("model")} onKeyDown={handleTabKey}>Model</button>
    </div>
    {tab === "history" ? <div id="evidence-history-panel" role="tabpanel" aria-labelledby="evidence-history-tab" className="drawer-list">
      <p className="drawer-note">Reference records for selected recipe · run / vial / relative-time provenance</p>
      {vial.history.map(record => <article key={record.history_id}><span className={`decision-chip decision-chip--${record.decision}`}>{record.decision.replace("_", " ")}</span><h3>{record.run_id} · {record.vial_id} · {record.time_seconds.value.toFixed(1)} s</h3><p>{record.explanation.text}</p><small title={record.provenance.note}>{record.provenance.kind} · {record.provenance.source_id}</small></article>)}
    </div> : <div id="evidence-model-panel" role="tabpanel" aria-labelledby="evidence-model-tab" className="drawer-list"><div className="wp6-table-wrap"><table className="wp6-table"><caption>{data.wp6.disclaimer}</caption><thead><tr><th scope="col">Method</th><th scope="col">Result</th><th scope="col">Method note</th><th scope="col">Failure note</th><th scope="col">Provenance</th></tr></thead><tbody>{data.wp6.rows.map(row => <tr key={row.method}><th scope="row">{row.label}</th><td>{row.passed} / {row.total}</td><td>{row.method_note}</td><td>{row.failure_note ?? "—"}</td><td title={row.provenance.note}>{row.provenance.kind} · {row.provenance.source_id}</td></tr>)}</tbody></table></div><section className="follow-up" aria-label="Follow-up evidence"><h3>{data.wp6.inn67.label}</h3><p>{data.wp6.inn67.note}</p></section></div>}
  </aside>;
}
