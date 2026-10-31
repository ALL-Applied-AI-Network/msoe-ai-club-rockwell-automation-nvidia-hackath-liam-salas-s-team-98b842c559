import type { DemoData } from "../data/adapter";

export function TopDownPanel({ data, selectedId, revealed, onSelect }: { readonly data: DemoData; readonly selectedId: string; readonly revealed: ReadonlySet<string>; readonly onSelect: (id: string) => void }) {
  return (
    <section className="panel top-down" aria-label="Top-down station view">
      <div className="section-heading"><div><span className="eyebrow">Physical station</span><h2>Top-down position map</h2></div><span className="status-dot">{data.top_down.state === "feed" ? "top-down feed connected" : "waiting for top-down feed"}</span></div>
      <div className="station-map">
        {data.top_down.state === "feed" && data.top_down.asset && <img className="top-down-feed" src={data.top_down.asset.src} alt="Top-down camera feed" />}
        <div className="station-rail" />
        {data.vials.map(vial => {
          const { bbox } = vial.top_down;
          const selected = selectedId === vial.vial_id;
          const isRevealed = revealed.has(vial.vial_id);
          const outcome = vial.decision.outcome.replace("_", " ");
          const delta = vial.delta_e ? vial.delta_e.value.toFixed(2) : "unavailable";
          const state = isRevealed ? `${outcome}, Delta E ${delta}, revealed` : "awaiting decision";
          return <button key={vial.vial_id} type="button" className={`station-position ${selected ? "is-selected" : ""} ${isRevealed ? `is-${vial.decision.outcome}` : ""}`} style={{ left: `${bbox.x * 100}%`, top: `${bbox.y * 100}%`, width: `${bbox.width * 100}%`, height: `${bbox.height * 100}%` }} aria-label={`Top-down position, slot ${vial.slot}, vial ${vial.vial_id}, ${state}, ${selected ? "selected" : "not selected"}`} aria-pressed={selected} onClick={() => onSelect(vial.vial_id)}><b>{vial.slot}</b><small>{isRevealed ? <>{outcome}<br />ΔE {delta}</> : "awaiting decision"}</small></button>;
        })}
      </div>
    </section>
  );
}
