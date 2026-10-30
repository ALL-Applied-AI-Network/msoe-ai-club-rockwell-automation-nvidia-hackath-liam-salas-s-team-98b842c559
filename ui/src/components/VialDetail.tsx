import type { EvidenceRecord, VialViewModel } from "../data/adapter";
import { ProvenanceChip } from "./ProvenanceChip";

const lab = (value: VialViewModel["target_lab"]) =>
  `L* ${value.l.toFixed(2)} · a* ${value.a.toFixed(2)} · b* ${value.b.toFixed(2)}`;

export function VialDetail({ vial, highlighted, revealed }: { readonly vial: VialViewModel; readonly highlighted: EvidenceRecord["target"] | null; readonly revealed: boolean }) {
  const isHighlighted = (target: EvidenceRecord["target"]) => highlighted === target ? "true" : undefined;
  return (
    <section className="panel vial-detail" aria-label="Vial detail">
      <div className="section-heading"><div><span className="eyebrow">Inspection focus</span><h2>Vial {vial.slot} · Recipe {vial.recipe.recipe_id}</h2></div>{revealed ? <span className={`decision-chip decision-chip--${vial.decision.outcome}`}>{vial.decision.outcome.replace("_", " ").toUpperCase()}</span> : <span className="pending-chip">PENDING CUE</span>}</div>
      <div className="inspection-frame">
        <img src={vial.rgb_frame.src} alt={`Vial ${vial.slot} measured RGB frame`} data-evidence-highlight={isHighlighted("rgb_frame")} />
        <img className="mask-overlay" src={vial.predicted_mask.src} alt={`Vial ${vial.slot} predicted mask overlay`} data-evidence-highlight={isHighlighted("predicted_mask")} />
        <span className="frame-corner">RGB + MASK</span>
      </div>
      {revealed ? <><div className="measure-grid">
        <div data-evidence-highlight={isHighlighted("target_lab")}><span>Target L*a*b*</span><strong>{lab(vial.target_lab)}</strong><ProvenanceChip provenance={vial.target_lab.provenance} /></div>
        <div data-evidence-highlight={isHighlighted("reference_lab")}><span>Reference L*a*b*</span><strong>{lab(vial.reference_lab)}</strong><ProvenanceChip provenance={vial.reference_lab.provenance} /></div>
        <div><span data-evidence-highlight={isHighlighted("delta_e")}>Delta E {vial.delta_e.value.toFixed(2)}</span><strong>{vial.delta_e.value.toFixed(2)}</strong><ProvenanceChip provenance={vial.delta_e.provenance} /></div>
        <div data-evidence-highlight={isHighlighted("threshold")}><span>Threshold</span><strong>{vial.threshold.value.toFixed(2)}</strong><ProvenanceChip provenance={vial.threshold.provenance} /></div>
      </div>
      <div className="limit-row"><span className="eyebrow">Limit codes</span>{vial.decision.limit_codes.map(code => <code key={code}>{code}</code>)}</div></> : <div className="pending-detail" aria-label="Decision details pending"><strong>Measurement captured</strong><span>Decision-bearing values appear when this vial reaches its playback cue.</span></div>}
    </section>
  );
}
