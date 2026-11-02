import { cantAssessMessage, pickExplanation, unavailableCode } from "../../model/plainMessage";
import { dispositionView, finalDisposition, vialLabel, vialView } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import { EvidenceGrid } from "./EvidenceGrid";
import { ModelCards } from "./ModelCards";
import { TrackerTimeline } from "./TrackerTimeline";
import "./detail.css";

export interface VialDetailScreenProps { readonly model: InspectionModel; readonly vialId: string | null; readonly onSelectVial: (vialId: string) => void }

/** L3 vial detail: evidence grid, per-source cards and tracker timeline for one vial. */
export function VialDetailScreen({ model, vialId, onSelectVial }: VialDetailScreenProps) {
  const runner = model.runner;
  const vial = runner && vialId ? runner.vials[vialId] : undefined;
  if (!runner || !runner.trayOrder.length || !vial) {
    return <section className="screen detail-screen" aria-label="Vial detail"><div className="panel detail-section"><h2>Vial detail</h2><p className="unavailable">No vial to show yet. Start or attach an inspection session, then pick a vial.</p></div></section>;
  }
  const view = vialView(vial, { runEnded: model.runPhase !== "running" && model.runPhase !== "starting", vialCount: model.vialCount });
  const disposition = finalDisposition(vial);
  const avg = vial.average;
  const score = avg && avg.disposition !== "insufficient_evidence" ? avg.scores[avg.disposition] : null;
  const picked = pickExplanation(vial);
  const line = picked && !picked.fallback ? picked.item.plain_language
    : disposition === "insufficient_evidence" ? cantAssessMessage(unavailableCode(vial))
      : picked?.item.plain_language ?? (disposition ? `${dispositionView(disposition).label}.` : "Waiting for a decision.");
  return (
    <section className="screen detail-screen" aria-label="Vial detail">
      <header className="panel detail-header">
        <div className="detail-selector" role="group" aria-label="Select vial">
          {runner.trayOrder.map(id => (
            <button key={id} type="button" className="hmi-button" aria-pressed={id === vial.vialId} onClick={() => onSelectVial(id)}>{vialLabel(runner.vials[id])}</button>
          ))}
        </div>
        <div className={`state state--${view.tone} detail-state`}><span className="state-shape" aria-hidden="true">{view.icon}</span><span>{view.label}</span></div>
        <p className="detail-line">{line}</p>
        {avg && <p className="detail-avg">equal-weight average {score!.toFixed(2)} → {dispositionView(avg.disposition).label.toLowerCase()} <span className="source-tag">discard score</span></p>}
      </header>
      <EvidenceGrid model={model} vial={vial} />
      <ModelCards model={model} vial={vial} />
      <TrackerTimeline model={model} vial={vial} />
    </section>
  );
}
