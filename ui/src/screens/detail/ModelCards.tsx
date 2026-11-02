import { dispositionView, SOURCE_NAMES, SOURCE_ORDER, finalRecordFor, modelRecordFor, ruleReceiptFor, sourceAgreement, sourceKeyForModel, type SourceKey } from "../../model/selectors";
import type { VialState } from "../../runner/contract";
import type { InspectionModel } from "../../model/useInspection";
import { fmt1 } from "./detailData";

const Vote = ({ d }: { d: string | null }) => { const v = dispositionView(d); return <span className={`state state--${v.tone}`}><span className="state-shape" aria-hidden="true">{v.icon}</span>{v.label}</span>; };

function RuleCard({ model, vial }: { model: InspectionModel; vial: VialState }) {
  const rule = ruleReceiptFor(model.decisions, vial.vialId);
  const disposition = rule?.disposition ?? vial.decision?.disposition ?? null;
  return (
    <article className="detail-card" aria-label="Rule source">
      <h3>Rule <span className="source-tag source-tag--measured">deterministic</span></h3>
      {disposition ? <Vote d={disposition} /> : <p className="unavailable">No vote received</p>}
      {rule && <>
        <p>discard score <strong>{rule.discard_score == null ? "—" : rule.discard_score.toFixed(2)}</strong> <span className="source-tag">rule receipt</span></p>
        <ul className="detail-rows">{rule.stages.map(s => <li key={s.stage_id}>{s.stage_id.replace(/^dispense-\d+-/, "").toUpperCase()}: <Vote d={s.disposition} /></li>)}</ul>
        {rule.receipt_sha256 && <p className="detail-sha">receipt {rule.receipt_sha256.slice(0, 10)}</p>}
      </>}
    </article>
  );
}

function ModelCard({ model, vial, sk }: { model: InspectionModel; vial: VialState; sk: SourceKey }) {
  const agreement = sourceAgreement(vial).find(a => a.key === sk)!;
  const rec = modelRecordFor(model.decisions, vial.vialId, sk);
  const disposition = agreement.disposition ?? rec?.disposition ?? null;
  const confidence = agreement.confidence ?? rec?.confidence ?? null;
  const streamed = vial.explanations.find(e => sourceKeyForModel(e.producer.model_id) === sk);
  const fallback = rec?.explanation_status === "deterministic_fallback";
  const reasons = rec?.explanation_guard_reasons.join(", ") || "no reason recorded";
  return (
    <article className="detail-card" aria-label={`${SOURCE_NAMES[sk]} source`}>
      <h3>{SOURCE_NAMES[sk]}</h3>
      {!disposition ? <p className="unavailable">No vote received</p> : (<>
        <Vote d={disposition} />
        <p>confidence <strong>{confidence == null ? "—" : confidence.toFixed(2)}</strong>{rec?.elapsed_seconds != null && <> · took {Math.round(rec.elapsed_seconds)} s</>}</p>
        <p>{agreement.admitted == null ? "Admission unknown" : agreement.admitted ? "Admitted to average" : `Not admitted: ${agreement.admissionReason ?? "no reason"}`}
          {agreement.agreesWithRule != null && <> · <strong>{agreement.agreesWithRule ? "agrees with rule" : "disagrees with rule"}</strong></>}</p>
      </>)}
      {rec ? (
        <div className="detail-explain">
          {fallback ? <p><span className="source-tag source-tag--deterministic_fallback">Fallback (model text rejected: {reasons})</span></p>
            : rec.explanation_status === "model_written" ? <p><span className="source-tag source-tag--inferred">model-written</span></p> : null}
          <p>{rec.explanation ?? "No explanation recorded."}</p>
          {fallback && rec.model_explanation && <details><summary>Model's rejected text (engineering)</summary><p>{rec.model_explanation}</p></details>}
          <p className="detail-physics">{rec.physics_cited.length ? <>Physics cited: {rec.physics_cited.join("; ")}</> : <span className="unavailable">Physics fields not provided yet (VLM schema v2 pending)</span>}</p>
          {rec.observations.length > 0 && <ul className="detail-rows">{rec.observations.map(o => <li key={o.stage_id}>{o.stage_id.replace(/^dispense-\d+-/, "").toUpperCase()}: {o.visual_assessment}</li>)}</ul>}
          {rec.per_stage_comparison.length > 0 && <ul className="detail-rows">{rec.per_stage_comparison.map(c => <li key={c.stage_id}>{c.stage_id.replace(/^dispense-\d+-/, "").toUpperCase()} reliability: {c.reliability ?? "—"} · ΔE76 {fmt1(c.delta_e76_vial2)} / {fmt1(c.delta_e76_vial3)}</li>)}</ul>}
        </div>
      ) : streamed ? (
        <div className="detail-explain"><p><span className={`source-tag source-tag--${streamed.producer.source}`}>{streamed.producer.source}</span></p><p>{streamed.plain_language}</p></div>
      ) : null}
    </article>
  );
}

export function ModelCards({ model, vial }: { model: InspectionModel; vial: VialState }) {
  const final = finalRecordFor(model.decisions, vial.vialId);
  return (
    <section className="panel detail-section" aria-label="Model sources">
      <p className="kicker">Sources · equal-weight average{final?.valid_source_count != null ? ` of ${final.valid_source_count} admitted` : ""}</p>
      <div className="detail-cards">
        {SOURCE_ORDER.map(sk => sk === "rule" ? <RuleCard key={sk} model={model} vial={vial} /> : <ModelCard key={sk} model={model} vial={vial} sk={sk} />)}
      </div>
    </section>
  );
}
