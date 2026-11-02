import { useState } from "react";

import { dispositionView, finalDisposition, sourceAgreement, vialLabel, vialRole, vialView, type SourceAgreement } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import type { Disposition } from "../../runner/contract";
import { cantAssessMessage, isInsufficient, pickExplanation, unavailableCode } from "./plainMessage";

const CHOICES: readonly { readonly value: Disposition; readonly label: string }[] = [
  { value: "keep", label: "Keep" }, { value: "discard", label: "Discard" }, { value: "insufficient_evidence", label: "Can't assess" }
];

function AgreementChip({ s }: { readonly s: SourceAgreement }) {
  const view = dispositionView(s.disposition);
  const note = s.disposition == null ? "no vote" : s.admitted === false ? "not admitted" : null;
  return (
    <li className={`station-chip station-chip--${s.disposition && s.admitted !== false ? view.tone : "pending"}`}>
      <span className="station-chip__name">{s.name}</span>
      {s.disposition && s.admitted !== false
        ? <span className={`state state--${view.tone}`}><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}{s.confidence != null && <span className="station-chip__conf"> {s.confidence.toFixed(1)}</span>}</span>
        : <span className="unavailable">{note}</span>}
    </li>
  );
}

export function CurrentVialCard({ model, operatorId, onOpenDetail }: { readonly model: InspectionModel; readonly operatorId: string; readonly onOpenDetail: (id: string) => void }) {
  const runner = model.runner;
  const id = model.selectedVialId ?? runner?.trayOrder.at(-1) ?? null;
  const vial = runner && id ? runner.vials[id] : null;
  const [overriding, setOverriding] = useState(false);
  const [choice, setChoice] = useState<Disposition | null>(null);
  const [reason, setReason] = useState("");
  const [recordedAt, setRecordedAt] = useState<string | null>(null);

  if (!runner || !vial) {
    return <section className="panel station-card" aria-label="Current vial"><p className="kicker">Current vial</p><p className="station-card__empty">No vial yet. Start or attach a session to see results here.</p></section>;
  }
  const ended = model.runPhase === "completed" || model.runPhase === "stopped" || model.runPhase === "failed";
  const view = vialView(vial, { runEnded: ended, vialCount: model.vialCount });
  const disposition = finalDisposition(vial);
  const role = vialRole(vial, model.vialCount);
  const explanation = pickExplanation(vial);
  const code = unavailableCode(vial);
  const average = vial.average && disposition ? vial.average.scores[disposition as "keep" | "check" | "discard"] : undefined;
  const records = model.operatorLog.filter(r => r.vial_id === vial.vialId);
  const decisionId = vial.average?.decision_id ?? vial.decision?.decision_id ?? null;
  const operator = operatorId || "operator";

  const stamp = () => setRecordedAt(new Date().toLocaleTimeString("en-GB", { hour12: false }));
  const agree = () => {
    if (!disposition) return;
    model.recordOperatorAction({ vial_id: vial.vialId, decision_id: decisionId, system_disposition: disposition, action: "agree", operator_disposition: null, reason: "", operator });
    stamp();
  };
  const confirm = () => {
    if (!disposition || !choice || !reason.trim()) return;
    model.recordOperatorAction({ vial_id: vial.vialId, decision_id: decisionId, system_disposition: disposition, action: "override", operator_disposition: choice, reason: reason.trim(), operator });
    stamp(); setOverriding(false); setChoice(null); setReason("");
  };

  return (
    <section className="panel station-card" aria-label="Current vial">
      <p className="kicker">Current vial · {vialLabel(vial)} · {role === "reference" ? "reference" : "judged"}</p>
      <p className={`state state--${view.tone} station-state`} data-testid="big-state"><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}</p>
      {role === "reference" && !disposition && <p className="station-plain">Reference vial: used to judge vial {Math.floor(vial.position.tray_index / model.vialCount) * model.vialCount + 1}, not judged itself.</p>}
      {isInsufficient(vial)
        ? <p className="station-plain">{cantAssessMessage(code)} {code && <small className="station-code">({code})</small>}</p>
        : explanation && <p className="station-plain">{explanation.item.plain_language} <span className={`source-tag source-tag--${explanation.fallback ? "deterministic_fallback" : "inferred"}`}>{explanation.fallback ? "template text" : "model-written"}</span></p>}
      {disposition && (
        <>
          <div className="station-agreement">
            <p className="kicker">Agreement</p>
            <ul className="station-chips">{sourceAgreement(vial).map(s => <AgreementChip key={s.key} s={s} />)}</ul>
          </div>
          <p className="station-average">
            {average != null ? <>Average {average.toFixed(2)} → {dispositionView(disposition).label.toLowerCase()} <span className="source-tag source-tag--inferred">inferred</span> <small>equal-weight average</small></>
              : <span className="unavailable">Average unavailable</span>}
          </p>
        </>
      )}
      <div className="station-actions">
        <button type="button" className="hmi-button hmi-button--primary" disabled={!disposition} onClick={agree}>Agree</button>
        <button type="button" className="hmi-button" disabled={!disposition} onClick={() => setOverriding(true)}>Override…</button>
        <button type="button" className="hmi-button" onClick={() => onOpenDetail(vial.vialId)}>Detail</button>
      </div>
      {overriding && (
        <form className="station-override" onSubmit={e => { e.preventDefault(); confirm(); }} aria-label="Override decision">
          <fieldset><legend>What should this vial be?</legend>
            {CHOICES.map(c => (
              <label key={c.value} className="station-radio hmi-control"><input type="radio" name="override-disposition" checked={choice === c.value} onChange={() => setChoice(c.value)} /> {c.label}</label>
            ))}
          </fieldset>
          <label className="station-reason">Reason (required)<textarea value={reason} onChange={e => setReason(e.target.value)} rows={2} /></label>
          <div className="station-actions">
            <button type="submit" className="hmi-button hmi-button--primary" disabled={!choice || !reason.trim()}>Confirm override</button>
            <button type="button" className="hmi-button" onClick={() => setOverriding(false)}>Cancel</button>
          </div>
        </form>
      )}
      {recordedAt && <p className="station-recorded" role="status">Recorded locally at {recordedAt} — the system decision is unchanged.</p>}
      {records.length > 0 && (
        <ul className="station-records" aria-label="Operator records">
          {records.map(r => <li key={r.record_id}>{new Date(r.recorded_at).toLocaleTimeString("en-GB", { hour12: false })} · {r.operator} · {r.action}{r.operator_disposition ? ` → ${dispositionView(r.operator_disposition).label}` : ""}{r.reason ? ` · ${r.reason}` : ""} <span className="source-tag source-tag--measured">this browser</span></li>)}
        </ul>
      )}
    </section>
  );
}
