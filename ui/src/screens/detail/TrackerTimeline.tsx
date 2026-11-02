import { frameFromId, labToHex } from "../../model/selectors";
import type { VialState } from "../../runner/contract";
import type { InspectionModel } from "../../model/useInspection";
import { fmt1, samplesFor, stagesFor } from "./detailData";

export function TrackerTimeline({ model, vial }: { model: InspectionModel; vial: VialState }) {
  const stages = stagesFor(model, vial);
  return (
    <section className="panel detail-section" aria-label="Tracker timeline">
      <p className="kicker">Tracker timeline · fill steps</p>
      {stages.length === 0 ? <p className="unavailable">No tracker samples yet.</p> : (
        <ol className="detail-timeline">
          {stages.map(stage => {
            const samples = samplesFor(vial, stage);
            const measured = [...samples].reverse().find(s => s.tracker.measured_volume_ml != null)?.tracker.measured_volume_ml ?? stage.bundle?.measured_volume_ml ?? null;
            const expected = stage.bundle?.expected_cumulative_ml ?? null;
            const stream = stage.bundle?.stream_median_lab ?? null;
            return (
              <li key={stage.id} className="detail-step" aria-label={`${stage.label} step`}>
                <h3>{stage.label}</h3>
                <p>Volume <strong>{fmt1(measured)} mL</strong> <span className="source-tag source-tag--measured">measured</span></p>
                <p>{expected != null ? <>Expected <strong>{fmt1(expected)} mL</strong> cumulative <span className="source-tag source-tag--inferred">expected (recipe)</span></> : <span className="unavailable">Expected volume not available</span>}</p>
                {stream && <p><span className="detail-swatch" style={{ background: labToHex(stream) }} aria-hidden="true" /> stream colour (bundle)</p>}
                <ul className="detail-ticks" aria-label={`${stage.label} samples`}>
                  {samples.map(s => {
                    const m = s.measurement; const frame = m ? frameFromId(m.measurement_id) : null;
                    const bad = !m || m.status !== "measured";
                    return <li key={s.sequence} className={bad ? "detail-tick detail-tick--bad" : "detail-tick"}>{bad ? "? " : ""}{frame != null ? `F${frame}` : `#${s.sequence}`}{bad && <small> {m?.unavailable_reason ?? s.tracker.unavailable_reason ?? s.stage}</small>}</li>;
                  })}
                </ul>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
