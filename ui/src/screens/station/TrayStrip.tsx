import { vialLabel, vialView } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";

export function TrayStrip({ model }: { readonly model: InspectionModel }) {
  const runner = model.runner;
  const ended = model.runPhase === "completed" || model.runPhase === "stopped" || model.runPhase === "failed";
  const ids = runner?.trayOrder ?? [];
  const waiting = Math.max(0, model.vialCount - ids.length);
  return (
    <section className="panel station-tray" aria-label="Tray">
      <p className="kicker station-tray__title">Tray</p>
      <div className="station-tray__slots">
        {ids.map((id, i) => {
          const vial = runner!.vials[id];
          const view = vialView(vial, { runEnded: ended, vialCount: model.vialCount });
          const selected = model.selectedVialId === id;
          return (
            <button key={id} type="button" aria-pressed={selected}
              aria-label={`Select tray position ${vial.position.tray_index + 1}, ${vialLabel(vial)}, ${view.label}`}
              className={`hmi-button station-slot station-slot--${view.tone}${selected ? " station-slot--selected" : ""}`}
              onClick={() => model.selectVial(id)}>
              <span className="station-slot__pos">{i + 1}</span>
              <span className={`state state--${view.tone}`}><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}</span>
            </button>
          );
        })}
        {Array.from({ length: Math.min(waiting, 2) }, (_, k) => (
          <div key={`wait-${k}`} className="station-slot station-slot--empty" aria-label={`Tray position ${ids.length + k + 1} ${ended ? "not reached" : "waiting"}`}>
            <span className="station-slot__pos">{ids.length + k + 1}</span><span className="unavailable">{ended ? "— not reached" : "○ waiting"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
