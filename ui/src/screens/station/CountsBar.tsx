import { counts, formatPercent } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";

export function CountsBar({ model }: { readonly model: InspectionModel }) {
  const c = counts(model.runner);
  const item = (label: string, value: string, cls = "") => (
    <div className={`station-count ${cls}`}><span className="station-count__value">{value}</span><span className="station-count__label">{label}</span></div>
  );
  return (
    <section className="panel station-counts" aria-label="Counts">
      {item("Inspected", String(c.inspected))}
      {item("Keep", String(c.keep))}
      {item("Discard", String(c.discard), "station-count--discard")}
      {item("Can't assess", String(c.cantAssess), "station-count--cant")}
      {item("Discard rate", formatPercent(c.discardRate))}
      <p className="station-counts__note">This session, judged vials only (reference vials are not counted).</p>
    </section>
  );
}
