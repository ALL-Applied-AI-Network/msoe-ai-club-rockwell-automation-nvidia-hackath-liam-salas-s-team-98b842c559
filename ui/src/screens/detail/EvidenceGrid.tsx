import { labToHex } from "../../model/selectors";
import type { VialState } from "../../runner/contract";
import type { InspectionModel } from "../../model/useInspection";
import { BAND_TEXT, bandOf, bundleImageId, bundleLab, deltaEFor, fmt1, fmtLab, stagesFor, streamedLab, vialAtPosition, type StageRef } from "./detailData";

const COLUMNS = [{ position: 1, name: "Vial 1", role: "target" }, { position: 2, name: "Vial 2", role: "reference" }, { position: 3, name: "Vial 3", role: "reference" }] as const;

function Cell({ model, vial, stage, position, role }: { model: InspectionModel; vial: VialState; stage: StageRef; position: number; role: "target" | "reference" }) {
  const name = `Vial ${position}`;
  const imageId = bundleImageId(model.bundle, stage.id, position);
  const src = imageId ? model.imageUrl(imageId) : null;
  const bLab = bundleLab(model.bundle, stage.id, position);
  const sLab = bLab ? null : streamedLab(vialAtPosition(model, vial, position), stage);
  const lab = bLab ?? sLab;
  const de = role === "target" ? deltaEFor(model, vial.vialId, stage.id) : null;
  const worst = de ? Math.max(...[de.v2, de.v3].filter((x): x is number => x != null)) : null;
  const tone = worst == null ? "" : bandOf(worst) === "discard" ? "detail-cell--discard" : bandOf(worst) === "check" ? "detail-cell--check" : "";
  return (
    <td className={`detail-cell ${tone}`} data-testid={`cell-${stage.label}-${position}`} data-outside-limits={worst != null && bandOf(worst) === "discard" ? "true" : undefined}>
      {src ? <img className="detail-crop" src={src} alt={`${name} end of ${stage.label} crop`} /> : <div className="detail-crop detail-crop--missing">Crop not exposed by this runner</div>}
      {lab ? (
        <div className="detail-swatch-row">
          <span className="detail-swatch" style={{ background: labToHex(lab) }} aria-hidden="true" />
          <span><span className="detail-lab">{fmtLab(lab)}</span><br /><span className="source-tag source-tag--measured">{bLab ? "measured · bundle median" : "measured · streamed frame"}</span></span>
        </div>
      ) : <p className="unavailable">Colour not available</p>}
      {role === "reference" ? <p className="detail-role">reference</p> : de ? (
        <ul className="detail-de">
          {([["Vial 2", de.v2], ["Vial 3", de.v3]] as const).map(([n, v]) => (
            <li key={n}>ΔE76 to {n}: <strong>{fmt1(v)}</strong> {v != null && <span className={`detail-band detail-band--${bandOf(v)}`}>{BAND_TEXT[bandOf(v)]}</span>}</li>
          ))}
          <li><span className="source-tag">{de.origin}</span></li>
        </ul>
      ) : <p className="unavailable">Rule ΔE not available from this runner</p>}
    </td>
  );
}

export function EvidenceGrid({ model, vial }: { model: InspectionModel; vial: VialState }) {
  const stages = stagesFor(model, vial);
  return (
    <section className="panel detail-section" aria-label="Evidence grid">
      <p className="kicker">Evidence grid · end-of-stage crops</p>
      {stages.length === 0 ? <p className="unavailable">No stages reported yet for this vial.</p> : (
        <table className="detail-grid">
          <thead><tr><th scope="col">Stage</th>{COLUMNS.map(c => <th scope="col" key={c.position}>{c.name} ({c.role})</th>)}</tr></thead>
          <tbody>
            {stages.map(stage => (
              <tr key={stage.id}>
                <th scope="row">{stage.label}</th>
                {COLUMNS.map(c => <Cell key={c.position} model={model} vial={vial} stage={stage} position={c.position} role={c.role} />)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="detail-caption">ΔE76 = CIELAB distance between vial medians at the same stage; limits 1.0 / 2.0. Colour swatches approximate the measured colour on screen; they are not a calibrated display.</p>
    </section>
  );
}
