import { DELTA_E_DISCARD_LIMIT, DELTA_E_KEEP_LIMIT, dispositionView, finalDisposition, vialDeltaE, vialLabel } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";

const W = 1200; const H = 360; const PAD = { l: 70, r: 20, t: 16, b: 52 };
const LO = 0.4; const HI = 60; const TICKS = [0.5, 1, 2, 5, 10, 20, 50];
const PLOT_BOTTOM = H - PAD.b - 34; // the "no ΔE" lane sits below this line

/** log10 axis: ΔE 0.4 to 60, so the 1.0 and 2.0 limits stay visible even when a value is ~30. */
function yOf(v: number): number {
  const c = Math.min(HI, Math.max(LO, v));
  const f = (Math.log10(c) - Math.log10(LO)) / (Math.log10(HI) - Math.log10(LO));
  return PLOT_BOTTOM - f * (PLOT_BOTTOM - PAD.t);
}

export interface DeltaETrendProps { readonly model: InspectionModel; readonly onOpenDetail: (vialId: string) => void }

export function DeltaETrend({ model, onOpenDetail }: DeltaETrendProps) {
  const runner = model.runner;
  const judged = (runner?.trayOrder ?? []).map(id => runner!.vials[id]).filter(v => finalDisposition(v));
  const recipe = model.runContext?.recipe_id ?? model.health?.frozen_profile?.recipe_id ?? null;
  const plotW = W - PAD.l - PAD.r;
  const xOf = (i: number) => PAD.l + ((i + 0.5) / Math.max(judged.length, 1)) * plotW;
  const yKeep = yOf(DELTA_E_KEEP_LIMIT); const yDisc = yOf(DELTA_E_DISCARD_LIMIT);
  const noLaneY = H - PAD.b - 12;
  return (
    <figure className="overview-trend" aria-label="ΔE trend">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="group" aria-label="ΔE per judged vial, log scale">
        <rect x={PAD.l} y={yKeep} width={plotW} height={PLOT_BOTTOM - yKeep} className="overview-band overview-band--keep" />
        <rect x={PAD.l} y={yDisc} width={plotW} height={yKeep - yDisc} className="overview-band overview-band--check" />
        <rect x={PAD.l} y={PAD.t} width={plotW} height={yDisc - PAD.t} className="overview-band overview-band--out" />
        <line x1={PAD.l} x2={PAD.l + plotW} y1={yDisc} y2={yDisc} className="overview-limit-line" />
        <text x={PAD.l + plotW - 6} y={yKeep + 18} textAnchor="end" className="overview-band-label">≤ 1.0 keep limit</text>
        <text x={PAD.l + plotW - 6} y={yDisc + 18} textAnchor="end" className="overview-band-label">1.0–2.0 check band</text>
        <text x={PAD.l + plotW - 6} y={PAD.t + 18} textAnchor="end" className="overview-band-label">&gt; 2.0 outside</text>
        {TICKS.map(t => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + plotW} y1={yOf(t)} y2={yOf(t)} className="overview-grid" />
            <text x={PAD.l - 8} y={yOf(t) + 5} textAnchor="end" className="overview-tick">{t}</text>
          </g>
        ))}
        <text x={16} y={(PAD.t + PLOT_BOTTOM) / 2} transform={`rotate(-90 16 ${(PAD.t + PLOT_BOTTOM) / 2})`} textAnchor="middle" className="overview-axis">ΔE76 (log scale)</text>
        <line x1={PAD.l} x2={PAD.l + plotW} y1={PLOT_BOTTOM + 6} y2={PLOT_BOTTOM + 6} className="overview-grid" />
        <text x={PAD.l - 8} y={noLaneY + 5} textAnchor="end" className="overview-tick">n/a</text>
        <text x={PAD.l + 8} y={noLaneY - 12} className="overview-band-label">ΔE unavailable</text>
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b + 4} className="overview-changeover" />
        <text x={PAD.l + 6} y={H - 8} className="overview-changeover-label">{recipe ? `Recipe ${recipe} start` : "Session start"}</text>
        <text x={PAD.l + plotW} y={H - 8} textAnchor="end" className="overview-axis">Judged vials in tray order →</text>
        {judged.map((vial, i) => {
          const { value } = vialDeltaE(vial, model.decisions);
          const view = dispositionView(finalDisposition(vial));
          const x = xOf(i); const y = value == null ? noLaneY : yOf(value);
          const label = `${vialLabel(vial)} · ${value == null ? "ΔE unavailable" : `ΔE ${value.toFixed(1)}`} · ${view.label}`;
          return (
            <g key={vial.vialId} className={`overview-dot overview-dot--${view.tone}`} role="button" tabIndex={0} aria-label={label}
              onClick={() => onOpenDetail(vial.vialId)}
              onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpenDetail(vial.vialId); } }}>
              <title>{label}</title>
              <circle cx={x} cy={y} r={24} fill="transparent" />
              <circle cx={x} cy={y} r={11} className="overview-dot-mark" />
              {view.tone !== "keep" && <text x={x} y={y + 5} textAnchor="middle" className="overview-dot-icon">{view.icon}</text>}
            </g>
          );
        })}
        {judged.length === 0 && <text x={PAD.l + plotW / 2} y={(PAD.t + PLOT_BOTTOM) / 2} textAnchor="middle" className="overview-empty">No judged vials yet.</text>}
      </svg>
      <figcaption>Dashed line: recipe changeover. Carryover risk is highest right after a changeover. Log scale; values outside 0.4–60 are clamped to the edge.</figcaption>
    </figure>
  );
}
