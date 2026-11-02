import { useEffect, useRef, useState } from "react";

import { frameFromId, vialLabel } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import type { StageSample, VialState } from "../../runner/contract";
import { frameToVialFrame, normalizedToDisplayRect, roiToDisplayRect, type Rect, type Size } from "./geometry";

/** Latest history sample whose measurement frame is at or before the local frame; null if none processed yet. */
export function sampleAtFrame(vial: VialState, localFrame: number): StageSample | null {
  let best: StageSample | null = null; let bestFrame = -1;
  for (const s of vial.history) {
    const f = s.measurement ? frameFromId(s.measurement.measurement_id) : null;
    if (f != null && f <= localFrame && f >= bestFrame) { best = s; bestFrame = f; }
  }
  return best;
}

function useBoxSize(ref: React.RefObject<HTMLElement | null>): Size {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const read = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read); ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export function VideoPanel({ model }: { readonly model: InspectionModel }) {
  const wrap = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const box = useBoxSize(wrap);
  const [time, setTime] = useState({ t: 0, duration: 0 });
  const fps = model.fps ?? 10;
  const runEnded = model.runPhase === "completed" || model.runPhase === "stopped" || model.runPhase === "failed";

  useEffect(() => {
    const el = video.current; if (!el || !model.videoUrl) return;
    let raf = 0;
    const read = () => setTime(p => (p.t === el.currentTime && p.duration === el.duration ? p : { t: el.currentTime, duration: el.duration }));
    const loop = () => { read(); raf = requestAnimationFrame(loop); };
    const start = () => { cancelAnimationFrame(raf); if (typeof requestAnimationFrame === "function") raf = requestAnimationFrame(loop); };
    const stop = () => { cancelAnimationFrame(raf); read(); };
    el.addEventListener("timeupdate", read); el.addEventListener("seeked", read); el.addEventListener("loadedmetadata", read);
    el.addEventListener("play", start); el.addEventListener("pause", stop); el.addEventListener("ended", stop);
    return () => {
      cancelAnimationFrame(raf);
      for (const [n, f] of [["timeupdate", read], ["seeked", read], ["loadedmetadata", read], ["play", start], ["pause", stop], ["ended", stop]] as const) el.removeEventListener(n, f);
    };
  }, [model.videoUrl]);

  const runner = model.runner;
  const hasVideo = !!model.videoUrl;
  const vf = frameToVialFrame(time.t, fps, time.duration, model.vialCount, model.framesPerVial);
  const byIndex = runner?.trayOrder.map(id => runner.vials[id]).find(v => v.position.tray_index === vf.trayIndex);
  const current: VialState | null = hasVideo ? (byIndex ?? null) : (runner && model.selectedVialId ? runner.vials[model.selectedVialId] ?? null : null);
  const sample = current ? (hasVideo ? sampleAtFrame(current, vf.localFrame) : current.history.at(-1) ?? null) : null;

  const bundle = model.bundle;
  const frameSize = bundle?.frame_size ? { width: bundle.frame_size[0], height: bundle.frame_size[1] } : null;
  let roi: Rect | null = null; let roiSource = "";
  if (hasVideo) {
    if (bundle?.vial_roi_xyxy && bundle.frame_size) { roi = roiToDisplayRect(bundle.vial_roi_xyxy, bundle.frame_size, box); roiSource = "evidence bundle"; }
    else if (current?.tracker?.locator.status === "located" && current.tracker.locator.bbox_normalized && frameSize) { roi = normalizedToDisplayRect(current.tracker.locator.bbox_normalized, frameSize, box); roiSource = "tracker locator"; }
  }
  const finalExpected = bundle?.stages.reduce<number | null>((m, s) => (s.expected_cumulative_ml != null && (m == null || s.expected_cumulative_ml > m) ? s.expected_cumulative_ml : m), null) ?? null;
  const measuredMl = sample?.tracker.measured_volume_ml ?? null;
  const fillY = roi && finalExpected && measuredMl != null ? roi.y + roi.height * (1 - Math.min(1, Math.max(0, measuredMl / finalExpected))) : null;
  const totalStages = bundle?.stages.length || null;
  const tracker = sample?.tracker;

  return (
    <section className="panel station-video" aria-label="Live camera">
      <div className="station-video__stage" ref={wrap}>
        {hasVideo
          ? <video ref={video} src={model.videoUrl ?? undefined} muted controls playsInline className="station-video__el" />
          : <div className="station-video__placeholder"><p>{runner ? "Live camera preview not streamed by runner — upload a video file to play it locally with overlays." : "No session. Start or attach a session, then upload a video file to preview it with overlays."}</p></div>}
        {hasVideo && roi && roi.width > 0 && (
          <svg className="station-video__overlay" width={box.width} height={box.height} aria-hidden="true">
            <rect x={roi.x} y={roi.y} width={roi.width} height={roi.height} className="station-roi" />
            {fillY != null && <g><line x1={roi.x - 6} x2={roi.x + roi.width + 6} y1={fillY} y2={fillY} className="station-fill-line" /><text x={roi.x + roi.width + 10} y={fillY + 4} className="station-fill-text">measured fill (approx.)</text></g>}
          </svg>
        )}
        <div className="station-hud" aria-label="Stream box">
          <p className="station-hud__vial">{current ? vialLabel(current) : hasVideo ? `Tray position ${vf.trayIndex + 1}` : "No vial"}</p>
          {current ? (
            sample || !hasVideo ? (
              <>
                <p>Tank: <strong>{tracker?.tank_id?.toUpperCase() ?? "—"}</strong></p>
                <p>Fill step: <strong>{tracker?.fill_step_id ? `Stage ${tracker.fill_step_id}${totalStages ? ` of ${totalStages}` : ""}` : "—"}</strong></p>
                <p>Measured volume: <strong>{measuredMl != null ? `${measuredMl.toFixed(1)} mL` : "—"}</strong> <span className={`source-tag source-tag--${measuredMl != null ? tracker?.volume_provenance ?? "measured" : "unavailable"}`}>{measuredMl != null ? tracker?.volume_provenance ?? "measured" : "unavailable"}</span></p>
              </>
            ) : <p className="station-hud__note">{runEnded ? "No runner measurement for this frame" : "Not processed yet"}</p>
          ) : <p className="station-hud__note">{!hasVideo ? "No event for this vial" : runEnded ? "No runner measurement for this frame" : "Not processed yet"}</p>}
        </div>
        {hasVideo && (
          <div className="station-chips-overlay">
            <span className="station-note">Liquid mask: not streamed</span>
            {!roi && <span className="station-note">ROI geometry not provided by runner</span>}
            {roi && <span className="station-note">ROI outline: {roiSource}</span>}
          </div>
        )}
      </div>
    </section>
  );
}
