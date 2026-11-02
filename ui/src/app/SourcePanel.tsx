import { type FormEvent, useState } from "react";

import type { RunnerSource } from "../runner/client";
import type { InspectionModel } from "../model/useInspection";

type SourceType = RunnerSource["type"] | "attach";

/** Source picker: upload a video (played locally for overlays), a V4L2 camera, RTSP, or attach to an existing session. */
export function SourcePanel({ model, operatorId, onOperatorId }: { readonly model: InspectionModel; readonly operatorId: string; readonly onOperatorId: (value: string) => void }) {
  const [sourceType, setSourceType] = useState<SourceType>("video");
  const [file, setFile] = useState<File | null>(null);
  const [device, setDevice] = useState("");
  const [rtspUrl, setRtspUrl] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [recipeId, setRecipeId] = useState("");
  const [trayId, setTrayId] = useState("");
  const [dispensePlan, setDispensePlan] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const canStart = model.pinStatus === "verified" || model.pinStatus === "dev_override";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    if (sourceType === "attach") {
      if (!/^[A-Za-z0-9_-]+$/.test(sessionId)) { setLocalError("Session ID must be letters, digits, '-' or '_'."); return; }
      await model.attach(sessionId, file);
      return;
    }
    if (sourceType === "v4l2" && !/^\/dev\/video\d+$/.test(device)) { setLocalError("Device path must use /dev/videoN, for example /dev/video0."); return; }
    if (sourceType === "rtsp" && !/^rtsps?:\/\//i.test(rtspUrl)) { setLocalError("RTSP URL must begin with rtsp:// or rtsps://."); return; }
    const runContext = {
      operator_run_id: operatorId,
      recipe_id: recipeId || null,
      tray_id: trayId,
      ...(dispensePlan ? { dispense_plan: dispensePlan } : {}),
    };
    const source: RunnerSource = sourceType === "video" ? { type: "video", file: file as File, runContext }
      : sourceType === "v4l2" ? { type: "v4l2", device, runContext } : { type: "rtsp", url: rtspUrl, runContext };
    await model.start(source);
  };

  const disabled = model.busy || (sourceType === "attach"
    ? !sessionId
    : !canStart || !operatorId || !trayId || (sourceType === "video" && !file) || (sourceType === "v4l2" && !device) || (sourceType === "rtsp" && !rtspUrl));

  return <details className="source-panel panel" open={!model.runner}>
    <summary>
      <span>Inspection source</span>
      {model.runner && <span className="active-run-summary">{model.sourceLabel ?? "source"} · Tray {model.runContext?.tray_id ?? "—"} · Recipe {model.runContext?.recipe_id ?? "not set"}</span>}
    </summary>
    <form className="source-picker" noValidate onSubmit={submit}>
      <fieldset><legend>Inspection source</legend><div className="source-options">
        <label><input type="radio" name="source" checked={sourceType === "video"} onChange={() => setSourceType("video")} /> Video upload</label>
        <label><input type="radio" name="source" checked={sourceType === "v4l2"} onChange={() => setSourceType("v4l2")} /> V4L2 camera</label>
        <label><input type="radio" name="source" checked={sourceType === "rtsp"} onChange={() => setSourceType("rtsp")} /> RTSP stream</label>
        <label><input type="radio" name="source" checked={sourceType === "attach"} onChange={() => setSourceType("attach")} /> Open existing session</label>
      </div></fieldset>
      {sourceType !== "attach" && <div className="run-context">
        <label className="source-input">Operator run ID<input aria-label="Operator run ID" value={operatorId} required onChange={e => onOperatorId(e.currentTarget.value)} /></label>
        <label className="source-input">Tray ID<input aria-label="Tray ID" value={trayId} required onChange={e => setTrayId(e.currentTarget.value)} /></label>
        <label className="source-input">Recipe ID (optional)<input aria-label="Recipe ID (optional)" value={recipeId} onChange={e => setRecipeId(e.currentTarget.value)} /></label>
        <label className="source-input">Dispense plan (optional)<textarea aria-label="Dispense plan (optional)" value={dispensePlan} placeholder={"T200:10>T400:10>T100:10 x3\nT200:30>T100:10 x3"} onChange={e => setDispensePlan(e.currentTarget.value)} /></label>
      </div>}
      {sourceType === "attach" && <label className="source-input">Session ID<input aria-label="Session ID" value={sessionId} required onChange={e => setSessionId(e.currentTarget.value.trim())} /></label>}
      {(sourceType === "video" || sourceType === "attach") && <label className="source-input">{sourceType === "attach" ? "Same video file (optional, for local playback)" : "Video file"}<input aria-label="Video file" type="file" accept="video/*,.avi" onChange={e => setFile(e.currentTarget.files?.[0] ?? null)} /></label>}
      {sourceType === "v4l2" && <label className="source-input">Device path<input aria-label="Device path" value={device} required placeholder="/dev/video0" onChange={e => setDevice(e.currentTarget.value)} /></label>}
      {sourceType === "rtsp" && <label className="source-input">RTSP URL<input aria-label="RTSP URL" type="url" value={rtspUrl} required placeholder="rtsp://camera.example/live" onChange={e => setRtspUrl(e.currentTarget.value)} /></label>}
      <button className="hmi-button hmi-button--primary" disabled={disabled}>{model.busy ? "Starting…" : sourceType === "attach" ? "Open session" : "Start live inspection"}</button>
      {localError && <p className="inline-error" role="alert">{localError}</p>}
    </form>
  </details>;
}
