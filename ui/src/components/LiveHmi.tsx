import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import type { RunContext, RunnerClient, RunnerSource } from "../runner/client";
import { initialRunnerState, reduceRunnerEvent, type Disposition, type RunnerState, type VialState } from "../runner/contract";

type SourceType = RunnerSource["type"];

function dispositionView(disposition?: Disposition) {
  if (disposition === "keep") return { icon: "✓", label: "Keep", tone: "neutral" };
  if (disposition === "check") return { icon: "◇", label: "Check", tone: "neutral" };
  if (disposition === "discard") return { icon: "!", label: "Discard", tone: "alarm" };
  if (disposition === "insufficient_evidence") return { icon: "▲", label: "Insufficient evidence", tone: "warning" };
  return { icon: "○", label: "In progress", tone: "neutral" };
}

function SourceTag({ source }: { readonly source?: string }) {
  if (!source) return null;
  return <span className={`source-tag source-tag--${source}`}>{source[0].toUpperCase() + source.slice(1)}</span>;
}

function EngineerDetails({ vial }: { readonly vial: VialState }) {
  const measurement = vial.measurement;
  const averageScore = vial.average && vial.average.disposition !== "insufficient_evidence" ? vial.average.scores[vial.average.disposition] : null;
  return <details className="engineer-details">
    <summary>Engineer details</summary>
    <dl>
      <div><dt>Lab</dt><dd>{measurement?.lab ? `L* ${measurement.lab[0]} · a* ${measurement.lab[1]} · b* ${measurement.lab[2]}` : "Unavailable"} <SourceTag source={measurement?.status} /></dd></div>
      <div><dt>Delta E</dt><dd>{measurement?.delta_e ?? "Unavailable"} <SourceTag source={measurement?.status} /></dd></div>
      <div><dt>Tracker volume</dt><dd>{vial.tracker?.measured_volume_ml == null ? "Unavailable" : `${vial.tracker.measured_volume_ml} ml`} <SourceTag source={vial.tracker?.volume_provenance} /></dd></div>
      <div><dt>Average</dt><dd>{averageScore ?? "Unavailable"} <SourceTag source={vial.average ? "inferred" : undefined} /></dd></div>
    </dl>
    <section aria-label="Model votes" className="engineer-section">
      <h3>Model votes</h3>
      {vial.votes.length === 0 ? <p>Unavailable</p> : vial.votes.map(vote => <p key={vote.vote_id}>
        <strong>{vote.model.model_id}</strong> {vote.model.revision} · {vote.disposition}{vote.confidence == null ? "" : ` · ${vote.confidence}`} <SourceTag source={vote.model.source} />
      </p>)}
    </section>
    <section aria-label="Evidence IDs" className="engineer-section">
      <h3>Evidence IDs</h3>
      {[...(measurement?.evidence_ids ?? []), ...(vial.tracker?.locator.evidence_ids ?? []), ...(vial.decision?.evidence_ids ?? []), ...(vial.explanation?.evidence_ids ?? [])].length === 0
        ? <p>Unavailable</p>
        : <ul>{Array.from(new Set([...(measurement?.evidence_ids ?? []), ...(vial.tracker?.locator.evidence_ids ?? []), ...(vial.decision?.evidence_ids ?? []), ...(vial.explanation?.evidence_ids ?? [])])).map(id => <li key={id}><code>{id}</code></li>)}</ul>}
    </section>
    <section aria-label="Tracker and locator" className="engineer-section"><h3>Tracker and locator</h3><p>Tank {vial.tracker?.tank_id ?? "unavailable"} · Fill step {vial.tracker?.fill_step_id ?? "unavailable"}</p><p>Bounding box {vial.tracker?.locator.bbox_normalized?.join(", ") ?? "unavailable"} · Centroid {vial.tracker?.locator.centroid_normalized?.join(", ") ?? "unavailable"}</p></section>
  </details>;
}

function VialDetails({ vial }: { readonly vial?: VialState }) {
  if (!vial) return <section className="hmi-section vial-details" aria-label="Vial details"><h2>Vial details</h2><p>Select a source to begin.</p></section>;
  const view = dispositionView(vial.average?.disposition ?? vial.decision?.disposition);
  return <section className="hmi-section vial-details" aria-label="Vial details">
    <div className="section-heading"><div><p className="kicker">Current vial</p><h2>{vial.phase === "measuring" ? `Measuring ${vial.vialId}` : vial.vialId}</h2></div><div className={`large-state large-state--${view.tone}`}><span aria-hidden="true">{view.icon}</span> {view.label}</div></div>
    {vial.phase === "waiting_for_references" && <p className="operator-message">Waiting for reference measurements.</p>}
    {vial.explanation && <div className="operator-explanation"><h3>What this means</h3><p>{vial.explanation.plain_language}</p>{vial.explanation.limits.map(limit => <p className="limit" key={limit}>Limit: {limit}</p>)}</div>}
    <EngineerDetails vial={vial} />
  </section>;
}

function TopDownLocator({ vial }: { readonly vial?: VialState }) {
  const vialId = vial?.vialId;
  const locator = vial?.tracker?.locator;
  const located = vialId && locator?.status === "located" && locator.bbox_normalized && locator.centroid_normalized;
  return <section className="hmi-section locator" aria-label="Top-down vial locator">
    <div className="section-heading"><div><p className="kicker">Top-down slot</p><h2>Vial locator</h2></div></div>
    {located ? <div className="locator-canvas">
      <div className="locator-box" role="img" aria-label={`${vialId} normalized bounding box ${locator.bbox_normalized.join(", ")}`} style={{ left: `${locator.bbox_normalized[0] * 100}%`, top: `${locator.bbox_normalized[1] * 100}%`, width: `${locator.bbox_normalized[2] * 100}%`, height: `${locator.bbox_normalized[3] * 100}%` }} />
      <p>Centroid {locator.centroid_normalized.join(", ")}</p>
    </div> : <div className="locator-empty"><span aria-hidden="true">＋</span><p>{locator?.unavailable_reason || "No locator geometry reported."}</p></div>}
  </section>;
}

export function LiveHmi({ client }: { readonly client: RunnerClient }) {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [sourceType, setSourceType] = useState<SourceType>("video");
  const [file, setFile] = useState<File | null>(null);
  const [device, setDevice] = useState("");
  const [rtspUrl, setRtspUrl] = useState("");
  const [operatorRunId, setOperatorRunId] = useState("");
  const [recipeId, setRecipeId] = useState("");
  const [trayId, setTrayId] = useState("");
  const [runner, setRunner] = useState<RunnerState | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [connection, setConnection] = useState("Choose a source to start live inspection.");
  const [startError, setStartError] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const closeStream = useRef<(() => void) | null>(null);

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => () => closeStream.current?.(), []);
  const selected = useMemo(() => runner && (runner.vials[selectedId ?? ""] ?? runner.vials[runner.trayOrder[0]]), [runner, selectedId]);

  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (sourceType === "v4l2" && !/^\/dev\/video\d+$/.test(device)) {
      setStartError("Device path must use /dev/videoN, for example /dev/video0.");
      return;
    }
    if (sourceType === "rtsp" && !/^rtsps?:\/\//i.test(rtspUrl)) {
      setStartError("RTSP URL must begin with rtsp:// or rtsps://.");
      return;
    }
    setBusy(true);
    setStartError(null);
    closeStream.current?.();
    try {
      const runContext: RunContext = { operator_run_id: operatorRunId, recipe_id: recipeId || null, tray_id: trayId };
      const source: RunnerSource = sourceType === "video"
        ? { type: "video", file: file as File, runContext }
        : sourceType === "v4l2"
          ? { type: "v4l2", device, runContext }
          : { type: "rtsp", url: rtspUrl, runContext };
      const session = await client.createSession(source);
      setRunner(initialRunnerState(session.session_id, session.test_only));
      setSelectedId(null);
      setAcknowledged(new Set());
      setConnection(`Connected · session ${session.session_id}`);
      closeStream.current = client.subscribe(
        session.session_id,
        0,
        next => setRunner(current => current ? reduceRunnerEvent(current, next) : current),
        message => setConnection(`${message}. No live updates are available.`)
      );
    } catch (reason) {
      setRunner(null);
      setConnection("Not connected.");
      setStartError(`Could not start live inspection. ${reason instanceof Error ? reason.message : String(reason)}`);
    } finally {
      setBusy(false);
    }
  };

  const acknowledge = async (alarmId: string) => {
    if (!runner) return;
    try {
      await client.acknowledge(runner.sessionId, alarmId, operatorRunId);
      setAcknowledged(current => new Set(current).add(alarmId));
    } catch (reason) {
      setStartError(`Could not acknowledge ${alarmId}. ${reason instanceof Error ? reason.message : String(reason)}`);
    }
  };

  const alarms = runner?.trayOrder.flatMap(vialId => {
    const vial = runner.vials[vialId];
    const alarm = vial.average?.alarm ?? vial.decision?.alarm;
    return alarm && alarm.state === "unacknowledged" && !acknowledged.has(alarm.alarm_id) ? [{ vialId, alarmId: alarm.alarm_id }] : [];
  }) ?? [];

  return <div className="hmi-shell">
    <header className="hmi-header">
      <div><p className="kicker">Fluid mixing inspection</p><h1>Live vial decisions</h1></div>
      <div className="header-status">
        {runner?.testOnly && <strong className="test-only">Test-only service</strong>}
        <span>{runner?.rate?.processing_fps == null ? "Rate unavailable" : `${runner.rate.processing_fps} frames/s`}</span>
        <button type="button" className="hmi-control" style={{ minHeight: 48 }} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "Light theme" : "Dark theme"}</button>
      </div>
    </header>

    <main>
      <form className="source-picker hmi-section" noValidate onSubmit={start}>
        <fieldset><legend>Inspection source</legend><div className="source-options">
          <label><input type="radio" name="source" checked={sourceType === "video"} onChange={() => setSourceType("video")} /> Video upload</label>
          <label><input type="radio" name="source" checked={sourceType === "v4l2"} onChange={() => setSourceType("v4l2")} /> V4L2 camera</label>
          <label><input type="radio" name="source" checked={sourceType === "rtsp"} onChange={() => setSourceType("rtsp")} /> RTSP stream</label>
        </div></fieldset>
        <div className="run-context"><label className="source-input">Operator run ID<input aria-label="Operator run ID" value={operatorRunId} required onChange={event => setOperatorRunId(event.currentTarget.value)} /></label><label className="source-input">Tray ID<input aria-label="Tray ID" value={trayId} required onChange={event => setTrayId(event.currentTarget.value)} /></label><label className="source-input">Recipe ID (optional)<input aria-label="Recipe ID (optional)" value={recipeId} onChange={event => setRecipeId(event.currentTarget.value)} /></label></div>
        {sourceType === "video" && <label className="source-input">Video file<input aria-label="Video file" type="file" accept="video/*,.avi" required onChange={event => setFile(event.currentTarget.files?.[0] ?? null)} /></label>}
        {sourceType === "v4l2" && <label className="source-input">Device path<input aria-label="Device path" value={device} required placeholder="/dev/video0" onChange={event => setDevice(event.currentTarget.value)} /></label>}
        {sourceType === "rtsp" && <label className="source-input">RTSP URL<input aria-label="RTSP URL" type="url" value={rtspUrl} required placeholder="rtsp://camera.example/live" onChange={event => setRtspUrl(event.currentTarget.value)} /></label>}
        <button className="hmi-control hmi-control--primary" style={{ minHeight: 48 }} disabled={busy || !operatorRunId || !trayId || (sourceType === "video" && !file) || (sourceType === "v4l2" && !device) || (sourceType === "rtsp" && !rtspUrl)}>{busy ? "Starting…" : "Start live inspection"}</button>
      </form>

      {startError && <div className="failure-banner" role="alert"><strong>Service request failed</strong><span>{startError}</span></div>}
      <p className="connection-status" role="status">{connection}</p>
      {runner?.protocolError && <div className="failure-banner" role="alert"><strong>Event stream stopped</strong><span>{runner.protocolError}</span></div>}
      {runner?.error && <div className="failure-banner" role="alert"><strong>Pipeline error</strong><span>{runner.error}</span></div>}
      {alarms.map(alarm => <div className="alarm-banner" role="alert" key={alarm.alarmId}><div><strong>Discard alarm · {alarm.vialId}</strong><span>Remove this vial from the keep path and acknowledge the alarm.</span></div><button className="hmi-control" style={{ minHeight: 48 }} onClick={() => void acknowledge(alarm.alarmId)}>Acknowledge</button></div>)}

      <div className="hmi-grid">
        <VialDetails vial={selected ?? undefined} />
        <TopDownLocator vial={selected ?? undefined} />
      </div>

      <section className="tray-strip" aria-label="Tray order">
        <div className="section-heading"><div><p className="kicker">Physical sequence</p><h2>Tray order</h2></div><span>{runner?.trayOrder.length ?? 0} vials</span></div>
        <div className="tray-slots">{runner?.trayOrder.map(vialId => {
          const vial = runner.vials[vialId];
          const view = dispositionView(vial.average?.disposition ?? vial.decision?.disposition);
          return <button type="button" className={`hmi-control tray-slot tray-slot--${view.tone}`} style={{ minHeight: 48 }} key={vialId} aria-label={`Select tray position ${vial.position.tray_index}, ${vialId}, ${view.label}`} aria-pressed={selected?.vialId === vialId} onClick={() => setSelectedId(vialId)}><span className="tray-position">{vial.position.tray_index}</span><span className="tray-vial">{vialId}</span><span className="tray-decision"><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}</span></button>;
        })}</div>
      </section>
    </main>
  </div>;
}
