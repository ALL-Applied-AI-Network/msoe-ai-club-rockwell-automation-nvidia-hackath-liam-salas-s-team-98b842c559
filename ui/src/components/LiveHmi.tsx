import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import type { RunContext, RunnerClient, RunnerHealth, RunnerSource } from "../runner/client";
import { initialRunnerState, reduceRunnerEvent, type Disposition, type RunnerState, type VialState } from "../runner/contract";

type SourceType = RunnerSource["type"];

function dispositionView(disposition?: Disposition) {
  if (disposition === "keep") return { icon: "✓", label: "Keep", tone: "neutral" };
  if (disposition === "check") return { icon: "◇", label: "Check", tone: "neutral" };
  if (disposition === "discard") return { icon: "!", label: "Discard", tone: "alarm" };
  if (disposition === "insufficient_evidence") return { icon: "▲", label: "Insufficient evidence", tone: "warning" };
  return { icon: "○", label: "In progress", tone: "neutral" };
}

function vialView(vial: VialState, runStopped: boolean) {
  const disposition = vial.average?.disposition ?? vial.decision?.disposition;
  if (runStopped && !disposition) return { icon: "—", label: "Not assessed (run stopped)", tone: "neutral" };
  return dispositionView(disposition);
}

function SourceTag({ source }: { readonly source?: string }) {
  if (!source) return null;
  const label = source.replaceAll("_", " ");
  return <span className={`source-tag source-tag--${source}`}>{label[0].toUpperCase() + label.slice(1)}</span>;
}

function operatorAbstentionMessage(reason?: string | null) {
  if (reason === "tracker_source_profile_unavailable") return "Can't assess this video: camera setup not recognised";
  if (reason) return "Can't assess this video: there isn't enough reliable evidence.";
  return null;
}

function explanationKind(modelId: string, source: string) {
  if (source === "deterministic_fallback") return "Deterministic fallback";
  if (modelId === "deterministic-template") return "Deterministic template";
  return "Model-written";
}

function operatorVialLabel(vial: VialState) {
  return `Vial ${vial.position.tray_index + 1}`;
}

function modelDisplayName(modelId: string) {
  if (modelId.includes("Qwen")) return "Qwen";
  if (modelId.includes("Cosmos")) return "Cosmos";
  if (modelId.includes("AnomalyThink")) return "AnomalyThink";
  return modelId;
}

function EngineerDetails({ vial }: { readonly vial: VialState }) {
  const measurement = vial.measurement;
  const averageScore = vial.average && vial.average.disposition !== "insufficient_evidence" ? vial.average.scores[vial.average.disposition].toFixed(2) : null;
  const explanationEvidenceIds = vial.explanations.flatMap(item => item.evidence_ids);
  const unavailableReason = vial.decision?.unavailable_reason ?? measurement?.unavailable_reason;
  const lab = measurement?.lab;
  const deltaE = measurement?.delta_e;
  const deltaEText = deltaE != null
    ? deltaE.toFixed(2)
    : measurement?.status === "measured"
      ? `Not provided for ${vial.tracker?.tank_id ?? "current tank"} · ${vial.tracker?.fill_step_id ?? "current fill step"}; runner event does not expose stage ΔE.`
      : `Unavailable${measurement?.unavailable_reason ? ` · ${measurement.unavailable_reason}` : ""}`;
  return <details className="engineer-details">
    <summary>Engineer details</summary>
    <section aria-label="Model votes" className="engineer-section engineer-section--votes">
      <h3>Model votes</h3>
      {vial.votes.length === 0 ? <p>Unavailable</p> : vial.votes.map(vote => <p key={vote.vote_id}>
        <strong>{vote.model.model_id}</strong> {vote.model.revision} · {vote.admission.admitted ? "Admitted" : "Not admitted"} ({vote.admission.reason}) · {vote.disposition}{vote.confidence == null ? "" : ` · ${vote.confidence}`} <SourceTag source={vote.model.source} />
      </p>)}
    </section>
    <dl>
      <div><dt>Lab (CIELAB; L* 0–100)</dt><dd>{lab ? `L* ${lab[0].toFixed(1)} · a* ${lab[1].toFixed(1)} · b* ${lab[2].toFixed(1)}` : "Unavailable"} <SourceTag source={measurement?.status} /></dd></div>
      <div><dt>Delta E (current event stage)</dt><dd>{deltaEText} <SourceTag source={deltaE == null ? "unavailable" : measurement?.status} /></dd></div>
      <div><dt>Tracker volume</dt><dd>{vial.tracker?.measured_volume_ml == null ? "Unavailable" : `${vial.tracker.measured_volume_ml.toFixed(1)} ml`} <SourceTag source={vial.tracker?.volume_provenance} /></dd></div>
      <div><dt>Average (equal weight)</dt><dd>{averageScore ?? "Unavailable"} <SourceTag source={vial.average ? "inferred" : undefined} /></dd></div>
      <div><dt>Reason</dt><dd>{unavailableReason ?? "None reported"}</dd></div>
      <div><dt>Case ID</dt><dd><code>{vial.vialId}</code></dd></div>
    </dl>
    <div className="engineer-grid">
      <section aria-label="Evidence IDs" className="engineer-section engineer-section--scroll">
        <h3>Evidence IDs</h3>
        {[...(measurement?.evidence_ids ?? []), ...(vial.tracker?.locator.evidence_ids ?? []), ...(vial.decision?.evidence_ids ?? []), ...explanationEvidenceIds].length === 0
          ? <p>Unavailable</p>
          : <ul>{Array.from(new Set([...(measurement?.evidence_ids ?? []), ...(vial.tracker?.locator.evidence_ids ?? []), ...(vial.decision?.evidence_ids ?? []), ...explanationEvidenceIds])).map(id => <li key={id}><code>{id}</code></li>)}</ul>}
      </section>
      <section aria-label="Explanation sources" className="engineer-section engineer-section--scroll">
        <h3>Explanation sources</h3>
        {vial.explanations.length === 0 ? <p>Unavailable</p> : vial.explanations.map(item => <div key={item.explanation_id}>
          <p><strong>{item.producer.model_id}</strong> {item.producer.revision} · {explanationKind(item.producer.model_id, item.producer.source)} <SourceTag source={item.producer.source} /></p>
          <p>{item.plain_language}</p>
        </div>)}
      </section>
      <section aria-label="Tracker and locator" className="engineer-section"><h3>Tracker and locator</h3><p>Tank {vial.tracker?.tank_id ?? "unavailable"} · Fill step {vial.tracker?.fill_step_id ?? "unavailable"}</p><p>Bounding box {vial.tracker?.locator.bbox_normalized?.join(", ") ?? "unavailable"} · Centroid {vial.tracker?.locator.centroid_normalized?.join(", ") ?? "unavailable"}</p></section>
    </div>
  </details>;
}

function VialDetails({ vial, runStopped }: { readonly vial?: VialState; readonly runStopped: boolean }) {
  if (!vial) return <section className="hmi-section vial-details" aria-label="Vial details"><h2>Vial details</h2><p>Select a source to begin.</p></section>;
  const view = vialView(vial, runStopped);
  const explanations = vial.explanations.length ? vial.explanations : (vial.explanation ? [vial.explanation] : []);
  const primaryExplanation = explanations.find(item => item.producer.source !== "deterministic_fallback" && item.producer.model_id !== "deterministic-template") ?? explanations.at(-1);
  const abstentionMessage = explanations.length === 0 && view.label === "Insufficient evidence"
    ? operatorAbstentionMessage(vial.decision?.unavailable_reason ?? vial.measurement?.unavailable_reason)
    : null;
  return <section className="hmi-section vial-details" aria-label="Vial details">
    <div className="section-heading"><div><p className="kicker">Current vial</p><h2>{vial.phase === "measuring" ? `Measuring ${operatorVialLabel(vial)}` : operatorVialLabel(vial)}</h2></div><div className={`large-state large-state--${view.tone}`}><span aria-hidden="true">{view.icon}</span> {view.label}</div></div>
    {vial.phase === "waiting_for_references" && <p className="operator-message">Waiting for reference measurements.</p>}
    {abstentionMessage && <p className="operator-message">{abstentionMessage}</p>}
    {primaryExplanation && <div className="operator-explanation"><h3>What this means</h3><p>{primaryExplanation.plain_language}</p><p className="explanation-source">Explanation source: {explanationKind(primaryExplanation.producer.model_id, primaryExplanation.producer.source)} · {modelDisplayName(primaryExplanation.producer.model_id)} <SourceTag source={primaryExplanation.producer.source} /></p>{primaryExplanation.limits.map(limit => <p className="limit" key={limit}>Limit: {limit}</p>)}</div>}
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
      <div className="locator-box" role="img" aria-label={`${operatorVialLabel(vial)} normalized bounding box ${locator.bbox_normalized.join(", ")}`} style={{ left: `${locator.bbox_normalized[0] * 100}%`, top: `${locator.bbox_normalized[1] * 100}%`, width: `${locator.bbox_normalized[2] * 100}%`, height: `${locator.bbox_normalized[3] * 100}%` }} />
      <p>Centroid {locator.centroid_normalized.join(", ")}</p>
    </div> : <div className="locator-empty"><span aria-hidden="true">＋</span><p>{locator?.unavailable_reason || "No locator geometry reported."}</p></div>}
  </section>;
}

export function LiveHmi({ client, expectedPipelineCommit }: { readonly client: RunnerClient; readonly expectedPipelineCommit: string }) {
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
  const [health, setHealth] = useState<RunnerHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [streamFailure, setStreamFailure] = useState<string | null>(null);
  const closeStream = useRef<(() => void) | null>(null);
  const sessionVersion = useRef(0);
  const mounted = useRef(true);

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => {
    let active = true;
    client.getHealth().then(value => {
      if (active) { setHealth(value); setHealthError(null); }
    }).catch(reason => {
      if (active) setHealthError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { active = false; };
  }, [client]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      sessionVersion.current += 1;
      closeStream.current?.();
    };
  }, []);
  const selected = useMemo(() => runner && (runner.vials[selectedId ?? ""] ?? runner.vials[runner.trayOrder[0]]), [runner, selectedId]);
  const healthVerified = health !== null && health.pipeline_commit === expectedPipelineCommit;
  const disclosedHybrid = health !== null && !health.test_only && health.pipeline_commit === "2344f3c655d9aa255eb576e6d2a280bd0fc9bdff";

  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (!healthVerified) {
      setStartError("Cannot start until runner health and pipeline commit are verified.");
      return;
    }
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
    setStreamFailure(null);
    closeStream.current?.();
    const version = ++sessionVersion.current;
    try {
      const runContext: RunContext = { operator_run_id: operatorRunId, recipe_id: recipeId || null, tray_id: trayId };
      const source: RunnerSource = sourceType === "video"
        ? { type: "video", file: file as File, runContext }
        : sourceType === "v4l2"
          ? { type: "v4l2", device, runContext }
          : { type: "rtsp", url: rtspUrl, runContext };
      const session = await client.createSession(source);
      if (!mounted.current || version !== sessionVersion.current) return;
      setRunner(initialRunnerState(session.session_id, session.test_only));
      setSelectedId(null);
      setAcknowledged(new Set());
      setConnection(`Connected · session ${session.session_id}`);
      closeStream.current = client.subscribe(
        session.session_id,
        0,
        next => {
          if (mounted.current && version === sessionVersion.current) setRunner(current => current ? reduceRunnerEvent(current, next) : current);
        },
        notice => {
          if (!mounted.current || version !== sessionVersion.current) return;
          if (notice.kind === "terminal") {
            setConnection("Event stream stopped.");
            setStreamFailure(notice.message);
            return;
          }
          setConnection("Connection interrupted — reconnecting.");
          void client.getSession(session.session_id).then(current => {
            if (!mounted.current || version !== sessionVersion.current) return;
            if (current.state === "completed") {
              closeStream.current?.();
              setConnection("Run completed");
            } else if (current.state === "stopped") {
              closeStream.current?.();
              setConnection("Run stopped");
            } else if (current.state === "failed") {
              closeStream.current?.();
              setConnection("Run failed");
              setStreamFailure("Run failed. The runner reported that this session failed.");
            } else {
              setConnection("Connection interrupted — reconnecting.");
            }
          }).catch(() => {
            if (mounted.current && version === sessionVersion.current) setConnection("Connection interrupted — reconnecting. Session state is unavailable.");
          });
        }
      );
    } catch (reason) {
      if (!mounted.current || version !== sessionVersion.current) return;
      setRunner(null);
      setConnection("Not connected.");
      setStartError(`Could not start live inspection. ${reason instanceof Error ? reason.message : String(reason)}`);
    } finally {
      if (mounted.current && version === sessionVersion.current) setBusy(false);
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
  const runStopped = connection === "Run stopped";

  return <div className="hmi-shell">
    <header className="hmi-header">
      <div><p className="kicker">Fluid mixing inspection</p><h1>Live vial decisions</h1></div>
      <div className="header-status">
        {healthVerified && <strong className="service-verified">Service verified · {health.status}</strong>}
        {(health?.test_only || runner?.testOnly) && <strong className="test-only">Test-only service</strong>}
        {health && <code className="pipeline-commit" aria-label={`Pipeline commit ${health.pipeline_commit}`}>{health.pipeline_commit}</code>}
        <span>{runner?.rate?.processing_fps == null ? "Rate unavailable" : `${runner.rate.processing_fps.toFixed(1)} frames/s`}</span>
        <button type="button" className="hmi-control" style={{ minHeight: 48 }} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "Light theme" : "Dark theme"}</button>
      </div>
    </header>

    <main>
      {disclosedHybrid && <div className="development-banner" role="note">Development data, frozen-cohort hybrid, service 2344f3c</div>}
      <details className="source-panel hmi-section" open={!runner}>
        <summary><span>Inspection source</span>{runner && <span className="active-run-summary">{sourceType === "video" ? "Video upload" : sourceType === "v4l2" ? "V4L2 camera" : "RTSP stream"} · Tray {trayId} · Recipe {recipeId || "not set"}</span>}</summary>
      <form className="source-picker" noValidate onSubmit={start}>
        <fieldset><legend>Inspection source</legend><div className="source-options">
          <label><input type="radio" name="source" checked={sourceType === "video"} onChange={() => setSourceType("video")} /> Video upload</label>
          <label><input type="radio" name="source" checked={sourceType === "v4l2"} onChange={() => setSourceType("v4l2")} /> V4L2 camera</label>
          <label><input type="radio" name="source" checked={sourceType === "rtsp"} onChange={() => setSourceType("rtsp")} /> RTSP stream</label>
        </div></fieldset>
        <div className="run-context"><label className="source-input">Operator run ID<input aria-label="Operator run ID" value={operatorRunId} required onChange={event => setOperatorRunId(event.currentTarget.value)} /></label><label className="source-input">Tray ID<input aria-label="Tray ID" value={trayId} required onChange={event => setTrayId(event.currentTarget.value)} /></label><label className="source-input">Recipe ID (optional)<input aria-label="Recipe ID (optional)" value={recipeId} onChange={event => setRecipeId(event.currentTarget.value)} /></label></div>
        {sourceType === "video" && <label className="source-input">Video file<input aria-label="Video file" type="file" accept="video/*,.avi" required onChange={event => setFile(event.currentTarget.files?.[0] ?? null)} /></label>}
        {sourceType === "v4l2" && <label className="source-input">Device path<input aria-label="Device path" value={device} required placeholder="/dev/video0" onChange={event => setDevice(event.currentTarget.value)} /></label>}
        {sourceType === "rtsp" && <label className="source-input">RTSP URL<input aria-label="RTSP URL" type="url" value={rtspUrl} required placeholder="rtsp://camera.example/live" onChange={event => setRtspUrl(event.currentTarget.value)} /></label>}
        <button className="hmi-control hmi-control--primary" style={{ minHeight: 48 }} disabled={busy || !healthVerified || !operatorRunId || !trayId || (sourceType === "video" && !file) || (sourceType === "v4l2" && !device) || (sourceType === "rtsp" && !rtspUrl)}>{busy ? "Starting…" : "Start live inspection"}</button>
      </form>
      </details>

      {health && !healthVerified && <div className="failure-banner" role="alert"><strong>Pipeline commit mismatch</strong><span>Expected {expectedPipelineCommit}; received {health.pipeline_commit}. Session start is blocked.</span></div>}
      {healthError && <div className="failure-banner" role="alert"><strong>Could not verify runner health</strong><span>{healthError}. Session start is blocked.</span></div>}
      {startError && <div className="failure-banner" role="alert"><strong>Service request failed</strong><span>{startError}</span></div>}
      {streamFailure && <div className="failure-banner" role="alert"><strong>Event stream stopped</strong><span>{streamFailure}</span></div>}
      <p className="connection-status" role="status">{connection}</p>
      {runner?.protocolError && <div className="failure-banner" role="alert"><strong>Event stream stopped</strong><span>{runner.protocolError}</span></div>}
      {runner?.error && <div className="failure-banner" role="alert"><strong>Pipeline error</strong><span>{runner.error}</span></div>}
      {alarms.map(alarm => <div className="alarm-banner" role="alert" key={alarm.alarmId}><div><strong>Discard alarm · {operatorVialLabel(runner!.vials[alarm.vialId])}</strong><span>Remove this vial from the keep path and acknowledge the alarm.</span></div><button className="hmi-control" style={{ minHeight: 48 }} onClick={() => void acknowledge(alarm.alarmId)}>Acknowledge</button></div>)}

      <div className="hmi-grid">
        <VialDetails vial={selected ?? undefined} runStopped={runStopped} />
        <TopDownLocator vial={selected ?? undefined} />
      </div>

      <section className="tray-strip" aria-label="Tray order">
        <div className="section-heading"><div><p className="kicker">Physical sequence</p><h2>Tray order</h2></div><span>{runner?.trayOrder.length ?? 0} vials</span></div>
        <div className="tray-slots">{runner?.trayOrder.map(vialId => {
          const vial = runner.vials[vialId];
          const view = vialView(vial, runStopped);
          const label = operatorVialLabel(vial);
          return <button type="button" className={`hmi-control tray-slot tray-slot--${view.tone}`} style={{ minHeight: 48 }} key={vialId} aria-label={`Select tray position ${vial.position.tray_index + 1}, ${label}, ${view.label}`} aria-pressed={selected?.vialId === vialId} onClick={() => setSelectedId(vialId)}><span className="tray-position">{vial.position.tray_index + 1}</span><span className="tray-vial">{label}</span><span className="tray-decision"><span className="state-shape" aria-hidden="true">{view.icon}</span>{view.label}</span></button>;
        })}</div>
      </section>
    </main>
  </div>;
}
