import { useEffect, useState } from "react";

import { AlarmDrawer } from "../screens/alarms/AlarmDrawer";
import { DiagnosticsScreen } from "../screens/diagnostics/DiagnosticsScreen";
import { VialDetailScreen } from "../screens/detail/VialDetailScreen";
import { OverviewScreen } from "../screens/overview/OverviewScreen";
import { StationScreen } from "../screens/station/StationScreen";
import type { RunnerClient } from "../runner/client";
import { alarmList, shortCommit } from "../model/selectors";
import { useInspection, type InspectionModel } from "../model/useInspection";
import { SourcePanel } from "./SourcePanel";

export type ScreenId = "overview" | "station" | "detail" | "diagnostics";

const RUN_TEXT: Readonly<Record<InspectionModel["runPhase"], string>> = {
  idle: "Idle", starting: "Starting", running: "Running", completed: "Run complete", stopped: "Run stopped", failed: "Run failed"
};

function Banners({ model }: { readonly model: InspectionModel }) {
  const { health } = model;
  return <div className="banner-stack">
    {health?.replay && <div className="development-banner development-banner--replay" role="note">
      <strong>Recorded replay, not live.</strong> Development data from session {health.replay.recorded_session_id} on service {shortCommit(health.replay.recorded_service_commit)}, recorded {health.replay.recorded_at}. {health.replay.note}
    </div>}
    {model.disclosedHybrid && <div className="development-banner" role="note">Development data, frozen-cohort hybrid, service 2344f3c</div>}
    {model.pinStatus === "dev_override" && health && <div className="development-banner development-banner--override" role="note">
      <strong>Unpinned development service.</strong> Runner reports {shortCommit(health.pipeline_commit)}; this UI is pinned to {shortCommit(model.expectedCommit)}. Allowed only by the VITE_DEV_ALLOWED_COMMITS override.
    </div>}
    {model.pinStatus === "mismatch" && health && <div className="failure-banner" role="alert"><strong>Pipeline commit mismatch</strong><span>Expected {model.expectedCommit}; received {health.pipeline_commit}. Session start is blocked.</span></div>}
    {model.healthError && <div className="failure-banner" role="alert"><strong>Could not verify runner health</strong><span>{model.healthError}. Session start is blocked.</span></div>}
    {model.startError && <div className="failure-banner" role="alert"><strong>Service request failed</strong><span>{model.startError}</span></div>}
    {model.streamFailure && <div className="failure-banner" role="alert"><strong>Event stream stopped</strong><span>{model.streamFailure}</span></div>}
    {model.runner?.protocolError && <div className="failure-banner" role="alert"><strong>Event stream stopped</strong><span>{model.runner.protocolError}</span></div>}
    {model.runner?.error && <div className="failure-banner" role="alert"><strong>Pipeline error</strong><span>{model.runner.error}</span></div>}
  </div>;
}

export function AppShell({ client, expectedCommit, devAllowedCommits = [] }: { readonly client: RunnerClient; readonly expectedCommit: string; readonly devAllowedCommits?: readonly string[] }) {
  const model = useInspection(client, expectedCommit, devAllowedCommits);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [screen, setScreen] = useState<ScreenId>("station");
  const [engineer, setEngineer] = useState(false);
  const [alarmsOpen, setAlarmsOpen] = useState(false);
  const [operatorId, setOperatorId] = useState("");

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => { if (!engineer && screen === "diagnostics") setScreen("station"); }, [engineer, screen]);

  const alarms = alarmList(model.runner, model.localAcks);
  const activeAlarms = alarms.filter(a => !a.acknowledged).length;
  const openDetail = (vialId: string) => { model.selectVial(vialId); setScreen("detail"); setAlarmsOpen(false); };
  const health = model.health;
  const serviceText = model.healthError ? "Service unreachable" : !health ? "Checking service" : `Service ${health.status === "healthy" ? "ok" : health.status}`;
  const tabs: readonly { readonly id: ScreenId; readonly label: string; readonly level: string }[] = [
    { id: "overview", label: "Line overview", level: "L1" },
    { id: "station", label: "Inspection station", level: "L2" },
    { id: "detail", label: "Vial detail", level: "L3" },
    ...(engineer ? [{ id: "diagnostics" as const, label: "Diagnostics", level: "L4" }] : [])
  ];

  return <div className="hmi-shell">
    <header className="hmi-header">
      <div className="hmi-title">
        <h1>Live vial decisions</h1>
        <p className="hmi-subtitle">Fill station 1 · Recipe {model.runContext?.recipe_id ?? health?.frozen_profile?.recipe_id ?? "—"} · <span className={`run-state run-state--${model.runPhase}`}>{RUN_TEXT[model.runPhase]}</span></p>
      </div>
      <nav className="screen-tabs" aria-label="Screens">
        {tabs.map(tab => <button key={tab.id} type="button" className="hmi-button screen-tab" aria-current={screen === tab.id ? "page" : undefined} onClick={() => setScreen(tab.id)}>
          <span className="screen-tab__level">{tab.level}</span>{tab.label}
        </button>)}
      </nav>
      <div className="header-status">
        <button type="button" className={`hmi-button alarm-button${activeAlarms ? " alarm-button--active" : ""}`} aria-haspopup="dialog" aria-expanded={alarmsOpen} onClick={() => setAlarmsOpen(true)}>
          Alarms ({activeAlarms})
        </button>
        <span className={`service-status service-status--${model.healthError ? "down" : health?.status ?? "unknown"}`}>
          {model.pinStatus === "verified" && <strong className="service-verified">Service verified · {health?.status}</strong>}
          {model.pinStatus !== "verified" && serviceText}
          {health && <code className="pipeline-commit" aria-label={`Pipeline commit ${health.pipeline_commit}`} title={health.pipeline_commit}>{shortCommit(health.pipeline_commit)}</code>}
        </span>
        {(health?.test_only || model.runner?.testOnly) && <strong className="test-only">Test-only service</strong>}
        <span className="rate">{model.runner?.rate?.processing_fps == null ? "Rate unavailable" : `${model.runner.rate.processing_fps.toFixed(1)} frames/s`}</span>
        <button type="button" className="hmi-button" aria-pressed={engineer} onClick={() => setEngineer(v => !v)}>{engineer ? "Engineer on" : "Engineer"}</button>
        <button type="button" className="hmi-button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "Light theme" : "Dark theme"}</button>
      </div>
    </header>

    <main className="hmi-main">
      <Banners model={model} />
      <SourcePanel model={model} operatorId={operatorId} onOperatorId={setOperatorId} />
      <p className="connection-status" role="status">{model.connection}</p>
      {screen === "overview" && <OverviewScreen model={model} onOpenDetail={openDetail} />}
      {screen === "station" && <StationScreen model={model} operatorId={operatorId} onOpenDetail={openDetail} />}
      {screen === "detail" && <VialDetailScreen model={model} vialId={model.selectedVialId} onSelectVial={model.selectVial} />}
      {screen === "diagnostics" && engineer && <DiagnosticsScreen model={model} />}
    </main>
    <AlarmDrawer model={model} alarms={alarms} open={alarmsOpen} operatorId={operatorId} onClose={() => setAlarmsOpen(false)} onOpenDetail={openDetail} />
  </div>;
}
