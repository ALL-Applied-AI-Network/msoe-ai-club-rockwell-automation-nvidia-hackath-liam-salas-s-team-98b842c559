import { useEffect, useMemo, useRef, useState } from "react";
import { explanationSourceLabel, type DemoData, type EvidenceRecord } from "../data/adapter";
import { SidePanel } from "./SidePanel";
import { TopDownPanel } from "./TopDownPanel";
import { VialDetail } from "./VialDetail";

export function OperatorDashboard({ data, demoMode }: { readonly data: DemoData; readonly demoMode?: boolean }) {
  const recordingMode = demoMode ?? (data.source.mode !== "mock" && new URLSearchParams(window.location.search).get("demo") === "1");
  const [selectedId, setSelectedId] = useState(data.vials[0].vial_id);
  const [revealed, setRevealed] = useState<ReadonlySet<string>>(new Set());
  const [highlighted, setHighlighted] = useState<EvidenceRecord["target"] | null>(null);
  const [highlightedExplanationId, setHighlightedExplanationId] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [tab, setTab] = useState<"history" | "model">("history");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [announcement, setAnnouncement] = useState("");
  const revealedRef = useRef<ReadonlySet<string>>(new Set());
  const videoRef = useRef<HTMLVideoElement>(null);
  const panelOpenerRef = useRef<HTMLButtonElement | null>(null);
  const selected = useMemo(() => data.vials.find(vial => vial.vial_id === selectedId) ?? data.vials[0], [data, selectedId]);

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => { if (!panelOpen && panelOpenerRef.current) { panelOpenerRef.current.focus(); panelOpenerRef.current = null; } }, [panelOpen]);
  useEffect(() => {
    const video = videoRef.current;
    if (!recordingMode || !video) return;
    video.muted = true;
    void video.play().catch(() => { /* Browser policy can reject autoplay; the recording view remains usable. */ });
  }, [recordingMode]);
  const select = (id: string) => { setSelectedId(id); setHighlighted(null); setHighlightedExplanationId(null); };
  const updateTime = (time: number) => {
    const next = new Set(revealedRef.current);
    const newlyRevealed = data.vials.filter(vial => time >= vial.cue.start_seconds && !next.has(vial.vial_id));
    newlyRevealed.forEach(vial => next.add(vial.vial_id));
    if (newlyRevealed.length > 0) {
      revealedRef.current = next;
      setRevealed(next);
      const latest = newlyRevealed[newlyRevealed.length - 1];
      setAnnouncement(`Vial ${latest.slot} outcome: ${latest.decision.outcome}`);
      if (recordingMode) {
        setSelectedId(latest.vial_id);
        if (latest.decision.outcome === "discard") {
          setHighlightedExplanationId(latest.vial_id);
          setHighlighted(latest.evidence[0]?.target ?? null);
        } else {
          setHighlightedExplanationId(null);
          setHighlighted(null);
        }
      }
    }
  };
  const restart = () => { const empty = new Set<string>(); revealedRef.current = empty; setRevealed(empty); setAnnouncement(""); setSelectedId(data.vials[0].vial_id); setHighlighted(null); setHighlightedExplanationId(null); try { if (videoRef.current) { videoRef.current.currentTime = 0; if (recordingMode) void videoRef.current.play().catch(() => { /* See autoplay handling above. */ }); } } catch { /* synthetic media property in tests */ } };
  const highlightEvidence = (vialId: string, target: EvidenceRecord["target"]) => { select(vialId); setHighlighted(target); };
  const openPanel = (opener: HTMLButtonElement) => { panelOpenerRef.current = opener; setPanelOpen(true); };
  const closePanel = () => setPanelOpen(false);

  return <div className={`app-shell ${recordingMode ? "is-recording" : ""}`}>
    <header className="app-header"><div><span className="eyebrow">FluidMix / operator evidence console</span><h1>{data.source.label} · live run</h1><p>Contract evidence · progressive decision cues</p></div><div className="header-actions"><span className="claim-label">No plant or accuracy claim</span><button type="button" className="control-button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☼ Light" : "◐ Dark"}</button><button type="button" className="control-button" aria-label="Open evidence panel" onClick={event => openPanel(event.currentTarget)}>Evidence panel</button></div></header>
    <main><div className="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</div>
      <div className="dashboard-grid">
        <section className="panel run-panel" aria-label="Live run"><div className="section-heading"><div><span className="eyebrow">Live stitched sequence</span><h2>{data.playback.title}</h2></div><button type="button" className="control-button" aria-label="Restart demo" onClick={restart}>↻ Restart demo</button></div><video ref={videoRef} controls={!recordingMode} autoPlay={recordingMode} muted={recordingMode} preload="metadata" aria-label={`${data.playback.run_id} stitched vial video`} onTimeUpdate={event => updateTime(event.currentTarget.currentTime)}><source src={data.playback.video.src} type={data.playback.video.media_type} /></video><div className="run-meta"><span>RUN {data.playback.run_id}</span><span>{data.playback.video.duration_seconds.value.toFixed(1)} SEC</span><span>{data.vials.length} VIAL SLOTS</span></div></section>
        <VialDetail vial={selected} highlighted={highlighted} revealed={revealed.has(selected.vial_id)} />
        <TopDownPanel data={data} selectedId={selectedId} revealed={revealed} onSelect={select} />
      </div>
      <section className="timeline" aria-label="Vial timeline"><div className="timeline-heading"><span className="eyebrow">Progressive decision stream</span><span>{revealed.size} / {data.vials.length} revealed</span></div><div className="vial-strip" style={{ gridTemplateColumns: `repeat(${data.vials.length}, minmax(120px, 1fr))` }}>{data.vials.map(vial => <article key={vial.vial_id} className={`vial-card ${selectedId === vial.vial_id ? "is-selected" : ""}`} aria-label={`Vial slot ${vial.slot}`}><button type="button" aria-label={`Select vial slot ${vial.slot}`} aria-pressed={selectedId === vial.vial_id} onClick={() => select(vial.vial_id)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") select(vial.vial_id); }}><span className="slot-number">{String(vial.slot).padStart(2, "0")}</span><span><small>ACTUAL RECIPE</small><b>{vial.recipe.recipe_id}</b></span>{revealed.has(vial.vial_id) ? <span className={`decision-chip decision-chip--${vial.decision.outcome}`}>{vial.decision.outcome}</span> : <span className="pending-chip">PENDING</span>}</button></article>)}</div>{data.vials.some(vial => revealed.has(vial.vial_id) && vial.decision.outcome === "discard") && <section className="explanation-stream" aria-label="Retained discard explanations">{data.vials.filter(vial => revealed.has(vial.vial_id) && vial.decision.outcome === "discard").map(vial => <article key={vial.vial_id} className="stream-explanation" aria-label={`Vial slot ${vial.slot} discard explanation`}><div className="stream-heading"><span className="stream-source">{explanationSourceLabel(vial.decision.explanation.source)}</span><b>VIAL {String(vial.slot).padStart(2, "0")}</b></div><p>{vial.decision.explanation.text}</p><div className="evidence-links">{vial.evidence.map(evidence => <button type="button" key={evidence.evidence_id} aria-label={`Evidence ${evidence.label} ${evidence.evidence_id}`} title={evidence.provenance.note} data-evidence-highlight={highlightedExplanationId === vial.vial_id ? "true" : undefined} onClick={() => highlightEvidence(vial.vial_id, evidence.target)}>{evidence.evidence_id}</button>)}</div></article>)}</section>}</section>
    </main>
    {panelOpen && <><button type="button" className="drawer-scrim" aria-label="Close evidence panel backdrop" onClick={closePanel} /><SidePanel data={data} vial={selected} revealed={revealed.has(selected.vial_id)} tab={tab} onTab={setTab} onClose={closePanel} /></>}
  </div>;
}
