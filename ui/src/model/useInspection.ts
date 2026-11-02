/**
 * Session lifecycle for the operator UI: health and pin check, session start or attach, the event
 * stream, decision receipts, the optional evidence bundle, alarm acknowledgement, operator records
 * and the locally played video. Screens receive the returned `InspectionModel` and never talk to
 * the runner directly.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { RunContext, RunnerClient, RunnerHealth, RunnerSource } from "../runner/client";
import { initialRunnerState, reduceRunnerEvent, type RunnerState } from "../runner/contract";
import type { DecisionRecords, EvidenceBundle } from "../runner/evidence";
import { loadOperatorLog, newRecordId, saveOperatorLog, type OperatorRecord } from "./operatorLog";
import { DEFAULT_VIAL_COUNT, finalDisposition } from "./selectors";

/** The service that produced the disclosed frozen-cohort hybrid development result. */
export const HYBRID_SERVICE_COMMIT = "2344f3c655d9aa255eb576e6d2a280bd0fc9bdff";

export type PinStatus = "unknown" | "verified" | "dev_override" | "mismatch";
export type RunPhase = "idle" | "starting" | "running" | "completed" | "stopped" | "failed";

export interface InspectionModel {
  readonly health: RunnerHealth | null;
  readonly healthError: string | null;
  readonly pinStatus: PinStatus;
  readonly expectedCommit: string;
  readonly disclosedHybrid: boolean;
  readonly runner: RunnerState | null;
  readonly runPhase: RunPhase;
  readonly connection: string;
  readonly startError: string | null;
  readonly streamFailure: string | null;
  readonly busy: boolean;
  readonly runContext: RunContext | null;
  readonly sourceKind: RunnerSource["type"] | "attached" | null;
  readonly sourceLabel: string | null;
  /** Object URL of the uploaded file, played locally; null for camera/RTSP or attached sessions. */
  readonly videoUrl: string | null;
  readonly decisions: DecisionRecords | null;
  readonly decisionsError: string | null;
  /** undefined = not fetched yet; null = runner does not expose it. */
  readonly bundle: EvidenceBundle | null | undefined;
  readonly vialCount: number;
  readonly fps: number | null;
  readonly framesPerVial: number | null;
  readonly selectedVialId: string | null;
  readonly localAcks: ReadonlySet<string>;
  readonly operatorLog: readonly OperatorRecord[];
  start(source: RunnerSource): Promise<void>;
  attach(sessionId: string, videoFile?: File | null): Promise<void>;
  setVideoFile(file: File | null): void;
  selectVial(vialId: string | null): void;
  acknowledge(alarmId: string, origin: "runner" | "ui", operator: string): Promise<void>;
  recordOperatorAction(input: Omit<OperatorRecord, "record_id" | "recorded_at" | "session_id">): OperatorRecord | null;
  imageUrl(imageId: string): string | null;
  refreshDecisions(): void;
}

export function useInspection(client: RunnerClient, expectedCommit: string, devAllowedCommits: readonly string[] = []): InspectionModel {
  const [health, setHealth] = useState<RunnerHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [runner, setRunner] = useState<RunnerState | null>(null);
  const [runPhase, setRunPhase] = useState<RunPhase>("idle");
  const [connection, setConnection] = useState("Choose a source to start live inspection.");
  const [startError, setStartError] = useState<string | null>(null);
  const [streamFailure, setStreamFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [runContext, setRunContext] = useState<RunContext | null>(null);
  const [sourceKind, setSourceKind] = useState<InspectionModel["sourceKind"]>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecords | null>(null);
  const [decisionsError, setDecisionsError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<EvidenceBundle | null | undefined>(undefined);
  const [selectedVialId, setSelectedVialId] = useState<string | null>(null);
  const [localAcks, setLocalAcks] = useState<ReadonlySet<string>>(new Set());
  const [operatorLog, setOperatorLog] = useState<OperatorRecord[]>(() => loadOperatorLog());
  const closeStream = useRef<(() => void) | null>(null);
  const sessionVersion = useRef(0);
  const mounted = useRef(true);
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    let active = true;
    client.getHealth().then(value => { if (active) { setHealth(value); setHealthError(null); } })
      .catch(reason => { if (active) setHealthError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { active = false; };
  }, [client]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; sessionVersion.current += 1; closeStream.current?.(); };
  }, []);

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL?.(videoUrl); }, [videoUrl]);

  const pinStatus: PinStatus = !health ? "unknown"
    : health.pipeline_commit === expectedCommit ? "verified"
      : devAllowedCommits.includes(health.pipeline_commit) ? "dev_override" : "mismatch";
  const disclosedHybrid = !!health && !health.test_only && health.pipeline_commit === HYBRID_SERVICE_COMMIT;

  const decisionsRequest = useRef(0);
  const fetchDecisions = useCallback((sessionId: string, version: number) => {
    const request = ++decisionsRequest.current;
    client.getDecisions(sessionId).then(value => {
      // Only the newest request may apply: an older, slower response must not overwrite a newer receipt.
      if (mounted.current && version === sessionVersion.current && request === decisionsRequest.current) { setDecisions(value); setDecisionsError(null); }
    }).catch(reason => {
      if (mounted.current && version === sessionVersion.current) setDecisionsError(reason instanceof Error ? reason.message : String(reason));
    });
    client.getEvidenceBundle(sessionId).then(value => {
      if (mounted.current && version === sessionVersion.current) setBundle(value);
    }).catch(() => {
      if (mounted.current && version === sessionVersion.current) setBundle(null);
    });
  }, [client]);

  const follow = useCallback((sessionId: string, testOnly: boolean, version: number) => {
    sessionIdRef.current = sessionId;
    setRunner(initialRunnerState(sessionId, testOnly));
    setSelectedVialId(null);
    setLocalAcks(new Set());
    setDecisions(null);
    setBundle(undefined);
    setRunPhase("running");
    setConnection(`Connected · session ${sessionId}`);
    fetchDecisions(sessionId, version);
    let lastSequence = 0;
    const connect = (after: number) => { closeStream.current = client.subscribe(sessionId, after, next => {
      if (!mounted.current || version !== sessionVersion.current) return;
      lastSequence = Math.max(lastSequence, next.sequence);
      setRunner(current => current ? reduceRunnerEvent(current, next) : current);
      if (next.type === "vial_decision" || next.type === "averaged_decision" || next.type === "explanation") fetchDecisions(sessionId, version);
    }, notice => {
      if (!mounted.current || version !== sessionVersion.current) return;
      if (notice.kind === "terminal") {
        setConnection("Event stream stopped.");
        setStreamFailure(notice.message);
        setRunPhase("failed");
        return;
      }
      setConnection("Connection interrupted — reconnecting.");
      void client.getSession(sessionId).then(current => {
        if (!mounted.current || version !== sessionVersion.current) return;
        if (current.state === "completed" || current.state === "stopped") {
          closeStream.current?.();
          setConnection(current.state === "completed" ? "Run completed" : "Run stopped");
          setRunPhase(current.state);
          fetchDecisions(sessionId, version);
        } else if (current.state === "failed") {
          closeStream.current?.();
          setConnection("Run failed");
          setRunPhase("failed");
          setStreamFailure("Run failed. The runner reported that this session failed.");
        } else {
          // Still running: resume from the last applied event (replay is idempotent in the reducer).
          setTimeout(() => { if (mounted.current && version === sessionVersion.current) connect(lastSequence); }, 1000);
        }
      }).catch(() => {
        if (mounted.current && version === sessionVersion.current) setConnection("Connection interrupted — reconnecting. Session state is unavailable.");
      });
    }); };
    connect(0);
  }, [client, fetchDecisions]);

  const start = useCallback(async (source: RunnerSource) => {
    if (pinStatus !== "verified" && pinStatus !== "dev_override") {
      setStartError("Cannot start until runner health and pipeline commit are verified.");
      return;
    }
    setBusy(true); setStartError(null); setStreamFailure(null); setRunPhase("starting");
    closeStream.current?.();
    const version = ++sessionVersion.current;
    try {
      const session = await client.createSession(source);
      if (!mounted.current || version !== sessionVersion.current) return;
      if (health && session.pipeline_commit !== health.pipeline_commit) throw new Error(`Session commit ${session.pipeline_commit.slice(0, 7)} differs from service health ${health.pipeline_commit.slice(0, 7)}`);
      setRunContext(source.runContext);
      setSourceKind(source.type);
      setSourceLabel(source.type === "video" ? source.file.name : source.type === "v4l2" ? source.device : source.url);
      setVideoUrl(source.type === "video" && typeof URL.createObjectURL === "function" ? URL.createObjectURL(source.file) : null);
      follow(session.session_id, session.test_only, version);
    } catch (reason) {
      if (!mounted.current || version !== sessionVersion.current) return;
      setRunner(null); setRunPhase("idle"); setConnection("Not connected.");
      setStartError(`Could not start live inspection. ${reason instanceof Error ? reason.message : String(reason)}`);
    } finally {
      if (mounted.current && version === sessionVersion.current) setBusy(false);
    }
  }, [client, follow, pinStatus, health]);

  const attach = useCallback(async (sessionId: string, videoFile?: File | null) => {
    if (pinStatus !== "verified" && pinStatus !== "dev_override") {
      setStartError("Cannot open a session until runner health and pipeline commit are verified.");
      return;
    }
    setBusy(true); setStartError(null); setStreamFailure(null);
    closeStream.current?.();
    const version = ++sessionVersion.current;
    try {
      const session = await client.getSession(sessionId);
      if (!mounted.current || version !== sessionVersion.current) return;
      if (health && session.pipeline_commit !== health.pipeline_commit) throw new Error(`Session commit ${session.pipeline_commit.slice(0, 7)} differs from service health ${health.pipeline_commit.slice(0, 7)}`);
      setRunContext(null);
      setSourceKind("attached");
      setSourceLabel(`session ${sessionId}`);
      setVideoUrl(videoFile && typeof URL.createObjectURL === "function" ? URL.createObjectURL(videoFile) : null);
      follow(session.session_id, session.test_only, version);
    } catch (reason) {
      if (!mounted.current || version !== sessionVersion.current) return;
      setRunPhase("idle");
      setStartError(`Could not open session ${sessionId}. ${reason instanceof Error ? reason.message : String(reason)}`);
    } finally {
      if (mounted.current && version === sessionVersion.current) setBusy(false);
    }
  }, [client, follow, pinStatus, health]);

  const setVideoFile = useCallback((file: File | null) => {
    setVideoUrl(file && typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : null);
  }, []);

  const acknowledge = useCallback(async (alarmId: string, origin: "runner" | "ui", operator: string) => {
    const sessionId = sessionIdRef.current;
    if (!sessionId) return;
    if (origin === "runner") {
      try {
        await client.acknowledge(sessionId, alarmId, operator || "operator");
      } catch (reason) {
        setStartError(`Could not acknowledge ${alarmId}. ${reason instanceof Error ? reason.message : String(reason)}`);
        return;
      }
    }
    setLocalAcks(current => new Set(current).add(alarmId));
  }, [client]);

  const recordOperatorAction = useCallback((input: Omit<OperatorRecord, "record_id" | "recorded_at" | "session_id">) => {
    const sessionId = sessionIdRef.current;
    if (!sessionId) return null;
    const record: OperatorRecord = { ...input, record_id: newRecordId(), recorded_at: new Date().toISOString(), session_id: sessionId };
    setOperatorLog(current => { const next = [...current, record]; saveOperatorLog(next); return next; });
    return record;
  }, []);

  const imageUrl = useCallback((imageId: string) => {
    const sessionId = sessionIdRef.current;
    return sessionId && bundle ? client.evidenceImageUrl(sessionId, imageId) : null;
  }, [bundle, client]);

  const refreshDecisions = useCallback(() => {
    if (sessionIdRef.current) fetchDecisions(sessionIdRef.current, sessionVersion.current);
  }, [fetchDecisions]);

  // Default selection: the newest judged vial, otherwise the newest vial seen.
  const effectiveSelected = useMemo(() => {
    if (!runner) return null;
    if (selectedVialId && runner.vials[selectedVialId]) return selectedVialId;
    const judged = runner.trayOrder.filter(id => finalDisposition(runner.vials[id]));
    return judged.at(-1) ?? runner.trayOrder.at(-1) ?? null;
  }, [runner, selectedVialId]);

  const profile = health?.frozen_profile;
  return {
    health, healthError, pinStatus, expectedCommit, disclosedHybrid, runner, runPhase, connection, startError, streamFailure, busy,
    runContext, sourceKind, sourceLabel, videoUrl, decisions, decisionsError, bundle,
    vialCount: profile?.vial_count ?? DEFAULT_VIAL_COUNT,
    fps: profile?.fps ?? bundle?.frame_rate_hz ?? null,
    framesPerVial: profile?.frames_per_vial ?? null,
    selectedVialId: effectiveSelected, localAcks, operatorLog,
    start, attach, setVideoFile, selectVial: setSelectedVialId, acknowledge, recordOperatorAction, imageUrl, refreshDecisions
  };
}
