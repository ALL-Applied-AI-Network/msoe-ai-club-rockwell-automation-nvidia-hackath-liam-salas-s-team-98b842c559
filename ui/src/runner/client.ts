import type { RunnerEvent } from "./contract";

export interface RunContext {
  readonly operator_run_id: string;
  readonly recipe_id: string | null;
  readonly tray_id: string;
}

export type RunnerSource =
  | { readonly type: "video"; readonly file: File; readonly runContext: RunContext }
  | { readonly type: "v4l2"; readonly device: string; readonly runContext: RunContext }
  | { readonly type: "rtsp"; readonly url: string; readonly runContext: RunContext };

export interface SessionCreated {
  readonly session_id: string;
  readonly test_only: boolean;
  readonly pipeline_commit: string;
}

export interface RunnerHealth {
  readonly status: "healthy" | "degraded" | "stalled";
  readonly service: string;
  readonly api_version: string;
  readonly pipeline_commit: string;
  readonly test_only: boolean;
  readonly inference: boolean;
  readonly active_sessions: number;
  readonly limits: readonly string[];
}

export interface RunnerClient {
  getHealth(): Promise<RunnerHealth>;
  createSession(source: RunnerSource): Promise<SessionCreated>;
  subscribe(
    sessionId: string,
    afterSequence: number,
    onEvent: (event: RunnerEvent) => void,
    onDisconnect: (message: string) => void
  ): () => void;
  acknowledge(sessionId: string, alarmId: string, acknowledgedBy: string): Promise<void>;
}

interface EventStream {
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
  close(): void;
}

type EventSourceFactory = (url: string) => EventStream;
const RUNNER_EVENT_TYPES = ["measuring", "waiting_for_references", "vial_decision", "model_vote", "averaged_decision", "explanation", "error", "rate_health"] as const;

function isRunnerEvent(value: unknown, expectedType: typeof RUNNER_EVENT_TYPES[number], sessionId: string): value is RunnerEvent {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const event = value as Record<string, unknown>;
  return event.schema_version === "runner-event.v1"
    && event.type === expectedType
    && event.session_id === sessionId
    && typeof event.pipeline_commit === "string"
    && Number.isInteger(event.sequence)
    && (event.sequence as number) > 0
    && !!event.payload
    && typeof event.payload === "object"
    && !Array.isArray(event.payload);
}

function joinUrl(origin: string, path: string): string {
  if (!origin) throw new Error("VITE_RUNNER_BASE_URL is not configured");
  return `${origin.replace(/\/$/, "")}${path}`;
}

async function requireOk(response: Response): Promise<Response> {
  if (!response.ok) throw new Error(`Runner request failed (HTTP ${response.status})`);
  return response;
}

export class HttpRunnerClient implements RunnerClient {
  constructor(
    private readonly origin: string,
    private readonly expectedPipelineCommit: string,
    private readonly fetcher: typeof fetch = fetch,
    private readonly eventSourceFactory: EventSourceFactory = url => new EventSource(url)
  ) {}

  async getHealth(): Promise<RunnerHealth> {
    const response = await requireOk(await this.fetcher(joinUrl(this.origin, "/api/health")));
    const value = await response.json() as Partial<RunnerHealth>;
    if ((value.status !== "healthy" && value.status !== "degraded" && value.status !== "stalled")
      || typeof value.service !== "string"
      || typeof value.api_version !== "string"
      || typeof value.pipeline_commit !== "string"
      || typeof value.test_only !== "boolean"
      || typeof value.inference !== "boolean"
      || !Number.isInteger(value.active_sessions)
      || !Array.isArray(value.limits)) throw new Error("Runner returned invalid health data");
    return value as RunnerHealth;
  }

  async createSession(source: RunnerSource): Promise<SessionCreated> {
    let init: RequestInit;
    if (source.type === "video") {
      const body = new FormData();
      body.set("source_kind", "file");
      body.set("file", source.file);
      body.set("run_context", JSON.stringify(source.runContext));
      init = { method: "POST", body };
    } else {
      const sourceBody = source.type === "v4l2"
        ? { kind: source.type, device: source.device }
        : { kind: source.type, url: source.url };
      init = {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ source: sourceBody, run_context: source.runContext })
      };
    }
    const response = await requireOk(await this.fetcher(joinUrl(this.origin, "/api/sessions"), init));
    const value = await response.json() as { session_id?: unknown; test_only?: unknown; pipeline_commit?: unknown };
    if (typeof value.session_id !== "string" || !value.session_id) throw new Error("Runner returned no session ID");
    if (typeof value.test_only !== "boolean" || typeof value.pipeline_commit !== "string") throw new Error("Runner returned an invalid session");
    if (value.pipeline_commit !== this.expectedPipelineCommit) throw new Error("Session pipeline commit does not match the expected pin");
    return { session_id: value.session_id, test_only: value.test_only, pipeline_commit: value.pipeline_commit };
  }

  subscribe(
    sessionId: string,
    afterSequence: number,
    onEvent: (event: RunnerEvent) => void,
    onDisconnect: (message: string) => void
  ): () => void {
    const path = `/api/sessions/${encodeURIComponent(sessionId)}/events?after=${afterSequence}`;
    const source = this.eventSourceFactory(joinUrl(this.origin, path));
    const listeners = RUNNER_EVENT_TYPES.map(type => {
      const listener: EventListener = rawEvent => {
        if (type === "error" && !(rawEvent instanceof MessageEvent)) {
          onDisconnect("Runner connection lost");
          return;
        }
        try {
          const value: unknown = JSON.parse((rawEvent as MessageEvent<string>).data);
          if (!isRunnerEvent(value, type, sessionId)) throw new Error("invalid envelope");
          if (value.pipeline_commit !== this.expectedPipelineCommit) {
            onDisconnect("Runner event pipeline commit does not match the expected pin");
            return;
          }
          onEvent(value);
        } catch {
          onDisconnect("Runner sent an invalid event");
        }
      };
      source.addEventListener(type, listener);
      return { type, listener };
    });
    return () => {
      listeners.forEach(({ type, listener }) => source.removeEventListener(type, listener));
      source.close();
    };
  }

  async acknowledge(sessionId: string, alarmId: string, acknowledgedBy: string): Promise<void> {
    await requireOk(await this.fetcher(
      joinUrl(this.origin, `/api/sessions/${encodeURIComponent(sessionId)}/alarms/${encodeURIComponent(alarmId)}/acknowledgement`),
      {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ acknowledged_by: acknowledgedBy })
      }
    ));
  }
}
