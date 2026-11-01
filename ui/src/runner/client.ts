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

export interface RunnerClient {
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
  onmessage: ((event: MessageEvent<string>) => void) | null;
  onerror: ((event: Event) => void) | null;
  close(): void;
}

type EventSourceFactory = (url: string) => EventStream;

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
    private readonly fetcher: typeof fetch = fetch,
    private readonly eventSourceFactory: EventSourceFactory = url => new EventSource(url)
  ) {}

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
    source.onmessage = message => {
      try {
        onEvent(JSON.parse(message.data) as RunnerEvent);
      } catch {
        onDisconnect("Runner sent an invalid event");
      }
    };
    source.onerror = () => onDisconnect("Runner connection lost");
    return () => source.close();
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
