import { describe, expect, it, vi } from "vitest";

import { HttpRunnerClient } from "./client";

const EXPECTED_COMMIT = "cf6829b99486b41f4190874dbd701ad812648217";

describe("HttpRunnerClient", () => {
  it("fails closed when the sole runner base URL setting is missing", async () => {
    const fetcher = vi.fn();
    const client = new HttpRunnerClient("", EXPECTED_COMMIT, fetcher as typeof fetch);
    const file = new File(["video"], "three-pour.avi", { type: "video/x-msvideo" });

    await expect(client.createSession({ type: "video", file, runContext: { operator_run_id: "run-1", recipe_id: null, tray_id: "tray-1" } })).rejects.toThrow(/VITE_RUNNER_BASE_URL/i);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("creates upload, V4L2 and RTSP sessions without a precomputed data request", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ session_id: "session-7", test_only: true, pipeline_commit: EXPECTED_COMMIT }) });
    const client = new HttpRunnerClient("https://runner.example/", EXPECTED_COMMIT, fetcher as typeof fetch);
    const video = new File(["video"], "three-pour.avi", { type: "video/x-msvideo" });
    const runContext = { operator_run_id: "run-7", recipe_id: null, tray_id: "tray-1" };

    expect(await client.createSession({ type: "video", file: video, runContext })).toMatchObject({ test_only: true });
    await client.createSession({ type: "v4l2", device: "/dev/video2", runContext });
    await client.createSession({ type: "rtsp", url: "rtsp://camera.local/live", runContext });

    const upload = fetcher.mock.calls[0];
    expect(upload[0]).toBe("https://runner.example/api/sessions");
    expect(upload[1].body).toBeInstanceOf(FormData);
    expect((upload[1].body as FormData).get("source_kind")).toBe("file");
    expect((upload[1].body as FormData).get("file")).toBe(video);
    expect(JSON.parse(String((upload[1].body as FormData).get("run_context")))).toEqual(runContext);
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ source: { kind: "v4l2", device: "/dev/video2" }, run_context: runContext });
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({ source: { kind: "rtsp", url: "rtsp://camera.local/live" }, run_context: runContext });
    expect(fetcher.mock.calls.flatMap(call => call).join(" ")).not.toMatch(/active-run|mock-run|final/i);
  });

  it("reads typed service health from the sole runner origin", async () => {
    const health = { status: "healthy", service: "fluidmix-pipeline-runner", api_version: "v1", pipeline_commit: EXPECTED_COMMIT, test_only: false, inference: true, active_sessions: 0, limits: [] };
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => health });
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, fetcher as typeof fetch);

    await expect(client.getHealth()).resolves.toEqual(health);
    expect(fetcher).toHaveBeenCalledWith("https://runner.example/api/health");
  });

  it("reads and validates the current session lifecycle state", async () => {
    const session = { session_id: "session/a", state: "completed", pipeline_commit: EXPECTED_COMMIT, test_only: false, last_sequence: 17 };
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => session });
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, fetcher as typeof fetch);

    await expect(client.getSession("session/a")).resolves.toEqual(session);
    expect(fetcher).toHaveBeenCalledWith("https://runner.example/api/sessions/session%2Fa");
  });

  it("rejects a session attested by a different pipeline commit", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ session_id: "session-7", test_only: false, pipeline_commit: "0000000000000000000000000000000000000000" }) });
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, fetcher as typeof fetch);
    const file = new File(["video"], "three-pour.avi", { type: "video/x-msvideo" });

    await expect(client.createSession({ type: "video", file, runContext: { operator_run_id: "run-1", recipe_id: null, tray_id: "tray-1" } })).rejects.toThrow(/session.*pipeline commit.*does not match/i);
  });

  it("acknowledges the exact alarm idempotently by alarm URL", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ alarm_id: "alarm-7", state: "acknowledged" }) });
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, fetcher as typeof fetch);

    await client.acknowledge("session/a", "alarm/7", "operator-7");

    expect(fetcher).toHaveBeenCalledWith(
      "https://runner.example/api/sessions/session%2Fa/alarms/alarm%2F7/acknowledgement",
      expect.objectContaining({ method: "PUT", body: JSON.stringify({ acknowledged_by: "operator-7" }) })
    );
  });

  it("receives named runner SSE events, guards envelopes and removes listeners on cleanup", () => {
    const listeners = new Map<string, EventListener>();
    const addEventListener = vi.fn((type: string, listener: EventListener) => listeners.set(type, listener));
    const removeEventListener = vi.fn((type: string, listener: EventListener) => {
      if (listeners.get(type) === listener) listeners.delete(type);
    });
    const closeSource = vi.fn<() => void>();
    const source = { close: closeSource, addEventListener, removeEventListener };
    const eventSourceFactory = vi.fn(() => source);
    const onEvent = vi.fn();
    const onDisconnect = vi.fn();
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, vi.fn() as unknown as typeof fetch, eventSourceFactory);

    const close = client.subscribe("session/a", 12, onEvent, onDisconnect);
    expect(eventSourceFactory).toHaveBeenCalledWith("https://runner.example/api/sessions/session%2Fa/events?after=12");
    expect(addEventListener.mock.calls.map(call => call[0])).toEqual([
      "measuring", "waiting_for_references", "vial_decision", "model_vote",
      "averaged_decision", "explanation", "error", "rate_health"
    ]);

    const valid = { schema_version: "runner-event.v1", event_id: "event-13", session_id: "session/a", sequence: 13, emitted_at: "2026-11-01T00:00:00Z", pipeline_commit: EXPECTED_COMMIT, source: { source_id: "source-1", kind: "file", label: "video.avi" }, type: "measuring", payload: {} };
    listeners.get("measuring")?.(new MessageEvent("measuring", { data: JSON.stringify(valid) }));
    expect(onEvent).toHaveBeenCalledWith(valid);
    listeners.get("error")?.(new Event("error"));
    expect(onDisconnect).toHaveBeenCalledWith({ kind: "transport", message: "Runner connection interrupted" });
    listeners.get("rate_health")?.(new MessageEvent("rate_health", { data: JSON.stringify({ type: "rate_health", sequence: 14 }) }));
    expect(onDisconnect).toHaveBeenCalledWith({ kind: "terminal", message: "Runner sent an invalid event" });
    expect(onEvent).toHaveBeenCalledTimes(1);
    listeners.get("rate_health")?.(new MessageEvent("rate_health", { data: JSON.stringify({ ...valid, event_id: "event-14", sequence: 14, type: "rate_health", pipeline_commit: "0000000000000000000000000000000000000000" }) }));
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onDisconnect).toHaveBeenCalledTimes(2);
    close();
    expect(removeEventListener).toHaveBeenCalledTimes(8);
    expect(listeners.size).toBe(0);
    expect(closeSource).toHaveBeenCalledOnce();
  });

  it("closes terminally on a commit mismatch and ignores every later event", () => {
    const listeners = new Map<string, EventListener>();
    const closeSource = vi.fn<() => void>();
    const source = {
      addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener),
      removeEventListener: vi.fn(),
      close: closeSource,
    };
    const onEvent = vi.fn();
    const onDisconnect = vi.fn();
    const client = new HttpRunnerClient("https://runner.example", EXPECTED_COMMIT, vi.fn() as unknown as typeof fetch, () => source);
    client.subscribe("session/a", 0, onEvent, onDisconnect);
    const valid = { schema_version: "runner-event.v1", event_id: "event-2", session_id: "session/a", sequence: 2, emitted_at: "2026-11-01T00:00:02Z", pipeline_commit: EXPECTED_COMMIT, source: { source_id: "source-1", kind: "file", label: "video.avi" }, type: "measuring", payload: {} };

    listeners.get("measuring")?.(new MessageEvent("measuring", { data: JSON.stringify({ ...valid, event_id: "event-1", sequence: 1, pipeline_commit: "0000000000000000000000000000000000000000" }) }));
    listeners.get("measuring")?.(new MessageEvent("measuring", { data: JSON.stringify(valid) }));

    expect(closeSource).toHaveBeenCalledOnce();
    expect(onDisconnect).toHaveBeenCalledOnce();
    expect(onDisconnect).toHaveBeenCalledWith({ kind: "terminal", message: "Runner event pipeline commit does not match the expected pin" });
    expect(onEvent).not.toHaveBeenCalled();
  });
});
