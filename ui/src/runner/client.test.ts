import { describe, expect, it, vi } from "vitest";

import { HttpRunnerClient } from "./client";

describe("HttpRunnerClient", () => {
  it("fails closed when the sole runner base URL setting is missing", async () => {
    const fetcher = vi.fn();
    const client = new HttpRunnerClient("", fetcher as typeof fetch);
    const file = new File(["video"], "three-pour.avi", { type: "video/x-msvideo" });

    await expect(client.createSession({ type: "video", file, runContext: { operator_run_id: "run-1", recipe_id: null, tray_id: "tray-1" } })).rejects.toThrow(/VITE_RUNNER_BASE_URL/i);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("creates upload, V4L2 and RTSP sessions without a precomputed data request", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ session_id: "session-7", test_only: true, pipeline_commit: "7bf35aeea21c36acf1a390661bd05fea2b07b190" }) });
    const client = new HttpRunnerClient("https://runner.example/", fetcher as typeof fetch);
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

  it("acknowledges the exact alarm idempotently by alarm URL", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ alarm_id: "alarm-7", state: "acknowledged" }) });
    const client = new HttpRunnerClient("https://runner.example", fetcher as typeof fetch);

    await client.acknowledge("session/a", "alarm/7", "operator-7");

    expect(fetcher).toHaveBeenCalledWith(
      "https://runner.example/api/sessions/session%2Fa/alarms/alarm%2F7/acknowledgement",
      expect.objectContaining({ method: "PUT", body: JSON.stringify({ acknowledged_by: "operator-7" }) })
    );
  });

  it("opens replayable SSE after the last applied sequence and reports disconnect", () => {
    const closeSource = vi.fn<() => void>();
    const source: { close: () => void; onmessage: ((event: MessageEvent<string>) => void) | null; onerror: ((event: Event) => void) | null } = { close: closeSource, onmessage: null, onerror: null };
    const eventSourceFactory = vi.fn(() => source);
    const onEvent = vi.fn();
    const onDisconnect = vi.fn();
    const client = new HttpRunnerClient("https://runner.example", vi.fn() as unknown as typeof fetch, eventSourceFactory);

    const close = client.subscribe("session/a", 12, onEvent, onDisconnect);
    expect(eventSourceFactory).toHaveBeenCalledWith("https://runner.example/api/sessions/session%2Fa/events?after=12");

    source.onerror?.(new Event("error"));
    expect(onDisconnect).toHaveBeenCalledWith("Runner connection lost");
    close();
    expect(closeSource).toHaveBeenCalledOnce();
  });
});
