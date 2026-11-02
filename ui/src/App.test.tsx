import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import pipelinePin from "../config/pipeline-pin.json";

const EXPECTED_COMMIT = pipelinePin.expected_pipeline_commit;

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe("App", () => {
  it("uses the same-origin runner route when no explicit backend URL is configured", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "healthy",
        service: "fluidmix-runner",
        api_version: "1.0",
        pipeline_commit: EXPECTED_COMMIT,
        test_only: false,
        inference: true,
        active_sessions: 0,
        limits: [],
      }),
    });
    vi.stubGlobal("fetch", fetcher);
    const { App } = await import("./App");

    render(<App />);

    expect(await screen.findByText(/service verified/i)).toBeVisible();
    expect(fetcher.mock.calls[0][0]).toBe("/api/health");
  });

  it("loads service health using the canonical pipeline pin without fetching Phase 2 artifacts", async () => {
    vi.stubEnv("VITE_RUNNER_BASE_URL", "https://runner.example");
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "healthy",
        service: "fluidmix-runner",
        api_version: "1.0",
        pipeline_commit: EXPECTED_COMMIT,
        test_only: false,
        inference: true,
        active_sessions: 0,
        limits: [],
      }),
    });
    vi.stubGlobal("fetch", fetcher);
    const { App } = await import("./App");

    render(<App />);

    expect(screen.getByRole("heading", { name: /live vial decisions/i })).toBeVisible();
    expect(screen.getByRole("group", { name: /inspection source/i })).toBeVisible();
    expect(await screen.findByText(/service verified/i)).toBeVisible();
    expect(screen.getByLabelText(`Pipeline commit ${EXPECTED_COMMIT}`)).toBeVisible();
    expect(screen.getByText("Development data, frozen-cohort hybrid, service 2344f3c", { exact: true })).toBeVisible();
    expect(fetcher.mock.calls[0][0]).toBe("https://runner.example/api/health");
    expect(fetcher.mock.calls.flat().join(" ")).not.toMatch(/active-run|committed-media|mock/i);
    expect(document.body).not.toHaveTextContent(/final data|demo mode/i);
  });
});
