import { StrictMode } from "react";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { RunnerClient, RunnerSession } from "../runner/client";
import "../styles.css";
import { FakeRunnerClient as BaseFake, HYBRID_COMMIT, TEST_COMMIT, TEST_HEALTH, runnerEvent } from "../test/fakeClient";
import { AppShell } from "./AppShell";

const EXPECTED_COMMIT = TEST_COMMIT;
const HEALTH = TEST_HEALTH;
type State = RunnerSession["state"];

class FakeRunnerClient extends BaseFake {
  constructor(health: typeof HEALTH | Error = HEALTH, sessionState: State = "running", sessionLookup?: (id: string) => Promise<RunnerSession>) {
    super({ health, sessionState, sessionLookup, sessionCommit: EXPECTED_COMMIT });
  }
}

function LiveHmi({ client, expectedPipelineCommit, devAllowedCommits }: { client: RunnerClient; expectedPipelineCommit: string; devAllowedCommits?: string[] }) {
  return <AppShell client={client} expectedCommit={expectedPipelineCommit} devAllowedCommits={devAllowedCommits} />;
}

async function enterRunContext() {
  await userEvent.type(screen.getByLabelText(/operator run id/i), "run-7");
  await userEvent.type(screen.getByLabelText(/tray id/i), "tray-1");
}

async function startUpload(client: FakeRunnerClient) {
  const file = new File(["video"], "three-pour.avi", { type: "video/x-msvideo" });
  await enterRunContext();
  await userEvent.upload(screen.getByLabelText(/video file/i), file);
  expect(await screen.findByText(/service verified/i)).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: /start live inspection/i }));
  expect(await screen.findByText(/session session-live/i)).toBeVisible();
  void client;
  return file;
}

describe("AppShell (session lifecycle)", () => {
  it("creates browser upload, V4L2 and RTSP sources", async () => {
    const uploadClient = new FakeRunnerClient();
    const first = render(<LiveHmi client={uploadClient} expectedPipelineCommit={EXPECTED_COMMIT} />);
    const file = await startUpload(uploadClient);
    const runContext = { operator_run_id: "run-7", recipe_id: null, tray_id: "tray-1" };
    expect(uploadClient.sources).toEqual([{ type: "video", file, runContext }]);
    first.unmount();

    const v4l2Client = new FakeRunnerClient();
    const second = render(<LiveHmi client={v4l2Client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await enterRunContext();
    await userEvent.click(screen.getAllByRole("radio", { name: /v4l2 camera/i }).at(-1)!);
    await userEvent.type(screen.getAllByLabelText(/device path/i).at(-1)!, "/dev/video2");
    await userEvent.click(screen.getAllByRole("button", { name: /start live inspection/i }).at(-1)!);
    expect(v4l2Client.sources).toEqual([{ type: "v4l2", device: "/dev/video2", runContext }]);
    second.unmount();

    const rtspClient = new FakeRunnerClient();
    render(<LiveHmi client={rtspClient} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await enterRunContext();
    await userEvent.click(screen.getAllByRole("radio", { name: /rtsp stream/i }).at(-1)!);
    await userEvent.type(screen.getAllByLabelText(/rtsp url/i).at(-1)!, "rtsp://camera.local/live");
    await userEvent.click(screen.getAllByRole("button", { name: /start live inspection/i }).at(-1)!);
    expect(rtspClient.sources).toEqual([{ type: "rtsp", url: "rtsp://camera.local/live", runContext }]);
  });

  it("shows matching service health and its attested pipeline commit", async () => {
    render(<LiveHmi client={new FakeRunnerClient()} expectedPipelineCommit={EXPECTED_COMMIT} />);

    expect(await screen.findByText(/service verified/i)).toBeVisible();
    expect(screen.getByText(/test-only service/i)).toBeVisible();
    expect(screen.getByLabelText(`Pipeline commit ${EXPECTED_COMMIT}`)).toHaveTextContent(EXPECTED_COMMIT.slice(0, 7));
    expect(screen.queryByText(/pipeline commit mismatch/i)).not.toBeInTheDocument();
  });

  it("labels the disclosed 2344 development result as a frozen-cohort hybrid", async () => {
    const health = { ...HEALTH, pipeline_commit: HYBRID_COMMIT, test_only: false, inference: true };
    render(<LiveHmi client={new FakeRunnerClient(health)} expectedPipelineCommit={HYBRID_COMMIT} />);

    expect(await screen.findByText("Development data, frozen-cohort hybrid, service 2344f3c", { exact: true })).toBeVisible();
  });

  it("warns and prevents session start when health reports a different commit", async () => {
    const client = new FakeRunnerClient({ ...HEALTH, pipeline_commit: "0000000000000000000000000000000000000000" });
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/pipeline commit mismatch.*expected.*cf6829.*received.*000000/is);
    await enterRunContext();
    await userEvent.upload(screen.getByLabelText(/video file/i), new File(["video"], "three-pour.avi", { type: "video/x-msvideo" }));
    expect(screen.getByRole("button", { name: /start live inspection/i })).toBeDisabled();
    expect(client.sources).toHaveLength(0);
  });

  it("fails closed when service health cannot be read", async () => {
    const client = new FakeRunnerClient(new Error("Health endpoint unavailable"));
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not verify.*health endpoint unavailable/i);
    expect(screen.getByRole("button", { name: /start live inspection/i })).toBeDisabled();
    expect(client.sources).toHaveLength(0);
  });

  it("rejects invalid or empty device and stream sources before calling the runner", async () => {
    const client = new FakeRunnerClient();
    const first = render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await enterRunContext();
    await userEvent.click(screen.getByRole("radio", { name: /v4l2 camera/i }));
    expect(screen.getByRole("button", { name: /start live inspection/i })).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/device path/i), "camera0");
    await userEvent.click(screen.getByRole("button", { name: /start live inspection/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/device path.*\/dev\/video/i);
    expect(client.sources).toHaveLength(0);
    first.unmount();

    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await enterRunContext();
    await userEvent.click(screen.getByRole("radio", { name: /rtsp stream/i }));
    expect(screen.getByRole("button", { name: /start live inspection/i })).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/rtsp url/i), "http://camera.local/live");
    await userEvent.click(screen.getByRole("button", { name: /start live inspection/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/rtsp.*rtsp:\/\//i);
    expect(client.sources).toHaveLength(0);
  });

  it("shows start failures and stream disconnects without substituting data", async () => {
    const failing: RunnerClient = {
      getHealth: vi.fn().mockResolvedValue(HEALTH),
      getSession: vi.fn(),
      createSession: vi.fn().mockRejectedValue(new Error("Camera refused the request")),
      subscribe: vi.fn(),
      acknowledge: vi.fn(),
      getDecisions: vi.fn(),
      getEvidenceBundle: vi.fn(),
      evidenceImageUrl: vi.fn()
    };
    const failingView = render(<LiveHmi client={failing} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await enterRunContext();
    await userEvent.click(screen.getByRole("radio", { name: /v4l2 camera/i }));
    await userEvent.type(screen.getByLabelText(/device path/i), "/dev/video0");
    await userEvent.click(screen.getByRole("button", { name: /start live inspection/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not start.*camera refused/i);
    expect(screen.queryByRole("button", { name: /select tray position/i })).not.toBeInTheDocument();
    failingView.unmount();

    const client = new FakeRunnerClient();
    const live = render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);
    act(() => client.emit(runnerEvent(1, "error", { code: "pipeline_failed", message: "Pipeline stopped unexpectedly", stage: "inference", retryable: false, evidence_ids: [] })));
    expect(await screen.findByRole("alert")).toHaveTextContent(/pipeline stopped unexpectedly/i);
    act(() => client.disconnect());
    expect(await screen.findByRole("status")).toHaveTextContent(/connection interrupted.*reconnecting/i);
    live.unmount();
  });

  it("reports a completed run instead of a lost connection when the stream ends", async () => {
    const client = new FakeRunnerClient(HEALTH, "completed");
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);

    act(() => client.disconnect());

    expect(await screen.findByRole("status")).toHaveTextContent(/^Run completed$/i);
    expect(screen.queryByText(/connection lost|no live updates/i)).not.toBeInTheDocument();
  });

  it("keeps lifecycle updates active through Strict Mode effect replay", async () => {
    const client = new FakeRunnerClient(HEALTH, "completed");
    render(<StrictMode><LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} /></StrictMode>);

    await startUpload(client);
    act(() => client.disconnect());

    expect(await screen.findByRole("status")).toHaveTextContent(/^Run completed$/i);
  });

  it("reports an operator-stopped run when the terminal session state is stopped", async () => {
    const client = new FakeRunnerClient(HEALTH, "stopped");
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);

    act(() => {
      client.emit(runnerEvent(1, "averaged_decision", { disposition: "discard", average: 0.9 }, "CASE-target", 0));
      client.emit(runnerEvent(2, "measuring", {}, "CASE-reference-2", 1));
      client.emit(runnerEvent(3, "measuring", {}, "CASE-reference-3", 2));
    });

    act(() => client.disconnect());

    expect(await screen.findByRole("status")).toHaveTextContent(/^Run stopped$/i);
  });

  it("shows a clear alert when the session state is failed", async () => {
    const client = new FakeRunnerClient(HEALTH, "failed");
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);

    act(() => client.disconnect());

    expect(await screen.findByRole("alert")).toHaveTextContent(/run failed.*session/i);
  });

  it("keeps fatal stream rejection terminal and visible", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);

    act(() => client.rejectStream("Runner sent an invalid event"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/event stream stopped.*invalid event/i);
  });

  it("ignores a delayed lifecycle result after replacing the session", async () => {
    let resolveLookup!: (value: { session_id: string; state: "completed"; pipeline_commit: string; test_only: boolean; last_sequence: number }) => void;
    const pendingLookup = new Promise<{ session_id: string; state: "completed"; pipeline_commit: string; test_only: boolean; last_sequence: number }>(resolve => { resolveLookup = resolve; });
    const client = new FakeRunnerClient(HEALTH, "running", () => pendingLookup);
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);
    act(() => client.disconnect());
    expect(await screen.findByRole("status")).toHaveTextContent(/reconnecting/i);

    await userEvent.click(screen.getByRole("button", { name: /start live inspection/i }));
    expect(await screen.findByText(/session session-live-2/i)).toBeVisible();
    await act(async () => resolveLookup({ session_id: "session-live", state: "completed", pipeline_commit: EXPECTED_COMMIT, test_only: true, last_sequence: 0 }));

    expect(screen.getByRole("status")).toHaveTextContent(/connected.*session-live-2/i);
  });

  it("starts in the light theme with touch-sized header controls", () => {
    render(<LiveHmi client={new FakeRunnerClient()} expectedPipelineCommit={EXPECTED_COMMIT} />);
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    const header = screen.getByRole("banner");
    header.querySelectorAll("button").forEach(button => expect(button.classList.contains("hmi-button")).toBe(true));
    expect(screen.getByRole("button", { name: /switch to dark theme/i })).toBeVisible();
  });

  it("labels the disclosed hybrid service and allows an explicit development override with a banner", async () => {
    const live = "10c106fbd38e3da34d902abb7d920c5a4041440e";
    const client = new FakeRunnerClient({ ...HEALTH, pipeline_commit: live, test_only: false });
    render(<LiveHmi client={client} expectedPipelineCommit={HYBRID_COMMIT} devAllowedCommits={[live]} />);
    expect(await screen.findByText(/unpinned development service/i)).toBeVisible();
    expect(screen.queryByText(/pipeline commit mismatch/i)).not.toBeInTheDocument();
    await enterRunContext();
    await userEvent.upload(screen.getByLabelText(/video file/i), new File(["v"], "a.mp4", { type: "video/mp4" }));
    expect(screen.getByRole("button", { name: /start live inspection/i })).toBeEnabled();
  });

  it("shows a recorded-replay banner when health discloses a replay", async () => {
    const replay = { recorded_session_id: "e7383eef", recorded_service_commit: HYBRID_COMMIT, recorded_at: "2026-11-01T17:39:29Z", source_label: "a.mp4", note: "No model runs here." };
    render(<LiveHmi client={new FakeRunnerClient({ ...HEALTH, replay })} expectedPipelineCommit={EXPECTED_COMMIT} />);
    expect(await screen.findByText(/recorded replay, not live/i)).toBeVisible();
  });

  it("gates diagnostics behind the Engineer toggle and counts alarms", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    expect(screen.queryByRole("button", { name: /diagnostics/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /^engineer$/i }));
    expect(screen.getByRole("button", { name: /diagnostics/i })).toBeVisible();
    await startUpload(client);
    act(() => client.emit(runnerEvent(1, "averaged_decision", { disposition: "discard", average: 0.9 }, "CASE-target", 0)));
    expect(await screen.findByRole("button", { name: "Alarms (1)" })).toBeVisible();
  });
});
