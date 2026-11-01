import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { RunnerClient, RunnerHealth, RunnerSource } from "../runner/client";
import type { RunnerEvent } from "../runner/contract";
import "../styles.css";
import { LiveHmi } from "./LiveHmi";

const EXPECTED_COMMIT = "cf6829b99486b41f4190874dbd701ad812648217";
const HEALTH: RunnerHealth = { status: "healthy", service: "fluidmix-pipeline-runner", api_version: "v1", pipeline_commit: EXPECTED_COMMIT, test_only: true, inference: false, active_sessions: 0, limits: [] };

class FakeRunnerClient implements RunnerClient {
  readonly sources: RunnerSource[] = [];
  readonly acknowledgements: string[] = [];
  private onEvent?: (event: RunnerEvent) => void;
  private onDisconnect?: (message: string) => void;

  constructor(private readonly health: RunnerHealth | Error = HEALTH) {}

  async getHealth() {
    if (this.health instanceof Error) throw this.health;
    return this.health;
  }

  async createSession(source: RunnerSource) {
    this.sources.push(source);
    return { session_id: "session-live", test_only: true, pipeline_commit: EXPECTED_COMMIT };
  }

  subscribe(_sessionId: string, _after: number, onEvent: (event: RunnerEvent) => void, onDisconnect: (message: string) => void) {
    this.onEvent = onEvent;
    this.onDisconnect = onDisconnect;
    return vi.fn();
  }

  async acknowledge(_sessionId: string, alarmId: string, _operatorId: string) {
    this.acknowledgements.push(alarmId);
  }

  emit(event: RunnerEvent) { this.onEvent?.(event); }
  disconnect(message = "Runner connection lost") { this.onDisconnect?.(message); }
}

function runnerEvent(sequence: number, type: RunnerEvent["type"], payload: Record<string, unknown>, vialId?: string, trayPosition?: number): RunnerEvent {
  const position = { tray_index: trayPosition ?? 0, vial_number: vialId ?? "vial-unknown", slot_label: `Slot ${(trayPosition ?? 0) + 1}` };
  const locator = payload.locator ?? { status: "unavailable", bbox_normalized: null, centroid_normalized: null, evidence_ids: [], unavailable_reason: "No top-down feed" };
  const tracker = { state: "tracking", geometry_source: "pipeline", fill_fraction: 0.4, confidence: 0.9, tank_id: "T-200", fill_step_id: "step-2", measured_volume_ml: payload.tracker_volume_ml ?? 4.2, volume_provenance: "measured", unavailable_reason: null, locator };
  const measurement = { measurement_id: "measurement-7", status: payload.source ?? "measured", rgb: [100, 90, 80], lab: payload.lab ?? [51.2, 12.4, -3.1], delta_e: payload.delta_e ?? 2.8, reference_ids: ["reference-1"], evidence_ids: payload.evidence_ids ?? ["measurement-7"], unavailable_reason: null };
  const disposition = payload.disposition ?? "keep";
  const alarm = disposition === "discard" ? { alarm_id: `alarm-${vialId}`, state: "unacknowledged", acknowledged_at: null, acknowledged_by: null } : null;
  const exactPayload: Record<string, unknown> = type === "measuring" ? { position, tracker, measurement, stage: "measuring" }
    : type === "vial_decision" ? { position, disposition, decision_id: `decision-${sequence}`, alarm, measurement, evidence_ids: measurement.evidence_ids, unavailable_reason: null }
    : type === "model_vote" ? { position, vote_id: `vote-${sequence}`, model: { model_id: payload.model_id, revision: payload.model_revision, role: "cnn", source: payload.source }, admission: { admitted: true, reason: "valid" }, disposition, confidence: payload.score, evidence_ids: ["measurement-7"], unavailable_reason: null }
    : type === "averaged_decision" ? { position, decision_id: `average-${sequence}`, method: "equal_weight", admitted_vote_ids: ["vote-2"], scores: { keep: disposition === "keep" ? payload.average : 0.1, check: 0.08, discard: disposition === "discard" ? payload.average : 0.1 }, disposition, alarm, evidence_ids: ["measurement-7"], unavailable_reason: null }
    : type === "explanation" ? { position, explanation_id: `explanation-${sequence}`, plain_language: payload.text, limits: payload.limits, evidence_ids: payload.evidence_ids, producer: { model_id: "explain", revision: "r2", source: payload.source }, unavailable_reason: null }
    : type === "rate_health" ? { status: "healthy", processing_fps: payload.processing_rate_fps, source_fps: 10, queue_depth: 0, active_model_id: "cnn", gpu_free_mib: 12000, gpu_status: "available", limits: [] }
    : payload;
  return {
    schema_version: "runner-event.v1", event_id: `event-${sequence}`, type,
    session_id: "session-live",
    sequence,
    emitted_at: `2026-10-31T12:00:${String(sequence).padStart(2, "0")}Z`,
    pipeline_commit: EXPECTED_COMMIT,
    source: { source_id: "source-1", kind: "file", label: "three-pour.avi" }, payload: exactPayload
  } as RunnerEvent;
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
  return file;
}

describe("LiveHmi", () => {
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
    expect(screen.getByLabelText(`Pipeline commit ${EXPECTED_COMMIT}`)).toHaveTextContent(EXPECTED_COMMIT.slice(0, 12));
    expect(screen.queryByText(/pipeline commit mismatch/i)).not.toBeInTheDocument();
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

  it("renders ordered state shapes, current rate and the test-only service label", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);

    act(() => {
      client.emit(runnerEvent(1, "vial_decision", { disposition: "discard" }, "vial-9", 9));
      client.emit(runnerEvent(2, "vial_decision", { disposition: "keep" }, "vial-1", 1));
      client.emit(runnerEvent(3, "vial_decision", { disposition: "insufficient_evidence" }, "vial-4", 4));
      client.emit(runnerEvent(4, "rate_health", { processing_rate_fps: 7.5, healthy: true, test_only: true }));
    });

    const slots = await screen.findAllByRole("button", { name: /select tray position/i });
    expect(slots.map(slot => slot.textContent)).toEqual([
      expect.stringMatching(/1.*keep/i),
      expect.stringMatching(/4.*insufficient evidence/i),
      expect.stringMatching(/9.*discard/i)
    ]);
    expect(within(slots[0]).getByText("✓", { selector: "span" })).toBeVisible();
    expect(within(slots[1]).getByText("▲", { selector: "span" })).toBeVisible();
    expect(within(slots[2]).getByText("!", { selector: "span" })).toBeVisible();
    expect(screen.getByText("7.5 frames/s")).toBeVisible();
    expect(screen.getByText(/test-only service/i)).toBeVisible();
    expect(screen.getByRole("region", { name: /top-down vial locator/i })).toHaveTextContent(/no locator geometry reported/i);
  });

  it("requires acknowledgement for each discard alarm", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);
    act(() => client.emit(runnerEvent(1, "averaged_decision", { disposition: "discard", average: 0.82 }, "vial-7", 7)));

    const alarm = await screen.findByRole("alert");
    expect(alarm).toHaveTextContent(/discard alarm.*vial-7/i);
    expect(screen.getByRole("button", { name: /select tray position 7.*discard/i })).toHaveTextContent(/!.*discard/i);
    await userEvent.click(within(alarm).getByRole("button", { name: /acknowledge/i }));

    expect(client.acknowledgements).toEqual(["alarm-vial-7"]);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps engineering evidence behind a disclosure with measured and inferred tags", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);
    act(() => {
      client.emit(runnerEvent(1, "measuring", {
        tracker_volume_ml: 4.2, lab: [51.2, 12.4, -3.1], delta_e: 2.8, source: "measured", evidence_ids: ["measurement-7"]
      }, "vial-7", 7));
      client.emit(runnerEvent(2, "model_vote", {
        model_id: "cnn", model_revision: "r4", disposition: "discard", source: "inferred", score: 0.82
      }, "vial-7", 7));
      client.emit(runnerEvent(3, "averaged_decision", { disposition: "discard", average: 0.82, source: "inferred" }, "vial-7", 7));
      client.emit(runnerEvent(4, "explanation", {
        text: "The measured color differs from the reference.", source: "inferred", evidence_ids: ["measurement-7"], limits: ["Reference quality is limited."]
      }, "vial-7", 7));
    });

    const detail = await screen.findByRole("region", { name: /vial details/i });
    expect(detail).toHaveTextContent(/measuring vial-7/i);
    expect(detail).toHaveTextContent(/measured color differs/i);
    expect(within(detail).getByText(/L\* 51\.2/)).not.toBeVisible();
    expect(within(detail).getByRole("region", { name: /model votes/i })).not.toBeVisible();
    expect(within(detail).getByText("measurement-7")).not.toBeVisible();

    await userEvent.click(within(detail).getByText(/engineer details/i));
    expect(detail).toHaveTextContent(/L\* 51\.2.*a\* 12\.4.*b\* -3\.1/s);
    expect(detail).toHaveTextContent(/Delta E.*2\.8/s);
    expect(detail).toHaveTextContent(/Tracker volume.*4\.2 ml/s);
    expect(detail).toHaveTextContent(/cnn.*r4.*discard.*0\.82/s);
    expect(detail).toHaveTextContent(/Average.*0\.82/s);
    within(detail).getAllByText("Measured").forEach(tag => expect(tag).toBeVisible());
    expect(within(detail).getAllByText("Inferred").length).toBeGreaterThan(0);
    expect(detail).toHaveTextContent("measurement-7");
  });

  it("renders reported normalized locator geometry for the selected vial", async () => {
    const client = new FakeRunnerClient();
    render(<LiveHmi client={client} expectedPipelineCommit={EXPECTED_COMMIT} />);
    await startUpload(client);
    act(() => client.emit(runnerEvent(1, "measuring", {
      locator: { status: "located", bbox_normalized: [0.25, 0.3, 0.1, 0.2], centroid_normalized: [0.3, 0.4], evidence_ids: ["locator-7"], unavailable_reason: null }
    }, "vial-7", 7)));

    const locator = screen.getByRole("region", { name: /top-down vial locator/i });
    const marker = within(locator).getByRole("img", { name: /vial-7.*normalized.*0\.25.*0\.3.*0\.1.*0\.2/i });
    expect(marker).toHaveStyle({ left: "25%", top: "30%", width: "10%", height: "20%" });
    expect(locator).toHaveTextContent(/centroid.*0\.3.*0\.4/i);
    expect(locator).not.toHaveTextContent(/no locator geometry reported/i);
  });

  it("shows start failures and stream disconnects without substituting data", async () => {
    const failing: RunnerClient = {
      getHealth: vi.fn().mockResolvedValue(HEALTH),
      createSession: vi.fn().mockRejectedValue(new Error("Camera refused the request")),
      subscribe: vi.fn(),
      acknowledge: vi.fn()
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
    expect(await screen.findByRole("status")).toHaveTextContent(/connection lost.*no live updates/i);
    live.unmount();
  });

  it("starts in dark theme and offers controls with the HMI touch-target class", () => {
    render(<LiveHmi client={new FakeRunnerClient()} expectedPipelineCommit={EXPECTED_COMMIT} />);
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    const buttons = screen.getAllByRole("button");
    expect(buttons.every(button => button.classList.contains("hmi-control"))).toBe(true);
    buttons.forEach(button => expect(getComputedStyle(button).minHeight).toBe("48px"));
    expect(screen.getByRole("button", { name: /switch to light theme/i })).toBeVisible();
  });
});
