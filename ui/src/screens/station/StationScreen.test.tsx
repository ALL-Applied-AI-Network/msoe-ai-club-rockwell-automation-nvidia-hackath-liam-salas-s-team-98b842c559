import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { hybridModel, modelFromEvents } from "../../test/modelFixture";
import { runnerEvent } from "../../test/fakeClient";
import { containedRect, frameToVialFrame, roiToDisplayRect } from "./geometry";
import { StationScreen } from "./StationScreen";

const renderStation = (model = hybridModel(), onOpenDetail = vi.fn()) => {
  render(<StationScreen model={model} operatorId="op1" onOpenDetail={onOpenDetail} />);
  return { model, onOpenDetail };
};

describe("StationScreen with the recorded hybrid session", () => {
  it("shows the big Discard state, agreement chips and the average", () => {
    renderStation();
    expect(within(screen.getByTestId("big-state")).getByText("Discard")).toBeInTheDocument();
    const card = screen.getByRole("region", { name: "Current vial" });
    for (const name of ["Rule", "Qwen", "Cosmos", "AnomalyThink"]) expect(within(card).getByText(name)).toBeInTheDocument();
    const confs = [...card.querySelectorAll(".station-chip__conf")].map(n => n.textContent?.trim());
    expect(confs).toEqual(["0.9", "0.8", "0.9"]);
    expect(within(screen.getByText("Qwen").closest("li") as HTMLElement).getByText("0.9")).toBeInTheDocument();
    expect(within(card).getByText(/Average 0\.90/)).toBeInTheDocument();
  });

  it("shows the tray with Vial 1 Discard and two references, and the counts", () => {
    renderStation();
    const tray = screen.getByRole("region", { name: "Tray" });
    expect(within(tray).getByRole("button", { name: /Vial 1, Discard/ })).toBeInTheDocument();
    expect(within(tray).getAllByRole("button", { name: /Reference/ })).toHaveLength(2);
    const counts = screen.getByRole("region", { name: "Counts" });
    expect(within(counts).getByText("Inspected").previousSibling).toHaveTextContent("1");
    expect(within(counts).getByText("Discard").previousSibling).toHaveTextContent("1");
    expect(within(counts).getByText("100%")).toBeInTheDocument();
  });

  it("selects a vial from the tray", async () => {
    const { model } = renderStation();
    await userEvent.click(screen.getAllByRole("button", { name: /Reference/ })[0]);
    expect(model.selectVial).toHaveBeenCalled();
  });

  it("Agree records an agree action with the system disposition", async () => {
    const { model } = renderStation();
    await userEvent.click(screen.getByRole("button", { name: "Agree" }));
    expect(model.recordOperatorAction).toHaveBeenCalledWith(expect.objectContaining({ action: "agree", system_disposition: "discard", operator: "op1" }));
    expect(screen.getByText(/Recorded locally at .* the system decision is unchanged/)).toBeInTheDocument();
  });

  it("Override needs a reason and does not change the displayed decision", async () => {
    const { model } = renderStation();
    await userEvent.click(screen.getByRole("button", { name: "Override…" }));
    await userEvent.click(screen.getByLabelText("Keep"));
    expect(screen.getByRole("button", { name: "Confirm override" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/Reason/), "looks clear by eye");
    await userEvent.click(screen.getByRole("button", { name: "Confirm override" }));
    expect(model.recordOperatorAction).toHaveBeenCalledWith(expect.objectContaining({ action: "override", operator_disposition: "keep", system_disposition: "discard", reason: "looks clear by eye" }));
    expect(within(screen.getByTestId("big-state")).getByText("Discard")).toBeInTheDocument();
  });

  it("Detail opens the vial detail", async () => {
    const { onOpenDetail } = renderStation();
    await userEvent.click(screen.getByRole("button", { name: "Detail" }));
    expect(onOpenDetail).toHaveBeenCalledWith("CASE_97a8c6cb8161881c");
  });

  it("shows a placeholder when there is no video", () => {
    renderStation();
    expect(screen.getByText(/Live camera preview not streamed by runner/)).toBeInTheDocument();
  });
});

describe("insufficient evidence", () => {
  it("shows amber Can't assess with a plain message", () => {
    const model = modelFromEvents([runnerEvent(1, "vial_decision", { disposition: "insufficient_evidence", unavailable_reason: "measurement_stage_unavailable" }, "unavailable", 0)]);
    renderStation(model);
    expect(within(screen.getByTestId("big-state")).getByText("Can't assess")).toBeInTheDocument();
    expect(screen.getByText(/did not capture a usable measurement for every stage/)).toBeInTheDocument();
    expect(screen.getByTestId("big-state")).toHaveClass("state--cant_assess");
  });
});

describe("geometry helpers", () => {
  it("computes letterboxing", () => {
    const r = containedRect({ width: 1920, height: 1080 }, { width: 1000, height: 1000 });
    expect(r.x).toBeCloseTo(0); expect(r.y).toBeCloseTo(218.75); expect(r.width).toBeCloseTo(1000); expect(r.height).toBeCloseTo(562.5);
  });
  it("maps an ROI into a letterboxed box", () => {
    const r = roiToDisplayRect([960, 540, 1920, 1080], [1920, 1080], { width: 1000, height: 1000 });
    expect(r.x).toBeCloseTo(500); expect(r.width).toBeCloseTo(500);
    expect(r.y).toBeCloseTo(218.75 + 281.25); expect(r.height).toBeCloseTo(281.25);
  });
  it("maps video time to tray index and local frame", () => {
    expect(frameToVialFrame(0, 10, 90, 3, null)).toEqual({ globalFrame: 0, trayIndex: 0, localFrame: 0, framesPerVial: 300 });
    expect(frameToVialFrame(35.04, 10, 90, 3, 300)).toEqual({ globalFrame: 350, trayIndex: 1, localFrame: 50, framesPerVial: 300 });
    expect(frameToVialFrame(10, null, NaN, 3, 100)).toMatchObject({ globalFrame: 100, trayIndex: 1, localFrame: 0 });
  });
});
