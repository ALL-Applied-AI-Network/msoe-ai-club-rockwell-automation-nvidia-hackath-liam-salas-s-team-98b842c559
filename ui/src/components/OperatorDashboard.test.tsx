import { fireEvent, render, screen, within } from "@testing-library/react";
import fixture from "../../../ui-data/mock-run.json";
import { describe, expect, it } from "vitest";

import { parseDemoData } from "../data/adapter";
import { OperatorDashboard } from "./OperatorDashboard";

const data = parseDemoData(fixture);

function renderDashboard() {
  render(<OperatorDashboard data={data} />);
}

function advanceTo(seconds: number) {
  const video = screen.getByLabelText("RUN-01 stitched vial video") as HTMLVideoElement;
  Object.defineProperty(video, "currentTime", { configurable: true, value: seconds });
  fireEvent.timeUpdate(video);
}

describe("OperatorDashboard", () => {
  it("renders the single live synthetic run, detail, top-down and vial strip shell", () => {
    renderDashboard();
    expect(screen.getByRole("heading", { name: /synthetic twin data/i })).toBeInTheDocument();
    expect(screen.getByText(/camera-only.*deterministic decision boundary/i)).toBeInTheDocument();
    expect(screen.getByText(/no plant or accuracy claim/i)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /live run/i })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /vial detail/i })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /top-down/i })).toHaveTextContent("waiting for top-down feed");
    expect(screen.getByRole("region", { name: /vial timeline/i })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /select vial slot/i })).toHaveLength(9);
  });

  it("reveals decisions at their video cue and retains earlier decisions", () => {
    renderDashboard();
    expect(screen.queryByText(/keep|discard|insufficient evidence/i, { selector: ".decision-chip" })).not.toBeInTheDocument();
    advanceTo(1.3);
    expect(within(screen.getByLabelText("Vial slot 1")).getByText("KEEP")).toBeVisible();
    advanceTo(5.9);
    expect(within(screen.getByLabelText("Vial slot 1")).getByText("KEEP")).toBeVisible();
    expect(within(screen.getByLabelText("Vial slot 2")).getByText("KEEP")).toBeVisible();
  });

  it("reveals the slot 7 discard explanation as a deterministic template with evidence links", () => {
    renderDashboard();
    advanceTo(32.5);
    const slot = screen.getByLabelText("Vial slot 7");
    expect(within(slot).getByText("DISCARD")).toBeVisible();
    expect(within(slot).getByText(/deterministic template/i)).toBeVisible();
    expect(within(slot).getByRole("button", { name: /evidence.*delta/i })).toBeVisible();
  });

  it("selects a vial and updates real frame, mask, Lab values and top-down highlight", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: "Select vial slot 7" }));
    expect(screen.getByRole("img", { name: /vial 7 measured rgb frame/i })).toHaveAttribute("src", "/assets/vials/vial-007-rgb.jpg");
    expect(screen.getByRole("img", { name: /vial 7 predicted mask overlay/i })).toHaveAttribute("src", "/assets/vials/vial-007-mask.png");
    expect(screen.getByText("L* 32.30 · a* 22.53 · b* 19.69")).toBeVisible();
    expect(screen.getByRole("button", { name: /top-down position 7.*selected/i })).toHaveAttribute("aria-pressed", "true");
  });

  it("highlights the cited image or metric when an evidence ID is clicked", () => {
    renderDashboard();
    advanceTo(32.5);
    const slot = screen.getByLabelText("Vial slot 7");
    fireEvent.click(within(slot).getByRole("button", { name: /evidence.*delta/i }));
    expect(screen.getByText(/delta e 29\.08/i)).toHaveAttribute("data-evidence-highlight", "true");
  });

  it("opens history and exact WP6 model evidence in place", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: /open evidence panel/i }));
    expect(screen.getByRole("dialog", { name: /evidence panel/i })).toHaveTextContent(/RUN-TRAIN-04.*V1.*1\.0 s/s);
    fireEvent.click(screen.getByRole("tab", { name: /model/i }));
    const panel = screen.getByRole("dialog", { name: /evidence panel/i });
    expect(panel).toHaveTextContent("limited all-abstention benchmark, no accuracy claim");
    expect(panel).toHaveTextContent(/Template.*16 \/ 16/s);
    expect(panel).toHaveTextContent(/Cosmos.*0 \/ 16/s);
    expect(panel).toHaveTextContent("INN-67 placeholder");
  });

  it("toggles theme and supports keyboard selection and restart", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: /switch to light theme/i }));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    const vial7 = screen.getByRole("button", { name: "Select vial slot 7" });
    vial7.focus();
    fireEvent.keyDown(vial7, { key: "Enter" });
    expect(vial7).toHaveAttribute("aria-pressed", "true");
    advanceTo(32.5);
    fireEvent.click(screen.getByRole("button", { name: /restart demo/i }));
    expect(screen.queryByText("DISCARD", { selector: ".decision-chip" })).not.toBeInTheDocument();
  });
});
