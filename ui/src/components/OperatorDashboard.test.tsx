import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import fixture from "../../../ui-data/mock-run.json";
import { describe, expect, it } from "vitest";

import { parseDemoData } from "../data/adapter";
import { OperatorDashboard } from "./OperatorDashboard";

const data = parseDemoData(fixture);

function renderDashboard() {
  render(<OperatorDashboard data={data} />);
}

function renderFixture(input: Record<string, any>) {
  render(<OperatorDashboard data={parseDemoData(input)} />);
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

  it("keeps decision-bearing detail private on initial load", () => {
    renderDashboard();
    const detail = screen.getByRole("region", { name: /vial detail/i });

    expect(detail).toHaveTextContent(/pending cue/i);
    expect(detail).not.toHaveTextContent(/delta e|threshold|limit codes|within_delta_e_limit/i);
    expect(detail).not.toHaveTextContent(/L\* 35\.97|L\* 33\.61/i);
  });

  it("keeps a manually selected future vial private until its cue", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: "Select vial slot 7" }));
    const detail = screen.getByRole("region", { name: /vial detail/i });

    expect(detail).toHaveTextContent(/pending cue/i);
    expect(detail).not.toHaveTextContent(/29\.08|discard|exceeds_delta_e_limit|threshold/i);
    expect(detail).not.toHaveTextContent(/L\* 32\.30|L\* 33\.61/i);
    expect(screen.getByRole("img", { name: /vial 7 measured rgb frame/i })).toBeVisible();
  });

  it("reveals the slot 7 discard explanation as a deterministic template with evidence links", () => {
    renderDashboard();
    advanceTo(32.5);
    expect(within(screen.getByLabelText("Vial slot 7")).getByText("DISCARD")).toBeVisible();
    const explanation = screen.getByRole("article", { name: "Vial slot 7 discard explanation" });
    expect(within(explanation).getByText(/deterministic template/i)).toBeVisible();
    expect(within(explanation).getByRole("button", { name: /evidence.*delta/i })).toBeVisible();
  });

  it("keeps all three revealed discard explanations in an attached, non-overlapping stream", () => {
    renderDashboard();
    advanceTo(48);
    const stream = screen.getByRole("region", { name: /retained discard explanations/i });
    const explanations = within(stream).getAllByRole("article");
    expect(explanations).toHaveLength(3);
    [7, 8, 9].forEach((slot, index) => {
      expect(explanations[index]).toHaveAccessibleName(`Vial slot ${slot} discard explanation`);
      expect(within(explanations[index]).getByText(/deterministic template/i)).toBeVisible();
      expect(within(explanations[index]).getByRole("button", { name: /evidence.*delta/i })).toBeEnabled();
    });
  });

  it("selects a vial and updates real frame, mask, Lab values and top-down highlight", () => {
    renderDashboard();
    advanceTo(32.5);
    fireEvent.click(screen.getByRole("button", { name: "Select vial slot 7" }));
    expect(screen.getByRole("img", { name: /vial 7 measured rgb frame/i })).toHaveAttribute("src", "/assets/vials/vial-007-rgb.jpg");
    expect(screen.getByRole("img", { name: /vial 7 predicted mask overlay/i })).toHaveAttribute("src", "/assets/vials/vial-007-mask.png");
    expect(screen.getByText("L* 32.30 · a* 22.53 · b* 19.69")).toBeVisible();
    expect(screen.getByRole("button", { name: /top-down position.*slot 7.*selected/i })).toHaveAttribute("aria-pressed", "true");
  });

  it("exposes top-down reveal, outcome, Delta E and selection without color alone", () => {
    renderDashboard();
    const awaiting = screen.getByRole("button", { name: /top-down position.*slot 7.*vial-007.*awaiting decision.*not selected/i });
    expect(awaiting).toHaveTextContent(/awaiting decision/i);
    expect(awaiting).not.toHaveAccessibleName(/discard|29\.08/i);
    advanceTo(32.5);
    const revealed = screen.getByRole("button", { name: /top-down position.*slot 7.*vial-007.*discard.*delta e 29\.08.*revealed.*not selected/i });
    expect(revealed).toHaveTextContent(/discard.*ΔE 29\.08/i);
    fireEvent.click(revealed);
    expect(screen.getByRole("button", { name: /top-down position.*slot 7.*vial-007.*discard.*delta e 29\.08.*revealed.*selected/i })).toHaveAttribute("aria-pressed", "true");
  });

  it("highlights the cited image or metric when an evidence ID is clicked", () => {
    renderDashboard();
    advanceTo(32.5);
    const explanation = screen.getByRole("article", { name: "Vial slot 7 discard explanation" });
    fireEvent.click(within(explanation).getByRole("button", { name: /evidence.*delta/i }));
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

  it("renders contract-driven explanation source labels for current and history explanations", () => {
    const input = structuredClone(fixture) as Record<string, any>;
    input.vials[6].decision.explanation.source = "cosmos_reason2";
    input.history[0].explanation.source = "cosmos_reason2";
    renderFixture(input);
    advanceTo(32.5);

    const current = screen.getByRole("article", { name: "Vial slot 7 discard explanation" });
    expect(within(current).getByText("Cosmos-Reason2")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /open evidence panel/i }));
    expect(screen.getByRole("dialog", { name: /evidence panel/i })).toHaveTextContent("Cosmos-Reason2");
  });

  it("renders a feed asset and preserves top-down vial linkage", () => {
    const input = structuredClone(fixture) as Record<string, any>;
    input.top_down.state = "feed";
    input.top_down.asset = {
      asset_id: "top-down-feed-001",
      src: "/assets/top-down/feed.jpg",
      provenance: { kind: "documented", source_id: "feed-001", note: "Documented test feed." }
    };
    renderFixture(input);

    const panel = screen.getByRole("region", { name: /top-down station view/i });
    expect(panel).toHaveTextContent(/top-down feed connected/i);
    expect(within(panel).getByRole("img", { name: /top-down camera feed/i })).toHaveAttribute("src", "/assets/top-down/feed.jpg");
    const position = within(panel).getByRole("button", { name: /slot 7.*vial-007.*awaiting decision/i });
    fireEvent.click(position);
    expect(position).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("region", { name: /vial detail/i })).toHaveTextContent(/vial 7/i);
  });

  it("exposes provenance source and note to keyboard users", async () => {
    const user = userEvent.setup();
    renderDashboard();
    advanceTo(1.3);
    const provenance = screen.getAllByRole("button", { name: /provenance.*measured/i })[0];

    await user.tab();
    provenance.focus();
    expect(provenance).toHaveAccessibleDescription(/source.*note/i);
  });

  it("exposes history and model provenance details to keyboard users", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: /open evidence panel/i }));
    const historyPanel = screen.getByRole("tabpanel");
    expect(within(historyPanel).getByRole("button", { name: /provenance.*inferred/i })).toHaveAccessibleDescription(/source.*note/i);
    fireEvent.click(screen.getByRole("tab", { name: /model/i }));
    const modelPanel = screen.getByRole("tabpanel");
    expect(within(modelPanel).getAllByRole("button", { name: /provenance.*documented/i })[0]).toHaveAccessibleDescription(/source.*note/i);
  });

  it("announces each newly revealed outcome once", () => {
    renderDashboard();
    const status = screen.getByRole("status");
    expect(status).toBeEmptyDOMElement();
    advanceTo(1.3);
    expect(status).toHaveTextContent(/vial 1 outcome: keep/i);
    advanceTo(1.3);
    expect(status).toHaveTextContent(/vial 1 outcome: keep/i);
    advanceTo(5.9);
    expect(status).toHaveTextContent(/vial 2 outcome: keep/i);
  });

  it("presents the four exact WP6 comparisons as a semantic table", () => {
    renderDashboard();
    fireEvent.click(screen.getByRole("button", { name: /open evidence panel/i }));
    fireEvent.click(screen.getByRole("tab", { name: /model/i }));
    const table = screen.getByRole("table", { name: /limited all-abstention benchmark, no accuracy claim/i });
    ["Method", "Result", "Method note", "Failure note", "Provenance"].forEach(header => {
      expect(within(table).getByRole("columnheader", { name: header })).toBeVisible();
    });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toHaveTextContent(/Template.*16 \/ 16/s);
    expect(rows[1]).toHaveTextContent(/Qwen3-VL.*16 \/ 16/s);
    expect(rows[2]).toHaveTextContent(/AnomalyThink.*0 \/ 16/s);
    expect(rows[3]).toHaveTextContent(/Cosmos.*0 \/ 16/s);
    expect(within(table).queryByText(/INN-67 placeholder/i)).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: /follow-up evidence/i })).toHaveTextContent("INN-67 placeholder");
  });

  it("implements keyboard tabs, Escape close, relationships and opener focus restoration", async () => {
    const user = userEvent.setup();
    renderDashboard();
    const opener = screen.getByRole("button", { name: /open evidence panel/i });
    opener.focus();
    await user.click(opener);
    const historyTab = screen.getByRole("tab", { name: /history/i });
    const modelTab = screen.getByRole("tab", { name: /model/i });
    expect(historyTab).toHaveFocus();
    expect(historyTab).toHaveAttribute("tabindex", "0");
    expect(modelTab).toHaveAttribute("tabindex", "-1");
    await user.keyboard("{ArrowRight}");
    expect(modelTab).toHaveFocus();
    expect(modelTab).toHaveAttribute("aria-selected", "true");
    const modelPanel = screen.getByRole("tabpanel");
    expect(modelTab).toHaveAttribute("aria-controls", modelPanel.id);
    expect(modelPanel).toHaveAttribute("aria-labelledby", modelTab.id);
    await user.keyboard("{ArrowRight}");
    expect(historyTab).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(modelTab).toHaveFocus();
    await user.keyboard("{Home}");
    expect(historyTab).toHaveFocus();
    expect(historyTab).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(modelTab).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: /evidence panel/i })).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
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
