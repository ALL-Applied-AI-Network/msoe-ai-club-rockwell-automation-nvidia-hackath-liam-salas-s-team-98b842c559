import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import fixture from "../../../ui-data/mock-run.json";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { parseDemoData, type DemoData } from "../data/adapter";
import { OperatorDashboard } from "./OperatorDashboard";

const data = parseDemoData(fixture);
let developmentData: DemoData;

beforeAll(() => {
  const root = resolve(import.meta.dirname, "../../..");
  const temporary = mkdtempSync(resolve(tmpdir(), "fluidmix-ui-development-"));
  try {
    const config = resolve(temporary, "source.json");
    const output = resolve(temporary, "output");
    writeFileSync(config, '{"source":"development"}', "utf8");
    execFileSync(resolve(root, ".venv/bin/python"), [
      resolve(root, "scripts/materialize_ui_data.py"),
      "--root", root,
      "--config", config,
      "--output", output
    ]);
    developmentData = parseDemoData(JSON.parse(readFileSync(resolve(output, "active-run.json"), "utf8")));
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
});

afterEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState({}, "", "/");
});

function renderDashboard() {
  render(<OperatorDashboard data={data} />);
}

function renderDevelopmentDashboard() {
  render(<OperatorDashboard data={developmentData} />);
}

function renderFixture(input: Record<string, any>) {
  render(<OperatorDashboard data={parseDemoData(input)} />);
}

function advanceTo(seconds: number) {
  const video = screen.getByLabelText(/stitched vial video/i) as HTMLVideoElement;
  Object.defineProperty(video, "currentTime", { configurable: true, value: seconds });
  fireEvent.timeUpdate(video);
}

describe("OperatorDashboard", () => {
  it("renders the single live synthetic run, detail, top-down and vial strip shell", () => {
    renderDashboard();
    expect(screen.getByRole("heading", { name: /synthetic twin data/i })).toBeInTheDocument();
    expect(screen.getByText(/contract evidence.*progressive decision cues/i)).toBeInTheDocument();
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
    expect(within(screen.getByLabelText("Vial slot 1")).getByText("keep")).toBeVisible();
    advanceTo(5.9);
    expect(within(screen.getByLabelText("Vial slot 1")).getByText("keep")).toBeVisible();
    expect(within(screen.getByLabelText("Vial slot 2")).getByText("keep")).toBeVisible();
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
    expect(screen.queryByRole("img", { name: /vial 7 measured rgb frame/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /vial 7 predicted mask overlay/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open evidence panel/i }));
    expect(screen.getByRole("dialog", { name: /evidence panel/i })).toHaveTextContent(/available after.*playback cue/i);
    expect(screen.getByRole("dialog", { name: /evidence panel/i })).not.toHaveTextContent(/evidence-PAIR|mock-only|29\.08/i);
  });

  it("renders the exact Development data label and all 18 source-backed vials", () => {
    renderDevelopmentDashboard();
    expect(screen.getByRole("heading", { level: 1, name: /development data/i })).toBeVisible();
    expect(screen.getByText("0 / 18 revealed")).toBeVisible();
    expect(screen.getAllByRole("button", { name: /select vial slot/i })).toHaveLength(18);
    expect(screen.getByText("18 VIAL SLOTS")).toBeVisible();
  });

  it("reveals nullable Development values as unavailable only after the vial cue", () => {
    renderDevelopmentDashboard();
    const detail = screen.getByRole("region", { name: /vial detail/i });
    expect(detail).not.toHaveTextContent(/reference L\*a\*b\*|Delta E|Threshold|Unavailable/i);
    advanceTo(developmentData.vials[0].cue.start_seconds);
    expect(detail).toHaveTextContent("insufficient_evidence");
    expect(detail).toHaveTextContent(/Reference L\*a\*b\*.*Unavailable/s);
    expect(detail).toHaveTextContent(/Delta E.*Unavailable/s);
    expect(detail).toHaveTextContent(/Threshold.*Unavailable/s);
    expect(detail).toHaveTextContent(developmentData.vials[0].decision.explanation.text);
    developmentData.vials[0].decision.limit_codes.forEach(code => expect(detail).toHaveTextContent(code));
  });

  it("keeps the variable-count top-down vial-position-decision linkage testable", () => {
    renderDevelopmentDashboard();
    const first = developmentData.vials[0];
    advanceTo(first.cue.start_seconds);
    const position = screen.getByRole("button", { name: new RegExp(`slot ${first.slot}.*${first.vial_id}.*${first.decision.outcome}.*Delta E unavailable.*revealed`, "i") });
    expect(position).toHaveTextContent(first.decision.outcome);
    fireEvent.click(position);
    expect(position).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("region", { name: /vial detail/i })).toHaveTextContent(`Vial ${first.slot}`);
  });

  it("reveals the slot 7 discard explanation as a deterministic template with evidence links", () => {
    renderDashboard();
    advanceTo(32.5);
    expect(within(screen.getByLabelText("Vial slot 7")).getByText("discard")).toBeVisible();
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
    expect(screen.getByText("29.08").closest("div")).toHaveAttribute("data-evidence-highlight", "true");
  });

  it("uses the mock only at the component boundary to stream a source-driven discard and highlight its citations", () => {
    vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    render(<OperatorDashboard data={data} demoMode />);
    advanceTo(data.vials[6].cue.start_seconds);
    const explanation = screen.getByRole("article", { name: "Vial slot 7 discard explanation" });
    expect(explanation).toHaveTextContent(data.vials[6].decision.explanation.text);
    data.vials[6].evidence.forEach(evidence => {
      expect(within(explanation).getByRole("button", { name: new RegExp(evidence.evidence_id) })).toHaveAttribute("data-evidence-highlight", "true");
    });
    expect(screen.getByRole("button", { name: "Select vial slot 7" })).toHaveAttribute("aria-pressed", "true");
  });

  it("runs Development recording mode muted with handled autoplay and progressive selection", async () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockRejectedValue(new DOMException("blocked", "NotAllowedError"));
    window.history.replaceState({}, "", "/?demo=1");
    renderDevelopmentDashboard();
    const video = screen.getByLabelText(/stitched vial video/i) as HTMLVideoElement;
    expect(video).toHaveProperty("muted", true);
    expect(video).toHaveAttribute("autoplay");
    expect(play).toHaveBeenCalledTimes(1);
    advanceTo(developmentData.vials[1].cue.start_seconds);
    expect(screen.getByRole("button", { name: "Select vial slot 2" })).toHaveAttribute("aria-pressed", "true");
    await Promise.resolve();
  });

  it("does not turn the mock source into a recording or autoplay source", () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    window.history.replaceState({}, "", "/?demo=1");
    renderDashboard();
    const video = screen.getByLabelText(/stitched vial video/i) as HTMLVideoElement;
    expect(video).toHaveAttribute("controls");
    expect(video).not.toHaveAttribute("autoplay");
    expect(video).toHaveProperty("muted", false);
    expect(play).not.toHaveBeenCalled();
  });

  it("progresses through the real all-abstention Development run without inventing a discard", () => {
    vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    window.history.replaceState({}, "", "/?demo=1");
    renderDevelopmentDashboard();
    const last = developmentData.vials.at(-1)!;
    advanceTo(last.cue.start_seconds);
    expect(screen.getByText("18 / 18 revealed")).toBeVisible();
    expect(screen.getByRole("button", { name: "Select vial slot 18" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByText("discard", { selector: ".decision-chip" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: /retained discard explanations/i })).not.toBeInTheDocument();
  });

  it("opens history and exact WP6 model evidence in place", () => {
    renderDashboard();
    advanceTo(1.3);
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
    expect(panel).toHaveTextContent(/top-down feed withheld pending cues/i);
    expect(within(panel).queryByRole("img", { name: /top-down camera feed/i })).not.toBeInTheDocument();
    advanceTo(48);
    expect(panel).toHaveTextContent(/top-down feed connected/i);
    expect(within(panel).getByRole("img", { name: /top-down camera feed/i })).toHaveAttribute("src", "/assets/top-down/feed.jpg");
    const position = within(panel).getByRole("button", { name: /slot 7.*vial-007.*discard.*revealed/i });
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
    advanceTo(1.3);
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
    advanceTo(1.3);
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
    advanceTo(1.3);
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
    expect(screen.queryByText("discard", { selector: ".decision-chip" })).not.toBeInTheDocument();
  });
});
