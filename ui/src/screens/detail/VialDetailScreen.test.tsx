import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { runnerEvent } from "../../test/fakeClient";
import { hybridModel, modelFromEvents } from "../../test/modelFixture";
import { VialDetailScreen } from "./VialDetailScreen";

const TARGET = "CASE_97a8c6cb8161881c";
const show = (model = hybridModel(), onSelect = vi.fn()) => render(<VialDetailScreen model={model} vialId={model.selectedVialId} onSelectVial={onSelect} />);

describe("evidence grid", () => {
  it("has 3 stage rows by 3 vial columns with target ΔE outside limits", () => {
    show();
    const grid = screen.getByRole("region", { name: "Evidence grid" });
    const rows = within(grid).getAllByRole("row");
    expect(rows).toHaveLength(4);
    expect(within(rows[1]).getAllByRole("cell")).toHaveLength(3);
    const cell = screen.getByTestId("cell-T200-1");
    expect(cell).toHaveTextContent("30.0");
    expect(cell).toHaveTextContent("29.1");
    expect(cell).toHaveAttribute("data-outside-limits", "true");
    expect(screen.getByTestId("cell-T200-2")).toHaveTextContent("reference");
  });
  it("uses imageUrl for crops", () => {
    show();
    const img = screen.getByAltText("Vial 1 end of T200 crop") as HTMLImageElement;
    expect(img.src).toContain(`v4-${TARGET}-F0116`);
  });
  it("labels missing crops and falls back to streamed swatch when the bundle is null", () => {
    show(hybridModel({ bundle: null, imageUrl: () => null }));
    expect(screen.getAllByText("Crop not exposed by this runner").length).toBeGreaterThan(0);
    expect(within(screen.getByTestId("cell-T200-1")).getByText("measured · streamed frame")).toBeInTheDocument();
    expect(screen.getByTestId("cell-T200-1")).toHaveTextContent("30.0");
  });
});

describe("model cards", () => {
  it("shows Qwen model-written explanation", () => {
    show();
    const card = screen.getByRole("article", { name: "Qwen source" });
    expect(within(card).getByText("model-written")).toBeInTheDocument();
    expect(within(card).getByText(/schema v2 pending/)).toBeInTheDocument();
  });
  it("labels Cosmos and AnomalyThink fallbacks and hides rejected text in details", () => {
    show();
    const cosmos = screen.getByRole("article", { name: "Cosmos source" });
    expect(within(cosmos).getByText(/Fallback \(model text rejected:.*cause_claim/)).toBeInTheDocument();
    expect(cosmos.querySelector("details")).not.toBeNull();
    expect(within(cosmos).getByText("Model's rejected text (engineering)")).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "AnomalyThink source" })).toHaveTextContent("hidden_simulator_state_claim");
  });
});

describe("tracker timeline", () => {
  it("shows the three steps with measured volumes and an invalid-sample marker", () => {
    show();
    const tl = screen.getByRole("region", { name: "Tracker timeline" });
    for (const t of ["T200", "T400", "T100"]) expect(within(tl).getByRole("heading", { name: t })).toBeInTheDocument();
    for (const ml of ["7.6 mL", "17.6 mL", "28.8 mL"]) expect(within(tl).getByText(ml)).toBeInTheDocument();
    expect(within(tl).getByText(/measurement_invalid/)).toBeInTheDocument();
    expect(within(tl).getByText("F116")).toBeInTheDocument();
  });
});

describe("header", () => {
  it("selecting another vial calls onSelectVial", () => {
    const onSelect = vi.fn();
    show(hybridModel(), onSelect);
    const buttons = within(screen.getByRole("group", { name: "Select vial" })).getAllByRole("button");
    expect(buttons).toHaveLength(3);
    fireEvent.click(buttons[1]);
    expect(onSelect).toHaveBeenCalledWith("CASE_59038b977a73fdfb");
    expect(buttons[0]).toHaveAttribute("aria-pressed", "true");
  });
  it("shows a friendly empty state", () => {
    show(modelFromEvents([]));
    expect(screen.getByText(/No vial to show yet/)).toBeInTheDocument();
  });
});

describe("VialDetailScreen review regressions", () => {
  it("does not crash when the equal-weight average abstains", () => {
    const model = modelFromEvents([
      runnerEvent(1, "vial_decision", { disposition: "discard" }, "vial-a", 0),
      runnerEvent(2, "averaged_decision", { disposition: "insufficient_evidence", average: 0 }, "vial-a", 0)
    ]);
    render(<VialDetailScreen model={model} vialId="vial-a" onSelectVial={() => undefined} />);
    expect(screen.getByText(/average unavailable/i)).toBeInTheDocument();
  });
});
