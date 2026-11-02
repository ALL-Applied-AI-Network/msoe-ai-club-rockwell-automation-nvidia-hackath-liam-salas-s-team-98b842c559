import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { hybridModel, modelFromEvents } from "../../test/modelFixture";
import { OverviewScreen } from "./OverviewScreen";

describe("OverviewScreen", () => {
  it("shows KPIs, trend dot and bands for the recorded session", async () => {
    const onOpenDetail = vi.fn();
    render(<OverviewScreen model={hybridModel()} onOpenDetail={onOpenDetail} />);
    expect(screen.getByRole("group", { name: "Inspected" })).toHaveTextContent("1");
    expect(screen.getByRole("group", { name: "Discard rate" })).toHaveTextContent("100%");
    expect(screen.getByText("≤ 1.0 keep limit")).toBeInTheDocument();
    expect(screen.getByText("1.0–2.0 check band")).toBeInTheDocument();
    expect(screen.getByText("Recipe 5 start")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Vial 1 · ΔE 30.0 · Discard" }));
    expect(onOpenDetail).toHaveBeenCalledWith("CASE_97a8c6cb8161881c");
  });

  it("shows an empty state without judged vials", () => {
    render(<OverviewScreen model={modelFromEvents([])} onOpenDetail={vi.fn()} />);
    expect(screen.getAllByText("No judged vials yet.").length).toBeGreaterThan(0);
    expect(screen.getByRole("group", { name: "Inspected" })).toHaveTextContent("0");
  });
});
