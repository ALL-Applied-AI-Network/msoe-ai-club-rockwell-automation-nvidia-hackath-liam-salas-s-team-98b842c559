import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { HYBRID_COMMIT } from "../../test/fakeClient";
import { hybridModel } from "../../test/modelFixture";
import { DiagnosticsScreen } from "./DiagnosticsScreen";

describe("DiagnosticsScreen", () => {
  it("shows service identity, event log, models and operator-record disclosure", () => {
    render(<DiagnosticsScreen model={hybridModel()} />);
    expect(screen.getAllByText(HYBRID_COMMIT).length).toBeGreaterThan(0);
    const log = screen.getByRole("region", { name: "Event log" });
    expect(within(log).getAllByText("averaged_decision").length).toBeGreaterThan(0);
    expect(within(screen.getByRole("region", { name: "Models" })).getAllByText(/qwen/i).length).toBeGreaterThan(0);
    const ops = screen.getByRole("region", { name: "Operator records" });
    expect(within(ops).getByText("No operator records.")).toBeInTheDocument();
    expect(within(ops).getByText(/Stored in this browser only \(localStorage\)/)).toBeInTheDocument();
  });
});
