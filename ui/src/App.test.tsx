import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("opens the live source picker without fetching Phase 2 run artifacts", () => {
    const fetcher = vi.spyOn(globalThis, "fetch");

    render(<App />);

    expect(screen.getByRole("heading", { name: /live vial decisions/i })).toBeVisible();
    expect(screen.getByRole("group", { name: /inspection source/i })).toBeVisible();
    expect(fetcher).not.toHaveBeenCalled();
    expect(document.body).not.toHaveTextContent(/development data|final data|demo mode/i);
    fetcher.mockRestore();
  });
});
