import { render, screen } from "@testing-library/react";
import fixture from "../../ui-data/mock-run.json";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

describe("App data loader", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads and parses only the browser-facing active run", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => fixture });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    expect(screen.getByRole("status")).toHaveTextContent(/loading run/i);
    expect(await screen.findByRole("heading", { name: /synthetic twin data/i })).toBeVisible();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("/active-run.json", expect.any(Object));
    expect(fetchMock).not.toHaveBeenCalledWith("/mock-run.json", expect.anything());
  });

  it("shows an actionable error for unavailable or invalid data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not load.*refresh.*active-run\.json/i);
  });

  it("shows the same actionable error when the local run violates its schema", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ schema_version: "broken" }) }));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not load.*refresh.*active-run\.json/i);
  });
});
