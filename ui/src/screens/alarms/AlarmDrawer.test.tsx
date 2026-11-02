import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { alarmList } from "../../model/selectors";
import { runnerEvent } from "../../test/fakeClient";
import { hybridModel, modelFromEvents } from "../../test/modelFixture";
import { AlarmDrawer } from "./AlarmDrawer";

const draw = (model = hybridModel(), extra: Partial<Parameters<typeof AlarmDrawer>[0]> = {}) => {
  const props = { model, alarms: alarmList(model.runner, model.localAcks), open: true, operatorId: "op1", onClose: vi.fn(), onOpenDetail: vi.fn(), ...extra };
  render(<AlarmDrawer {...props} />);
  return props;
};

describe("AlarmDrawer", () => {
  it("lists a High unacknowledged discard alarm and acknowledges on the runner", async () => {
    const props = draw();
    const row = screen.getByText("Vial 1").closest("tr")!;
    expect(within(row).getByText(/High/)).toBeInTheDocument();
    expect(within(row).getByText("Unacknowledged")).toBeInTheDocument();
    await userEvent.click(within(row).getByRole("button", { name: "Acknowledge" }));
    expect(props.model.acknowledge).toHaveBeenCalledWith("alarm-final-c1-CASE_97a8c6cb8161881c", "runner", "op1");
  });

  it("closes on Escape", async () => {
    const props = draw();
    screen.getByRole("dialog", { name: "Alarm list" }).focus();
    await userEvent.keyboard("{Escape}");
    expect(props.onClose).toHaveBeenCalled();
  });

  it("shows a Medium UI alarm for can't assess", () => {
    const model = modelFromEvents([runnerEvent(1, "vial_decision", { disposition: "insufficient_evidence", unavailable_reason: "measurement_stage_unavailable" }, "unavailable", 0)]);
    const props = draw(model);
    expect(props.alarms[0]).toMatchObject({ priority: "medium", origin: "ui" });
    expect(screen.getByText(/Medium/)).toBeInTheDocument();
  });
});
