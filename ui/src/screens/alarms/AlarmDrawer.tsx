import { useEffect, useRef } from "react";

import type { AlarmItem, AlarmPriority } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";
import { clockTime } from "../overview/format";
import "./alarms.css";

export interface AlarmDrawerProps {
  readonly model: InspectionModel; readonly alarms: readonly AlarmItem[]; readonly open: boolean; readonly operatorId: string;
  readonly onClose: () => void; readonly onOpenDetail: (vialId: string) => void;
}

const PRIORITY: Record<AlarmPriority, { label: string; shape: string; tone: "red" | "amber" | "plain" }> = {
  critical: { label: "Critical", shape: "▲", tone: "red" }, high: { label: "High", shape: "◆", tone: "red" },
  medium: { label: "Medium", shape: "●", tone: "amber" }, low: { label: "Low", shape: "○", tone: "plain" }
};

/** ISA-18.2 style alarm list in a right-side drawer. */
export function AlarmDrawer({ model, alarms, open, operatorId, onClose, onOpenDetail }: AlarmDrawerProps) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => { if (open) ref.current?.focus(); }, [open]);
  if (!open) return null;
  return (
    <aside ref={ref} className="alarm-drawer panel" role="dialog" aria-modal="true" aria-label="Alarm list" tabIndex={-1}
      onKeyDown={e => { if (e.key === "Escape") onClose(); }}>
      <header className="alarm-head"><h2>Alarms</h2><button type="button" className="hmi-button" onClick={onClose}>Close</button></header>
      {alarms.length === 0 ? <p className="unavailable">No alarms.</p> : (
        <table className="alarm-table">
          <thead><tr><th>Priority</th><th>Time</th><th>Vial</th><th>Message</th><th>State</th><th>Action</th></tr></thead>
          <tbody>
            {alarms.map(a => {
              const p = PRIORITY[a.priority];
              return (
                <tr key={a.id} className={a.acknowledged ? "alarm-row" : "alarm-row alarm-row--open"}>
                  <td><span className={`alarm-priority alarm-priority--${p.tone}`}><span aria-hidden="true">{p.shape}</span> {p.label}</span></td>
                  <td>{clockTime(a.at)}</td>
                  <td>{a.vialLabel ?? "—"}</td>
                  <td>{a.message}</td>
                  <td>{a.acknowledged ? "Acknowledged" : "Unacknowledged"}</td>
                  <td className="alarm-actions">
                    <button type="button" className="hmi-button" disabled={a.acknowledged} onClick={() => { void model.acknowledge(a.id, a.origin, operatorId || "operator"); }}>Acknowledge</button>
                    {a.vialId && <button type="button" className="hmi-button" onClick={() => onOpenDetail(a.vialId!)}>Open vial</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="alarm-note">UI alarms (can't assess, service, errors) are acknowledged in this browser only; discard alarms are acknowledged on the runner.</p>
    </aside>
  );
}
