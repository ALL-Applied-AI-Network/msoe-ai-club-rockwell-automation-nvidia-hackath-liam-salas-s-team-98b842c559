import type { AlarmItem } from "../../model/selectors";
import type { InspectionModel } from "../../model/useInspection";

export interface AlarmDrawerProps {
  readonly model: InspectionModel; readonly alarms: readonly AlarmItem[]; readonly open: boolean; readonly operatorId: string;
  readonly onClose: () => void; readonly onOpenDetail: (vialId: string) => void;
}

/** ISA-18.2 style alarm list drawer. Stub: lane 3 implements. */
export function AlarmDrawer({ open, onClose }: AlarmDrawerProps) {
  if (!open) return null;
  return <aside role="dialog" aria-label="Alarm list"><button type="button" onClick={onClose}>Close</button></aside>;
}
