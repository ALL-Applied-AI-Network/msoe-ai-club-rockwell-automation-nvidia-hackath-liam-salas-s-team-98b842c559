import type { InspectionModel } from "../../model/useInspection";

export interface StationScreenProps { readonly model: InspectionModel; readonly operatorId: string; readonly onOpenDetail: (vialId: string) => void }

/** L2 inspection station (home). Stub: lane 1 implements. */
export function StationScreen({ model }: StationScreenProps) {
  return <section className="screen" aria-label="Inspection station"><h2>Inspection station</h2><p>{model.runner ? `${model.runner.trayOrder.length} vials` : "No session."}</p></section>;
}
