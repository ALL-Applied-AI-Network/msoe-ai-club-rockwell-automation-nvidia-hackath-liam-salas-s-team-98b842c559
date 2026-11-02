import type { InspectionModel } from "../../model/useInspection";
import { CountsBar } from "./CountsBar";
import { CurrentVialCard } from "./CurrentVialCard";
import { TrayStrip } from "./TrayStrip";
import { VideoPanel } from "./VideoPanel";
import "./station.css";

export interface StationScreenProps { readonly model: InspectionModel; readonly operatorId: string; readonly onOpenDetail: (vialId: string) => void }

/** L2 inspection station: live video with honest overlays, current vial verdict, tray strip and session counts. */
export function StationScreen({ model, operatorId, onOpenDetail }: StationScreenProps) {
  return (
    <section className="screen station" aria-label="Inspection station">
      <div className="station-top">
        <VideoPanel model={model} />
        <CurrentVialCard key={model.selectedVialId ?? "none"} model={model} operatorId={operatorId} onOpenDetail={onOpenDetail} />
      </div>
      <TrayStrip model={model} />
      <CountsBar model={model} />
    </section>
  );
}
