import type { InspectionModel } from "../../model/useInspection";

export interface VialDetailScreenProps { readonly model: InspectionModel; readonly vialId: string | null; readonly onSelectVial: (vialId: string) => void }

/** L3 vial detail. Stub: lane 2 implements. */
export function VialDetailScreen({ vialId }: VialDetailScreenProps) {
  return <section className="screen" aria-label="Vial detail"><h2>Vial detail</h2><p>{vialId ?? "No vial selected."}</p></section>;
}
