import type { InspectionModel } from "../../model/useInspection";

export interface OverviewScreenProps { readonly model: InspectionModel; readonly onOpenDetail: (vialId: string) => void }

/** L1 line overview. Stub: lane 3 implements. */
export function OverviewScreen(_props: OverviewScreenProps) {
  return <section className="screen" aria-label="Line overview"><h2>Line overview</h2></section>;
}
