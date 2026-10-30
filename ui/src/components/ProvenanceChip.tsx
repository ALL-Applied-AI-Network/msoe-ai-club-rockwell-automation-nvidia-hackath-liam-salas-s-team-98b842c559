import type { Provenance } from "../data/adapter";

export function ProvenanceChip({ provenance }: { readonly provenance: Provenance }) {
  return (
    <span className={`provenance provenance--${provenance.kind}`} title={`${provenance.source_id} — ${provenance.note}`}>
      {provenance.kind}
    </span>
  );
}
