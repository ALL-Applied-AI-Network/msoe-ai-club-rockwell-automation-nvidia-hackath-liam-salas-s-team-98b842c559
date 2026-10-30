import type { Provenance } from "../data/adapter";

export function ProvenanceChip({ provenance }: { readonly provenance: Provenance }) {
  return (
    <button type="button" className={`provenance provenance--${provenance.kind}`} aria-label={`Provenance ${provenance.kind}`} aria-description={`Source: ${provenance.source_id}. Note: ${provenance.note}`} title={`${provenance.source_id} — ${provenance.note}`}>
      {provenance.kind}
    </button>
  );
}
