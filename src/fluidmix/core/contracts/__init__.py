"""Frozen Part A and Part B contracts with lazy module boundaries."""

from __future__ import annotations

from importlib import import_module


_EVIDENCE_EXPORTS = {
    "publish_evidence_derivation_v1",
    "validate_evidence_record_v1",
    "validate_evidence_sequence_v1",
    "verify_evidence_artifacts",
    "verify_evidence_derivation_v1",
}
_SIM_EXPORTS = {
    "validate_sim_metadata_binding_v1",
    "validate_sim_metadata_runtime_schedule_v1",
    "validate_sim_metadata_v1",
    "verify_sim_metadata_artifacts",
}
__all__ = sorted(_EVIDENCE_EXPORTS | _SIM_EXPORTS)


def __getattr__(name: str):
    if name in _EVIDENCE_EXPORTS:
        module = import_module(".evidence", __name__)
    elif name in _SIM_EXPORTS:
        module = import_module(".sim_metadata", __name__)
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value
