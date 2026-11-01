"""Inject memory-backed peer selection into validation-stage measurement."""

from __future__ import annotations

from fluidmix.measurement.validation_stage_measurement import (
    write_validation_stage_evidence as _write_validation_stage_evidence,
)
from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan


def write_validation_stage_evidence(
    *args: object, **kwargs: object
) -> dict[str, object]:
    """Publish validation evidence with the memory peer selector injected."""

    return _write_validation_stage_evidence(
        *args,
        **kwargs,
        model_plan_evaluator=evaluate_model_plan,
    )


__all__ = ["write_validation_stage_evidence"]
