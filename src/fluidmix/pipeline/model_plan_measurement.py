"""Orchestrate memory-backed peer selection for synthetic plan measurement."""

from __future__ import annotations

from fluidmix.measurement.vision_measurement import (
    collect_model_plan_peer_requests,
    evaluate_model_plan as _evaluate_model_plan,
    write_measured_evidence as _write_measured_evidence,
)
from fluidmix.memory.history.vision_dataset_plan import select_peer_reference
from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1
from fluidmix.storage.artifacts.capture_manifest_adapter import (
    build_measured_evidence as _build_measured_evidence,
)


def evaluate_model_plan(model_plan: object, image_locators: object) -> dict[str, object]:
    """Measure a plan with peer values selected through the memory boundary."""

    requests = collect_model_plan_peer_requests(
        model_plan,
        image_locators,
        prediction_registry_factory=build_trusted_predicted_mask_registry_v1,
    )
    references = {
        request_sha256: select_peer_reference(
            request["candidates"],
            target_mask_sha256=request["target_mask_sha256"],
            target_batch_id=request["target_batch_id"],
            target_run_id=request["target_run_id"],
            decision_config=request["decision_config"],
        )
        for request_sha256, request in requests.items()
    }
    return _evaluate_model_plan(
        model_plan,
        image_locators,
        peer_references=references,
        prediction_registry_factory=build_trusted_predicted_mask_registry_v1,
    )


def write_measured_evidence(*args: object, **kwargs: object) -> dict[str, object]:
    """Publish measured evidence with the memory peer selector injected."""

    return _write_measured_evidence(
        *args,
        **kwargs,
        model_plan_evaluator=evaluate_model_plan,
    )


def build_measured_evidence(*args: object, **kwargs: object) -> dict[str, object]:
    """Bind capture inputs and measure them with memory-selected peers."""

    return _build_measured_evidence(
        *args,
        **kwargs,
        model_plan_evaluator=evaluate_model_plan,
    )


__all__ = [
    "build_measured_evidence",
    "evaluate_model_plan",
    "write_measured_evidence",
]
