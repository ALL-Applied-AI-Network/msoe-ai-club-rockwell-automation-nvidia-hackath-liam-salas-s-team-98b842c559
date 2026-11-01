"""Bind capture manifests before building explanation model-input plans."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

from fluidmix.core.io import load_strict_json_snapshot
from fluidmix.explanation.vlm.model_input_plan_builder import (
    ModelInputPlanError,
    build_model_input_plan as _build_model_input_plan,
    validate_model_plan_context,
)
from fluidmix.measurement.vision_measurement import _publish_json_no_clobber
from fluidmix.storage.artifacts.capture_manifest_adapter import (
    bind_stage_capture_manifest,
)


def _bind_captures(
    capture_paths: Sequence[str | os.PathLike[str]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if not capture_paths:
        raise ModelInputPlanError("capture_manifests_missing")
    bound: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for item in capture_paths:
        path = Path(item)
        try:
            payload, snapshot = load_strict_json_snapshot(path)
            capture = bind_stage_capture_manifest(payload, base_dir=path.parent)
        except (OSError, ValueError) as error:
            raise ModelInputPlanError("capture_manifest_invalid") from error
        batch_id = capture.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ModelInputPlanError("capture batch_id_invalid")
        if batch_id in seen:
            raise ModelInputPlanError("duplicate_capture_batch")
        seen.add(batch_id)
        hashes[batch_id] = str(snapshot["sha256"])
        bound.append(capture)
    return bound, hashes


def build_model_input_plan(
    capture_manifests: Sequence[str | os.PathLike[str]],
    batch_context: object,
) -> dict[str, object]:
    """Pass storage-bound capture values into the pure plan builder."""

    context_rows, context_hash = validate_model_plan_context(batch_context)
    captures, capture_hashes = _bind_captures(capture_manifests)
    return _build_model_input_plan(
        captures,
        capture_hashes,
        context_rows,
        context_hash,
    )


def write_model_input_plan(
    capture_manifests: Sequence[str | os.PathLike[str]],
    batch_context: object,
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Build and atomically publish a new plan, refusing an existing target."""

    plan = build_model_input_plan(capture_manifests, batch_context)
    _publish_json_no_clobber(Path(output_path), plan)
    return plan


__all__ = ["build_model_input_plan", "write_model_input_plan"]
