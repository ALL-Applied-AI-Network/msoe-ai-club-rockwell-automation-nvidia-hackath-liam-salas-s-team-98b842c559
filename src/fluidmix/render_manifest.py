"""Strict contract for the bounded FillCam renderer feasibility run."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .twin_material import DirectMaterial, make_direct_material


_KIT_ARGS = (
    "--/telemetry/enableAnonymousData=false",
    "--/structuredLog/enable=false",
)
_TOP_KEYS = {
    "schema_version", "camera", "subframes", "timeout_seconds", "kit_args", "smoke", "batch"
}
_JOB_KEYS = {"id", "frame", "material_color", "opacity"}
_EXPECTED_SMOKE = (("S00", 686, None),)
_EXPECTED_BATCH = (
    ("M00", 686, ((0.006, 0.2, 0.3), 0.6)),
    ("M01", 686, ((0.45, 0.12, 0.01), 0.6)),
    ("M02", 686, ((0.0615, 0.19, 0.26375), 0.6)),
    ("M03", 686, ((0.117, 0.18, 0.2275), 0.6)),
    ("M04", 686, ((0.228, 0.16, 0.155), 0.6)),
    ("M05", 686, ((0.28, 0.02, 0.22), 0.6)),
    ("F00", 400, None),
    ("F01", 500, None),
)


@dataclass(frozen=True)
class RenderJob:
    variant_id: str
    frame: int
    material: DirectMaterial | None


@dataclass(frozen=True)
class RenderManifest:
    schema_version: int
    camera: str
    subframes: int
    timeout_seconds: int
    kit_args: tuple[str, ...]
    smoke: tuple[RenderJob, ...]
    batch: tuple[RenderJob, ...]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _job(payload: Any) -> RenderJob:
    if not isinstance(payload, dict):
        raise ValueError("each render job must be an object")
    extra = set(payload) - _JOB_KEYS
    if extra:
        raise ValueError(f"render job has unexpected fields: {sorted(extra)}")
    if "id" not in payload or "frame" not in payload:
        raise ValueError("each render job requires id and frame")
    frame = _positive_int(payload["frame"], "frame")
    if frame > 686:
        raise ValueError("FillCam frame must be in 1..686")

    has_color = "material_color" in payload
    has_opacity = "opacity" in payload
    if has_color != has_opacity:
        raise ValueError("direct material jobs require both material_color and opacity")
    if has_color:
        material = make_direct_material(
            payload["material_color"], payload["opacity"], payload["id"]
        )
        variant_id = material.variant_id
    else:
        # Reuse the neutral-ID validator without assigning this default-stage job a material.
        variant_id = make_direct_material((0, 0, 0), 0, payload["id"]).variant_id
        material = None
    return RenderJob(variant_id=variant_id, frame=frame, material=material)


def _job_signature(job: RenderJob) -> tuple:
    material = None
    if job.material is not None:
        material = (job.material.color, job.material.opacity)
    return job.variant_id, job.frame, material


def load_render_manifest(path: str | Path) -> RenderManifest:
    """Load the frozen schema and reject any unreviewed expansion."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read render manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("render manifest must be an object")
    if set(payload) != _TOP_KEYS:
        raise ValueError("render manifest fields do not match schema version 1")
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if payload["camera"] != "FillCam":
        raise ValueError("camera must be the fixed FillCam view")
    subframes = _positive_int(payload["subframes"], "subframes")
    timeout_seconds = _positive_int(payload["timeout_seconds"], "timeout_seconds")
    if timeout_seconds != 1200:
        raise ValueError("timeout_seconds must remain frozen at 1200")
    if not isinstance(payload["kit_args"], list) or tuple(payload["kit_args"]) != _KIT_ARGS:
        raise ValueError("kit args must remain the reviewed telemetry-off pair")
    if not isinstance(payload["smoke"], list) or len(payload["smoke"]) != 1:
        raise ValueError("smoke phase must contain exactly one job")
    if not isinstance(payload["batch"], list) or len(payload["batch"]) > 8:
        raise ValueError("batch may contain at most eight jobs")

    smoke = tuple(_job(item) for item in payload["smoke"])
    batch = tuple(_job(item) for item in payload["batch"])
    ids = [job.variant_id for job in smoke + batch]
    if len(ids) != len(set(ids)):
        raise ValueError("render job IDs must be unique across phases")
    if tuple(map(_job_signature, smoke)) != _EXPECTED_SMOKE or tuple(
        map(_job_signature, batch)
    ) != _EXPECTED_BATCH:
        raise ValueError("jobs do not match the frozen render matrix")
    return RenderManifest(
        schema_version=1,
        camera="FillCam",
        subframes=subframes,
        timeout_seconds=timeout_seconds,
        kit_args=_KIT_ARGS,
        smoke=smoke,
        batch=batch,
    )
