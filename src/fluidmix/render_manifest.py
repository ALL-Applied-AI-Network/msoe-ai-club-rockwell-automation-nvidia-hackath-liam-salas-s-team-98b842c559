"""Strict manifest loader for the single FillCam recovery smoke attempt."""

from dataclasses import dataclass
import json
from pathlib import Path


_EXPECTED_ATTEMPT_ID = "fillcam-recovery-s00-20261024-a1"
_EXPECTED_KIT_ARGS = (
    "--/telemetry/enableAnonymousData=false",
    "--/structuredLog/enable=false",
    "--/exts/isaacsim.core.throttling/enable_async=false",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "attempt_id",
    "camera",
    "renderer",
    "resolution",
    "subframes",
    "timeout_seconds",
    "kit_args",
    "smoke",
}


@dataclass(frozen=True)
class RenderJob:
    """The one immutable smoke render request."""

    variant_id: str
    frame: int


@dataclass(frozen=True)
class RenderManifest:
    """Frozen inputs for the one permitted recovery renderer spawn."""

    schema_version: int
    attempt_id: str
    camera: str
    renderer: str
    resolution: tuple[int, int]
    subframes: int
    timeout_seconds: int
    kit_args: tuple[str, ...]
    smoke: RenderJob


def _require_exact(value: object, expected: object, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"recovery manifest {field} must be exactly {expected!r}")


def load_render_manifest(path: str | Path) -> RenderManifest:
    """Load the exact reviewed recovery contract; reject every extension or drift."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"recovery manifest must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(candidate.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recovery manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recovery manifest root must be an object")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError(
            "recovery manifest top-level fields must be exactly "
            f"{sorted(_TOP_LEVEL_KEYS)!r}"
        )

    _require_exact(payload["schema_version"], 2, "schema_version")
    _require_exact(payload["attempt_id"], _EXPECTED_ATTEMPT_ID, "attempt_id")
    _require_exact(payload["camera"], "FillCam", "camera")
    _require_exact(payload["renderer"], "PathTracing", "renderer")
    _require_exact(payload["resolution"], [640, 480], "resolution")
    _require_exact(payload["subframes"], 32, "subframes")
    _require_exact(payload["timeout_seconds"], 1200, "timeout_seconds")
    _require_exact(payload["kit_args"], list(_EXPECTED_KIT_ARGS), "kit_args")

    smoke = payload["smoke"]
    if not isinstance(smoke, dict) or set(smoke) != {"id", "frame"}:
        raise ValueError("recovery manifest smoke fields must be exactly ['frame', 'id']")
    _require_exact(smoke["id"], "S00", "smoke.id")
    _require_exact(smoke["frame"], 686, "smoke.frame")

    return RenderManifest(
        schema_version=2,
        attempt_id=_EXPECTED_ATTEMPT_ID,
        camera="FillCam",
        renderer="PathTracing",
        resolution=(640, 480),
        subframes=32,
        timeout_seconds=1200,
        kit_args=_EXPECTED_KIT_ARGS,
        smoke=RenderJob(variant_id="S00", frame=686),
    )
