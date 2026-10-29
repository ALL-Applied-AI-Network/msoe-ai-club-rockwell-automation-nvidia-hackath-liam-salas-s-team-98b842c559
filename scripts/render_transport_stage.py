#!/usr/bin/env python3
"""Render immutable transport state layers with frozen camera/light and runtime evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Protocol

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.render_capture import resolve_instance_paths  # noqa: E402
from fluidmix.rigid_vial_transport import (  # noqa: E402
    INCOMING_PRIM_PATHS,
    OUTGOING_PRIM_PATHS,
    QUEUED_PRIM_PATHS,
    load_rigid_transport_inputs,
)
from fluidmix.transport_stage import (  # noqa: E402
    ALL_STATE_INDICES,
    CAMERA_AND_LIGHT_PATHS,
    capture_transport_implementation,
    capture_evidence_closure,
    freeze_runtime_partition,
    validate_evidence_closure,
    validate_pilot_visual_review,
    validate_runtime_partition,
    validate_wp1_fill_readiness,
)


PILOT_STATE_INDICES = (0, 6, 12)
CAMERA_PATH = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
RUNTIME_PRODUCT_NAME = "FluidmixTransportProduct"
FRAME_SHAPE = (480, 640, 3)
RTX_RENDER_MODE_SETTING = "/rtx/rendermode"
RTX_RENDER_MODE_TARGET = "RaytracedLighting"
RTX_RENDER_MODE_MAX_APP_UPDATES = 120
REPLICATOR_EXTRA_ARGS = (
    "--/exts/isaacsim.core.throttling/enable_async=false",
    "--/persistent/rtx/modes/rt/enabled=true",
    "--/persistent/rtx/modes/pt/enabled=true",
)
POST_SIMULATION_APP_MARKER = "TRANSPORT_LAUNCHER_SIMULATION_APP_RETURNED"


class TransportStageRuntime(Protocol):
    def preflight_state(self, state_layer: Path, state_index: int) -> Mapping[str, Any]: ...

    def capture_state(self, state_layer: Path, state_index: int) -> Mapping[str, Any]: ...

    def attest_current(self) -> Mapping[str, Any]: ...

    def finalize_for_publication(self) -> Sequence[Mapping[str, Any]]: ...

    def close(self) -> None: ...


def _transition_rtx_render_mode(
    *,
    settings: Any,
    app: Any,
    records: list[dict[str, Any]],
    max_updates: int = RTX_RENDER_MODE_MAX_APP_UPDATES,
    state_index: int | None = None,
    opening_kind: str | None = None,
) -> dict[str, Any]:
    """Request exact RT2 mode and pump Kit until its async readback settles."""

    if (
        isinstance(max_updates, bool)
        or not isinstance(max_updates, int)
        or max_updates < 0
    ):
        raise ValueError("RTX render-mode max_updates must be a non-negative integer")
    before = str(settings.get_as_string(RTX_RENDER_MODE_SETTING))
    settings.set_string(RTX_RENDER_MODE_SETTING, RTX_RENDER_MODE_TARGET)
    actual = str(settings.get_as_string(RTX_RENDER_MODE_SETTING))
    app_updates = 0
    while actual != RTX_RENDER_MODE_TARGET and app_updates < max_updates:
        app.update()
        app_updates += 1
        actual = str(settings.get_as_string(RTX_RENDER_MODE_SETTING))
    record = {
        "setting_path": RTX_RENDER_MODE_SETTING,
        "requested": RTX_RENDER_MODE_TARGET,
        "before": before,
        "actual_readback": actual,
        "app_updates": app_updates,
        "max_app_updates": max_updates,
        "passed": actual == RTX_RENDER_MODE_TARGET,
    }
    if state_index is not None or opening_kind is not None:
        if type(state_index) is not int or opening_kind not in {"preflight", "capture"}:
            raise ValueError("RTX render-mode opening context is invalid")
        record = {
            "state_index": state_index,
            "opening_kind": opening_kind,
            **record,
        }
    records.append(record)
    if not record["passed"]:
        raise RuntimeError(
            "RTX render mode never reached exact RaytracedLighting after "
            f"{app_updates} app updates; actual readback was {actual!r}"
        )
    return record


def _validated_render_mode_transitions(
    value: object, *, expected_state_indices: Sequence[int]
) -> list[dict[str, Any]]:
    required = {
        "state_index",
        "opening_kind",
        "setting_path",
        "requested",
        "before",
        "actual_readback",
        "app_updates",
        "max_app_updates",
        "passed",
    }
    expected_openings = [
        *(("preflight", index) for index in expected_state_indices),
        *(("capture", index) for index in expected_state_indices),
    ]
    if not expected_openings or not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ) or len(value) != len(expected_openings):
        raise ValueError("transport render-mode transition membership is invalid")
    result: list[dict[str, Any]] = []
    for row, (opening_kind, state_index) in zip(value, expected_openings, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != required
            or type(row.get("state_index")) is not int
            or row.get("state_index") != state_index
            or row.get("opening_kind") != opening_kind
            or row.get("setting_path") != RTX_RENDER_MODE_SETTING
            or row.get("requested") != RTX_RENDER_MODE_TARGET
            or not isinstance(row.get("before"), str)
            or row.get("actual_readback") != RTX_RENDER_MODE_TARGET
            or type(row.get("app_updates")) is not int
            or not 0 <= row["app_updates"] <= RTX_RENDER_MODE_MAX_APP_UPDATES
            or row.get("max_app_updates") != RTX_RENDER_MODE_MAX_APP_UPDATES
            or row.get("passed") is not True
        ):
            raise ValueError("transport render-mode transition evidence is invalid")
        result.append(copy.deepcopy(dict(row)))
    return result


def _register_reviewed_mdl_search_path(
    *,
    app: Any,
    mdl_file: Path,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> tuple[Any, Any]:
    """Register only the directory containing the reviewed transport MDL file."""

    reviewed_mdl = mdl_file.resolve(strict=True)
    if not reviewed_mdl.is_file():
        raise RuntimeError("reviewed spatial OmniSurface MDL asset is not a file")
    manager = app.get_extension_manager()
    extension = "omni.mdl.neuraylib"
    if not manager.is_extension_enabled(extension):
        manager.set_extension_enabled_immediate(extension, True)
    if not manager.is_extension_enabled(extension):
        raise RuntimeError("omni.mdl.neuraylib could not be enabled")
    module = module_loader(extension)
    neuraylib = module.get_neuraylib()
    registration = neuraylib.addMdlSearchPath(os.fspath(reviewed_mdl.parent))
    if not registration:
        raise RuntimeError("reviewed spatial OmniSurface MDL path registration failed")
    return neuraylib, registration


def _remove_reviewed_mdl_search_path(neuraylib: Any, registration: Any) -> None:
    """Remove the exact MDL search-path registration returned by Kit."""

    if not neuraylib.removeMdlSearchPath(registration):
        raise RuntimeError("reviewed spatial OmniSurface MDL path removal failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _launcher_trace_path(output_dir: Path) -> Path:
    output = output_dir.resolve()
    return output.with_name(output.name + ".launcher-trace.jsonl")


class _LauncherTrace:
    """Exclusive, owner-bound launcher journal; each event is durable before return."""

    def __init__(self, path: Path, *, owner_thread_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.owner_thread_id = owner_thread_id
        self.sequence_index = 0
        self.phase = "preflight"
        self._handle = path.open("x", encoding="utf-8")

    def append(
        self, event_type: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        phases = {
            "simulation_app_construction_started": "simulation_app_construction",
            "simulation_app_construction_returned": "kit_app_acquisition",
            "kit_app_acquired_running": "kit_app_acquisition",
            "runtime_construction_started": "runtime_construction",
            "runtime_construction_returned": "transport_run",
            "transport_run_started": "transport_run",
            "transport_run_returned": "transport_run",
            "close_started": "close",
            "close_returned": "close",
        }
        self.phase = phases.get(event_type, self.phase)
        event = {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_launcher_trace_event_v1",
            "owner_thread_id": self.owner_thread_id,
            "sequence_index": self.sequence_index,
            "event_type": event_type,
            "details": copy.deepcopy(dict(details or {})),
        }
        encoded = json.dumps(
            event, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        self._handle.write(encoded + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.sequence_index += 1

    def close(self) -> None:
        self._handle.close()


class _KitStartupFailure(RuntimeError):
    def __init__(self, message: str, *, owned_app: Any) -> None:
        super().__init__(message)
        self.owned_app = owned_app


def _start_or_reuse_kit(
    *, record_event: Callable[..., None] | None = None,
) -> tuple[Any, Any | None]:
    """Return a running Kit app and the SimulationApp owned by this launcher, if any."""

    emit = record_event or (lambda *_args, **_kwargs: None)
    try:
        import omni.kit.app

        current = omni.kit.app.get_app()
        if current is not None and bool(current.is_running()):
            emit("kit_app_acquired_running", details={"owned": False, "reused": True})
            return current, None
    except Exception:
        pass

    emit("simulation_app_construction_started")
    from isaacsim import SimulationApp

    owned = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "RaytracedLighting",
            "extra_args": list(REPLICATOR_EXTRA_ARGS),
        }
    )
    print(POST_SIMULATION_APP_MARKER, flush=True)
    emit("simulation_app_construction_returned")

    import omni.kit.app

    app = omni.kit.app.get_app()
    if app is None or not bool(app.is_running()):
        raise _KitStartupFailure(
            "SimulationApp returned without a running Kit application",
            owned_app=owned,
        )
    emit("kit_app_acquired_running", details={"owned": True, "reused": False})
    return app, owned


def _canonical_authored_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("session spec contains a nonfinite authored value")
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, Mapping):
        rows = [
            {
                "key": _canonical_authored_value(key),
                "value": _canonical_authored_value(item),
            }
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: json.dumps(row["key"], sort_keys=True, separators=(",", ":")))
        return {"type": "mapping", "items": rows}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "type": "sequence",
            "items": [_canonical_authored_value(item) for item in value],
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "text": str(value),
    }


def _session_spec_record(spec: Any) -> dict[str, Any]:
    keys = sorted(str(key) for key in spec.ListInfoKeys())
    return {
        "path": str(spec.path),
        "fields": [
            {"key": key, "value": _canonical_authored_value(spec.GetInfo(key))}
            for key in keys
        ],
    }


def _attest_scene_session_specs(root_prims: Sequence[Any]) -> dict[str, Any]:
    """Attest every authored field on every non-render session-layer spec."""

    prim_specs: list[dict[str, Any]] = []
    property_specs: list[dict[str, Any]] = []

    def visit(spec: Any) -> None:
        path = str(spec.path)
        if path == "/Render" or path.startswith("/Render/"):
            return
        prim_specs.append(_session_spec_record(spec))
        for item in spec.properties:
            property_path = str(item.path)
            if not (
                property_path.startswith("/Render/")
                or property_path.startswith("/Render.")
            ):
                property_specs.append(_session_spec_record(item))
        for child in spec.nameChildren:
            visit(child)

    for root in root_prims:
        visit(root)
    prim_specs.sort(key=lambda row: row["path"])
    property_specs.sort(key=lambda row: row["path"])
    payload: dict[str, Any] = {
        "prim_specs": prim_specs,
        "property_specs": property_specs,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _records_to_paths(value: object, label: str, expected_count: int | None = None) -> list[Path]:
    if not isinstance(value, list) or (
        expected_count is not None and len(value) != expected_count
    ):
        raise ValueError(f"{label} evidence membership is invalid")
    paths: list[Path] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError(f"{label} evidence record is malformed")
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or type(row["bytes"]) is not int
            or path.stat().st_size != row["bytes"]
            or _sha256(path) != row["sha256"]
        ):
            raise ValueError(f"{label} evidence differs from its build closure")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label} evidence paths are duplicated")
    return paths


def _load_build(segment_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Path]]]:
    build_path = segment_dir / "transport-build-receipt.json"
    build = _strict_object(build_path, "transport build receipt")
    if (
        build.get("schema_version") != 1
        or build.get("artifact_type") != "fluidmix_transport_stage_build_receipt_v1"
        or build.get("status") != "pass"
        or build.get("source_time_code") != 1
        or build.get("physics_steps") != 0
        or build.get("camera_and_light_specs_authored") != []
    ):
        raise ValueError("transport build receipt is not an accepted immutable build")
    manifest = build.get("layer_manifest")
    closure = build.get("pre_output_evidence_closure")
    if not isinstance(manifest, dict) or not isinstance(closure, dict):
        raise ValueError("transport build receipt lacks manifest or evidence closure")
    if (
        not isinstance(build.get("owner_thread_id"), str)
        or manifest.get("owner_thread_id") != build.get("owner_thread_id")
    ):
        raise ValueError("transport build owner tag is missing or inconsistent")
    implementation = build.get("transport_implementation")
    if (
        not isinstance(implementation, Mapping)
        or dict(implementation) != capture_transport_implementation(ROOT)
        or manifest.get("transport_implementation_sha256")
        != implementation.get("sha256")
    ):
        raise ValueError("transport implementation bytes differ from the build")
    state_rows = manifest.get("state_layers")
    if not isinstance(state_rows, list) or len(state_rows) != 13:
        raise ValueError("transport layer manifest does not bind 13 states")
    state_paths: list[Path] = []
    for expected_index, row in enumerate(state_rows):
        if not isinstance(row, Mapping) or row.get("state_index") != expected_index:
            raise ValueError("transport state layer order is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("transport state layer path is invalid")
        path = (segment_dir / relative).resolve()
        try:
            path.relative_to(segment_dir)
        except ValueError as error:
            raise ValueError("transport state layer escapes its build root") from error
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError("transport state layer differs from its manifest")
        state_paths.append(path)
    evidence_paths = {
        "source_files": _records_to_paths(closure.get("source_files"), "source file"),
        "snapshot_arrays": _records_to_paths(
            closure.get("snapshot_arrays"), "snapshot array", 13
        ),
        "state_layers": _records_to_paths(closure.get("state_layers"), "state layer", 13),
        "dependency_files": _records_to_paths(
            closure.get("dependency_files"), "dependency file"
        ),
    }
    if evidence_paths["state_layers"] != sorted(state_paths):
        raise ValueError("transport state manifest and closure differ")
    validate_evidence_closure(
        closure,
        source_files=evidence_paths["source_files"],
        snapshot_arrays=evidence_paths["snapshot_arrays"],
        state_layers=evidence_paths["state_layers"],
        dependency_files=evidence_paths["dependency_files"],
        counters=closure["counters"],
    )
    return build, closure, evidence_paths


def _validated_attestation(value: object, state_index: int) -> dict[str, Any]:
    required = {
        "state_index",
        "source_time_code",
        "physics_steps",
        "physics_callbacks",
        "timeline_playing",
        "runtime_partition",
        "camera_light_before_annotators_sha256",
        "camera_light_sha256",
        "scene_session_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("transport runtime attestation schema is malformed")
    result = copy.deepcopy(dict(value))
    if (
        result["state_index"] != state_index
        or result["source_time_code"] != 1
        or result["physics_steps"] != 0
        or result["physics_callbacks"] != 0
        or result["timeline_playing"] is not False
        or not isinstance(result["camera_light_before_annotators_sha256"], str)
        or len(result["camera_light_before_annotators_sha256"]) != 64
        or not isinstance(result["camera_light_sha256"], str)
        or len(result["camera_light_sha256"]) != 64
        or not isinstance(result["scene_session_sha256"], str)
        or len(result["scene_session_sha256"]) != 64
        or not isinstance(result["runtime_partition"], Mapping)
    ):
        raise ValueError("transport runtime attestation is not frozen at source time 1")
    validate_runtime_partition(
        result["runtime_partition"],
        result["runtime_partition"].get("prim_paths", []),
        result["runtime_partition"].get("property_paths", []),
    )
    return result


def _save_capture(root: Path, state_index: int, value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"rgb", "instance_ids", "instance_id_to_path", "group_masks", "attestation"}:
        raise ValueError("transport capture schema is malformed")
    rgb = value["rgb"]
    instance = value["instance_ids"]
    mapping = value["instance_id_to_path"]
    masks = value["group_masks"]
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.shape != FRAME_SHAPE
        or not isinstance(instance, np.ndarray)
        or instance.shape != FRAME_SHAPE[:2]
        or not np.issubdtype(instance.dtype, np.integer)
        or not isinstance(mapping, Mapping)
        or not isinstance(masks, Mapping)
        or set(masks) != {"outgoing", "incoming", "queued", "incoming_labels", "queue_tail"}
    ):
        raise ValueError("transport capture arrays or mappings are malformed")
    states = root / "states"
    states.mkdir(exist_ok=True)
    prefix = f"state-{state_index:02d}"
    rgb_path = states / f"{prefix}.png"
    if not cv2.imwrite(os.fspath(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("transport RGB could not be written")
    instance_path = states / f"{prefix}-instance.npy"
    with instance_path.open("xb") as handle:
        np.save(handle, np.asarray(instance, dtype=np.uint32), allow_pickle=False)
    mapping_path = states / f"{prefix}-mapping.json"
    _write_json_new(mapping_path, {str(key): str(item) for key, item in mapping.items()})
    mask_records: dict[str, dict[str, object]] = {}
    for name, mask in masks.items():
        if (
            not isinstance(mask, np.ndarray)
            or mask.shape != FRAME_SHAPE[:2]
            or mask.dtype != np.uint8
            or not set(np.unique(mask)).issubset({0, 255})
        ):
            raise ValueError("transport group mask is not a binary uint8 image")
        path = states / f"{prefix}-{name}.png"
        if not cv2.imwrite(os.fspath(path), mask):
            raise RuntimeError("transport group mask could not be written")
        mask_records[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
        }
    return {
        "state_index": state_index,
        "rgb_path": rgb_path.relative_to(root).as_posix(),
        "rgb_sha256": _sha256(rgb_path),
        "instance_path": instance_path.relative_to(root).as_posix(),
        "instance_sha256": _sha256(instance_path),
        "mapping_path": mapping_path.relative_to(root).as_posix(),
        "mapping_sha256": _sha256(mapping_path),
        "group_masks": mask_records,
        "attestation": _validated_attestation(value["attestation"], state_index),
    }


def _make_contact_sheet(root: Path, captures: Sequence[Mapping[str, Any]]) -> Path:
    thumbnails: list[np.ndarray] = []
    for row in captures:
        image = cv2.imread(os.fspath(root / str(row["rgb_path"])), cv2.IMREAD_COLOR)
        if image is None or image.shape != FRAME_SHAPE:
            raise RuntimeError("transport contact-sheet source is not decodable")
        thumb = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            f'state {int(row["state_index"]):02d}',
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    columns = min(5, len(thumbnails))
    rows = math.ceil(len(thumbnails) / columns)
    sheet = np.zeros((rows * 240, columns * 320, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        row, column = divmod(index, columns)
        sheet[row * 240 : (row + 1) * 240, column * 320 : (column + 1) * 320] = thumb
    path = root / "contact-sheet.png"
    if not cv2.imwrite(os.fspath(path), sheet):
        raise RuntimeError("transport contact sheet could not be written")
    return path


def _run_transport_stage(
    *,
    segment_dir: Path,
    output_dir: Path,
    state_indices: Sequence[int],
    runtime: TransportStageRuntime,
    check_output: Callable[..., Mapping[str, Any]],
    wp1_fill_readiness_path: Path,
    wp1_fill_readiness_sha256: str,
    pilot_visual_review_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze first, capture second, then revalidate after the independent checker."""

    segment = segment_dir.resolve()
    final = output_dir.resolve()
    indices = tuple(state_indices)
    if indices not in (ALL_STATE_INDICES, PILOT_STATE_INDICES):
        raise ValueError("transport render must be full states 0..12 or pilot states 0,6,12")
    if final.exists():
        raise FileExistsError("transport render output must not already exist")
    build, closure, evidence_paths = _load_build(segment)
    build_path = segment / "transport-build-receipt.json"
    frozen_build_receipt_sha256 = _sha256(build_path)
    wp1_fill_readiness = validate_wp1_fill_readiness(
        wp1_fill_readiness_path,
        declared_sha256=wp1_fill_readiness_sha256,
        owner_thread_id=str(build["owner_thread_id"]),
    )
    if indices == ALL_STATE_INDICES:
        if pilot_visual_review_path is None:
            raise ValueError("full transport requires an accepted pilot visual review")
        pilot_visual_review = validate_pilot_visual_review(
            pilot_visual_review_path,
            transport_implementation_sha256=str(
                build["transport_implementation"]["sha256"]
            ),
            owner_thread_id=str(build["owner_thread_id"]),
        )
    else:
        if pilot_visual_review_path is not None:
            raise ValueError("pilot transport cannot consume its own visual review")
        pilot_visual_review = None
    state_paths = [segment / row["path"] for row in build["layer_manifest"]["state_layers"]]

    # Required ordering: annotators are attached in every preflight before any output
    # directory is created, then the exact /Render partition is frozen.
    preflights: list[dict[str, Any]] = []
    frozen_partition: dict[str, Any] | None = None
    camera_light_sha256: str | None = None
    camera_light_before_annotators_sha256: str | None = None
    scene_session_sha256: str | None = None
    for index in indices:
        row = _validated_attestation(runtime.preflight_state(state_paths[index], index), index)
        if row["camera_light_before_annotators_sha256"] != row["camera_light_sha256"]:
            raise ValueError("annotator attachment changed camera/light effective state")
        if frozen_partition is None:
            frozen_partition = dict(row["runtime_partition"])
            camera_light_before_annotators_sha256 = str(
                row["camera_light_before_annotators_sha256"]
            )
            camera_light_sha256 = str(row["camera_light_sha256"])
            scene_session_sha256 = str(row["scene_session_sha256"])
        else:
            if row["runtime_partition"] != frozen_partition:
                raise ValueError("post-annotator runtime partition differs across states")
            if row["camera_light_sha256"] != camera_light_sha256:
                raise ValueError("camera/light effective state differs across transport states")
            if (
                row["camera_light_before_annotators_sha256"]
                != camera_light_before_annotators_sha256
            ):
                raise ValueError("pre-annotator camera/light state differs across states")
            if row["scene_session_sha256"] != scene_session_sha256:
                raise ValueError("nuisance session partition differs across transport states")
        preflights.append(row)
    assert frozen_partition is not None
    validate_evidence_closure(
        closure,
        source_files=evidence_paths["source_files"],
        snapshot_arrays=evidence_paths["snapshot_arrays"],
        state_layers=evidence_paths["state_layers"],
        dependency_files=evidence_paths["dependency_files"],
        counters=closure["counters"],
    )

    final.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=final.name + ".partial-", dir=final.parent))
    try:
        captures: list[dict[str, Any]] = []
        for index in indices:
            value = dict(runtime.capture_state(state_paths[index], index))
            attestation = _validated_attestation(value.get("attestation"), index)
            if (
                attestation["runtime_partition"] != frozen_partition
                or attestation["camera_light_before_annotators_sha256"]
                != camera_light_before_annotators_sha256
                or attestation["camera_light_sha256"] != camera_light_sha256
                or attestation["scene_session_sha256"] != scene_session_sha256
            ):
                raise ValueError("transport capture drifted from its pre-output frozen state")
            value["attestation"] = attestation
            captures.append(_save_capture(partial, index, value))
        rgb_hashes = [row["rgb_sha256"] for row in captures]
        if len(set(rgb_hashes)) != len(rgb_hashes):
            raise ValueError("rendered transport states are not byte-unique")
        contact_sheet = _make_contact_sheet(partial, captures)
        current = _validated_attestation(runtime.attest_current(), indices[-1])
        if (
            current["runtime_partition"] != frozen_partition
            or current["camera_light_sha256"] != camera_light_sha256
            or current["scene_session_sha256"] != scene_session_sha256
        ):
            raise ValueError("transport runtime drifted before its checker")
        render_mode_transitions = _validated_render_mode_transitions(
            runtime.finalize_for_publication(), expected_state_indices=indices
        )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "fluidmix_transport_stage_render_receipt_v1",
            "status": "pilot" if indices == PILOT_STATE_INDICES else "candidate",
            "mode": build["mode"],
            "owner_thread_id": build["owner_thread_id"],
            "segment_dir": os.fspath(segment),
            "build_receipt_sha256": frozen_build_receipt_sha256,
            "pre_output_evidence_closure_sha256": closure["sha256"],
            "post_checker_evidence_closure": copy.deepcopy(closure),
            "state_indices": list(indices),
            "source_time_code": 1,
            "physics_steps": 0,
            "physics_callbacks": 0,
            "runtime_partition": frozen_partition,
            "render_mode_transitions": render_mode_transitions,
            "camera_light_before_annotators_sha256": camera_light_before_annotators_sha256,
            "camera_light_sha256": camera_light_sha256,
            "scene_session_sha256": scene_session_sha256,
            "preflight_attestations": preflights,
            "final_runtime_attestation": current,
            "pilot_visual_review": pilot_visual_review,
            "wp1_fill_readiness": wp1_fill_readiness,
            "captures": captures,
            "contact_sheet_path": contact_sheet.relative_to(partial).as_posix(),
            "contact_sheet_sha256": _sha256(contact_sheet),
            "rendered_states_byte_unique": True,
            "excluded_from_model_inputs": True,
        }
        _write_json_new(partial / "transport-render-receipt.json", receipt)
        evaluated = dict(
            check_output(
                output_dir=partial,
                segment_dir=segment,
                expected_state_indices=indices,
                evaluate_only=True,
            )
        )
        receipt["checker_evaluation"] = evaluated
        (partial / "transport-render-receipt.json").unlink()
        _write_json_new(partial / "transport-render-receipt.json", receipt)
        before_checker = {
            path.relative_to(partial).as_posix(): _sha256(path)
            for path in sorted(partial.rglob("*"))
            if path.is_file()
        }
        checked = dict(
            check_output(
                output_dir=partial,
                segment_dir=segment,
                expected_state_indices=indices,
                evaluate_only=False,
            )
        )
        if checked != evaluated:
            raise ValueError("independent transport checker changed its evaluation")
        after_checker = {
            path.relative_to(partial).as_posix(): _sha256(path)
            for path in sorted(partial.rglob("*"))
            if path.is_file()
        }
        if after_checker != before_checker:
            raise ValueError("independent transport checker mutated candidate output")
        if _sha256(build_path) != frozen_build_receipt_sha256:
            raise ValueError("transport build receipt evidence drifted")
        validate_evidence_closure(
            closure,
            source_files=evidence_paths["source_files"],
            snapshot_arrays=evidence_paths["snapshot_arrays"],
            state_layers=evidence_paths["state_layers"],
            dependency_files=evidence_paths["dependency_files"],
            counters=closure["counters"],
        )
        if pilot_visual_review is not None:
            current_review = validate_pilot_visual_review(
                Path(str(pilot_visual_review["review_path"])),
                transport_implementation_sha256=str(
                    build["transport_implementation"]["sha256"]
                ),
                owner_thread_id=str(build["owner_thread_id"]),
            )
            if current_review != pilot_visual_review:
                raise ValueError("pilot visual review changed during full transport")
        current_readiness = validate_wp1_fill_readiness(
            Path(str(wp1_fill_readiness["receipt_path"])),
            declared_sha256=str(wp1_fill_readiness["receipt_sha256"]),
            owner_thread_id=str(build["owner_thread_id"]),
        )
        if current_readiness != wp1_fill_readiness:
            raise ValueError("WP1 fill readiness changed during transport")
        final_attestation = _validated_attestation(runtime.attest_current(), indices[-1])
        if final_attestation != current:
            raise ValueError("camera/light, partition, or counters changed during checker")
        partial.rename(final)
        return receipt
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def run_transport_stage(
    *,
    segment_dir: Path,
    output_dir: Path,
    state_indices: Sequence[int],
    runtime: TransportStageRuntime,
    check_output: Callable[..., Mapping[str, Any]],
    wp1_fill_readiness_path: Path,
    wp1_fill_readiness_sha256: str,
    pilot_visual_review_path: Path | None = None,
) -> dict[str, Any]:
    """Run one render attempt and close its runtime on every return path."""

    try:
        return _run_transport_stage(
            segment_dir=segment_dir,
            output_dir=output_dir,
            state_indices=state_indices,
            runtime=runtime,
            check_output=check_output,
            wp1_fill_readiness_path=wp1_fill_readiness_path,
            wp1_fill_readiness_sha256=wp1_fill_readiness_sha256,
            pilot_visual_review_path=pilot_visual_review_path,
        )
    finally:
        runtime.close()


class IsaacTransportStageRuntime:
    """Isaac adapter that composes one immutable state layer per same-state capture."""

    def __init__(self, *, app: Any, segment_dir: Path, accepted_case_output: Path):
        import carb
        import omni.replicator.core as rep
        import omni.physx
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Sdf, Usd, UsdGeom

        from fluidmix.nuisance_authoring import apply_nuisance_session_layer
        import physx_three_pour_sequence as f2

        self.app = app
        self.segment_dir = segment_dir
        self.rep = rep
        self.SimulationManager = SimulationManager
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.Sdf = Sdf
        self.context = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()
        self.physx = omni.physx.get_physx_interface()
        self.settings = carb.settings.get_settings()
        self.f2 = f2
        self.apply_nuisance = apply_nuisance_session_layer
        self._mdl_registration: tuple[Any, Any] | None = None
        self._render_mode_transitions: list[dict[str, Any]] = []
        self.rgb_annotator = None
        self.instance_annotator = None
        self.render_product = None
        self.stage = None
        self.session = None
        self._callback_count = 0
        self._current_state_index: int | None = None
        self._current_attestation: dict[str, Any] | None = None
        self._current_nuisance: dict[str, Any] | None = None
        self._current_tone_readback: dict[str, Any] | None = None
        self._subscription = None
        self._mdl_registration = _register_reviewed_mdl_search_path(
            app=app,
            mdl_file=f2.SPATIAL_OMNISURFACE_MDL,
        )
        try:
            source_stage = Path(
                _strict_object(
                    segment_dir / "transport-build-receipt.json", "build receipt"
                )["layer_manifest"]["source_layers"]["fill_station.usda"]["path"]
            )
            self.loaded = dict(
                load_rigid_transport_inputs(accepted_case_output, source_stage)
            )
            self.receipt = self.loaded["receipt"]
            self._subscription = self.physx.subscribe_physics_step_events(
                self._on_physics_step
            )
        except BaseException:
            self._remove_mdl_registration()
            raise

    def _on_physics_step(self, _dt: float) -> None:
        self._callback_count += 1

    def _close_product(self) -> None:
        if self.rgb_annotator is not None and self.render_product is not None:
            self.rgb_annotator.detach([self.render_product])
        if self.instance_annotator is not None and self.render_product is not None:
            self.instance_annotator.detach([self.render_product])
        self.rgb_annotator = None
        self.instance_annotator = None
        self.render_product = None

    def _session_inventory(self) -> tuple[list[str], list[str]]:
        prims: list[str] = []
        properties: list[str] = []

        def visit(spec: Any) -> None:
            prims.append(str(spec.path))
            properties.extend(str(item.path) for item in spec.properties)
            for child in spec.nameChildren:
                visit(child)

        for root in self.session.rootPrims:
            visit(root)
        return sorted(prims), sorted(properties)

    def _camera_light_hash(self) -> str:
        rows: list[dict[str, Any]] = []
        for path in CAMERA_AND_LIGHT_PATHS:
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"reviewed camera/light prim is missing: {path}")
            matrix = self.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                self.Usd.TimeCode(1)
            )
            attrs: dict[str, Any] = {}
            for attr in prim.GetAttributes():
                value = attr.Get(self.Usd.TimeCode(1))
                if value is not None:
                    attrs[str(attr.GetName())] = str(value)
            rows.append(
                {
                    "path": path,
                    "type": str(prim.GetTypeName()),
                    "world_matrix": [
                        [float(matrix[row][column]) for column in range(4)]
                        for row in range(4)
                    ],
                    "attributes": attrs,
                }
            )
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _open_fresh_session_stage(self, state_layer: Path) -> tuple[Any, Any]:
        session = self.Sdf.Layer.CreateAnonymous(
            "fluidmix-transport-runtime-session.usda"
        )
        if (
            session is None
            or session.anonymous is not True
            or list(session.rootPrims)
        ):
            raise RuntimeError("transport could not create a fresh anonymous session layer")

        result: list[tuple[bool, str]] = []

        def finished(success: bool, error_message: str) -> None:
            result.append((bool(success), str(error_message)))

        started = self.context.open_stage_with_session_layer(
            os.fspath(state_layer), str(session.identifier), finished
        )
        if not started:
            raise RuntimeError(f"transport state layer could not be opened: {state_layer}")
        while not result:
            self.app.update()
        if not result[-1][0]:
            message = result[-1][1] if result else "unknown error"
            raise RuntimeError(
                f"transport state layer could not be opened: {state_layer}: {message}"
            )
        while self.context.get_stage_loading_status()[2] > 0:
            self.app.update()
        stage = self.context.get_stage()
        if stage is None:
            raise RuntimeError("transport state did not open a USD stage")
        observed_session = stage.GetSessionLayer()
        if (
            observed_session is None
            or observed_session.anonymous is not True
            or str(observed_session.identifier) != str(session.identifier)
        ):
            raise RuntimeError(
                "transport state did not open with the requested fresh session layer"
            )
        return stage, observed_session

    def _open(
        self, state_layer: Path, state_index: int, *, opening_kind: str = "preflight"
    ) -> dict[str, Any]:
        self._close_product()
        self.stage, self.session = self._open_fresh_session_stage(state_layer)
        self.timeline.stop()
        tcps = float(self.stage.GetTimeCodesPerSecond())
        self.timeline.set_current_time(1.0 / tcps)
        arguments = self.receipt.get("arguments")
        binding = self.receipt.get("nuisance_profile_binding")
        nuisance = None
        if isinstance(binding, Mapping):
            nuisance = self.apply_nuisance(
                self.stage,
                self.Usd,
                binding,
                case_id=str(arguments["nuisance_case_id"]),
                seed=int(arguments["seed"]),
            )
            if nuisance != self.receipt.get("nuisance_runtime"):
                raise RuntimeError("transport nuisance state differs from the accepted fill")
        _transition_rtx_render_mode(
            settings=self.settings,
            app=self.app,
            records=self._render_mode_transitions,
            state_index=state_index,
            opening_kind=opening_kind,
        )
        camera_prim = self.stage.GetPrimAtPath(CAMERA_PATH)
        tone = self.f2._pin_tone_exposure_state(self.settings, camera_prim, self.Sdf)
        accepted_tone = (
            self.receipt.get("render", {}).get("tone_exposure_state")
            if isinstance(self.receipt.get("render"), Mapping)
            else None
        )
        accepted_tone_readback = (
            accepted_tone.get("readback") if isinstance(accepted_tone, Mapping) else None
        )
        tone_readback = tone.get("readback") if isinstance(tone, Mapping) else None
        if (
            tone.get("passed") is not True
            or not isinstance(tone_readback, Mapping)
            or tone_readback != accepted_tone_readback
        ):
            raise RuntimeError("transport tone/exposure differs from the accepted fill")
        camera_light_before_annotators_sha256 = self._camera_light_hash()
        self.render_product = self.rep.create.render_product(
            CAMERA_PATH, (640, 480), name=RUNTIME_PRODUCT_NAME
        )
        self.rgb_annotator = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.instance_annotator = self.rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        self.rgb_annotator.attach([self.render_product])
        self.instance_annotator.attach([self.render_product])
        camera_light_sha256 = self._camera_light_hash()
        if camera_light_sha256 != camera_light_before_annotators_sha256:
            raise RuntimeError("annotator attachment changed camera/light effective state")
        prims, properties = self._session_inventory()
        render_prims = [path for path in prims if path == "/Render" or path.startswith("/Render/")]
        render_properties = [
            path
            for path in properties
            if path.startswith("/Render/") or path.startswith("/Render.")
        ]
        runtime_partition = freeze_runtime_partition(render_prims, render_properties)
        scene_hash = str(_attest_scene_session_specs(self.session.rootPrims)["sha256"])
        source_time = int(self.timeline.time_to_time_code(self.timeline.get_current_time()))
        attestation = {
            "state_index": state_index,
            "source_time_code": source_time,
            "physics_steps": int(self.SimulationManager.get_num_physics_steps()),
            "physics_callbacks": self._callback_count,
            "timeline_playing": bool(self.timeline.is_playing()),
            "runtime_partition": runtime_partition,
            "camera_light_before_annotators_sha256": camera_light_before_annotators_sha256,
            "camera_light_sha256": camera_light_sha256,
            "scene_session_sha256": scene_hash,
        }
        self._current_state_index = state_index
        self._current_nuisance = copy.deepcopy(nuisance)
        self._current_tone_readback = copy.deepcopy(dict(tone_readback))
        self._current_attestation = attestation
        return copy.deepcopy(attestation)

    def preflight_state(self, state_layer: Path, state_index: int) -> Mapping[str, Any]:
        return self._open(state_layer, state_index, opening_kind="preflight")

    def _group_mask(
        self, data: np.ndarray, records: Sequence[Any], prefixes: Sequence[str]
    ) -> np.ndarray:
        ids = [
            record.instance_id
            for record in records
            if any(record.path == path or record.path.startswith(path + "/") for path in prefixes)
        ]
        return (np.isin(data, ids).astype(np.uint8) * 255)

    def capture_state(self, state_layer: Path, state_index: int) -> Mapping[str, Any]:
        before = self._open(state_layer, state_index, opening_kind="capture")
        guard = self.f2._replicator_same_state_step_checked_in_session(
            stage=self.stage,
            session_layer=self.session,
            Usd=self.Usd,
            rep=self.rep,
            rt_subframes=32,
            callback_count=lambda: self._callback_count,
            timeline=self.timeline,
            physics_state=lambda: {
                "num_physics_steps": int(self.SimulationManager.get_num_physics_steps()),
                "simulation_time_s": float(self.SimulationManager.get_simulation_time()),
            },
        )
        if guard["physics_steps_before"] != guard["physics_steps_after"] or guard[
            "callbacks_before"
        ] != guard["callbacks_after"]:
            raise RuntimeError("transport same-state capture advanced physics")
        rgb = np.asarray(self.rgb_annotator.get_data())
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        instance_raw = self.instance_annotator.get_data()
        if not isinstance(instance_raw, Mapping) or not isinstance(
            instance_raw.get("info"), Mapping
        ):
            raise RuntimeError("transport instance annotator returned no mapping")
        data = np.asarray(instance_raw.get("data"))
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[..., 0]
        records = resolve_instance_paths(instance_raw["info"]["idToLabels"])
        mapping = {str(row.instance_id): row.path for row in records}
        prefixes = {
            "outgoing": tuple(sorted(OUTGOING_PRIM_PATHS)),
            "incoming": tuple(sorted(INCOMING_PRIM_PATHS)),
            "queued": tuple(sorted(QUEUED_PRIM_PATHS)),
            "incoming_labels": (
                "/FillStation/FillStation_LocalRoot/FluidmixTransport/IncomingLabels",
            ),
            "queue_tail": (
                "/FillStation/FillStation_LocalRoot/FluidmixTransport/QueueTail",
            ),
        }
        after = self.attest_current()
        if after != before:
            raise RuntimeError("transport runtime state changed during capture")
        return {
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "instance_ids": np.asarray(data, dtype=np.uint32),
            "instance_id_to_path": mapping,
            "group_masks": {
                name: self._group_mask(data, records, paths)
                for name, paths in prefixes.items()
            },
            "attestation": after,
        }

    def attest_current(self) -> Mapping[str, Any]:
        if self._current_state_index is None:
            raise RuntimeError("transport runtime has no current state")
        prims, properties = self._session_inventory()
        render_prims = [path for path in prims if path == "/Render" or path.startswith("/Render/")]
        render_properties = [
            path
            for path in properties
            if path.startswith("/Render/") or path.startswith("/Render.")
        ]
        current = copy.deepcopy(self._current_attestation)
        current["runtime_partition"] = freeze_runtime_partition(render_prims, render_properties)
        current["scene_session_sha256"] = str(
            _attest_scene_session_specs(self.session.rootPrims)["sha256"]
        )
        current["camera_light_sha256"] = self._camera_light_hash()
        current["source_time_code"] = int(
            self.timeline.time_to_time_code(self.timeline.get_current_time())
        )
        current["physics_steps"] = int(self.SimulationManager.get_num_physics_steps())
        current["physics_callbacks"] = self._callback_count
        current["timeline_playing"] = bool(self.timeline.is_playing())
        return current

    def close(self) -> None:
        self._close_product()
        self._subscription = None
        self._remove_mdl_registration()

    def finalize_for_publication(self) -> Sequence[Mapping[str, Any]]:
        """Release process-global resources before the candidate can be published."""

        self.close()
        return copy.deepcopy(self._render_mode_transitions)

    def _remove_mdl_registration(self) -> None:
        if self._mdl_registration is None:
            return
        registration = self._mdl_registration
        _remove_reviewed_mdl_search_path(*registration)
        self._mdl_registration = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-visual-review", type=Path)
    parser.add_argument("--wp1-fill-readiness", type=Path, required=True)
    parser.add_argument("--wp1-fill-readiness-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_path = _launcher_trace_path(args.output_dir)
    try:
        segment = args.segment_dir.resolve()
        if args.output_dir.resolve().exists():
            raise FileExistsError("transport render output must not already exist")
        if trace_path.exists() or trace_path.is_symlink():
            raise FileExistsError("transport launcher trace must not already exist")
        build, _closure, _paths = _load_build(segment)
        validate_wp1_fill_readiness(
            args.wp1_fill_readiness,
            declared_sha256=args.wp1_fill_readiness_sha256,
            owner_thread_id=str(build["owner_thread_id"]),
        )
        if args.pilot:
            if args.pilot_visual_review is not None:
                raise ValueError("pilot transport cannot consume its own visual review")
        else:
            if args.pilot_visual_review is None:
                raise ValueError("full transport requires an accepted pilot visual review")
            validate_pilot_visual_review(
                args.pilot_visual_review,
                transport_implementation_sha256=str(
                    build["transport_implementation"]["sha256"]
                ),
                owner_thread_id=str(build["owner_thread_id"]),
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"transport render failed before SimulationApp: {error}", file=sys.stderr)
        return 2

    try:
        trace = _LauncherTrace(
            trace_path, owner_thread_id=str(build["owner_thread_id"])
        )
    except OSError as error:
        print(f"transport render failed before SimulationApp: {error}", file=sys.stderr)
        return 2
    trace.append("preflight_passed")

    owned_app = None
    runtime = None
    runtime_delegated = False
    receipt = None
    exit_code = 2
    try:
        trace.phase = "kit_app_acquisition"
        app, owned_app = _start_or_reuse_kit(record_event=trace.append)
        trace.append("runtime_construction_started")
        build = _strict_object(
            args.segment_dir / "transport-build-receipt.json", "transport build receipt"
        )
        runtime = IsaacTransportStageRuntime(
            app=app,
            segment_dir=args.segment_dir.resolve(),
            accepted_case_output=Path(build["case_output"]),
        )
        trace.append("runtime_construction_returned")
        from check_transport_stage import check_transport_output

        trace.append("transport_run_started")
        runtime_delegated = True
        receipt = run_transport_stage(
            segment_dir=args.segment_dir,
            output_dir=args.output_dir,
            state_indices=PILOT_STATE_INDICES if args.pilot else ALL_STATE_INDICES,
            runtime=runtime,
            check_output=check_transport_output,
            wp1_fill_readiness_path=args.wp1_fill_readiness,
            wp1_fill_readiness_sha256=args.wp1_fill_readiness_sha256,
            pilot_visual_review_path=args.pilot_visual_review,
        )
        published_receipt = _strict_object(
            args.output_dir.resolve() / "transport-render-receipt.json",
            "published transport render receipt",
        )
        normalized_receipt = json.loads(
            json.dumps(receipt, allow_nan=False, sort_keys=True)
        )
        checker_evaluation = published_receipt.get("checker_evaluation")
        if published_receipt != normalized_receipt:
            raise RuntimeError("published transport receipt differs from launcher result")
        if (
            not isinstance(checker_evaluation, Mapping)
            or checker_evaluation.get("status") != "pass"
        ):
            raise RuntimeError("published transport receipt lacks a passing checker")
        trace.append("transport_run_returned")
        exit_code = 0
    except SystemExit as error:
        trace.append(
            "launcher_failed",
            details={
                "exception_type": "SystemExit",
                "exit_code": error.code,
                "phase": trace.phase,
            },
        )
        print(
            f"transport render failed during {trace.phase}: SystemExit: {error.code}",
            file=sys.stderr,
        )
        exit_code = 2
    except Exception as error:
        if isinstance(error, _KitStartupFailure):
            owned_app = error.owned_app
        trace.append(
            "launcher_failed",
            details={
                "exception_type": type(error).__name__,
                "message": str(error),
                "phase": trace.phase,
            },
        )
        print(f"transport render failed: {error}", file=sys.stderr)
        exit_code = 2
    finally:
        if runtime is not None and not runtime_delegated:
            try:
                runtime.close()
            except Exception as error:
                trace.append(
                    "launcher_failed",
                    details={
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "phase": "runtime_cleanup",
                    },
                )
                exit_code = 2
        if owned_app is not None:
            trace.append("close_started", details={"exit_code": exit_code})
            try:
                owned_app.close(exit_code=exit_code)
            except SystemExit as error:
                if error.code is not None and error.code != exit_code:
                    trace.append(
                        "launcher_failed",
                        details={
                            "exception_type": "SystemExit",
                            "exit_code": error.code,
                            "phase": "close",
                        },
                    )
                    exit_code = 2
                else:
                    trace.append(
                        "close_returned", details={"system_exit_code": error.code}
                    )
            except Exception as error:
                trace.append(
                    "launcher_failed",
                    details={
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "phase": "close",
                    },
                )
                exit_code = 2
            else:
                trace.append("close_returned")
        trace.close()
    if receipt is None or exit_code != 0:
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
