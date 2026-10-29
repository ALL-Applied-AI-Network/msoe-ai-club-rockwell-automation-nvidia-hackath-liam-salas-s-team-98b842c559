#!/usr/bin/env python3
"""Build one source-bound, 13-state transport overlay without rendering."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.rigid_vial_transport import (  # noqa: E402
    COUNT_BODY_MESH_PATH,
    COUNT_BODY_PATH,
    GLOBAL_MATERIAL_PATH,
    MAX_TRUSTED_JSON_BYTES,
    SOURCE_LAYER_BYTES,
    SOURCE_LAYER_HASHES,
    SPATIAL_MATERIAL_PATH,
    load_rigid_transport_inputs,
    strict_json_loads,
)
from fluidmix.transport_stage import (  # noqa: E402
    CAMERA_AND_LIGHT_PATHS,
    SOURCE_CURVE_APPROVED_SHA256,
    SOURCE_CURVE_CANONICAL_13_SHA256,
    SOURCE_CURVE_CANONICAL_37_SHA256,
    SOURCE_CURVE_DELTA_37_SHA256,
    SOURCE_CURVE_SAMPLES,
    TRANSPORT_IMPLEMENTATION_PATHS,
    build_transport_layers,
    capture_transport_implementation,
    capture_evidence_closure,
    capture_file_records,
    validate_owner_thread_id,
)


SOURCE_VIAL_PATH = "/FillStation/FillStation_LocalRoot/Fill_vial_neighbour_2"
SNAPSHOT_ARRAY_NAMES = (
    "birth_compositions",
    "birth_event_ids",
    "birth_steps",
    "current_compositions",
    "ids",
    "orientations",
    "positions",
    "pour_event_ids",
    "proto_indices",
    "scales",
    "scheduled_sources",
    "simulation_points",
    "velocities",
)
COUNT_BODY_DERIVED_COMPARATOR_VERSION = (
    "count-body-derived-elementwise-rtol-1e-12-atol-1e-12-v1"
)
COUNT_BODY_DERIVED_RELATIVE_TOLERANCE = 1.0e-12
COUNT_BODY_DERIVED_ABSOLUTE_TOLERANCE = 1.0e-12
COUNT_BODY_SOURCE_DERIVED_FIELDS = (
    "source_world_bottom_z_m",
    "source_world_height_m",
    "source_signed_volume_m3",
    "source_effective_area_m2",
    "xy_area_correction_scale",
)
COUNT_BODY_SOURCE_EXACT_FIELDS = (
    "resolved_points_time_code",
    "resolved_normals_time_code",
    "resolved_extent_time_code",
    "source_point_count",
    "source_face_count",
    "source_corner_count",
    "source_fan_triangle_count",
    "source_geometry_sha256",
    "source_topology_sha256",
    "copied_primvar_names",
    "initial_visibility",
    "body_color_mode",
)
COUNT_BODY_INITIAL_POINTS_HASH_FIELD = "initial_points_sha256"
COUNT_BODY_CAPTURE_DERIVED_FIELDS = (
    "cross_section_area_m2",
    "post_transform_signed_volume_m3",
    "post_transform_volume_residual_m3",
    "target_height_m",
    "target_volume_m3",
    "world_bottom_z_m",
    "world_extent_m",
    "world_height_m",
)
COUNT_BODY_CAPTURE_EXACT_FIELDS = (
    "body_material_color",
    "body_material_color_sha256",
    "body_visibility",
    "contained_particle_count",
    "extent_sha256",
    "particle_volume_m3",
    "phase",
    "physics_step",
    "points_sha256",
    "representation",
    "volume_proof_coordinate_path",
    "world_points_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _is_lower_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_derived_array(value: object) -> np.ndarray | None:
    def valid_tree(item: object) -> bool:
        if isinstance(item, bool):
            return False
        if isinstance(item, Real):
            return True
        return bool(
            isinstance(item, (list, tuple))
            and item
            and all(valid_tree(child) for child in item)
        )

    if not valid_tree(value):
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.dtype != np.float64 or not np.all(np.isfinite(array)):
        return None
    return array


def _compare_finite_derived_value(
    expected: object, observed: object
) -> dict[str, object]:
    expected_array = _finite_derived_array(expected)
    observed_array = _finite_derived_array(observed)
    result: dict[str, object] = {
        "expected": expected,
        "observed": observed,
        "shape": None,
        "elementwise_absolute_errors": None,
        "elementwise_tolerance_limits": None,
        "elementwise_within_tolerance": None,
        "maximum_absolute_error": None,
        "passed": False,
        "failure_reasons": [],
    }
    failures: list[str] = []
    if expected_array is None:
        failures.append("expected_nonfinite_or_type_invalid")
    if observed_array is None:
        failures.append("observed_nonfinite_or_type_invalid")
    if expected_array is None or observed_array is None:
        result["failure_reasons"] = failures
        return result
    if expected_array.shape != observed_array.shape:
        result["failure_reasons"] = ["shape_mismatch"]
        return result
    errors = np.abs(expected_array - observed_array)
    limits = np.maximum(
        COUNT_BODY_DERIVED_ABSOLUTE_TOLERANCE,
        COUNT_BODY_DERIVED_RELATIVE_TOLERANCE
        * np.maximum(np.abs(expected_array), np.abs(observed_array)),
    )
    within = errors <= limits
    if not bool(np.all(within)):
        failures.append("element_outside_tolerance")
    result.update(
        {
            "shape": list(expected_array.shape),
            "elementwise_absolute_errors": errors.tolist(),
            "elementwise_tolerance_limits": limits.tolist(),
            "elementwise_within_tolerance": within.tolist(),
            "maximum_absolute_error": float(np.max(errors)),
            "passed": not failures,
            "failure_reasons": failures,
        }
    )
    return result


def _exact_json_equal(expected: object, observed: object) -> bool:
    try:
        return json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) == json.dumps(
            observed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _attest_count_body_source_evidence(
    accepted: object, observed: object
) -> dict[str, object]:
    """Allow only bounded platform drift in source-derived numeric evidence."""

    required_fields = (
        *COUNT_BODY_SOURCE_EXACT_FIELDS,
        *COUNT_BODY_SOURCE_DERIVED_FIELDS,
        COUNT_BODY_INITIAL_POINTS_HASH_FIELD,
    )
    failures: list[str] = []
    if not isinstance(accepted, Mapping):
        accepted_row: Mapping[str, object] = {}
        failures.append("accepted_source_evidence_invalid")
    else:
        accepted_row = accepted
    if not isinstance(observed, Mapping):
        observed_row: Mapping[str, object] = {}
        failures.append("observed_source_evidence_invalid")
    else:
        observed_row = observed
    missing_accepted = [field for field in required_fields if field not in accepted_row]
    missing_observed = [field for field in required_fields if field not in observed_row]
    unexpected_observed = sorted(set(observed_row) - set(required_fields))
    if missing_accepted:
        failures.append("accepted_source_evidence_fields_missing")
    if missing_observed or unexpected_observed:
        failures.append("observed_source_evidence_shape_mismatch")

    exact_fields: dict[str, object] = {}
    for field in COUNT_BODY_SOURCE_EXACT_FIELDS:
        expected_value = accepted_row.get(field)
        observed_value = observed_row.get(field)
        passed = _exact_json_equal(expected_value, observed_value)
        exact_fields[field] = {
            "expected": expected_value,
            "observed": observed_value,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{field}_exact_mismatch")

    derived_fields: dict[str, object] = {}
    for field in COUNT_BODY_SOURCE_DERIVED_FIELDS:
        comparison = _compare_finite_derived_value(
            accepted_row.get(field), observed_row.get(field)
        )
        derived_fields[field] = comparison
        if comparison["passed"] is not True:
            failures.append(f"{field}_derived_mismatch")

    expected_points_hash = accepted_row.get(COUNT_BODY_INITIAL_POINTS_HASH_FIELD)
    observed_points_hash = observed_row.get(COUNT_BODY_INITIAL_POINTS_HASH_FIELD)
    valid_points_hashes = _is_lower_sha256(expected_points_hash) and _is_lower_sha256(
        observed_points_hash
    )
    if not valid_points_hashes:
        failures.append("initial_points_hash_invalid")
    failures = list(dict.fromkeys(failures))
    return {
        "attestation_version": COUNT_BODY_DERIVED_COMPARATOR_VERSION,
        "relative_tolerance": COUNT_BODY_DERIVED_RELATIVE_TOLERANCE,
        "absolute_tolerance": COUNT_BODY_DERIVED_ABSOLUTE_TOLERANCE,
        "required_source_fields": list(required_fields),
        "missing_accepted_fields": missing_accepted,
        "missing_observed_fields": missing_observed,
        "unexpected_observed_fields": unexpected_observed,
        "exact_fields": exact_fields,
        "derived_fields": derived_fields,
        "accepted_initial_points_sha256": expected_points_hash,
        "recomputed_initial_points_sha256": observed_points_hash,
        "initial_points_sha256_exact_match": (
            expected_points_hash == observed_points_hash
        ),
        "passed": not failures,
        "failure_reasons": failures,
    }


def _attest_count_body_capture_geometry(
    accepted: object, observed: object
) -> dict[str, object]:
    """Attest one reconstructed capture while keeping its geometry hashes exact."""

    required_fields = (
        *COUNT_BODY_CAPTURE_EXACT_FIELDS,
        *COUNT_BODY_CAPTURE_DERIVED_FIELDS,
    )
    failures: list[str] = []
    if not isinstance(accepted, Mapping):
        accepted_row: Mapping[str, object] = {}
        failures.append("accepted_capture_geometry_invalid")
    else:
        accepted_row = accepted
    if not isinstance(observed, Mapping):
        observed_row: Mapping[str, object] = {}
        failures.append("observed_capture_geometry_invalid")
    else:
        observed_row = observed
    missing_accepted = [field for field in required_fields if field not in accepted_row]
    missing_observed = [field for field in required_fields if field not in observed_row]
    unexpected_accepted = sorted(set(accepted_row) - set(required_fields))
    unexpected_observed = sorted(set(observed_row) - set(required_fields))
    if missing_accepted or unexpected_accepted:
        failures.append("accepted_capture_geometry_shape_mismatch")
    if missing_observed or unexpected_observed:
        failures.append("observed_capture_geometry_shape_mismatch")

    exact_fields: dict[str, object] = {}
    for field in COUNT_BODY_CAPTURE_EXACT_FIELDS:
        expected_value = accepted_row.get(field)
        observed_value = observed_row.get(field)
        passed = _exact_json_equal(expected_value, observed_value)
        exact_fields[field] = {
            "expected": expected_value,
            "observed": observed_value,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{field}_exact_mismatch")

    derived_fields: dict[str, object] = {}
    for field in COUNT_BODY_CAPTURE_DERIVED_FIELDS:
        comparison = _compare_finite_derived_value(
            accepted_row.get(field), observed_row.get(field)
        )
        derived_fields[field] = comparison
        if comparison["passed"] is not True:
            failures.append(f"{field}_derived_mismatch")
    failures = list(dict.fromkeys(failures))
    return {
        "attestation_version": COUNT_BODY_DERIVED_COMPARATOR_VERSION,
        "relative_tolerance": COUNT_BODY_DERIVED_RELATIVE_TOLERANCE,
        "absolute_tolerance": COUNT_BODY_DERIVED_ABSOLUTE_TOLERANCE,
        "required_capture_fields": list(required_fields),
        "missing_accepted_fields": missing_accepted,
        "missing_observed_fields": missing_observed,
        "unexpected_accepted_fields": unexpected_accepted,
        "unexpected_observed_fields": unexpected_observed,
        "exact_fields": exact_fields,
        "derived_fields": derived_fields,
        "passed": not failures,
        "failure_reasons": failures,
    }


def _snapshot_array_paths(loaded: Mapping[str, Any]) -> list[Path]:
    snapshot_path = Path(loaded["snapshot_path"]).resolve()
    case_root = Path(loaded["case_output"]).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    arrays = snapshot.get("arrays") if isinstance(snapshot, dict) else None
    if not isinstance(arrays, dict) or tuple(sorted(arrays)) != SNAPSHOT_ARRAY_NAMES:
        raise ValueError("final snapshot does not bind the exact 13-array closure")
    result: list[Path] = []
    for name in SNAPSHOT_ARRAY_NAMES:
        row = arrays[name]
        relative = row.get("path") if isinstance(row, dict) else None
        expected = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"snapshot array {name} path is invalid")
        path = (case_root / relative).resolve()
        try:
            path.relative_to(case_root)
        except ValueError as error:
            raise ValueError(f"snapshot array {name} escapes the accepted case") from error
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"snapshot array {name} differs from its accepted hash")
        result.append(path)
    return result


def _extract_and_validate_source_curve(stage_path: Path) -> dict[str, object]:
    try:
        from pxr import Usd
    except ImportError as error:
        raise RuntimeError("USD Python is required to inspect the source curve") from error
    stage = Usd.Stage.Open(os.fspath(stage_path))
    if stage is None:
        raise RuntimeError("reviewed source stage could not be opened")
    prim = stage.GetPrimAtPath(SOURCE_VIAL_PATH)
    attribute = prim.GetAttribute("xformOp:translate") if prim.IsValid() else None
    times = attribute.GetTimeSamples() if attribute else []
    motion_times = [int(value) for value in times if 180 <= value <= 216]
    if motion_times != list(range(180, 217)):
        raise RuntimeError("source conveyor curve no longer has exact t=180..216 keys")
    full_rows: list[dict[str, object]] = []
    start_y = float(attribute.Get(180)[1])
    end_y = float(attribute.Get(216)[1])
    if not math.isclose(start_y - end_y, 4.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("source conveyor curve no longer moves exactly -4 local Y")
    for time_code in motion_times:
        value = attribute.Get(time_code)
        y = float(value[1])
        full_rows.append(
            {
                "time_code": time_code,
                "local_xyz": [float(value[0]), y, float(value[2])],
                "progress": (start_y - y) / (start_y - end_y),
            }
        )
    observed = [full_rows[index] for index in range(0, 37, 3)]
    expected = [asdict(row) for row in SOURCE_CURVE_SAMPLES]
    for actual, wanted in zip(observed, expected, strict=True):
        if (
            actual["time_code"] != wanted["time_code"]
            or actual["local_xyz"][1] != wanted["local_y"]
            or actual["progress"] != wanted["progress"]
        ):
            raise RuntimeError("source conveyor 10 fps sample table drifted")
    canonical = json.dumps(
        full_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    canonical_37_sha256 = hashlib.sha256(canonical).hexdigest()
    canonical_13_sha256 = hashlib.sha256(
        json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if canonical_37_sha256 != SOURCE_CURVE_CANONICAL_37_SHA256:
        raise RuntimeError("canonical 37-sample source conveyor curve drifted")
    if canonical_13_sha256 != SOURCE_CURVE_CANONICAL_13_SHA256:
        raise RuntimeError("canonical 13-state source conveyor curve drifted")
    delta_rows = [
        {
            "source_time_code": int(row["time_code"]),
            "local_y_delta": float(row["local_xyz"][1]) - start_y,
        }
        for row in full_rows
    ]
    delta_37_sha256 = hashlib.sha256(
        json.dumps(
            delta_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    delta_13_sha256 = hashlib.sha256(
        json.dumps(
            delta_rows[::3], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if delta_37_sha256 != SOURCE_CURVE_DELTA_37_SHA256:
        raise RuntimeError("canonical 37-key delta source conveyor curve drifted")
    if delta_13_sha256 != SOURCE_CURVE_APPROVED_SHA256:
        raise RuntimeError("approved 13-state delta source conveyor curve drifted")
    return {
        "source_prim_path": SOURCE_VIAL_PATH,
        "approved_extraction_sha256": SOURCE_CURVE_APPROVED_SHA256,
        "canonical_37_sample_sha256": canonical_37_sha256,
        "canonical_13_sample_sha256": canonical_13_sha256,
        "canonical_delta_37_sha256": delta_37_sha256,
        "approved_delta_13_sha256": delta_13_sha256,
        "full_30fps_samples": full_rows,
        "used_10fps_samples": observed,
    }


def _accepted_liquid_contract(loaded: Mapping[str, Any]) -> dict[str, Any]:
    receipt = loaded["receipt"]
    final_step = loaded["final_physics_step"]
    count_body = receipt.get("count_driven_liquid_body")
    capture_rows = count_body.get("capture_geometry") if isinstance(count_body, Mapping) else None
    accepted_geometry = [
        row
        for row in capture_rows or []
        if isinstance(row, Mapping) and row.get("physics_step") == final_step
    ]
    band_rows = count_body.get("stage_band_evidence") if isinstance(count_body, Mapping) else None
    accepted_bands = [
        row
        for row in band_rows or []
        if isinstance(row, Mapping) and row.get("physics_step") == final_step
    ]
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
    accepted_pair = [
        row
        for row in pairs or []
        if isinstance(row, Mapping) and row.get("physics_step") == final_step
    ]
    plan = receipt.get("plan")
    palette = plan.get("run_palette") if isinstance(plan, Mapping) else None
    transmissions = palette.get("transmission_1cm") if isinstance(palette, Mapping) else None
    treatment = plan.get("render_treatment") if isinstance(plan, Mapping) else None
    if (
        not isinstance(count_body, Mapping)
        or len(accepted_geometry) != 1
        or len(accepted_bands) != 1
        or len(accepted_pair) != 1
        or not isinstance(transmissions, Mapping)
        or not isinstance(treatment, Mapping)
        or type(treatment.get("particle_volume_m3")) is not float
    ):
        raise ValueError("accepted count-body reconstruction contract is incomplete")
    material_state = accepted_pair[0].get("body_material_render_state")
    if not isinstance(material_state, Mapping):
        raise ValueError("accepted count-body material state is missing")
    return {
        "count_body": count_body,
        "capture_geometry": accepted_geometry[0],
        "band_evidence": accepted_bands[0],
        "material_state": material_state,
        "transmissions": transmissions,
        "particle_volume_m3": treatment["particle_volume_m3"],
    }


def _restore_accepted_band_display_colors(
    *,
    runtime: Mapping[str, Any],
    live_geometry: Mapping[str, Any],
    band_evidence: Mapping[str, Any],
    expected_display_color_sha256: object,
    f2: Any,
    Usd: Any,
    UsdGeom: Any,
    Gf: Any,
    Vt: Any,
) -> dict[str, object]:
    """Restore the exact accepted band colors when snapshot diffusion is later."""

    colors = np.asarray(band_evidence.get("band_transmission_colors"), dtype=np.float64)
    band_count = band_evidence.get("band_count")
    z_min = band_evidence.get("particle_z_min_m")
    z_max = band_evidence.get("particle_z_max_m")
    if (
        type(band_count) is not int
        or band_count <= 0
        or colors.shape != (band_count, 3)
        or not np.all(np.isfinite(colors))
        or isinstance(z_min, bool)
        or not isinstance(z_min, (int, float))
        or isinstance(z_max, bool)
        or not isinstance(z_max, (int, float))
        or not math.isfinite(float(z_min))
        or not math.isfinite(float(z_max))
        or float(z_max) <= float(z_min)
    ):
        raise RuntimeError("accepted final height-band color evidence is malformed")
    if live_geometry.get("world_points_sha256") != band_evidence.get(
        "mesh_vertices_sha256"
    ):
        raise RuntimeError("accepted band evidence binds different liquid mesh vertices")
    mesh = runtime["mesh"]
    local_points = np.asarray(
        [tuple(float(value) for value in row) for row in mesh.GetPointsAttr().Get()],
        dtype=np.float64,
    )
    world_points = f2._mesh_world_points(
        mesh,
        local_points,
        time_code=f2.COUNT_DRIVEN_SOURCE_TIME_CODE,
        Usd=Usd,
        UsdGeom=UsdGeom,
        Gf=Gf,
    )
    normalized = (world_points[:, 2] - float(z_min)) / (float(z_max) - float(z_min))
    vertex_bands = np.clip(
        np.floor(normalized * band_count).astype(np.int64), 0, band_count - 1
    )
    indices_sha256 = hashlib.sha256(
        np.asarray(vertex_bands, dtype="<i8", order="C").tobytes()
    ).hexdigest()
    vertex_colors = colors[vertex_bands]
    colors_sha256 = hashlib.sha256(
        np.asarray(vertex_colors, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    if (
        indices_sha256 != band_evidence.get("vertex_band_indices_sha256")
        or colors_sha256 != band_evidence.get("vertex_colors_sha256")
    ):
        raise RuntimeError("accepted final height-band projection evidence drifted")
    primvar = runtime.get("display_color_primvar")
    if primvar is None:
        raise RuntimeError("accepted count body lacks its displayColor primvar")
    primvar.Set(Vt.Vec3fArray([Gf.Vec3f(*row) for row in vertex_colors]))
    readback = np.asarray(
        [tuple(float(value) for value in row) for row in primvar.Get()], dtype="<f4"
    )
    display_sha256 = hashlib.sha256(readback.tobytes()).hexdigest()
    if display_sha256 != expected_display_color_sha256:
        raise RuntimeError("accepted final displayColor bytes could not be restored")
    return {
        "band_count": band_count,
        "vertex_band_indices_sha256": indices_sha256,
        "vertex_colors_sha256": colors_sha256,
        "display_color_sha256": display_sha256,
        "source": "accepted_final_stage_band_evidence",
    }


def _cpu_legacy_semantics_adapter(Sdf: Any) -> Any:
    """Provide the two legacy semantics properties without loading SimulationApp."""

    class BoundSemanticsAPI:
        def __init__(self, prim: Any, instance_name: str):
            self.prim = prim
            self.instance_name = instance_name

        def _attribute(self, leaf: str) -> Any:
            return self.prim.CreateAttribute(
                f"semantic:{self.instance_name}:params:{leaf}",
                Sdf.ValueTypeNames.String,
                False,
            )

        def CreateSemanticTypeAttr(self) -> Any:
            return self._attribute("semanticType")

        def CreateSemanticDataAttr(self) -> Any:
            return self._attribute("semanticData")

        def GetSemanticTypeAttr(self) -> Any:
            return self.prim.GetAttribute(
                f"semantic:{self.instance_name}:params:semanticType"
            )

        def GetSemanticDataAttr(self) -> Any:
            return self.prim.GetAttribute(
                f"semantic:{self.instance_name}:params:semanticData"
            )

    class SemanticsAPI:
        @staticmethod
        def Apply(prim: Any, instance_name: str) -> BoundSemanticsAPI:
            if not prim.AddAppliedSchema(f"SemanticsAPI:{instance_name}"):
                raise RuntimeError("could not apply the legacy semantics schema token")
            return BoundSemanticsAPI(prim, instance_name)

    class Adapter:
        pass

    Adapter.SemanticsAPI = SemanticsAPI
    return Adapter


def _accepted_endpoints(loaded: Mapping[str, Any]) -> dict[str, object]:
    receipt = loaded["receipt"]
    case_root = Path(loaded["case_output"]).resolve()
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
    final_step = loaded["final_physics_step"]
    final_rows = [
        row
        for row in pairs or []
        if isinstance(row, Mapping) and row.get("physics_step") == final_step
    ]
    if len(final_rows) != 1:
        raise ValueError("accepted fill does not bind exactly one final RGB endpoint")
    final_row = final_rows[0]
    final_relative = final_row.get("rgb_path")
    if not isinstance(final_relative, str) or Path(final_relative).is_absolute():
        raise ValueError("accepted final RGB endpoint path is invalid")
    final_path = (case_root / final_relative).resolve()
    if not final_path.is_file() or _sha256(final_path) != final_row.get("rgb_sha256"):
        raise ValueError("accepted final RGB endpoint differs from its receipt")
    video_path = Path(loaded["video_path"]).resolve()
    if _sha256(video_path) != loaded["video_sha256"]:
        raise ValueError("accepted first-frame endpoint video differs from its receipt")
    return {
        "case_output": os.fspath(case_root),
        "final_rgb": {
            "path": os.fspath(final_path),
            "sha256": _sha256(final_path),
            "video_path": os.fspath(video_path),
            "video_sha256": _sha256(video_path),
            "frame_index": -1,
        },
        "first_pre_pour": {
            "video_path": os.fspath(video_path),
            "video_sha256": _sha256(video_path),
            "frame_index": 0,
        },
    }


def _source_layer_key(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.name == "fill_station.usda":
        return "fill_station.usda"
    if path.name == "fill_station.usdc" and path.parent.name == "geometry":
        return "geometry/fill_station.usdc"
    return None


def _load_accepted_first_frame_endpoint(case_output: Path) -> dict[str, object]:
    """Bind a next fill's first frame without consuming its 13 snapshot arrays."""

    case_root = Path(case_output).resolve()
    receipt_path = case_root / "receipt.json"
    if not case_root.is_dir() or not receipt_path.is_file():
        raise ValueError("next accepted case output or receipt is missing")
    raw = receipt_path.read_bytes()
    if len(raw) > MAX_TRUSTED_JSON_BYTES:
        raise ValueError("next accepted receipt exceeds the bounded JSON size limit")
    try:
        value = strict_json_loads(raw.decode("utf-8"), "next accepted receipt")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("next accepted receipt is unreadable") from error
    if not isinstance(value, Mapping):
        raise ValueError("next accepted receipt must contain one object")
    if (
        value.get("schema_version") != 20
        or value.get("experiment") != "physx_three_pour_f2_paint_ball_emitter"
        or value.get("status") != "pass"
    ):
        raise ValueError("next accepted receipt identity or pass status is invalid")
    scene = value.get("scene_contract")
    records = scene.get("source_layer_records") if isinstance(scene, Mapping) else None
    if (
        not isinstance(scene, Mapping)
        or scene.get("source_layers_unchanged") is not True
        or not isinstance(records, list)
        or len(records) != 2
    ):
        raise ValueError("next accepted receipt source contract is invalid")
    observed: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "identifier",
            "resolved_path",
            "bytes",
            "sha256",
        }:
            raise ValueError("next accepted source layer record is malformed")
        key = _source_layer_key(record.get("resolved_path"))
        if key is None or _source_layer_key(record.get("identifier")) != key or key in observed:
            raise ValueError("next accepted source layer membership is invalid")
        if (
            record.get("bytes") != SOURCE_LAYER_BYTES[key]
            or record.get("sha256") != SOURCE_LAYER_HASHES[key]
        ):
            raise ValueError("next accepted source layer evidence differs")
        observed[key] = SOURCE_LAYER_HASHES[key]
    if observed != SOURCE_LAYER_HASHES:
        raise ValueError("next accepted source layer membership is incomplete")
    render = value.get("render")
    relative = render.get("video_path") if isinstance(render, Mapping) else None
    if (
        not isinstance(render, Mapping)
        or not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or render.get("video_decodable") is not True
        or type(render.get("video_decoded_frame_count")) is not int
        or render.get("video_decoded_frame_count", 0) <= 0
    ):
        raise ValueError("next accepted video manifest is invalid")
    video_path = (case_root / relative).resolve()
    try:
        video_path.relative_to(case_root)
    except ValueError as error:
        raise ValueError("next accepted video escapes its case output") from error
    if not video_path.is_file() or _sha256(video_path) != render.get("video_sha256"):
        raise ValueError("next accepted video differs from its receipt")
    return {
        "case_output": os.fspath(case_root),
        "first_pre_pour": {
            "video_path": os.fspath(video_path),
            "video_sha256": _sha256(video_path),
            "frame_index": 0,
        },
        "dependencies": [receipt_path.resolve(), video_path],
    }


def author_transport_liquid_layer(
    *,
    loaded: Mapping[str, Any],
    stage_path: Path,
    output_path: Path,
    owner_thread_id: str,
    transport_implementation_sha256: str,
) -> dict[str, object]:
    """Reconstruct the accepted final liquid once into a non-session layer."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSemantics, UsdShade, Vt
        try:
            from pxr import Semantics as LegacySemantics
        except ImportError:
            LegacySemantics = _cpu_legacy_semantics_adapter(Sdf)
    except ImportError as error:
        raise RuntimeError("USD Python is required to author the transport liquid") from error
    import physx_three_pour_sequence as f2

    owner = validate_owner_thread_id(owner_thread_id)
    contract = _accepted_liquid_contract(loaded)
    root_layer = Sdf.Layer.FindOrOpen(os.fspath(stage_path))
    session = Sdf.Layer.CreateAnonymous("transport-liquid.usda")
    stage = Usd.Stage.Open(root_layer, session)
    if stage is None:
        raise RuntimeError("reviewed source stage could not be opened for liquid authoring")
    stage.SetEditTarget(Usd.EditTarget(session))
    if stage.GetEditTarget().GetLayer() != session:
        raise RuntimeError("transport liquid edit target did not retain its private layer")
    session.customLayerData = {
        "ownerThreadId": owner,
        "transportImplementationSha256": transport_implementation_sha256,
    }
    transmissions = np.asarray(
        [contract["transmissions"][tank] for tank in f2.COMPOSITION_COMPONENTS],
        dtype=np.float64,
    )
    with Usd.EditContext(stage, session):
        UsdGeom.Scope.Define(stage, "/FillStation/FluidmixF2")
        spatial_material = UsdShade.Material.Define(stage, SPATIAL_MATERIAL_PATH)
        global_material = UsdShade.Material.Define(stage, GLOBAL_MATERIAL_PATH)
        global_shader = f2._author_omnisurface(
            stage,
            global_material,
            tuple(float(value) for value in transmissions[0]),
            Sdf,
            Gf,
            UsdShade,
        )
        f2._author_direct_liquid_semantics(global_material.GetPrim(), UsdSemantics)
        spatial_shader = f2._author_spatial_omnisurface(
            stage,
            spatial_material,
            tuple(float(value) for value in transmissions[0]),
            Sdf,
            Gf,
            UsdShade,
        )
        runtime, source_evidence = f2._author_count_driven_twin_liquid_body(
            stage=stage,
            body_material=spatial_material,
            body_shader=spatial_shader,
            height_banded=True,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            UsdSemantics=UsdSemantics,
            LegacySemantics=LegacySemantics,
            Sdf=Sdf,
            Gf=Gf,
            Vt=Vt,
            requested_source_time_code=f2.COUNT_DRIVEN_SOURCE_TIME_CODE,
            particle_volume_m3=contract["particle_volume_m3"],
        )
        source_evidence_attestation = _attest_count_body_source_evidence(
            contract["count_body"], source_evidence
        )
        if source_evidence_attestation["passed"] is not True:
            raise RuntimeError(
                "accepted count-body source geometry evidence drifted: "
                + ", ".join(source_evidence_attestation["failure_reasons"])
            )
        positions = loaded["arrays"]["positions"]
        compositions = loaded["arrays"]["current_compositions"]
        contained = f2._count_driven_contained_mask(positions)
        if (
            not isinstance(contained, np.ndarray)
            or contained.dtype != np.bool_
            or not np.any(contained)
        ):
            raise RuntimeError("accepted final snapshot has no contained liquid")
        live_geometry = f2._update_count_driven_twin_liquid_body(
            runtime,
            contained_positions=positions[contained],
            contained_compositions=compositions[contained],
            global_shader=global_shader,
            source_transmissions=transmissions,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Gf=Gf,
            Vt=Vt,
        )
        selected = f2._select_count_driven_body_capture_geometry(
            live_geometry,
            physics_step=int(loaded["final_physics_step"]),
            phase=str(contract["capture_geometry"].get("phase")),
        )
        capture_geometry_attestation = _attest_count_body_capture_geometry(
            contract["capture_geometry"], selected
        )
        if capture_geometry_attestation["passed"] is not True:
            raise RuntimeError(
                "accepted final count-body geometry drifted: "
                + ", ".join(capture_geometry_attestation["failure_reasons"])
            )
        accepted_band_restore = _restore_accepted_band_display_colors(
            runtime=runtime,
            live_geometry=live_geometry,
            band_evidence=contract["band_evidence"],
            expected_display_color_sha256=contract["material_state"].get(
                "display_color_sha256"
            ),
            f2=f2,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Gf=Gf,
            Vt=Vt,
        )
        material_state = f2._read_count_driven_body_material_state(
            runtime["mesh"], spatial_shader, UsdShade, UsdGeom
        )
        if material_state != contract["material_state"]:
            changed = sorted(
                key
                for key in set(material_state) | set(contract["material_state"])
                if material_state.get(key) != contract["material_state"].get(key)
            )
            raise RuntimeError(
                "accepted final count-body material state drifted in fields: "
                + ", ".join(changed)
                + f" (expected={contract['material_state'].get('display_color_sha256')}, "
                + f"actual={material_state.get('display_color_sha256')})"
            )
        body_prim = runtime["mesh"].GetPrim()
        api_schemas = body_prim.GetMetadata("apiSchemas")
        applied_tokens = (
            [str(value) for value in api_schemas.GetAddedOrExplicitItems()]
            if api_schemas is not None
            else []
        )
        legacy_type = body_prim.GetAttribute(
            "semantic:Semantics:params:semanticType"
        ).Get()
        legacy_data = body_prim.GetAttribute(
            "semantic:Semantics:params:semanticData"
        ).Get()
        if (
            "SemanticsAPI:Semantics" not in applied_tokens
            or str(legacy_type) != "class"
            or str(legacy_data) != "liquid"
        ):
            raise RuntimeError("transport liquid legacy semantic binding drifted")
        legacy_semantics = {
            "api_schema": "SemanticsAPI:Semantics",
            "semantic_type": str(legacy_type),
            "semantic_data": str(legacy_data),
        }

    prim_paths: list[str] = []
    property_paths: list[str] = []

    def visit(spec: Any) -> None:
        prim_paths.append(str(spec.path))
        property_paths.extend(str(item.path) for item in spec.properties)
        for child in spec.nameChildren:
            visit(child)

    for root_spec in session.rootPrims:
        visit(root_spec)
    if any(
        path == protected or path.startswith(protected + "/")
        for path in prim_paths
        for protected in CAMERA_AND_LIGHT_PATHS
    ):
        raise RuntimeError("transport liquid layer authored a camera/light spec")
    required = {COUNT_BODY_PATH, COUNT_BODY_MESH_PATH, SPATIAL_MATERIAL_PATH, GLOBAL_MATERIAL_PATH}
    if not required.issubset(prim_paths):
        raise RuntimeError("transport liquid layer lacks its required reconstruction prims")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or not session.Export(os.fspath(output_path)):
        raise RuntimeError("transport liquid layer could not be exported once")
    return {
        "path": os.fspath(output_path.resolve()),
        "owner_thread_id": owner,
        "sha256": _sha256(output_path),
        "prim_paths": sorted(prim_paths),
        "property_paths": sorted(property_paths),
        "contained_particle_count": int(np.count_nonzero(contained)),
        "source_geometry_attestation": source_evidence_attestation,
        "capture_geometry_attestation": capture_geometry_attestation,
        "capture_geometry": selected,
        "material_state_sha256": hashlib.sha256(
            json.dumps(
                material_state, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
        "accepted_band_display_color": accepted_band_restore,
        "legacy_semantics": legacy_semantics,
    }


def build_transport_stage(
    *,
    mode: str,
    owner_thread_id: str,
    stage: Path,
    case_output: Path,
    output_dir: Path,
    next_case_output: Path | None = None,
) -> dict[str, object]:
    stage = stage.resolve()
    case_output = case_output.resolve()
    output_dir = output_dir.resolve()
    owner = validate_owner_thread_id(owner_thread_id)
    implementation = capture_transport_implementation(ROOT)
    if _sha256(stage) != SOURCE_LAYER_HASHES["fill_station.usda"]:
        raise ValueError("source stage differs from the reviewed hash")
    geometry = stage.parent / "geometry" / "fill_station.usdc"
    if _sha256(geometry) != SOURCE_LAYER_HASHES["geometry/fill_station.usdc"]:
        raise ValueError("source geometry differs from the reviewed hash")
    loaded = dict(load_rigid_transport_inputs(case_output, stage))
    if mode == "between" and next_case_output is None:
        raise ValueError("between transport requires the actual next accepted fill")
    if mode == "exit" and next_case_output is not None:
        raise ValueError("exit transport cannot bind a replacement fill")
    if mode == "entry" and next_case_output is not None:
        raise ValueError("entry uses its primary accepted fill as the next endpoint")
    next_endpoint = (
        _load_accepted_first_frame_endpoint(next_case_output.resolve())
        if next_case_output is not None
        else None
    )
    primary_endpoints = _accepted_endpoints(loaded)
    adjacent_endpoints = {
        "accepted_start": (
            primary_endpoints["final_rgb"] if mode in {"between", "exit"} else None
        ),
        "accepted_end": (
            (
                next_endpoint["first_pre_pour"]
                if next_endpoint is not None
                else primary_endpoints["first_pre_pour"]
            )
            if mode in {"entry", "between"}
            else None
        ),
        "outgoing_case_output": os.fspath(case_output) if mode in {"between", "exit"} else None,
        "next_case_output": (
            os.fspath(next_case_output.resolve())
            if next_case_output is not None
            else (os.fspath(case_output) if mode == "entry" else None)
        ),
    }
    curve = _extract_and_validate_source_curve(stage)
    array_paths = _snapshot_array_paths(loaded)
    input_dependencies = [
        Path(loaded["receipt_path"]),
        Path(loaded["snapshot_path"]),
        Path(loaded["video_path"]),
    ]
    if adjacent_endpoints["accepted_start"] is not None:
        input_dependencies.append(Path(adjacent_endpoints["accepted_start"]["path"]))
    if adjacent_endpoints["accepted_end"] is not None:
        input_dependencies.append(Path(adjacent_endpoints["accepted_end"]["video_path"]))
    if next_endpoint is not None:
        input_dependencies.extend(next_endpoint["dependencies"])
    input_dependencies = list(
        dict.fromkeys(path.resolve() for path in input_dependencies)
    )
    pre_source_records = capture_file_records([stage, geometry], "source file")
    pre_array_records = capture_file_records(array_paths, "snapshot array")
    pre_input_dependency_records = capture_file_records(
        input_dependencies, "accepted input dependency"
    )
    with tempfile.TemporaryDirectory(prefix="fluidmix-transport-liquid-") as raw:
        liquid_path = Path(raw) / "transport-liquid.usda"
        liquid = author_transport_liquid_layer(
            loaded=loaded,
            stage_path=stage,
            output_path=liquid_path,
            owner_thread_id=owner,
            transport_implementation_sha256=str(implementation["sha256"]),
        )
        manifest = build_transport_layers(
            mode=mode,
            owner_thread_id=owner,
            transport_implementation_sha256=str(implementation["sha256"]),
            source_stage=stage,
            source_geometry=geometry,
            liquid_layer=liquid_path,
            output_dir=output_dir,
            liquid_evidence_sha256=str(liquid["sha256"]),
        )
    state_paths = [output_dir / row["path"] for row in manifest["state_layers"]]
    dependencies = [
        *input_dependencies,
        output_dir / "transport-stage.usda",
        output_dir / "transport-liquid.usda",
        output_dir / "transport-manifest.json",
        *(ROOT / relative for relative in TRANSPORT_IMPLEMENTATION_PATHS),
    ]
    dependencies = list(dict.fromkeys(path.resolve() for path in dependencies))
    counters = {
        "physics_steps": 0,
        "physics_callbacks": 0,
        "timeline_playing": False,
        "source_time_code": 1,
    }
    try:
        closure = capture_evidence_closure(
            source_files=[stage, geometry],
            snapshot_arrays=array_paths,
            state_layers=state_paths,
            dependency_files=dependencies,
            counters=counters,
        )
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    input_set = set(input_dependencies)
    post_input_dependency_records = [
        row
        for row in closure["dependency_files"]
        if Path(str(row["path"])) in input_set
    ]
    if (
        closure["source_files"] != pre_source_records
        or closure["snapshot_arrays"] != pre_array_records
        or post_input_dependency_records != pre_input_dependency_records
        or capture_transport_implementation(ROOT) != implementation
    ):
        shutil.rmtree(output_dir)
        raise RuntimeError("transport inputs changed while authoring the frozen build")
    result = {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_stage_build_receipt_v1",
        "status": "pass",
        "mode": mode,
        "owner_thread_id": owner,
        "transport_implementation": implementation,
        "case_output": os.fspath(case_output),
        "source_curve": curve,
        "liquid_layer": liquid,
        "layer_manifest": manifest,
        "adjacent_endpoints": adjacent_endpoints,
        "pre_output_evidence_closure": closure,
        "source_time_code": 1,
        "physics_steps": 0,
        "camera_and_light_specs_authored": [],
        "excluded_from_model_inputs": True,
    }
    try:
        _write_json_new(output_dir / "transport-build-receipt.json", result)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("entry", "between", "exit"), required=True)
    parser.add_argument("--owner-thread-id", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--next-case-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_transport_stage(
            mode=args.mode,
            owner_thread_id=args.owner_thread_id,
            stage=args.stage,
            case_output=args.case_output,
            output_dir=args.output_dir,
            next_case_output=args.next_case_output,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"transport build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
