#!/usr/bin/env python3
"""Build a deterministic offline review of one frozen F2 volume gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix.physx_volume_gate import (  # noqa: E402
    FILL_CAM_HEIGHT_PX,
    FILL_CAM_WIDTH_PX,
    find_dense_surface_row,
    height_within_tolerance,
    projected_dodecagon_vertices_px,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)


CROP_XYXY = (278, 208, 342, 424)
COUNT_DRIVEN_RENDER_PATH = "count_driven_twin_liquid_mesh"
COUNT_DRIVEN_PARTICLE_VOLUME_M3 = 27.0e-9
COUNT_DRIVEN_SIGNED_VOLUME_ABS_TOLERANCE_M3 = 2.0e-10
COUNT_DRIVEN_RESIDUAL_ABS_TOLERANCE_M3 = 2.0e-10
COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M = 1.0e-6
CONTOUR_COLORS_RGB = {
    "floor": (0, 180, 255),
    "lower_tolerance": (255, 210, 0),
    "expected": (0, 255, 0),
    "upper_tolerance": (255, 210, 0),
    "detected_surface_row": (255, 0, 0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _receipt_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _count_driven_body_evidence(
    *,
    receipt: Mapping[str, object],
    target_step: int,
    target_gate: Mapping[str, object],
    paired_state: Mapping[str, object],
    containment: Mapping[str, object],
    bottom_z_m: float,
    cross_section_area_m2: float,
    programmed_expected_height_m: float,
    height_tolerance_m: float,
) -> dict[str, object] | None:
    """Fail closed on the hybrid PBD-state/count-driven-render evidence."""

    if receipt.get("render_path") != COUNT_DRIVEN_RENDER_PATH:
        return None

    body = _mapping(
        receipt.get("count_driven_liquid_body"),
        "count-driven liquid body",
    )
    if (
        body.get("representation") != COUNT_DRIVEN_RENDER_PATH
        or body.get("semantic_label") != "liquid"
        or body.get("sphere_instances_hidden") is not True
        or body.get("extent_recomputed_each_update") is not True
    ):
        raise ValueError("count-driven liquid body contract is incomplete")
    body_path = body.get("body_mesh_path")
    if not isinstance(body_path, str) or not body_path.startswith("/FillStation/"):
        raise ValueError("count-driven body mesh path is missing")

    expected_count = _integer(
        target_gate.get("expected_cumulative_particle_count"),
        "count-driven expected particle count",
        positive=True,
    )
    particle_volume_m3 = _finite(
        body.get("particle_volume_m3"),
        "count-driven particle volume",
        positive=True,
    )
    if not math.isclose(
        particle_volume_m3,
        COUNT_DRIVEN_PARTICLE_VOLUME_M3,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ):
        raise ValueError("count-driven particle volume is not the frozen value")
    body_area_m2 = _finite(
        body.get("cross_section_area_m2"),
        "count-driven cross-section area",
        positive=True,
    )
    if not math.isclose(
        body_area_m2, cross_section_area_m2, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("count-driven body and containment areas disagree")

    capture_records = containment.get("capture_records")
    if not isinstance(capture_records, list):
        raise ValueError("count-driven containment capture records are missing")
    target_captures = [
        row
        for row in capture_records
        if isinstance(row, Mapping) and row.get("physics_step") == target_step
    ]
    if len(target_captures) != 1:
        raise ValueError("count-driven target must have exactly one containment record")
    target_capture = target_captures[0]
    for field in (
        "emitted_particle_count",
        "active_particle_count",
        "inside_vial_interior_count",
    ):
        if _integer(target_capture.get(field), f"count-driven {field}") != expected_count:
            raise ValueError(f"count-driven {field} disagrees with the target count")
    for field in (
        "in_flight_above_mouth_count",
        "outside_or_escaped_particle_count",
    ):
        if _integer(target_capture.get(field), f"count-driven {field}") != 0:
            raise ValueError(f"count-driven {field} must be zero at the target stage")
    if target_capture.get("finite_state") is not True:
        raise ValueError("count-driven target particle state is not finite")

    stage_rows = body.get("stage_geometry")
    if not isinstance(stage_rows, list):
        raise ValueError("count-driven stage geometry is missing")
    target_rows = [
        row
        for row in stage_rows
        if isinstance(row, Mapping) and row.get("physics_step") == target_step
    ]
    if len(target_rows) != 1:
        raise ValueError("count-driven target must have exactly one stage geometry row")
    stage = target_rows[0]
    contained_count = _integer(
        stage.get("contained_particle_count"),
        "count-driven contained particle count",
        positive=True,
    )
    if contained_count != expected_count:
        raise ValueError("count-driven body is not driven by the live contained count")

    target_volume_m3 = contained_count * particle_volume_m3
    recorded_target_volume_m3 = _finite(
        stage.get("target_volume_m3"),
        "count-driven target volume",
        positive=True,
    )
    if not math.isclose(
        recorded_target_volume_m3,
        target_volume_m3,
        rel_tol=0.0,
        abs_tol=COUNT_DRIVEN_SIGNED_VOLUME_ABS_TOLERANCE_M3,
    ):
        raise ValueError("count-driven target volume contradicts the contained count")
    target_height_m = target_volume_m3 / body_area_m2
    recorded_target_height_m = _finite(
        stage.get("target_height_m"),
        "count-driven target height",
        positive=True,
    )
    if not math.isclose(
        recorded_target_height_m, target_height_m, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("count-driven target height contradicts its volume and area")
    if abs(target_height_m - programmed_expected_height_m) > height_tolerance_m:
        raise ValueError("count-driven count-derived height misses the programmed tolerance")

    signed_volume_m3 = _finite(
        stage.get("post_transform_signed_volume_m3"),
        "count-driven signed volume",
    )
    residual_m3 = _finite(
        stage.get("post_transform_volume_residual_m3"),
        "count-driven volume residual",
    )
    recomputed_residual_m3 = abs(abs(signed_volume_m3) - target_volume_m3)
    if not math.isclose(
        abs(signed_volume_m3),
        target_volume_m3,
        rel_tol=0.0,
        abs_tol=COUNT_DRIVEN_SIGNED_VOLUME_ABS_TOLERANCE_M3,
    ):
        raise ValueError("count-driven signed volume misses its target")
    if (
        abs(residual_m3) > COUNT_DRIVEN_RESIDUAL_ABS_TOLERANCE_M3
        or not math.isclose(
            residual_m3,
            recomputed_residual_m3,
            rel_tol=0.0,
            abs_tol=COUNT_DRIVEN_SIGNED_VOLUME_ABS_TOLERANCE_M3,
        )
    ):
        raise ValueError("count-driven volume residual is missing or contradictory")

    source_bottom_z_m = _finite(
        body.get("source_world_bottom_z_m"),
        "count-driven source bottom",
    )
    stage_bottom_z_m = _finite(
        stage.get("world_bottom_z_m"),
        "count-driven stage bottom",
    )
    if not (
        math.isclose(
            source_bottom_z_m,
            bottom_z_m,
            rel_tol=0.0,
            abs_tol=COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M,
        )
        and math.isclose(
            stage_bottom_z_m,
            bottom_z_m,
            rel_tol=0.0,
            abs_tol=COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M,
        )
    ):
        raise ValueError("count-driven body bottom moved away from the vial floor")
    extent = np.asarray(stage.get("world_extent_m"), dtype=np.float64)
    if extent.shape != (2, 3) or not np.all(np.isfinite(extent)):
        raise ValueError("count-driven world extent must be a finite 2x3 array")
    if np.any(extent[1] <= extent[0]):
        raise ValueError("count-driven world extent must have positive dimensions")
    if (
        not math.isclose(
            float(extent[0, 2]),
            bottom_z_m,
            rel_tol=0.0,
            abs_tol=COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M,
        )
        or not math.isclose(
            float(extent[1, 2] - extent[0, 2]),
            target_height_m,
            rel_tol=0.0,
            abs_tol=COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M,
        )
    ):
        raise ValueError("count-driven world extent does not preserve bottom and height")
    points_sha256 = _receipt_sha256(
        stage.get("points_sha256"), "count-driven body points hash"
    )
    extent_sha256 = _receipt_sha256(
        stage.get("extent_sha256"), "count-driven body extent hash"
    )

    paired_hashes: dict[str, str] = {}
    for field in (
        "body_points",
        "body_extent",
        "body_visibility",
        "body_material_color",
        "stream_visibility",
    ):
        before = _receipt_sha256(
            paired_state.get(f"{field}_sha256_before"),
            f"count-driven paired {field} before hash",
        )
        after = _receipt_sha256(
            paired_state.get(f"{field}_sha256_after"),
            f"count-driven paired {field} after hash",
        )
        if before != after:
            raise ValueError(f"count-driven paired {field} changed between passes")
        paired_hashes[field] = before

    semantics = _mapping(receipt.get("semantics"), "semantics")
    control = _mapping(
        semantics.get("hidden_particle_control"),
        "count-driven hidden-particle control",
    )
    target_ids = control.get("body_only_target_ids")
    control_path = control.get("control_semantic_path")
    control_sha256 = _receipt_sha256(
        control.get("control_semantic_sha256"),
        "count-driven hidden-particle control hash",
    )
    if (
        semantics.get("count_driven_body_semantic_bound") is not True
        or semantics.get("point_instancer_rgb_visible") is not False
        or semantics.get("point_instancer_semantic_visible") is not False
        or control.get("physics_step") != target_step
        or control.get("emitted_particle_count") != expected_count
        or control.get("inside_vial_interior_count") != expected_count
        or control.get("body_and_stream_hidden_liquid_pixel_count") != 0
        or control.get("hidden_sphere_prototypes") is not True
        or not isinstance(control_path, str)
        or not control_path
        or control.get("body_only_target_path") != body_path
        or not isinstance(target_ids, list)
        or not target_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in target_ids
        )
        or control.get("body_only_capture_step") != target_step
        or not isinstance(control.get("body_only_liquid_pixel_count"), int)
        or control.get("body_only_liquid_pixel_count") <= 0
        or control.get("passed") is not True
    ):
        raise ValueError("count-driven semantic isolation evidence is incomplete")

    return {
        "representation": COUNT_DRIVEN_RENDER_PATH,
        "physics_step": target_step,
        "live_contained_particle_count": contained_count,
        "particle_volume_m3": particle_volume_m3,
        "target_volume_m3": target_volume_m3,
        "target_height_m": target_height_m,
        "post_transform_signed_volume_m3": signed_volume_m3,
        "post_transform_volume_residual_m3": residual_m3,
        "signed_volume_abs_tolerance_m3": (
            COUNT_DRIVEN_SIGNED_VOLUME_ABS_TOLERANCE_M3
        ),
        "volume_residual_abs_tolerance_m3": (
            COUNT_DRIVEN_RESIDUAL_ABS_TOLERANCE_M3
        ),
        "geometry_abs_tolerance_m": COUNT_DRIVEN_GEOMETRY_ABS_TOLERANCE_M,
        "world_bottom_z_m": stage_bottom_z_m,
        "world_extent_m": extent.tolist(),
        "points_sha256": points_sha256,
        "extent_sha256": extent_sha256,
        "paired_render_state_sha256": paired_hashes,
        "body_only_target_ids": list(target_ids),
        "hidden_particle_control_path": control_path,
        "hidden_particle_control_sha256": control_sha256,
        "hidden_particle_liquid_pixel_count": 0,
    }


def _bound_source_path(
    *,
    receipt_dir: Path,
    recorded_path: object,
    supplied_path: Path,
    recorded_sha256: object,
    name: str,
) -> tuple[Path, str]:
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError(f"target {name} path is missing from the receipt")
    relative = Path(recorded_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"target {name} receipt path must stay below app-output")
    expected = (receipt_dir / relative).resolve(strict=True)
    supplied = supplied_path.resolve(strict=True)
    if supplied != expected:
        raise ValueError(
            f"target {name} path mismatch: receipt binds {expected}, got {supplied}"
        )
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ValueError(f"target {name} receipt hash is missing or malformed")
    actual_hash = _sha256(supplied)
    if actual_hash != recorded_sha256:
        raise ValueError(
            f"target {name} hash mismatch: receipt has {recorded_sha256}, "
            f"file has {actual_hash}"
        )
    return supplied, actual_hash


def _load_png(path: Path, *, expected_mode: str, name: str) -> Image.Image:
    with Image.open(path) as opened:
        if opened.format != "PNG":
            raise ValueError(f"target {name} must be a PNG")
        if opened.mode != expected_mode:
            raise ValueError(
                f"target {name} must have mode {expected_mode}, got {opened.mode}"
            )
        if opened.size != (FILL_CAM_WIDTH_PX, FILL_CAM_HEIGHT_PX):
            raise ValueError(
                f"target {name} must be the frozen 640x480 FillCam raster"
            )
        return opened.copy()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _project_contour(
    height_m: float,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    circumradius_m: float,
    wall_count: int,
) -> list[list[float]]:
    vertices = projected_dodecagon_vertices_px(
        height_m,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        circumradius_m=circumradius_m,
        wall_count=wall_count,
    )
    return [[float(x), float(y)] for x, y in vertices]


def build_volume_review(
    *,
    receipt_path: Path,
    rgb_path: Path,
    liquid_mask_path: Path,
    output_dir: Path,
    target_step: int | None = None,
) -> dict[str, object]:
    """Validate bound source evidence and write review-only artifacts."""

    receipt_path = receipt_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"review output already exists: {output_dir}")
    receipt_hash = _sha256(receipt_path)
    try:
        receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt is not readable JSON") from error
    receipt = _mapping(receipt_value, "receipt")
    if receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter":
        raise ValueError("receipt is not an F2 Paint Ball Emitter run")
    qualification = _mapping(receipt.get("qualification"), "qualification")
    explicit_target = target_step is not None
    target_step_value = target_step if explicit_target else qualification.get("target_step")
    if (
        isinstance(target_step_value, bool)
        or not isinstance(target_step_value, int)
        or target_step_value < 0
    ):
        name = "target_step" if explicit_target else "qualification.target_step"
        raise ValueError(f"{name} must be a non-negative integer")
    target_step = target_step_value

    if explicit_target:
        semantics = _mapping(receipt.get("semantics"), "semantics")
        pairs = semantics.get("pairs")
        pair_name = "semantics.pairs"
    else:
        if qualification.get("first_stage_only") is not True:
            raise ValueError("receipt is not a first-stage qualification run")
        pairs = qualification.get("paired_capture_evidence")
        pair_name = "qualification.paired_capture_evidence"
    if not isinstance(pairs, list):
        raise ValueError(f"{pair_name} must be a list")
    targets = [
        item
        for item in pairs
        if isinstance(item, Mapping) and item.get("physics_step") == target_step
    ]
    if len(targets) != 1:
        raise ValueError("receipt must contain exactly one target paired capture")
    target_pair = targets[0]
    rgb_path, rgb_hash = _bound_source_path(
        receipt_dir=receipt_path.parent,
        recorded_path=target_pair.get("rgb_path"),
        supplied_path=rgb_path,
        recorded_sha256=target_pair.get("rgb_sha256"),
        name="RGB",
    )
    liquid_mask_path, mask_hash = _bound_source_path(
        receipt_dir=receipt_path.parent,
        recorded_path=target_pair.get("mask_path"),
        supplied_path=liquid_mask_path,
        recorded_sha256=target_pair.get("mask_sha256"),
        name="liquid mask",
    )
    if target_pair.get("paired_same_physics_state_rgb_semantic_capture") is not True:
        raise ValueError("target paired-capture contract is not true")
    paired_state_passed = _bool(
        target_pair.get("paired_capture_state_passed"),
        "target paired_capture_state_passed",
    )
    paired_state = _mapping(
        target_pair.get("paired_capture_state_evidence"),
        "target paired_capture_state_evidence",
    )
    if paired_state.get("passed") is not paired_state_passed:
        raise ValueError("target paired-capture state contradicts its evidence")
    if paired_state.get("physics_step") != target_step:
        raise ValueError("target paired-capture state has the wrong physics step")

    if explicit_target:
        captures = receipt.get("captures")
        if not isinstance(captures, list):
            raise ValueError("captures must be a list")
        capture_targets = [
            item
            for item in captures
            if isinstance(item, Mapping)
            and item.get("physics_step") == target_step
            and isinstance(item.get("volume_gate"), Mapping)
            and item["volume_gate"].get("required") is True
        ]
        if len(capture_targets) != 1:
            raise ValueError(
                "receipt must contain exactly one required target capture volume gate"
            )
        target_gate = _mapping(
            capture_targets[0].get("volume_gate"),
            "target capture volume_gate",
        )
        target_binding_source = "semantics.pairs+captures.volume_gate"
    else:
        target_gate = _mapping(
            qualification.get("target_volume_evidence"),
            "qualification.target_volume_evidence",
        )
        target_binding_source = (
            "qualification.paired_capture_evidence+qualification.target_volume_evidence"
        )
    if target_gate.get("required") is not True:
        raise ValueError("target volume evidence is not marked required")
    state_passed = _bool(target_gate.get("state_passed"), "volume state_passed")
    visible_passed = _bool(
        target_gate.get("visible_passed"), "volume visible_passed"
    )
    gate_passed = _bool(target_gate.get("passed"), "volume passed")
    if gate_passed is not (state_passed and visible_passed):
        raise ValueError("target volume gate decision is internally contradictory")
    failure_reasons = target_gate.get("failure_reasons")
    if not isinstance(failure_reasons, list) or any(
        not isinstance(reason, str) for reason in failure_reasons
    ):
        raise ValueError("target volume failure_reasons must be a string list")
    if gate_passed and failure_reasons:
        raise ValueError("passed target volume gate has failure reasons")

    runner_fields = {
        "volume_state_gate_passed": state_passed,
        "volume_visible_gate_passed": visible_passed,
        "volume_gate_passed": gate_passed,
    }
    if not explicit_target:
        for name, nested_value in runner_fields.items():
            if _bool(qualification.get(name), f"qualification.{name}") is not nested_value:
                raise ValueError(f"qualification.{name} contradicts target volume evidence")
    run_status = receipt.get("status")
    qualification_status = qualification.get("status")
    if not isinstance(run_status, str) or not isinstance(qualification_status, str):
        raise ValueError("runner status fields are missing")

    containment = _mapping(receipt.get("containment"), "containment")
    geometry_value = _mapping(
        containment.get("vial_interior_bounds_m"),
        "containment.vial_interior_bounds_m",
    )
    center_value = geometry_value.get("center_xy_m")
    if not isinstance(center_value, list) or len(center_value) != 2:
        raise ValueError("vial center_xy_m must have two values")
    center_xy_m = (
        _finite(center_value[0], "vial center x"),
        _finite(center_value[1], "vial center y"),
    )
    bottom_z_m = _finite(geometry_value.get("bottom_z_m"), "vial bottom z")
    mouth_z_m = _finite(geometry_value.get("mouth_z_m"), "vial mouth z")
    circumradius_m = _finite(
        geometry_value.get("circumradius_m"),
        "vial circumradius",
        positive=True,
    )
    cross_section_area_m2 = (
        _finite(
            geometry_value.get("cross_section_area_m2"),
            "vial cross-section area",
            positive=True,
        )
        if receipt.get("render_path") == COUNT_DRIVEN_RENDER_PATH
        else 0.0
    )
    wall_count_value = geometry_value.get("wall_count")
    if (
        isinstance(wall_count_value, bool)
        or not isinstance(wall_count_value, int)
        or wall_count_value < 3
    ):
        raise ValueError("vial wall_count must be an integer of at least three")
    wall_count = wall_count_value
    maximum_height_m = mouth_z_m - bottom_z_m
    if maximum_height_m <= 0.0:
        raise ValueError("vial mouth must be above its bottom")

    expected_height_m = _finite(
        target_gate.get("expected_height_m"),
        "expected target height",
        positive=True,
    )
    tolerance_m = _finite(
        target_gate.get("height_tolerance_m"), "height tolerance"
    )
    if not math.isclose(tolerance_m, 0.006, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("height tolerance is not the frozen 0.006 m")
    lower_height_m = expected_height_m - tolerance_m
    upper_height_m = expected_height_m + tolerance_m
    if lower_height_m < 0.0 or upper_height_m > maximum_height_m:
        raise ValueError("target height tolerance contours fall outside the vial")

    count_driven_evidence = _count_driven_body_evidence(
        receipt=receipt,
        target_step=target_step,
        target_gate=target_gate,
        paired_state=paired_state,
        containment=containment,
        bottom_z_m=bottom_z_m,
        cross_section_area_m2=cross_section_area_m2,
        programmed_expected_height_m=expected_height_m,
        height_tolerance_m=tolerance_m,
    )

    rgb = _load_png(rgb_path, expected_mode="RGB", name="RGB")
    liquid_image = _load_png(
        liquid_mask_path, expected_mode="L", name="liquid mask"
    )
    liquid_mask = np.asarray(liquid_image)
    roi = projected_vial_geometry_roi_mask(
        liquid_mask.shape,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        mouth_z_m=mouth_z_m,
        circumradius_m=circumradius_m,
        wall_count=wall_count,
    )
    detected_surface = find_dense_surface_row(
        liquid_mask,
        geometry_roi_mask=roi,
    )
    observed_height_m = visible_height_from_surface_row_m(
        detected_surface["surface_row_px"],
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        circumradius_m=circumradius_m,
        maximum_height_m=maximum_height_m,
        wall_count=wall_count,
    )
    detected_surface = {
        **detected_surface,
        "observed_height_m": observed_height_m,
        "absolute_error_m": abs(observed_height_m - expected_height_m),
    }
    recorded_visible = _mapping(
        target_gate.get("visible_mask_evidence"),
        "target visible_mask_evidence",
    )
    if (
        recorded_visible.get("required") is not True
        or recorded_visible.get("status") != "measured"
        or _bool(recorded_visible.get("passed"), "visible evidence passed")
        is not visible_passed
    ):
        raise ValueError("target visible-mask evidence is incomplete or contradictory")
    for name, value in detected_surface.items():
        recorded = recorded_visible.get(name)
        if isinstance(value, float):
            if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
                raise ValueError(f"visible-mask evidence is missing {name}")
            if not math.isclose(float(recorded), value, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"visible-mask evidence {name} contradicts the mask")
        elif recorded != value:
            raise ValueError(f"visible-mask evidence {name} contradicts the mask")
    recomputed_visible_passed = height_within_tolerance(
        observed_height_m,
        expected_height_m,
        tolerance_m,
    )
    if recomputed_visible_passed is not visible_passed:
        raise ValueError("receipt visible decision contradicts the bound liquid mask")

    geometry_kwargs = {
        "center_xy_m": center_xy_m,
        "bottom_z_m": bottom_z_m,
        "circumradius_m": circumradius_m,
        "wall_count": wall_count,
    }
    contours_px = {
        "floor": _project_contour(0.0, **geometry_kwargs),
        "lower_tolerance": _project_contour(lower_height_m, **geometry_kwargs),
        "expected": _project_contour(expected_height_m, **geometry_kwargs),
        "upper_tolerance": _project_contour(upper_height_m, **geometry_kwargs),
    }
    surface_row_px = int(detected_surface["surface_row_px"])
    row_x = np.flatnonzero(roi[surface_row_px])
    if len(row_x) == 0:
        raise ValueError("detected surface row has no projected geometry span")
    detected_span_xyxy = [
        int(row_x[0]),
        surface_row_px,
        int(row_x[-1]) + 1,
        surface_row_px + 1,
    ]

    overlay = rgb.copy()
    drawing = ImageDraw.Draw(overlay)
    for name in ("floor", "lower_tolerance", "expected", "upper_tolerance"):
        points = [tuple(point) for point in contours_px[name]]
        drawing.line(
            points + [points[0]],
            fill=CONTOUR_COLORS_RGB[name],
            width=2,
            joint="curve",
        )
    drawing.line(
        (
            detected_span_xyxy[0],
            surface_row_px,
            detected_span_xyxy[2] - 1,
            surface_row_px,
        ),
        fill=CONTOUR_COLORS_RGB["detected_surface_row"],
        width=2,
    )

    crop = tuple(CROP_XYXY)
    raw_crop = rgb.crop(crop)
    overlay_crop = overlay.crop(crop)
    comparison = Image.new(
        "RGB", (raw_crop.width + overlay_crop.width, raw_crop.height)
    )
    comparison.paste(raw_crop, (0, 0))
    comparison.paste(overlay_crop, (raw_crop.width, 0))

    metrics: dict[str, object] = {
        "schema_version": 1,
        "review_type": "offline_geometry_only_volume_review",
        "source_evidence": {
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_hash,
            "rgb_path": str(rgb_path),
            "rgb_sha256": rgb_hash,
            "liquid_mask_path": str(liquid_mask_path),
            "liquid_mask_sha256": mask_hash,
            "target_step": target_step,
            "target_binding_source": target_binding_source,
        },
        "runner_decision": {
            "run_status": run_status,
            "qualification_status": qualification_status,
            **runner_fields,
        },
        "review_contract": {
            "decision_source": "receipt_only",
            "does_not_override_runner": True,
        },
        "target_volume_evidence": dict(target_gate),
        "paired_capture_state_evidence": dict(paired_state),
        "geometry": {
            "center_xy_m": list(center_xy_m),
            "bottom_z_m": bottom_z_m,
            "mouth_z_m": mouth_z_m,
            "circumradius_m": circumradius_m,
            "wall_count": wall_count,
        },
        "contour_heights_m": {
            "floor": 0.0,
            "lower_tolerance": lower_height_m,
            "expected": expected_height_m,
            "upper_tolerance": upper_height_m,
        },
        "contour_colors_rgb": {
            name: list(color) for name, color in CONTOUR_COLORS_RGB.items()
        },
        "contours_px": contours_px,
        "detected_surface": {
            **detected_surface,
            "span_xyxy": detected_span_xyxy,
        },
        "crop_xyxy": list(CROP_XYXY),
    }
    if count_driven_evidence is not None:
        metrics["count_driven_body_evidence"] = count_driven_evidence

    output_dir.mkdir(parents=True)
    overlay_path = output_dir / "target-volume-overlay.png"
    crop_path = output_dir / "target-rgb-and-overlay-crop.png"
    metrics_path = output_dir / "metrics.json"
    overlay.save(overlay_path, format="PNG", compress_level=9, optimize=False)
    comparison.save(crop_path, format="PNG", compress_level=9, optimize=False)
    _write_json(metrics_path, metrics)
    artifact_names = (
        "metrics.json",
        "target-volume-overlay.png",
        "target-rgb-and-overlay-crop.png",
    )
    hashes: dict[str, object] = {
        "schema_version": 1,
        "source_receipt_sha256": receipt_hash,
        "artifacts": {
            name: {
                "path": name,
                "sha256": _sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        },
    }
    _write_json(output_dir / "artifact-hashes.json", hashes)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--liquid-mask", type=Path, required=True)
    parser.add_argument(
        "--target-step",
        type=int,
        help=(
            "explicit full-run physics step bound through semantics.pairs and "
            "captures.volume_gate; omitted preserves qualification mode"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_volume_review(
        receipt_path=args.receipt,
        rgb_path=args.rgb,
        liquid_mask_path=args.liquid_mask,
        output_dir=args.out,
        target_step=args.target_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
