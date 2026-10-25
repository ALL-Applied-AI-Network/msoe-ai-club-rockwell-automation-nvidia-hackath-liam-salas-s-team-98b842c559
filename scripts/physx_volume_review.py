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
    if qualification.get("first_stage_only") is not True:
        raise ValueError("receipt is not a first-stage qualification run")
    target_step_value = qualification.get("target_step")
    if (
        isinstance(target_step_value, bool)
        or not isinstance(target_step_value, int)
        or target_step_value < 0
    ):
        raise ValueError("qualification.target_step must be a non-negative integer")
    target_step = target_step_value

    pairs = qualification.get("paired_capture_evidence")
    if not isinstance(pairs, list):
        raise ValueError("qualification.paired_capture_evidence must be a list")
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

    target_gate = _mapping(
        qualification.get("target_volume_evidence"),
        "qualification.target_volume_evidence",
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
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_volume_review(
        receipt_path=args.receipt,
        rgb_path=args.rgb,
        liquid_mask_path=args.liquid_mask,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
