"""Camera-derived fill-step contract for WP7 development evidence.

This module intentionally contains no video decoder or model runner.  It joins
already measured stream colour and accepted CNN-mask height to frozen physical
and recipe bindings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from statistics import median


ACTIVE_TANK_CYCLE = ("t100", "t200", "t400")
FROZEN_STREAM_ROI_XYXY = (300, 80, 320, 190)
CALIBRATION_SCHEMA = "wp7-fill-height-volume-calibration-v1"
RECEIPT_SCHEMA = "wp7-camera-fill-step-receipt-v1"
CLAIM_BOUNDARY = "camera_twin_geometry_equivalent_not_physical_ground_truth"
FROZEN_FILLCAM_CAMERA_SHA256 = (
    "772bc8a3d9bb20c8970d9e3ee160d444e68d58418147f498457bfc162e03dfcf"
)
FROZEN_GEOMETRY_SHA256 = (
    "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
)
FROZEN_VOLUME_CURVE_SHA256 = (
    "7add82ae73bcda63f667c755b20fd1d7077a763028979b03e7d1d241c92762f9"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FillStepTrackerError(ValueError):
    """The fill-step input or frozen binding failed closed."""


@dataclass(frozen=True)
class PixelToMetreCalibration:
    schema_version: str
    calibration_id: str
    camera_effective_state_sha256: str
    geometry_sha256: str
    volume_curve_sha256: str
    fill_height_px_samples: tuple[float, ...]
    visible_height_m_samples: tuple[float, ...]
    twin_geometry_equivalent_volume_ml_samples: tuple[float, ...]
    artifact_sha256: str


@dataclass(frozen=True)
class RecipePour:
    tank_id: str
    dispense_ml: float


@dataclass(frozen=True)
class TrackerFreeze:
    stream_roi_xyxy: tuple[int, int, int, int]
    stream_rgb_change_threshold: int
    stream_minimum_changed_pixels: int
    stream_minimum_consecutive_frames: int
    stream_change_delta_e76: float
    tank_cluster_delta_e76: float
    volume_match_tolerance_ml: float
    calibration: PixelToMetreCalibration
    tank_lab_centres: Mapping[str, tuple[float, float, float]]
    recipe_artifact_sha256: str
    volume_curve_sha256: str


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    stream_lab: tuple[float, float, float]
    stream_spread_lab: tuple[float, float, float]
    visible_height_px: float | None


@dataclass(frozen=True)
class StageFrameSelection:
    fill_step: int
    early_frame_index: int
    mid_frame_index: int
    end_frame_index: int


@dataclass(frozen=True)
class StageState:
    tank_number: int
    tank_id: str
    stream_step_index: int
    expected_dispense_ml: float
    expected_cumulative_ml: float
    twin_geometry_equivalent_volume_ml: float
    fill_step: int | None
    agreement: bool
    tank_cluster_consistent: bool
    stream_lab: tuple[float, float, float]
    stream_spread_lab: tuple[float, float, float]
    selected_frames: StageFrameSelection


@dataclass(frozen=True)
class FillStepResult:
    tank_number: int
    tank_id: str
    stream_step_index: int
    measured_volume_ml: float
    twin_geometry_equivalent_volume_ml: float
    expected_dispense_ml: float
    expected_cumulative_ml: float
    fill_step: int | None
    agreement: bool
    tank_cluster_consistent: bool
    stream_lab: tuple[float, float, float]
    stream_spread_lab: tuple[float, float, float]
    selected_frames: tuple[StageFrameSelection, ...]
    stages: tuple[StageState, ...]
    limit_codes: tuple[str, ...]


def calibration_payload_sha256(calibration: PixelToMetreCalibration) -> str:
    """Hash a calibration's closed payload, excluding its self-hash."""

    payload = asdict(calibration)
    payload.pop("artifact_sha256")
    return _canonical_sha256(payload)


def track_fill_step(
    frames: Sequence[FrameEvidence],
    recipe_pours: Sequence[RecipePour],
    freeze: TrackerFreeze,
    *,
    previous_final_tank_id: str,
) -> FillStepResult:
    """Return the agreed camera fill step, or an explicit disagreement abstention."""

    checked_freeze = _validate_freeze(freeze)
    pours = _validate_pours(recipe_pours)
    checked_frames = _validate_frames(frames)
    expected_previous = ACTIVE_TANK_CYCLE[
        (ACTIVE_TANK_CYCLE.index(pours[0].tank_id) - 1) % len(ACTIVE_TANK_CYCLE)
    ]
    if previous_final_tank_id != expected_previous:
        raise FillStepTrackerError(
            "previous final tank does not bind the recipe cycle offset"
        )
    segments = _completed_stage_segments(
        checked_frames,
        pours,
        checked_freeze,
        previous_final_tank_id=previous_final_tank_id,
    )
    stream_step = len(segments)

    selected = tuple(
        StageFrameSelection(
            fill_step=index,
            early_frame_index=segment[0].frame_index,
            mid_frame_index=segment[len(segment) // 2].frame_index,
            end_frame_index=segment[-1].frame_index,
        )
        for index, segment in enumerate(segments, start=1)
    )
    segment = segments[-1]
    stream_lab = _coordinate_median(item.stream_lab for item in segment)
    stream_spread = _coordinate_median(item.stream_spread_lab for item in segment)
    pour = pours[stream_step - 1]
    cluster_consistent = (
        math.dist(stream_lab, checked_freeze.tank_lab_centres[pour.tank_id])
        <= checked_freeze.tank_cluster_delta_e76
    )
    height_px = segment[-1].visible_height_px
    if height_px is None:
        raise FillStepTrackerError("current stage lacks accepted CNN fill height")
    measured = _interpolate_calibrated_volume(
        float(height_px), checked_freeze.calibration
    )
    cumulative: list[float] = []
    running = 0.0
    for item in pours:
        running += float(item.dispense_ml)
        cumulative.append(running)
    matches = [
        index for index, expected in enumerate(cumulative, start=1)
        if abs(measured - expected) <= checked_freeze.volume_match_tolerance_ml
    ]
    volume_step = matches[0] if len(matches) == 1 else None
    volume_agreement = volume_step == stream_step
    agreement = volume_agreement and cluster_consistent
    stage_states: list[StageState] = []
    for index, (stage_segment, selection) in enumerate(
        zip(segments, selected, strict=True), start=1
    ):
        stage_height = stage_segment[-1].visible_height_px
        if stage_height is None:
            raise FillStepTrackerError("stage lacks accepted CNN fill height")
        stage_volume = _interpolate_calibrated_volume(
            float(stage_height), checked_freeze.calibration
        )
        stage_pour = pours[index - 1]
        stage_lab = _coordinate_median(item.stream_lab for item in stage_segment)
        stage_spread = _coordinate_median(
            item.stream_spread_lab for item in stage_segment
        )
        stage_matches = [
            matched_index
            for matched_index, expected in enumerate(cumulative, start=1)
            if abs(stage_volume - expected)
            <= checked_freeze.volume_match_tolerance_ml
        ]
        stage_cluster_consistent = (
            math.dist(
                stage_lab,
                checked_freeze.tank_lab_centres[stage_pour.tank_id],
            ) <= checked_freeze.tank_cluster_delta_e76
        )
        stage_agreement = stage_matches == [index] and stage_cluster_consistent
        stage_states.append(
            StageState(
                tank_number=int(stage_pour.tank_id[1:]),
                tank_id=stage_pour.tank_id,
                stream_step_index=index,
                expected_dispense_ml=float(stage_pour.dispense_ml),
                expected_cumulative_ml=cumulative[index - 1],
                twin_geometry_equivalent_volume_ml=stage_volume,
                fill_step=index if stage_agreement else None,
                agreement=stage_agreement,
                tank_cluster_consistent=stage_cluster_consistent,
                stream_lab=stage_lab,
                stream_spread_lab=stage_spread,
                selected_frames=selection,
            )
        )
    limits: list[str] = []
    if not volume_agreement:
        limits.append("stream_volume_step_disagreement")
    if not cluster_consistent:
        limits.append("stream_lab_tank_cluster_inconsistent")
    return FillStepResult(
        tank_number=int(pour.tank_id[1:]),
        tank_id=pour.tank_id,
        stream_step_index=stream_step,
        measured_volume_ml=measured,
        twin_geometry_equivalent_volume_ml=measured,
        expected_dispense_ml=float(pour.dispense_ml),
        expected_cumulative_ml=cumulative[stream_step - 1],
        fill_step=stream_step if agreement else None,
        agreement=agreement,
        tank_cluster_consistent=cluster_consistent,
        stream_lab=stream_lab,
        stream_spread_lab=stream_spread,
        selected_frames=selected,
        stages=tuple(stage_states),
        limit_codes=tuple(limits),
    )


def adapt_runtime_frames(
    runtime_measurements: Sequence[object],
    stream_rows: Sequence[Mapping[str, object]],
) -> tuple[FrameEvidence, ...]:
    """Join accepted WP7 CNN measurements to separately measured stream Lab rows."""

    from .wp7_rgb_runtime import RuntimeVisionMeasurement

    measurements = tuple(runtime_measurements)
    for item in measurements:
        if (
            not isinstance(item, RuntimeVisionMeasurement)
            or item.status != "visible_measured"
            or item.visible_height_px is None
            or item.limit_codes
        ):
            raise FillStepTrackerError(
                "runtime measurement is not an accepted visible CNN mask"
            )
    by_id: dict[str, Mapping[str, object]] = {}
    for row in stream_rows:
        if not isinstance(row, Mapping):
            raise FillStepTrackerError("stream Lab row must be a mapping")
        if "pour_index" in row:
            raise FillStepTrackerError("external stream step ordinal is forbidden")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in by_id:
            raise FillStepTrackerError("stream Lab image_id is missing or duplicated")
        by_id[image_id] = row
    if set(by_id) != {item.image_id for item in measurements}:
        raise FillStepTrackerError("stream Lab rows are not exactly runtime-mask bound")
    result: list[FrameEvidence] = []
    for item in measurements:
        row = by_id[item.image_id]
        if row.get("frame_index") != item.frame_index:
            raise FillStepTrackerError("stream Lab frame index mismatches runtime mask")
        result.append(
            FrameEvidence(
                frame_index=item.frame_index,
                stream_lab=_triplet(row.get("median_lab"), "median_lab"),
                stream_spread_lab=_triplet(
                    row.get("spread_lab"), "spread_lab", nonnegative=True
                ),
                visible_height_px=float(item.visible_height_px),
            )
        )
    return tuple(result)


def build_fill_step_receipt(
    *,
    cohort_id: str,
    vial_id: str,
    result: FillStepResult,
    freeze: TrackerFreeze,
    image_id_by_frame: Mapping[int, str],
    source_artifact_sha256: str,
) -> dict[str, object]:
    """Build the closed, path-free Task-1A receipt consumed by the VLM lane."""

    _identifier(cohort_id, "cohort_id")
    _identifier(vial_id, "vial_id")
    _validate_freeze(freeze)
    if not isinstance(result, FillStepResult) or not result.stages:
        raise FillStepTrackerError("fill-step result is missing or invalid")
    if _SHA256.fullmatch(source_artifact_sha256) is None:
        raise FillStepTrackerError("source artifact SHA-256 is invalid")
    if any(not stage.agreement or stage.fill_step is None for stage in result.stages):
        raise FillStepTrackerError("disagreed fill stage cannot enter a VLM receipt")
    stages: list[dict[str, object]] = []
    for stage in result.stages:
        selection = stage.selected_frames
        try:
            images = {
                "early": image_id_by_frame[selection.early_frame_index],
                "mid": image_id_by_frame[selection.mid_frame_index],
                "end": image_id_by_frame[selection.end_frame_index],
            }
        except KeyError as error:
            raise FillStepTrackerError("selected frame lacks an opaque image ID") from error
        for image_id in images.values():
            _identifier(image_id, "image ID")
        stages.append(
            {
                "fill_step": stage.fill_step,
                "tank_id": stage.tank_id,
                "tank_number": stage.tank_number,
                "stream_step_index": stage.stream_step_index,
                "expected_dispense_ml": stage.expected_dispense_ml,
                "expected_cumulative_ml": stage.expected_cumulative_ml,
                "twin_geometry_equivalent_volume_ml": (
                    stage.twin_geometry_equivalent_volume_ml
                ),
                "measured_volume_ml": stage.twin_geometry_equivalent_volume_ml,
                "agreement": stage.agreement,
                "tank_cluster_consistent": stage.tank_cluster_consistent,
                "stream_lab": list(stage.stream_lab),
                "stream_spread_lab": list(stage.stream_spread_lab),
                "images": images,
            }
        )
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_type": "wp7_camera_fill_step_receipt",
        "cohort_id": cohort_id,
        "vial_id": vial_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "stages": stages,
        "calibration_sha256": freeze.calibration.artifact_sha256,
        "recipe_artifact_sha256": freeze.recipe_artifact_sha256,
        "source_artifact_sha256": source_artifact_sha256,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return validate_fill_step_receipt(receipt)


def validate_fill_step_receipt(value: object) -> dict[str, object]:
    """Validate and return one closed Task-1A receipt."""

    if not isinstance(value, Mapping):
        raise FillStepTrackerError("fill-step receipt must be a mapping")
    expected = {
        "schema_version", "artifact_type", "cohort_id", "vial_id",
        "claim_boundary", "stages", "calibration_sha256",
        "recipe_artifact_sha256", "source_artifact_sha256", "receipt_sha256",
    }
    if set(value) != expected:
        raise FillStepTrackerError("fill-step receipt fields are not closed")
    if value["schema_version"] != RECEIPT_SCHEMA or value["artifact_type"] != "wp7_camera_fill_step_receipt":
        raise FillStepTrackerError("fill-step receipt schema is invalid")
    if value["claim_boundary"] != CLAIM_BOUNDARY:
        raise FillStepTrackerError("fill-step receipt claim boundary is invalid")
    _identifier(value["cohort_id"], "cohort_id")
    _identifier(value["vial_id"], "vial_id")
    for name in ("calibration_sha256", "recipe_artifact_sha256", "source_artifact_sha256", "receipt_sha256"):
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            raise FillStepTrackerError(f"{name} is invalid")
    body = dict(value)
    claimed = body.pop("receipt_sha256")
    if _canonical_sha256(body) != claimed:
        raise FillStepTrackerError("fill-step receipt hash mismatch")
    stages_value = value["stages"]
    if not isinstance(stages_value, list) or not stages_value:
        raise FillStepTrackerError("fill-step receipt stages are missing")
    stage_fields = {
        "fill_step", "tank_id", "tank_number", "stream_step_index",
        "expected_dispense_ml", "expected_cumulative_ml",
        "twin_geometry_equivalent_volume_ml", "measured_volume_ml",
        "agreement", "tank_cluster_consistent", "stream_lab",
        "stream_spread_lab", "images",
    }
    first_tank = (
        stages_value[0].get("tank_id")
        if isinstance(stages_value[0], Mapping)
        else None
    )
    if first_tank not in ACTIVE_TANK_CYCLE:
        raise FillStepTrackerError("fill-step start tank is invalid")
    start_offset = ACTIVE_TANK_CYCLE.index(first_tank)
    previous_cumulative = 0.0
    for ordinal, stage in enumerate(stages_value, start=1):
        if not isinstance(stage, Mapping) or set(stage) != stage_fields:
            raise FillStepTrackerError("fill-step stage fields are not closed")
        if stage["fill_step"] != ordinal or stage["stream_step_index"] != ordinal:
            raise FillStepTrackerError("fill-step stage order is invalid")
        expected_tank = ACTIVE_TANK_CYCLE[(start_offset + ordinal - 1) % 3]
        if stage["tank_id"] != expected_tank or stage["tank_number"] != int(expected_tank[1:]):
            raise FillStepTrackerError("fill-step tank cycle is invalid")
        if stage["agreement"] is not True or type(stage["tank_cluster_consistent"]) is not bool:
            raise FillStepTrackerError("fill-step agreement fields are invalid")
        for name in ("expected_dispense_ml", "expected_cumulative_ml", "twin_geometry_equivalent_volume_ml", "measured_volume_ml"):
            _positive_finite(stage[name], name)
        expected_cumulative = previous_cumulative + float(stage["expected_dispense_ml"])
        if float(stage["expected_cumulative_ml"]) != expected_cumulative:
            raise FillStepTrackerError("expected cumulative recipe volume is inconsistent")
        previous_cumulative = expected_cumulative
        if stage["measured_volume_ml"] != stage["twin_geometry_equivalent_volume_ml"]:
            raise FillStepTrackerError("measured volume alias is inconsistent")
        _triplet(stage["stream_lab"], "stream_lab")
        _triplet(stage["stream_spread_lab"], "stream_spread_lab", nonnegative=True)
        images = stage["images"]
        if not isinstance(images, Mapping) or set(images) != {"early", "mid", "end"}:
            raise FillStepTrackerError("stage image roles are invalid")
        for image_id in images.values():
            _identifier(image_id, "image ID")
    return json.loads(json.dumps(value, allow_nan=False))


def development_support_summary(
    *, total_vial_ids: Sequence[str], before_supported_vial_ids: Sequence[str],
    after_supported_vial_ids: Sequence[str],
) -> dict[str, object]:
    """Publish exact before/after supported-vial identities and counts."""

    total = tuple(total_vial_ids)
    before = tuple(before_supported_vial_ids)
    after = tuple(after_supported_vial_ids)
    if (
        len(set(total)) != len(total)
        or len(set(before)) != len(before)
        or len(set(after)) != len(after)
        or set(before) - set(total)
        or set(after) - set(total)
    ):
        raise FillStepTrackerError("development support identities are invalid")
    return {
        "schema_version": "wp7-fill-step-support-summary-v1",
        "total_vial_ids": list(total),
        "before_supported_vial_ids": list(before),
        "after_supported_vial_ids": list(after),
        "total_vial_count": len(total),
        "before_supported_vial_count": len(before),
        "after_supported_vial_count": len(after),
    }


def _completed_stage_segments(
    frames: tuple[FrameEvidence, ...],
    pours: tuple[RecipePour, ...],
    freeze: TrackerFreeze,
    *,
    previous_final_tank_id: str,
) -> list[list[FrameEvidence]]:
    """Derive recipe-step groups from one global debounced camera-state sequence."""

    colour_segments: list[list[FrameEvidence]] = []
    for frame in frames:
        if not colour_segments:
            colour_segments.append([frame])
            continue
        centre = _coordinate_median(item.stream_lab for item in colour_segments[-1])
        if math.dist(frame.stream_lab, centre) >= freeze.stream_change_delta_e76:
            colour_segments.append([frame])
        else:
            colour_segments[-1].append(frame)

    stable_runs: list[tuple[str, list[FrameEvidence]]] = []
    for segment in colour_segments:
        if len(segment) < freeze.stream_minimum_consecutive_frames:
            continue
        centre = _coordinate_median(item.stream_lab for item in segment)
        distances = sorted(
            (math.dist(centre, tank_centre), tank_id)
            for tank_id, tank_centre in freeze.tank_lab_centres.items()
        )
        if len(distances) > 1 and distances[0][0] == distances[1][0]:
            raise FillStepTrackerError("stream Lab tank classification is tied")
        tank_id = distances[0][1]
        if stable_runs and stable_runs[-1][0] == tank_id:
            stable_runs[-1][1].extend(segment)
        else:
            stable_runs.append((tank_id, list(segment)))
    if stable_runs and stable_runs[0][0] == previous_final_tank_id:
        stable_runs = stable_runs[1:]
    if not stable_runs:
        raise FillStepTrackerError("camera stream never reaches a recipe tank state")
    if len(stable_runs) > len(pours):
        raise FillStepTrackerError("camera stream states exceed recipe order")
    expected = [pour.tank_id for pour in pours[: len(stable_runs)]]
    observed = [tank_id for tank_id, _ in stable_runs]
    if observed != expected:
        raise FillStepTrackerError("debounced camera stream states do not match recipe order")
    return [run_frames for _, run_frames in stable_runs]


def _validate_freeze(value: object) -> TrackerFreeze:
    if not isinstance(value, TrackerFreeze):
        raise FillStepTrackerError("tracker freeze is missing or invalid")
    if value.stream_roi_xyxy != FROZEN_STREAM_ROI_XYXY:
        raise FillStepTrackerError("frozen upper-stream ROI drifted")
    if type(value.stream_rgb_change_threshold) is not int or not 0 < value.stream_rgb_change_threshold <= 255:
        raise FillStepTrackerError("stream RGB change threshold is invalid")
    if type(value.stream_minimum_changed_pixels) is not int or value.stream_minimum_changed_pixels <= 0:
        raise FillStepTrackerError("stream changed-pixel minimum is invalid")
    if (
        type(value.stream_minimum_consecutive_frames) is not int
        or value.stream_minimum_consecutive_frames <= 0
    ):
        raise FillStepTrackerError("stream debounce frame minimum is invalid")
    for name in (
        "stream_change_delta_e76", "tank_cluster_delta_e76",
        "volume_match_tolerance_ml",
    ):
        _positive_finite(getattr(value, name), f"freeze {name}")
    _validate_calibration(value.calibration)
    if value.volume_curve_sha256 != FROZEN_VOLUME_CURVE_SHA256:
        raise FillStepTrackerError("frozen volume curve binding drifted")
    if _SHA256.fullmatch(value.recipe_artifact_sha256) is None:
        raise FillStepTrackerError("recipe artifact binding is invalid")
    if not isinstance(value.tank_lab_centres, Mapping) or set(value.tank_lab_centres) != set(ACTIVE_TANK_CYCLE):
        raise FillStepTrackerError("freeze requires one Lab centre per active tank")
    for tank, centre in value.tank_lab_centres.items():
        _triplet(centre, f"{tank} Lab centre")
    return value


def _validate_calibration(value: object) -> PixelToMetreCalibration:
    if not isinstance(value, PixelToMetreCalibration):
        raise FillStepTrackerError("pixel-to-metre calibration is missing")
    if value.schema_version != CALIBRATION_SCHEMA or not value.calibration_id:
        raise FillStepTrackerError("pixel-to-metre calibration schema is invalid")
    if value.camera_effective_state_sha256 != FROZEN_FILLCAM_CAMERA_SHA256:
        raise FillStepTrackerError("frozen camera calibration drifted")
    if value.geometry_sha256 != FROZEN_GEOMETRY_SHA256:
        raise FillStepTrackerError("frozen calibration geometry drifted")
    if value.volume_curve_sha256 != FROZEN_VOLUME_CURVE_SHA256:
        raise FillStepTrackerError("frozen calibration volume curve drifted")
    if value.artifact_sha256 != calibration_payload_sha256(value):
        raise FillStepTrackerError("calibration artifact hash mismatch")
    vectors = (
        value.fill_height_px_samples,
        value.visible_height_m_samples,
        value.twin_geometry_equivalent_volume_ml_samples,
    )
    if any(not isinstance(vector, tuple) for vector in vectors):
        raise FillStepTrackerError("calibration samples must be frozen tuples")
    if len({len(vector) for vector in vectors}) != 1 or len(vectors[0]) < 2:
        raise FillStepTrackerError("calibration needs aligned sample vectors")
    for vector in vectors:
        for item in vector:
            _positive_finite(item, "calibration sample")
        if any(right <= left for left, right in zip(vector, vector[1:])):
            raise FillStepTrackerError("calibration samples must be strictly increasing")
    return value


def _validate_pours(value: object) -> tuple[RecipePour, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise FillStepTrackerError("recipe pours must be a non-empty sequence")
    pours = tuple(value)
    if not isinstance(pours[0], RecipePour) or pours[0].tank_id not in ACTIVE_TANK_CYCLE:
        raise FillStepTrackerError("recipe start tank is not in the active-tank cycle")
    start_offset = ACTIVE_TANK_CYCLE.index(pours[0].tank_id)
    for index, pour in enumerate(pours):
        if not isinstance(pour, RecipePour):
            raise FillStepTrackerError("recipe pour has invalid type")
        if pour.tank_id != ACTIVE_TANK_CYCLE[(start_offset + index) % 3]:
            raise FillStepTrackerError(
                "recipe pours must follow the exact active-tank cycle; T300 is excluded"
            )
        _positive_finite(pour.dispense_ml, "recipe dispense amount")
    return pours


def _validate_frames(value: object) -> tuple[FrameEvidence, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise FillStepTrackerError("frame evidence must be a non-empty sequence")
    frames = tuple(value)
    previous = -1
    for frame in frames:
        if not isinstance(frame, FrameEvidence):
            raise FillStepTrackerError("frame evidence has invalid type")
        if type(frame.frame_index) is not int or frame.frame_index <= previous:
            raise FillStepTrackerError("frame indices must be strictly increasing integers")
        previous = frame.frame_index
        _triplet(frame.stream_lab, "stream_lab")
        _triplet(frame.stream_spread_lab, "stream_spread_lab", nonnegative=True)
        if frame.visible_height_px is not None:
            _positive_finite(frame.visible_height_px, "CNN fill height")
    return frames


def _interpolate_calibrated_volume(
    height_px: float, calibration: PixelToMetreCalibration
) -> float:
    _validate_calibration(calibration)
    xs = calibration.fill_height_px_samples
    ys = calibration.twin_geometry_equivalent_volume_ml_samples
    if height_px < xs[0] or height_px > xs[-1]:
        raise FillStepTrackerError("CNN fill height is outside frozen calibration domain")
    for left_x, right_x, left_y, right_y in zip(xs, xs[1:], ys, ys[1:]):
        if height_px <= right_x:
            fraction = (height_px - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y)
    raise AssertionError("validated interpolation domain was not covered")


def _positive_finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FillStepTrackerError(f"{role} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise FillStepTrackerError(f"{role} must be finite and positive")
    return number


def _triplet(value: object, role: str, *, nonnegative: bool = False) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise FillStepTrackerError(f"{role} must contain three finite channels")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FillStepTrackerError(f"{role} must contain three finite channels")
        number = float(item)
        if not math.isfinite(number) or (nonnegative and number < 0.0):
            raise FillStepTrackerError(f"{role} must contain three finite channels")
        converted.append(number)
    return converted[0], converted[1], converted[2]


def _coordinate_median(values: object) -> tuple[float, float, float]:
    rows = tuple(values)  # type: ignore[arg-type]
    return tuple(median(row[index] for row in rows) for index in range(3))  # type: ignore[return-value]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _identifier(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or any(token in value for token in ("/", "\\", "..")):
        raise FillStepTrackerError(f"{role} must be an opaque non-path string")
    return value
