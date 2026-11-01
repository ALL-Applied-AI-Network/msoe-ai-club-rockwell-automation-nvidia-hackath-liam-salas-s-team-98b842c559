"""Frozen, camera-only WP7 fill-step production consumers.

This module consumes reviewed artifacts.  It does not train, tune, import Part-A
truth, or infer liquid state from programmed recipe amounts.  Recipe amounts
enter only the final independent volume/stream cross-check in
``track_fill_step``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

from .empty_reference_rgb_predictor import CLAMP_ROWS, IMAGE_SHAPE, fillcam_masks
from .wp7_fill_step_tracker import (
    ACTIVE_TANK_CYCLE,
    FROZEN_STREAM_ROI_XYXY,
    FrameEvidence,
    PixelToMetreCalibration,
    RecipePour,
    StableVolumeWindow,
    TrackerFreeze,
    build_fill_step_receipt,
    calibration_payload_sha256,
    track_fill_step,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOPOLOGY_CONFIG_FIELDS = {
    "dense_row_fraction", "sustained_dense_rows", "bottom_zone_start_row",
    "lower_edge_support_rows", "upper_sustained_dense_rows",
}
_CALIBRATION_FIELDS = set(PixelToMetreCalibration.__dataclass_fields__)
class ProductionFillStepError(ValueError):
    """A production artifact or camera observation failed closed."""


@dataclass(frozen=True)
class TopologyFreeze:
    artifact_sha256: str
    receipt_sha256: str
    dense_row_fraction: float
    sustained_dense_rows: int
    bottom_zone_start_row: int
    lower_edge_support_rows: int
    upper_sustained_dense_rows: int

    def parameters(self) -> dict[str, object]:
        return {
            key: getattr(self, key)
            for key in sorted(_TOPOLOGY_CONFIG_FIELDS)
        }


@dataclass(frozen=True)
class MaskTopologyResult:
    mask: np.ndarray
    surface_row: int | None
    bottom_row: int | None
    reason_code: str


@dataclass(frozen=True)
class TopologyHeightMeasurement:
    status: str
    surface_row: int | None
    visible_height_px: int | None
    surface_source: str


@dataclass(frozen=True)
class CompletedPhaseObservation:
    image_id: str
    frame_index: int
    phase: str
    rgb: np.ndarray
    mask: np.ndarray


def _canonical_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ProductionFillStepError("artifact is not canonical JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionFillStepError(f"{role} must be a lowercase SHA-256")
    return value


def _read_bound_json(path: Path, expected_sha256: str, role: str) -> dict[str, object]:
    expected = _sha(expected_sha256, f"expected {role} SHA-256")
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ProductionFillStepError(f"{role} must be a regular file") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionFillStepError(f"{role} must be a regular non-symlink file")
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ProductionFillStepError(f"{role} raw-file SHA-256 mismatch")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionFillStepError(f"{role} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProductionFillStepError(f"{role} must contain an object")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionFillStepError(f"duplicate artifact field: {key}")
        result[key] = value
    return result


def load_topology_freeze(path: Path, *, expected_sha256: str) -> TopologyFreeze:
    """Load the reviewed train-only topology freeze by exact file hash."""

    artifact = _read_bound_json(path, expected_sha256, "topology freeze")
    expected_fields = {
        "artifact_type", "claim_boundary", "device", "grid_candidate_count",
        "grid_scores", "receipt_sha256", "replay_equivalence", "rule",
        "runtime_rule_input_contract",
        "same_stage_limitation", "schema_version", "selected_config",
        "selected_train_score", "selection_objective", "source_bindings",
        "source_commit", "threshold_grid", "train_sample_count",
        "train_sample_provenance", "train_selection_input_contract",
        "train_vial_count", "train_vials_with_stable_window_samples",
        "train_vials_without_stable_window_samples", "validation_access",
    }
    if set(artifact) != expected_fields:
        raise ProductionFillStepError("topology freeze fields are not closed")
    if (
        artifact["schema_version"] != "wp7-cnn-mask-topology-freeze-v1"
        or artifact["artifact_type"]
        != "development_train_only_topology_threshold_freeze"
    ):
        raise ProductionFillStepError("topology freeze identity is invalid")
    runtime = artifact["runtime_rule_input_contract"]
    access = artifact["validation_access"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"allowed", "forbidden_runtime_inputs"}
        or runtime["allowed"] != [
            "frozen_cnn_boolean_mask", "fixed_vial_roi", "fixed_clamp_raster"
        ]
        or "recipe_ml" not in runtime["forbidden_runtime_inputs"]
        or not isinstance(access, Mapping)
        or access.get("selection_partition") != "train"
        or access.get("validation_rgb_decoded") is not False
        or access.get("validation_semantic_masks_opened") is not False
    ):
        raise ProductionFillStepError("topology freeze runtime boundary is invalid")
    receipt = _sha(artifact["receipt_sha256"], "topology receipt SHA-256")
    body = dict(artifact)
    body.pop("receipt_sha256")
    if _canonical_hash(body) != receipt:
        raise ProductionFillStepError("topology freeze receipt hash mismatch")
    config = artifact["selected_config"]
    if not isinstance(config, Mapping) or set(config) != _TOPOLOGY_CONFIG_FIELDS:
        raise ProductionFillStepError("topology freeze selected config is not closed")
    result = TopologyFreeze(
        artifact_sha256=_sha(expected_sha256, "topology artifact SHA-256"),
        receipt_sha256=receipt,
        dense_row_fraction=float(config["dense_row_fraction"]),
        sustained_dense_rows=config["sustained_dense_rows"],
        bottom_zone_start_row=config["bottom_zone_start_row"],
        lower_edge_support_rows=config["lower_edge_support_rows"],
        upper_sustained_dense_rows=config["upper_sustained_dense_rows"],
    )
    # The topology function owns the complete numeric validation.
    postprocess_cnn_mask_topology(
        np.zeros(IMAGE_SHAPE, dtype=bool), **result.parameters()
    )
    return result


def load_pixel_to_metre_calibration(
    path: Path, *, expected_sha256: str
) -> PixelToMetreCalibration:
    artifact = _read_bound_json(path, expected_sha256, "pixel calibration")
    if set(artifact) != _CALIBRATION_FIELDS:
        raise ProductionFillStepError("pixel calibration fields are not closed")
    try:
        calibration = PixelToMetreCalibration(
            **{
                **artifact,
                "fill_height_px_samples": tuple(artifact["fill_height_px_samples"]),
                "visible_height_m_samples": tuple(artifact["visible_height_m_samples"]),
                "twin_geometry_equivalent_volume_ml_samples": tuple(
                    artifact["twin_geometry_equivalent_volume_ml_samples"]
                ),
            }
        )
    except (TypeError, ValueError) as error:
        raise ProductionFillStepError("pixel calibration payload is invalid") from error
    if calibration.artifact_sha256 != calibration_payload_sha256(calibration):
        raise ProductionFillStepError("pixel calibration payload hash mismatch")
    return calibration


def load_tracker_freeze(
    materialization_receipt_path: Path,
    *,
    expected_materialization_sha256: str,
    train_freeze_path: Path,
    expected_train_freeze_sha256: str,
    topology: TopologyFreeze,
    calibration: PixelToMetreCalibration,
    target_recipe_artifact_sha256: str,
) -> TrackerFreeze:
    """Load the accepted attempt04 freeze without reconstructing tuned values."""

    train = _read_bound_json(
        train_freeze_path, expected_train_freeze_sha256, "attempt04 train freeze"
    )
    train_fields = {
        "receipt_sha256", "schema_version", "selected_volume_tolerance_ml",
        "source_commit", "topology_config", "topology_freeze_path",
        "topology_freeze_sha256", "validation_rgb_decoded_before_freeze",
        "volume_tolerance_train_scores",
    }
    if set(train) != train_fields:
        raise ProductionFillStepError("attempt04 train freeze fields are not closed")
    if (
        train["schema_version"] != "wp7-fill-step-attempt04-train-freeze-v1"
        or train["source_commit"]
        != "4ab6d9b198e6d079165b9e40c21361e3cd745065"
        or train["validation_rgb_decoded_before_freeze"] is not False
    ):
        raise ProductionFillStepError("attempt04 train freeze identity is invalid")
    claimed = _sha(train["receipt_sha256"], "attempt04 train freeze receipt SHA-256")
    body = dict(train)
    body.pop("receipt_sha256")
    if _canonical_hash(body) != claimed:
        raise ProductionFillStepError("attempt04 train freeze receipt hash mismatch")
    if (
        train["topology_config"] != topology.parameters()
        or train["topology_freeze_sha256"] != topology.receipt_sha256
    ):
        raise ProductionFillStepError("attempt04 topology binding mismatch")

    artifact = _read_bound_json(
        materialization_receipt_path,
        expected_materialization_sha256,
        "attempt04 materialization receipt",
    )
    materialization_fields = {
        "artifact_type", "claim_boundary", "device", "excluded_partitions",
        "execution", "partition_counts", "per_vial", "receipt_sha256",
        "schema_version", "source_bindings", "source_commit", "support_summary",
        "supported_fill_step_receipts", "train_only_freeze",
    }
    if set(artifact) != materialization_fields:
        raise ProductionFillStepError("materialization receipt fields are not closed")
    if (
        artifact["schema_version"]
        != "wp7-fill-step-development-materialization-v1"
        or artifact["artifact_type"]
        != "development_only_v3_v4_fill_step_comparison"
        or artifact["source_commit"]
        != "4ab6d9b198e6d079165b9e40c21361e3cd745065"
    ):
        raise ProductionFillStepError("materialization receipt identity is invalid")
    receipt_hash = _sha(
        artifact["receipt_sha256"], "materialization receipt SHA-256"
    )
    receipt_body = dict(artifact)
    receipt_body.pop("receipt_sha256")
    if _canonical_hash(receipt_body) != receipt_hash:
        raise ProductionFillStepError("materialization receipt hash mismatch")
    tracker = artifact["train_only_freeze"]
    expected_freeze_fields = {
        "calibration_rule", "calibration_sample_count", "calibration_sha256",
        "rgb_change_threshold_grid", "selected_minimum_changed_pixels",
        "selected_minimum_consecutive_frames", "selected_rgb_change_threshold",
        "selected_stream_change_delta_e76", "selected_volume_tolerance_ml",
        "stable_volume_window", "tank_cluster_delta_e76",
        "tank_cluster_threshold_rule", "tank_lab_centres", "topology_config",
        "topology_freeze_sha256", "train_stable_segment_centre_distance",
        "train_vial_count", "volume_tolerance_train_scores",
    }
    if not isinstance(tracker, Mapping) or set(tracker) != expected_freeze_fields:
        raise ProductionFillStepError("accepted tracker freeze fields are not closed")
    if (
        tracker["calibration_sha256"] != calibration.artifact_sha256
        or tracker["topology_config"] != topology.parameters()
        or tracker["topology_freeze_sha256"] != topology.receipt_sha256
        or tracker["selected_volume_tolerance_ml"]
        != train["selected_volume_tolerance_ml"]
        or tracker["volume_tolerance_train_scores"]
        != train["volume_tolerance_train_scores"]
    ):
        raise ProductionFillStepError("accepted freeze binding mismatch")
    stable_window = tracker["stable_volume_window"]
    expected_stable_window = {
        "mask_topology_policy": "sealed_camera_only_cnn_mask_topology_freeze",
        "maximum_height_and_surface_delta_px": 5.0,
        "selection": "last_three_consecutive_frames_of_camera_state_tail",
        "window_length": 3,
    }
    if (
        not isinstance(stable_window, Mapping)
        or dict(stable_window) != expected_stable_window
        or type(stable_window.get("maximum_height_and_surface_delta_px"))
        is not float
        or type(stable_window.get("window_length")) is not int
    ):
        raise ProductionFillStepError("accepted stable volume window drifted")
    receipts = artifact["supported_fill_step_receipts"]
    if not isinstance(receipts, Mapping) or not receipts:
        raise ProductionFillStepError("accepted receipt set is missing")
    recipe_hashes = {
        row.get("recipe_artifact_sha256")
        for row in receipts.values() if isinstance(row, Mapping)
    }
    if len(recipe_hashes) != 1:
        raise ProductionFillStepError("accepted recipe binding is ambiguous")
    _sha(recipe_hashes.pop(), "development recipe artifact SHA-256")
    recipe_hash = _sha(
        target_recipe_artifact_sha256, "target recipe artifact SHA-256"
    )
    centres = tracker["tank_lab_centres"]
    if not isinstance(centres, Mapping) or set(centres) != set(ACTIVE_TANK_CYCLE):
        raise ProductionFillStepError("tracker tank Lab centres are not closed")
    try:
        freeze = TrackerFreeze(
            stream_roi_xyxy=FROZEN_STREAM_ROI_XYXY,
            stream_rgb_change_threshold=tracker["selected_rgb_change_threshold"],
            stream_minimum_changed_pixels=tracker["selected_minimum_changed_pixels"],
            stream_minimum_consecutive_frames=tracker[
                "selected_minimum_consecutive_frames"
            ],
            stream_change_delta_e76=float(
                tracker["selected_stream_change_delta_e76"]
            ),
            tank_cluster_delta_e76=float(tracker["tank_cluster_delta_e76"]),
            volume_match_tolerance_ml=float(
                tracker["selected_volume_tolerance_ml"]
            ),
            calibration=calibration,
            tank_lab_centres={key: tuple(centres[key]) for key in ACTIVE_TANK_CYCLE},
            recipe_artifact_sha256=recipe_hash,
            volume_curve_sha256=calibration.volume_curve_sha256,
            stable_volume_window=StableVolumeWindow(**stable_window),
        )
        # Reuse the production tracker's complete freeze validator without a
        # recipe-dependent measurement decision.
        from .wp7_fill_step_tracker import _validate_freeze
        return _validate_freeze(freeze)
    except (TypeError, ValueError) as error:
        raise ProductionFillStepError("tracker freeze payload is invalid") from error


def postprocess_cnn_mask_topology(
    mask: object,
    *,
    dense_row_fraction: float,
    sustained_dense_rows: int,
    bottom_zone_start_row: int,
    lower_edge_support_rows: int,
    upper_sustained_dense_rows: int,
) -> MaskTopologyResult:
    """Keep supported CNN pixels and reject narrow/disconnected mask spurs."""

    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.shape != IMAGE_SHAPE:
        raise ProductionFillStepError("topology mask is invalid")
    if (
        isinstance(dense_row_fraction, bool)
        or not isinstance(dense_row_fraction, (int, float))
        or not math.isfinite(float(dense_row_fraction))
        or not 0.0 < float(dense_row_fraction) <= 1.0
        or type(sustained_dense_rows) is not int
        or sustained_dense_rows <= 0
        or type(bottom_zone_start_row) is not int
        or not CLAMP_ROWS[1] < bottom_zone_start_row < IMAGE_SHAPE[0]
        or type(lower_edge_support_rows) is not int
        or lower_edge_support_rows <= 0
        or type(upper_sustained_dense_rows) is not int
        or upper_sustained_dense_rows <= 0
    ):
        raise ProductionFillStepError("topology config is invalid")
    from scipy import ndimage

    roi, clamp = fillcam_masks()
    candidates = values & roi & ~clamp
    roi_widths = np.count_nonzero(roi, axis=1)

    def dense_rows(component: np.ndarray) -> np.ndarray:
        counts = np.count_nonzero(component, axis=1)
        return (roi_widths > 0) & (
            counts >= np.ceil(roi_widths * float(dense_row_fraction))
        )

    def first_run(rows: np.ndarray, length: int) -> int | None:
        for start in np.flatnonzero(rows):
            if start + length <= len(rows) and np.all(rows[start:start + length]):
                return int(start)
        return None

    lower_start = CLAMP_ROWS[1] + 1
    lower_candidates = candidates.copy()
    lower_candidates[:lower_start] = False
    labels, count = ndimage.label(
        lower_candidates, structure=np.ones((3, 3), dtype=np.uint8)
    )
    eligible: list[tuple[int, int, np.ndarray]] = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        rows = np.flatnonzero(np.any(component, axis=1))
        if rows.size and int(rows[-1]) >= bottom_zone_start_row:
            eligible.append((int(np.count_nonzero(component)), int(rows[-1]), component))
    if not eligible:
        return MaskTopologyResult(
            np.zeros(IMAGE_SHAPE, dtype=bool), None, None,
            "bottom_zone_component_missing",
        )
    lower = max(eligible, key=lambda item: (item[0], item[1]))[2]
    lower_dense = dense_rows(lower)
    lower_surface = first_run(lower_dense, sustained_dense_rows)
    if lower_surface is None:
        return MaskTopologyResult(
            np.zeros(IMAGE_SHAPE, dtype=bool), None, None,
            "lower_sustained_dense_surface_missing",
        )
    selected = lower.copy()
    selected[:lower_surface] = False
    bottom = int(np.flatnonzero(np.any(selected, axis=1))[-1])
    upper_selected: np.ndarray | None = None
    upper_surface: int | None = None
    edge_stop = lower_start + lower_edge_support_rows
    if edge_stop <= IMAGE_SHAPE[0] and np.all(lower_dense[lower_start:edge_stop]):
        upper_candidates = candidates.copy()
        upper_candidates[CLAMP_ROWS[0]:] = False
        upper_labels, upper_count = ndimage.label(
            upper_candidates, structure=np.ones((3, 3), dtype=np.uint8)
        )
        upper: list[tuple[int, int, np.ndarray]] = []
        for label_id in range(1, upper_count + 1):
            component = upper_labels == label_id
            surface = first_run(dense_rows(component), upper_sustained_dense_rows)
            if surface is not None:
                upper.append((int(np.count_nonzero(component)), surface, component))
        if upper:
            _, upper_surface, upper_selected = max(
                upper, key=lambda item: (item[0], -item[1])
            )
            upper_selected = upper_selected.copy()
            upper_selected[:upper_surface] = False
            selected |= upper_selected
    selected &= ~clamp
    if upper_selected is not None and upper_surface is not None:
        return MaskTopologyResult(
            selected, upper_surface, bottom, "upper_and_lower_bodies_selected"
        )
    return MaskTopologyResult(selected, lower_surface, bottom, "lower_body_selected")


def apply_camera_hidden_band_surface(
    topology: MaskTopologyResult,
    *,
    temporal_status: str,
    temporal_surface_row: int | None,
) -> TopologyHeightMeasurement:
    """Apply only the accepted RGB-temporal clamp-hidden surface override."""

    if topology.surface_row is None or topology.bottom_row is None:
        return TopologyHeightMeasurement(
            topology.reason_code, None, None, "unavailable"
        )
    surface = topology.surface_row
    status = "visible_measured"
    source = "cnn_topology_dense_row"
    if temporal_status == "clamp_hidden_inferred":
        if (
            type(temporal_surface_row) is not int
            or not CLAMP_ROWS[0] <= temporal_surface_row <= CLAMP_ROWS[1]
        ):
            raise ProductionFillStepError("hidden-band surface is invalid")
        surface = temporal_surface_row
        status = temporal_status
        source = "rgb_temporal_hidden_band"
    return TopologyHeightMeasurement(
        status, surface, topology.bottom_row - surface + 1, source
    )


def measure_stream_lab(
    current_rgb: object,
    reference_rgb: object,
    freeze: TrackerFreeze,
) -> tuple[bool, int, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Measure changed stream pixels inside the frozen pixel ROI only."""

    current = _rgb(current_rgb)
    reference = _rgb(reference_rgb)
    x0, y0, x1, y1 = freeze.stream_roi_xyxy
    if tuple(freeze.stream_roi_xyxy) != FROZEN_STREAM_ROI_XYXY:
        raise ProductionFillStepError("stream ROI differs from frozen contract")
    roi = current[y0:y1, x0:x1]
    ref = reference[y0:y1, x0:x1]
    changed = np.max(
        np.abs(roi.astype(np.int16) - ref.astype(np.int16)), axis=2
    ) > freeze.stream_rgb_change_threshold
    count = int(np.count_nonzero(changed))
    if count < freeze.stream_minimum_changed_pixels:
        return False, count, None, None
    import cv2
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    values = lab[changed]
    centre = np.median(values, axis=0)
    spread = np.median(np.abs(values - centre), axis=0)
    return (
        True, count, tuple(float(item) for item in centre),
        tuple(float(item) for item in spread),
    )


class IncrementalFillStepTracker:
    """Incremental observer that emits completed early/mid/end camera stages."""

    def __init__(
        self,
        *,
        freeze: TrackerFreeze,
        topology: TopologyFreeze,
        recipe_pours: Sequence[RecipePour],
        previous_final_tank_id: str,
        same_run_empty_reference_rgb: object,
        predictor: Callable[[np.ndarray, np.ndarray], object],
        cohort_id: str,
        case_id: str,
        vial_id: str,
        vial_position: int,
        source_artifact_sha256: str | None,
        vlm_cohort_context: Mapping[str, object] | None = None,
        vlm_vial: Mapping[str, object] | None = None,
        allow_unavailable_selected_phase: bool = False,
        image_index_width: int = 6,
        defer_source_artifact_binding: bool = False,
        frame_rate_hz: float = 10.0,
    ) -> None:
        self.freeze = freeze
        self.topology = topology
        self.recipe_pours = tuple(recipe_pours)
        self.previous_final_tank_id = previous_final_tank_id
        self.reference = _rgb(same_run_empty_reference_rgb).copy()
        self.predictor = predictor
        self.cohort_id = cohort_id
        if (
            isinstance(frame_rate_hz, bool)
            or not isinstance(frame_rate_hz, (int, float))
            or not math.isfinite(float(frame_rate_hz))
            or float(frame_rate_hz) <= 0.0
        ):
            raise ProductionFillStepError("frame rate must be finite and positive")
        self.frame_rate_hz = float(frame_rate_hz)
        if not isinstance(case_id, str) or not case_id:
            raise ProductionFillStepError("case ID is invalid")
        self.case_id = case_id
        self.vial_id = vial_id
        self.vial_position = vial_position
        if type(defer_source_artifact_binding) is not bool:
            raise ProductionFillStepError("source artifact binding policy is invalid")
        if defer_source_artifact_binding:
            if source_artifact_sha256 is not None:
                raise ProductionFillStepError("deferred source artifact must start unbound")
            self.source_artifact_sha256: str | None = None
        else:
            self.source_artifact_sha256 = _sha(
                source_artifact_sha256, "source artifact SHA-256"
            )
        self.defer_source_artifact_binding = defer_source_artifact_binding
        if type(allow_unavailable_selected_phase) is not bool:
            raise ProductionFillStepError("selected-phase policy is invalid")
        if type(image_index_width) is not int or image_index_width not in {4, 6}:
            raise ProductionFillStepError("image index width is invalid")
        self.allow_unavailable_selected_phase = allow_unavailable_selected_phase
        self.image_index_width = image_index_width
        if (vlm_cohort_context is None) != (vlm_vial is None):
            raise ProductionFillStepError(
                "VLM cohort context and vial template must be supplied together"
            )
        if vlm_vial is not None and vlm_vial.get("vial_id") != vial_id:
            raise ProductionFillStepError("VLM vial template identity differs")
        self.vlm_cohort_context = vlm_cohort_context
        self.vlm_vial = vlm_vial
        self._frames: list[FrameEvidence] = []
        self._captures: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._topology_by_frame: dict[int, MaskTopologyResult] = {}
        self._height_by_frame: dict[int, TopologyHeightMeasurement] = {}
        self._sequence_frames: list[np.ndarray] = [self.reference]
        self._sequence_indices: list[int | None] = [None]
        self._emitted_steps = 0
        self._last_frame_index = -1

    def bind_source_artifact_sha256(self, value: object) -> None:
        """Bind one rolling source-window identity before the terminal close."""

        if not self.defer_source_artifact_binding:
            raise ProductionFillStepError("source artifact binding is not deferred")
        if self.source_artifact_sha256 is not None:
            raise ProductionFillStepError("source artifact is already bound")
        self.source_artifact_sha256 = _sha(value, "source artifact SHA-256")

    def observe(
        self, frame: object, *, frame_index: int, elapsed_seconds: float
    ) -> tuple[Mapping[str, object], ...]:
        """Observe one RGB frame and retain evidence for close-time replay."""

        if type(frame_index) is not int or frame_index <= self._last_frame_index:
            raise ProductionFillStepError("frame indices must increase")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0.0
        ):
            raise ProductionFillStepError("elapsed seconds is invalid")
        self._last_frame_index = frame_index
        rgb = _rgb(frame)
        self._sequence_frames.append(rgb.copy())
        self._sequence_indices.append(frame_index)
        active, _count, centre, spread = measure_stream_lab(
            rgb, self.reference, self.freeze
        )
        if not active or centre is None or spread is None:
            return ()
        prediction = self.predictor(rgb, self.reference)
        topology = postprocess_cnn_mask_topology(
            getattr(prediction, "mask", None), **self.topology.parameters()
        )
        height = apply_camera_hidden_band_surface(
            topology,
            temporal_status="visible_measured",
            temporal_surface_row=None,
        )
        self._frames.append(FrameEvidence(
            frame_index, centre, spread,
            (
                None
                if height.visible_height_px is None
                else float(height.visible_height_px)
            ),
            None if height.surface_row is None else float(height.surface_row),
        ))
        self._captures[frame_index] = (rgb.copy(), topology.mask.copy())
        self._topology_by_frame[frame_index] = topology
        self._height_by_frame[frame_index] = height
        return ()

    def _replay_temporal_heights(self) -> None:
        """Apply the temporal authority once to the complete closed vial."""

        from .empty_reference_rgb_predictor import predict_fill_sequence
        temporal = predict_fill_sequence(
            self._sequence_frames,
            reference_index=0,
            expected_reference_rgb_sha256=hashlib.sha256(
                self.reference.tobytes()
            ).hexdigest(),
            programmed_ml=None,
            frame_rate_hz=self.frame_rate_hz,
        )
        by_frame = {item.frame_index: item for item in self._frames}
        for sequence_index, temporal_item in zip(
            self._sequence_indices, temporal, strict=True
        ):
            if sequence_index not in by_frame or sequence_index not in self._topology_by_frame:
                continue
            corrected = apply_camera_hidden_band_surface(
                self._topology_by_frame[sequence_index],
                temporal_status=temporal_item.status,
                temporal_surface_row=temporal_item.surface_row,
            )
            self._height_by_frame[sequence_index] = corrected
            old = by_frame[sequence_index]
            by_frame[sequence_index] = FrameEvidence(
                old.frame_index, old.stream_lab, old.stream_spread_lab,
                (
                    None
                    if corrected.visible_height_px is None
                    else float(corrected.visible_height_px)
                ),
                (
                    None
                    if corrected.surface_row is None
                    else float(corrected.surface_row)
                ),
            )
        self._frames = [by_frame[item.frame_index] for item in self._frames]

    def close(self) -> tuple[Mapping[str, object], ...]:
        """Close the vial so its final stable stream segment has a real end."""

        if self.source_artifact_sha256 is None:
            raise ProductionFillStepError("source artifact is unbound at vial close")
        self._replay_temporal_heights()
        result = track_fill_step(
            self._frames, self.recipe_pours, self.freeze,
            previous_final_tank_id=self.previous_final_tank_id,
            frame_rate_hz=self.frame_rate_hz,
        )
        if not result.agreement or len(result.stages) != len(self.recipe_pours):
            raise ProductionFillStepError(
                "vial closed without complete tracker agreement"
            )
        return self._emit(
            result, completed_count=len(result.stages), closed=True
        )

    def _emit(
        self, result: object, *, completed_count: int, closed: bool
    ) -> tuple[Mapping[str, object], ...]:
        new_stages = result.stages[self._emitted_steps:completed_count]
        records: list[Mapping[str, object]] = []
        for stage in new_stages:
            selection = stage.selected_frames
            phase_indices = (
                ("early", selection.early_frame_index),
                ("mid", selection.mid_frame_index),
                ("end", selection.end_frame_index),
            )
            for phase, selected_index in phase_indices:
                captured_rgb, captured_mask = self._captures[selected_index]
                selected_height = self._height_by_frame[selected_index]
                if (
                    selected_height.surface_row is None
                    or selected_height.visible_height_px is None
                ) and not self.allow_unavailable_selected_phase:
                    raise ProductionFillStepError(
                        "selected phase lacks accepted CNN height evidence"
                    )
                image_id = (
                    f"{self.case_id}-F{selected_index:0{self.image_index_width}d}"
                )
                unavailable = (
                    selected_height.surface_row is None
                    or selected_height.visible_height_px is None
                )
                record: dict[str, object] = {
                    "vial_id": self.vial_id,
                    "vial_position": self.vial_position,
                    "fill_step": stage.fill_step,
                    "stage_id": f"STAGE-{self.vial_id}-{stage.fill_step}",
                    "tank_id": stage.tank_id,
                    "evidence_id": f"EVID-{image_id}",
                    "image_id": image_id,
                    "phase": phase,
                    "frame_index": selected_index,
                    "same_run_empty_reference_rgb": self.reference,
                    "observation_rgb": captured_rgb,
                    "prediction_mask": captured_mask,
                    "surface_row": selected_height.surface_row,
                    "visible_height_px": selected_height.visible_height_px,
                    "height_status": selected_height.status,
                    "agreement": stage.agreement,
                    "measured_volume_ml": stage.twin_geometry_equivalent_volume_ml,
                    "twin_geometry_equivalent_volume_ml": (
                        stage.twin_geometry_equivalent_volume_ml
                    ),
                    "expected_dispense_ml": stage.expected_dispense_ml,
                    "expected_cumulative_ml": stage.expected_cumulative_ml,
                    "stream_step_index": stage.stream_step_index,
                    "tank_cluster_consistent": stage.tank_cluster_consistent,
                    "limit_codes": (
                        [f"cnn_{selected_height.status}"] if unavailable else []
                    ),
                    "closed": False,
                }
                if self.vlm_cohort_context is not None:
                    record["vlm_cohort_context"] = self.vlm_cohort_context
                records.append(record)
        self._emitted_steps = completed_count
        if closed:
            if not records:
                raise ProductionFillStepError("tracker vial was already closed")
            image_ids = {
                record["frame_index"]: record["image_id"] for record in records
            }
            # Include earlier emitted stages as well.
            for stage in result.stages:
                for phase, index in (
                    ("early", stage.selected_frames.early_frame_index),
                    ("mid", stage.selected_frames.mid_frame_index),
                    ("end", stage.selected_frames.end_frame_index),
                ):
                    image_ids.setdefault(
                        index,
                        f"{self.case_id}-F{index:0{self.image_index_width}d}",
                    )
            receipt = build_fill_step_receipt(
                cohort_id=self.cohort_id,
                vial_id=self.vial_id,
                result=result,
                freeze=self.freeze,
                image_id_by_frame=image_ids,
                source_artifact_sha256=self.source_artifact_sha256,
            )
            final = dict(records[-1])
            final["closed"] = True
            final["tracker_receipt"] = receipt
            if self.vlm_vial is not None:
                final["vlm_vial"] = self.vlm_vial
            records[-1] = final
        return tuple(records)


def persist_completed_observations(
    *, output_root: Path, observations: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Exclusively persist native RGB/mask files and a trusted binding registry."""

    root = Path(output_root).expanduser().absolute()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ProductionFillStepError("tracker output root already exists") from error
    from io import BytesIO
    from PIL import Image
    bindings: dict[str, dict[str, object]] = {}
    try:
        for row in observations:
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id or image_id in bindings:
                raise ProductionFillStepError("observation image identity is invalid")
            rgb = _rgb(row.get("observation_rgb"))
            mask = np.asarray(row.get("prediction_mask"))
            if mask.dtype != np.bool_ or mask.shape != IMAGE_SHAPE:
                raise ProductionFillStepError("observation mask is invalid")
            rgb_buffer, mask_buffer = BytesIO(), BytesIO()
            Image.fromarray(rgb, mode="RGB").save(rgb_buffer, format="PNG")
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                mask_buffer, format="PNG"
            )
            rgb_path = root / "rgb" / f"{image_id}.png"
            mask_path = root / "mask" / f"{image_id}.png"
            roi, _clamp = fillcam_masks()
            occupied_roi = np.argwhere(roi)
            y0, x0 = occupied_roi.min(axis=0)
            y1, x1 = occupied_roi.max(axis=0) + 1
            crop_buffer = BytesIO()
            Image.fromarray(rgb[y0:y1, x0:x1], mode="RGB").save(
                crop_buffer, format="PNG"
            )
            crop_path = root / "crop" / f"{image_id}.png"
            _write_new(rgb_path, rgb_buffer.getvalue())
            _write_new(mask_path, mask_buffer.getvalue())
            _write_new(crop_path, crop_buffer.getvalue())
            bindings[image_id] = {
                "frame_index": row["frame_index"], "phase": row["phase"],
                "rgb_path": str(rgb_path),
                "rgb_sha256": hashlib.sha256(rgb_buffer.getvalue()).hexdigest(),
                "mask_path": str(mask_path),
                "mask_sha256": hashlib.sha256(mask_buffer.getvalue()).hexdigest(),
                "crop_path": str(crop_path),
                "crop_sha256": hashlib.sha256(crop_buffer.getvalue()).hexdigest(),
                "crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            }
        registry: dict[str, object] = {
            "schema_version": "wp7-fill-step-camera-registry-v1",
            "artifact_type": "wp7_fill_step_camera_registry",
            "images": bindings,
        }
        registry["registry_sha256"] = _canonical_hash(registry)
        _write_new(
            root / "registry.json",
            json.dumps(registry, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n",
        )
        return registry
    except BaseException:
        # Never overwrite or silently reuse a partial claim.  The fresh root is
        # deliberately retained as evidence of a failed attempt.
        raise


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _rgb(value: object) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.shape != (*IMAGE_SHAPE, 3):
        raise ProductionFillStepError("camera frame must be 480x640 uint8 RGB")
    return frame


__all__ = [
    "CompletedPhaseObservation", "IncrementalFillStepTracker",
    "MaskTopologyResult", "ProductionFillStepError", "TopologyFreeze",
    "TopologyHeightMeasurement", "apply_camera_hidden_band_surface",
    "load_pixel_to_metre_calibration", "load_topology_freeze",
    "load_tracker_freeze", "measure_stream_lab",
    "persist_completed_observations", "postprocess_cnn_mask_topology",
]
