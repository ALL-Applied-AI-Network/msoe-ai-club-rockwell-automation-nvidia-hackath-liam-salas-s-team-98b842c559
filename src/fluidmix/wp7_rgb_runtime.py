"""Production RGB frame-selection building blocks for the WP7 runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .empty_reference_rgb_predictor import (
    CHANGE_THRESHOLD,
    IMAGE_SHAPE,
    STREAM_MINIMUM_CHANGED_PIXELS,
    STREAM_MINIMUM_CONSECUTIVE_FRAMES,
    STREAM_ROI_XYXY,
    fillcam_masks,
    predict_fill_sequence,
)
from .predicted_mask_materializer import (
    CNN_CHECKPOINT_SHA256,
    CNN_REPLAY_SHA256,
    CNN_REVIEW_SHA256,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VISIBLE_STATUS = "visible_measured"
_MINIMUM_MASK_IOU = 0.90
_MAXIMUM_SURFACE_DELTA_PX = 2
_MAXIMUM_HEIGHT_DELTA_PX = 2
_TANK_IDS = frozenset({"t100", "t200", "t400"})
_REAL_TRUST_TOKEN = object()
_FORBIDDEN_PROVENANCE_TOKENS = frozenset(
    {"oracle", "simulation", "sim", "truth", "label", "labels", "part_a", "sim_metadata"}
)
RUNTIME_CONFIG = {
    "input_fields": ["current_rgb", "same_run_pre_pour_rgb"],
    "composition_start": "encoded_transition_stop_minus_one",
    "composition_stop": "next_encoded_transition_start",
    "stream_roi_xyxy": list(STREAM_ROI_XYXY),
    "stream_absolute_change_threshold": CHANGE_THRESHOLD,
    "stream_minimum_changed_pixels": STREAM_MINIMUM_CHANGED_PIXELS,
    "stream_minimum_consecutive_frames": STREAM_MINIMUM_CONSECUTIVE_FRAMES,
    "endpoint_minimum_mask_iou": _MINIMUM_MASK_IOU,
    "endpoint_maximum_surface_delta_px": _MAXIMUM_SURFACE_DELTA_PX,
    "endpoint_maximum_height_delta_px": _MAXIMUM_HEIGHT_DELTA_PX,
    "postprocessing": "accepted_vial_roi_and_clamp_exclusion",
    "temporal_hidden_band": "accepted_rgb_growth_stream_interpolation",
}


def _accepted_cnn_lineage() -> dict[str, str]:
    return {
        "checkpoint": CNN_CHECKPOINT_SHA256,
        "replay": CNN_REPLAY_SHA256,
        "review": CNN_REVIEW_SHA256,
    }


class RGBRuntimeError(ValueError):
    """A production RGB runtime contract failed closed."""


@dataclass(frozen=True)
class FillWindow:
    """One receipt-bound half-open fill span in the composed run video."""

    fill_index: int
    case_id: str
    start_frame_index: int
    stop_frame_index: int
    frame_count: int


@dataclass(frozen=True)
class EmptyReference:
    """Same-run pre-pour RGB bound to one composed fill start."""

    fill_index: int
    case_id: str
    frame_index: int
    rgb_sha256: str
    rgb: np.ndarray


@dataclass(frozen=True)
class StreamBout:
    """One maximal half-open run of sustained RGB change in the nozzle ROI."""

    start_frame_index: int
    stop_frame_index: int
    frame_count: int


@dataclass(frozen=True)
class SelectedEndpoint:
    """First post-bout CNN-visible frame confirmed by its next frame."""

    bout_index: int
    tank_id: str
    frame_index: int
    agreement_frame_index: int
    mask_sha256: str
    agreement_mask_sha256: str
    surface_row: int
    visible_height_px: int


@dataclass(frozen=True)
class SelectionResult:
    """Selected endpoints or an explicit fail-closed abstention."""

    status: str
    bouts: tuple[StreamBout, ...]
    endpoints: tuple[SelectedEndpoint, ...]
    limit_codes: tuple[str, ...]


@dataclass(frozen=True)
class AdapterResult:
    """Result of a production capability adapter that cannot safely proceed."""

    status: str
    limit_codes: tuple[str, ...]


@dataclass(frozen=True)
class LoadedFrozenCNN:
    """Hash-bound accepted CNN plus its reviewed lineage metadata."""

    model: object
    producer: Mapping[str, object]
    config: Mapping[str, object]
    accepted_evidence: Mapping[str, object]
    checkpoint_sha256: str


class TrustedRealRGBMaskEntryV1:
    """Opaque, factory-created real-camera current/reference/mask binding."""

    __slots__ = ("_record", "_registry_sha256", "_factory_token")

    def __init__(
        self,
        record: Mapping[str, object],
        registry_sha256: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _REAL_TRUST_TOKEN:
            raise TypeError("trusted real-RGB mask entries are factory-only")
        object.__setattr__(self, "_record", dict(record))
        object.__setattr__(self, "_registry_sha256", registry_sha256)
        object.__setattr__(self, "_factory_token", _REAL_TRUST_TOKEN)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("trusted real-RGB mask entries are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("trusted real-RGB mask entries are immutable")

    @property
    def image_id(self) -> str:
        return str(self._record["image_id"])

    @property
    def case_id(self) -> str:
        return str(self._record["case_id"])

    @property
    def frame_index(self) -> int:
        return int(self._record["frame_index"])

    @property
    def reference_frame_index(self) -> int:
        return int(self._record["reference_frame_index"])

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    def locator(self) -> dict[str, object]:
        return dict(self._record)


class TrustedRealRGBMaskRegistryV1:
    """Read-only registry distinct from the existing twin-only schema."""

    __slots__ = ("_entries", "_registry_sha256", "_factory_token")

    def __init__(
        self,
        entries: Mapping[str, TrustedRealRGBMaskEntryV1],
        registry_sha256: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _REAL_TRUST_TOKEN:
            raise TypeError("trusted real-RGB mask registries are factory-only")
        object.__setattr__(self, "_entries", dict(entries))
        object.__setattr__(self, "_registry_sha256", registry_sha256)
        object.__setattr__(self, "_factory_token", _REAL_TRUST_TOKEN)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("trusted real-RGB mask registries are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("trusted real-RGB mask registries are immutable")

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    def entry(self, image_id: str) -> TrustedRealRGBMaskEntryV1:
        try:
            return self._entries[image_id]
        except KeyError as error:
            raise RGBRuntimeError(f"real-RGB registry is missing image_id: {image_id}") from error


@dataclass(frozen=True)
class RuntimeVisionMeasurement:
    """Hash-bound CNN mask measurement with accepted temporal status."""

    image_id: str
    case_id: str
    frame_index: int
    status: str
    surface_row: int | None
    visible_height_px: int | None
    mask_sha256: str
    limit_codes: tuple[str, ...]
    citation_sha256: str


class TrustedRuntimeVisionMeasurementsV1:
    """Factory-only sequence of runtime vision-measurement capabilities."""

    __slots__ = ("_measurements", "_registry_sha256", "_factory_token")

    def __init__(
        self,
        measurements: tuple[RuntimeVisionMeasurement, ...],
        registry_sha256: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _REAL_TRUST_TOKEN:
            raise TypeError("runtime vision measurements are factory-only")
        object.__setattr__(self, "_measurements", measurements)
        object.__setattr__(self, "_registry_sha256", registry_sha256)
        object.__setattr__(self, "_factory_token", _REAL_TRUST_TOKEN)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("runtime vision measurements are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("runtime vision measurements are immutable")

    @property
    def measurements(self) -> tuple[RuntimeVisionMeasurement, ...]:
        return self._measurements

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256


def _rows(value: object, role: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RGBRuntimeError(f"{role} must be a sequence")
    return value


def _integer(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        raise RGBRuntimeError(f"{role} must be a non-negative integer")
    return value


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RGBRuntimeError(f"{role} must be a lowercase SHA-256")
    return value


def _rgb(value: object, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (*IMAGE_SHAPE, 3):
        raise RGBRuntimeError(f"{role} must be one 640x480 uint8 RGB array")
    return array


def _frame_sequence(value: object) -> Sequence[object]:
    frames = _rows(value, "RGB frames")
    if not frames:
        raise RGBRuntimeError("RGB frames must be nonempty")
    return frames


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _bgr_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value[..., ::-1].copy().tobytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


RUNTIME_CONFIG_SHA256 = _canonical_sha256(RUNTIME_CONFIG)


def runtime_source_sha256() -> str:
    """Hash the exact runtime module bytes used for a provenance record."""

    path = Path(__file__)
    if path.is_symlink() or not path.is_file():
        raise RGBRuntimeError("runtime source must be a regular symlink-free file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_fill_windows(
    encoded_transition_evidence: object, fill_clips: object
) -> tuple[FillWindow, ...]:
    """Return receipt-bound half-open fill windows."""

    transitions = _rows(encoded_transition_evidence, "encoded transition evidence")
    fills = _rows(fill_clips, "fill clips")
    if not fills or len(transitions) != len(fills) + 1:
        raise RGBRuntimeError("transition/fill composition count mismatch")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise RGBRuntimeError("encoded transition row must be an object")
        transition_start = _integer(
            transition.get("start_frame_index"), "transition start frame"
        )
        transition_stop = _integer(
            transition.get("stop_frame_index"), "transition stop frame"
        )
        transition_count = _integer(
            transition.get("frame_count"), "transition frame count"
        )
        hashes = transition.get("decoded_bgr_sha256")
        if (
            transition.get("segment_index") != index
            or transition_stop - transition_start != transition_count
            or not isinstance(hashes, list)
            or len(hashes) != transition_count
        ):
            raise RGBRuntimeError("encoded transition frame contract is invalid")
    windows: list[FillWindow] = []
    for index, raw_fill in enumerate(fills):
        current = transitions[index]
        following = transitions[index + 1]
        if not all(isinstance(row, Mapping) for row in (raw_fill, current, following)):
            raise RGBRuntimeError("composition row must be an object")
        assert isinstance(raw_fill, Mapping)
        assert isinstance(current, Mapping)
        assert isinstance(following, Mapping)
        start = _integer(current.get("stop_frame_index"), "transition stop frame") - 1
        stop = _integer(following.get("start_frame_index"), "transition start frame")
        frame_count = _integer(raw_fill.get("frame_count"), "fill frame count")
        if start < 0 or stop <= start or stop - start != frame_count:
            raise RGBRuntimeError(
                f"fill frame count mismatch at index {index}: "
                f"receipt={frame_count}, composed={stop - start}"
            )
        case_id = raw_fill.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RGBRuntimeError("fill case_id must be nonempty text")
        windows.append(FillWindow(index, case_id, start, stop, frame_count))
    return tuple(windows)


def bind_same_run_empty_references(
    rgb_frames: object,
    encoded_transition_evidence: object,
    fill_windows: object,
) -> tuple[EmptyReference, ...]:
    """Bind each fill's first frame to the preceding transition's last hash.

    Composition deliberately uses ``transition.stop_frame_index - 1``.  That
    frame is duplicated at the transition/fill seam and is the receipt-bound
    same-run pre-pour reference; no still or test annotation participates.
    """

    frames = _frame_sequence(rgb_frames)
    transitions = _rows(encoded_transition_evidence, "encoded transition evidence")
    windows = _rows(fill_windows, "fill windows")
    if len(transitions) != len(windows) + 1:
        raise RGBRuntimeError("transition/reference count mismatch")
    references: list[EmptyReference] = []
    for index, window in enumerate(windows):
        if not isinstance(window, FillWindow):
            raise RGBRuntimeError("fill windows must contain FillWindow values")
        if window.fill_index != index or window.stop_frame_index > len(frames):
            raise RGBRuntimeError("fill window cannot address its pre-pour frame")
        transition = transitions[index]
        if not isinstance(transition, Mapping):
            raise RGBRuntimeError("encoded transition row must be an object")
        hashes = transition.get("decoded_bgr_sha256")
        if (
            not isinstance(hashes, list)
            or not hashes
            or any(
                not isinstance(item, str) or _SHA256.fullmatch(item) is None
                for item in hashes
            )
        ):
            raise RGBRuntimeError("transition decoded frame hashes are invalid")
        frame = _rgb(frames[window.start_frame_index], "pre-pour frame")
        if _bgr_sha256(frame) != hashes[-1]:
            raise RGBRuntimeError(
                f"pre-pour frame hash mismatch at fill index {window.fill_index}"
            )
        frozen = frame.copy()
        frozen.setflags(write=False)
        references.append(
            EmptyReference(
                fill_index=index,
                case_id=window.case_id,
                frame_index=window.start_frame_index,
                rgb_sha256=_array_sha256(frame),
                rgb=frozen,
            )
        )
    return tuple(references)


def _stream_changed(frame: np.ndarray, reference: np.ndarray) -> bool:
    x0, y0, x1, y1 = STREAM_ROI_XYXY
    difference = np.max(
        np.abs(
            frame[y0:y1, x0:x1].astype(np.int16)
            - reference[y0:y1, x0:x1].astype(np.int16)
        ),
        axis=2,
    )
    return int(np.count_nonzero(difference > CHANGE_THRESHOLD)) >= STREAM_MINIMUM_CHANGED_PIXELS


def detect_stream_bouts(
    rgb_frames: object,
    reference_rgb: object,
    *,
    start_frame_index: int,
    stop_frame_index: int,
) -> tuple[StreamBout, ...]:
    """Detect maximal sustained stream bouts from RGB change only."""

    frames = _frame_sequence(rgb_frames)
    reference = _rgb(reference_rgb, "reference RGB")
    start = _integer(start_frame_index, "start frame index")
    stop = _integer(stop_frame_index, "stop frame index")
    if stop <= start or stop > len(frames):
        raise RGBRuntimeError("stream detection frame bounds are invalid")
    flags = [
        _stream_changed(_rgb(frames[index], "RGB frame"), reference)
        for index in range(start, stop)
    ]
    bouts: list[StreamBout] = []
    offset = 0
    while offset < len(flags):
        if not flags[offset]:
            offset += 1
            continue
        run_start = offset
        while offset < len(flags) and flags[offset]:
            offset += 1
        count = offset - run_start
        if count >= STREAM_MINIMUM_CONSECUTIVE_FRAMES:
            bouts.append(StreamBout(start + run_start, start + offset, count))
    return tuple(bouts)


def _visible_prediction(value: object) -> tuple[int, int, np.ndarray] | None:
    if getattr(value, "status", None) != _VISIBLE_STATUS:
        return None
    surface = getattr(value, "surface_row", None)
    height = getattr(value, "visible_height_px", None)
    mask = np.asarray(getattr(value, "mask", None))
    if (
        type(surface) is not int
        or type(height) is not int
        or surface < 0
        or height <= 0
        or mask.dtype != np.bool_
        or mask.shape != IMAGE_SHAPE
        or not np.any(mask)
    ):
        raise RGBRuntimeError("CNN visible prediction is malformed")
    return surface, height, mask


def _predictions_agree(
    left: tuple[int, int, np.ndarray], right: tuple[int, int, np.ndarray]
) -> bool:
    left_surface, left_height, left_mask = left
    right_surface, right_height, right_mask = right
    union = int(np.count_nonzero(left_mask | right_mask))
    intersection = int(np.count_nonzero(left_mask & right_mask))
    iou = intersection / union if union else 0.0
    return bool(
        abs(left_surface - right_surface) <= _MAXIMUM_SURFACE_DELTA_PX
        and abs(left_height - right_height) <= _MAXIMUM_HEIGHT_DELTA_PX
        and iou >= _MINIMUM_MASK_IOU
    )


def select_dispense_endpoints(
    rgb_frames: object,
    reference_rgb: object,
    *,
    dispense_order: object,
    start_frame_index: int,
    stop_frame_index: int,
    predictor: Callable[[np.ndarray, np.ndarray], object],
) -> SelectionResult:
    """Select the first next-frame-stable CNN-visible frame after each bout."""

    frames = _frame_sequence(rgb_frames)
    reference = _rgb(reference_rgb, "reference RGB")
    order = _rows(dispense_order, "dispense order")
    if (
        not order
        or any(not isinstance(item, str) or item not in _TANK_IDS for item in order)
        or len(set(order)) != len(order)
    ):
        raise RGBRuntimeError("dispense order must contain unique accepted tank IDs")
    if not callable(predictor):
        raise RGBRuntimeError("predictor must be callable")
    bouts = detect_stream_bouts(
        frames,
        reference,
        start_frame_index=start_frame_index,
        stop_frame_index=stop_frame_index,
    )
    if not bouts:
        return SelectionResult("abstain", (), (), ("dispense_stream_bout_missing",))
    if len(bouts) != len(order):
        return SelectionResult(
            "abstain", bouts, (), ("dispense_stream_bout_count_ambiguous",)
        )
    endpoints: list[SelectedEndpoint] = []
    for bout_index, bout in enumerate(bouts):
        search_stop = (
            bouts[bout_index + 1].start_frame_index
            if bout_index + 1 < len(bouts)
            else stop_frame_index
        )
        selected: SelectedEndpoint | None = None
        visible_candidate_seen = False
        for frame_index in range(bout.stop_frame_index, search_stop - 1):
            current = _rgb(frames[frame_index], "RGB frame")
            following = _rgb(frames[frame_index + 1], "RGB frame")
            current_prediction = _visible_prediction(predictor(current, reference))
            following_prediction = _visible_prediction(predictor(following, reference))
            visible_candidate_seen = visible_candidate_seen or any(
                item is not None for item in (current_prediction, following_prediction)
            )
            if (
                current_prediction is None
                or following_prediction is None
                or not _predictions_agree(current_prediction, following_prediction)
            ):
                continue
            surface, height, mask = current_prediction
            following_mask = following_prediction[2]
            selected = SelectedEndpoint(
                bout_index=bout_index,
                tank_id=str(order[bout_index]),
                frame_index=frame_index,
                agreement_frame_index=frame_index + 1,
                mask_sha256=_array_sha256(mask),
                agreement_mask_sha256=_array_sha256(following_mask),
                surface_row=surface,
                visible_height_px=height,
            )
            break
        if selected is None:
            limit_code = (
                "stable_post_bout_endpoint_ambiguous"
                if visible_candidate_seen
                else "stable_post_bout_endpoint_missing"
            )
            return SelectionResult(
                "abstain",
                bouts,
                (),
                (limit_code,),
            )
        endpoints.append(selected)
    return SelectionResult("selected", bouts, tuple(endpoints), ())


def fail_closed_temporal_hidden_band_adapter(
    registry: object,
    image_ids: object,
    *,
    programmed_ml: object | None = None,
) -> TrustedRuntimeVisionMeasurementsV1:
    """Compatibility name for the now-reviewed fail-closed temporal adapter."""

    return build_runtime_vision_measurements(
        registry, image_ids, programmed_ml=programmed_ml
    )


def _reject_forbidden_provenance(value: object, path: str = "") -> None:
    def forbidden(text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        bounded = f"_{normalized}_"
        return any(f"_{token}_" in bounded for token in _FORBIDDEN_PROVENANCE_TOKENS)

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RGBRuntimeError("real-RGB provenance fields must be text")
            if forbidden(key):
                raise RGBRuntimeError(f"forbidden real-RGB provenance field: {path}{key}")
            _reject_forbidden_provenance(child, f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_provenance(child, f"{path}{index}.")
    elif isinstance(value, str):
        if forbidden(value):
            raise RGBRuntimeError(f"forbidden real-RGB provenance value at {path.rstrip('.')}")


def _verified_bound_file(value: object, digest: object, role: str) -> Path:
    path = _regular_file(value, role)
    expected = _sha(digest, role)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RGBRuntimeError(f"{role} SHA-256 mismatch")
    return path


def build_trusted_real_rgb_mask_registry_v1(
    artifact: object,
) -> TrustedRealRGBMaskRegistryV1:
    """Validate and mint a real-camera registry without changing the twin schema."""

    _reject_forbidden_provenance(artifact)
    top_fields = {
        "schema_version",
        "artifact_type",
        "data_origin",
        "producer",
        "lineage",
        "images",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != top_fields:
        raise RGBRuntimeError("real-RGB registry fields are invalid")
    if (
        artifact.get("schema_version") != "wp7-real-rgb-mask-registry-v1"
        or artifact.get("artifact_type") != "real_camera_rgb_mask_registry"
        or artifact.get("data_origin") != "real_camera"
    ):
        raise RGBRuntimeError("real-RGB registry identity is invalid")
    producer = artifact.get("producer")
    producer_fields = {
        "producer_id",
        "checkpoint_sha256",
        "replay_sha256",
        "review_sha256",
        "probability_threshold",
        "input_fields",
        "mask_source",
    }
    if not isinstance(producer, Mapping) or set(producer) != producer_fields:
        raise RGBRuntimeError("real-RGB producer fields are invalid")
    accepted_lineage = _accepted_cnn_lineage()
    if producer != {
        "producer_id": "frozen-wp5-six-channel-cnn",
        "checkpoint_sha256": accepted_lineage["checkpoint"],
        "replay_sha256": accepted_lineage["replay"],
        "review_sha256": accepted_lineage["review"],
        "probability_threshold": 0.5,
        "input_fields": ["current_rgb", "same_run_pre_pour_rgb"],
        "mask_source": "frozen_cnn_prediction",
    }:
        raise RGBRuntimeError("real-RGB producer differs from the accepted CNN")
    lineage = artifact.get("lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(accepted_lineage):
        raise RGBRuntimeError("real-RGB CNN lineage fields are invalid")
    lineage_receipt: dict[str, dict[str, str]] = {}
    for role, expected in accepted_lineage.items():
        binding = lineage.get(role)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise RGBRuntimeError(f"real-RGB {role} lineage binding is invalid")
        if binding.get("sha256") != expected:
            raise RGBRuntimeError(f"real-RGB {role} lineage hash is not accepted")
        path = _verified_bound_file(binding.get("path"), expected, f"CNN {role}")
        lineage_receipt[role] = {"path": str(path), "sha256": expected}
    images = artifact.get("images")
    if not isinstance(images, Mapping) or not images:
        raise RGBRuntimeError("real-RGB registry images must be nonempty")
    image_fields = {
        "case_id",
        "frame_index",
        "current_rgb_path",
        "current_rgb_sha256",
        "reference_frame_index",
        "reference_rgb_path",
        "reference_rgb_sha256",
        "mask_path",
        "mask_sha256",
    }
    records: dict[str, dict[str, object]] = {}
    for image_id in sorted(images):
        locator = images[image_id]
        if not isinstance(image_id, str) or not image_id:
            raise RGBRuntimeError("real-RGB image_id must be nonempty text")
        if not isinstance(locator, Mapping) or set(locator) != image_fields:
            raise RGBRuntimeError("real-RGB image locator fields are invalid")
        case_id = locator.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RGBRuntimeError("real-RGB case_id must be nonempty text")
        frame_index = _integer(locator.get("frame_index"), "current frame index")
        reference_index = _integer(
            locator.get("reference_frame_index"), "reference frame index"
        )
        record: dict[str, object] = {
            "image_id": image_id,
            "case_id": case_id,
            "frame_index": frame_index,
            "reference_frame_index": reference_index,
        }
        for prefix in ("current_rgb", "reference_rgb", "mask"):
            digest = _sha(locator.get(f"{prefix}_sha256"), f"{prefix} hash")
            path = _verified_bound_file(
                locator.get(f"{prefix}_path"), digest, prefix
            )
            record[f"{prefix}_path"] = str(path)
            record[f"{prefix}_sha256"] = digest
        records[image_id] = record
    registry_sha256 = _canonical_sha256(
        {
            "schema_version": "trusted-real-rgb-mask-registry-v1",
            "lineage": lineage_receipt,
            "entries": [records[key] for key in sorted(records)],
        }
    )
    entries = {
        image_id: TrustedRealRGBMaskEntryV1(
            record,
            registry_sha256,
            _factory_token=_REAL_TRUST_TOKEN,
        )
        for image_id, record in records.items()
    }
    return TrustedRealRGBMaskRegistryV1(
        entries,
        registry_sha256,
        _factory_token=_REAL_TRUST_TOKEN,
    )


def adapt_real_rgb_measurements(artifact: object) -> TrustedRealRGBMaskRegistryV1:
    """Public adapter for the distinct real-camera trusted-mask schema."""

    return build_trusted_real_rgb_mask_registry_v1(artifact)


def build_runtime_vision_measurements(
    registry: object,
    image_ids: object,
    *,
    programmed_ml: object | None = None,
) -> TrustedRuntimeVisionMeasurementsV1:
    """Build trusted CNN/temporal measurements for one contiguous fill sequence."""

    if (
        not isinstance(registry, TrustedRealRGBMaskRegistryV1)
        or registry._factory_token is not _REAL_TRUST_TOKEN
    ):
        raise TypeError("runtime measurement requires a trusted real-RGB registry")
    ids = _rows(image_ids, "runtime image IDs")
    if (
        not ids
        or any(not isinstance(image_id, str) or not image_id for image_id in ids)
        or tuple(ids) != registry.image_ids
    ):
        raise RGBRuntimeError("runtime image IDs must exactly match registry order")
    from PIL import Image, UnidentifiedImageError

    entries = [registry.entry(str(image_id)) for image_id in ids]
    locators = [entry.locator() for entry in entries]
    case_ids = {entry.case_id for entry in entries}
    reference_indices = {entry.reference_frame_index for entry in entries}
    reference_paths = {str(locator["reference_rgb_path"]) for locator in locators}
    reference_hashes = {str(locator["reference_rgb_sha256"]) for locator in locators}
    frame_indices = [entry.frame_index for entry in entries]
    if (
        len(case_ids) != 1
        or len(reference_indices) != 1
        or len(reference_paths) != 1
        or len(reference_hashes) != 1
        or frame_indices != list(range(frame_indices[0], frame_indices[0] + len(entries)))
        or frame_indices[0] != entries[0].reference_frame_index
    ):
        raise RGBRuntimeError("runtime sequence/reference binding is invalid")

    def load_rgb(path_text: object, digest: object, role: str) -> np.ndarray:
        path = _verified_bound_file(path_text, digest, role)
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
                    raise RGBRuntimeError(f"{role} must be an exact 640x480 RGB PNG")
                image.load()
                return np.asarray(image, dtype=np.uint8).copy()
        except (OSError, UnidentifiedImageError) as error:
            raise RGBRuntimeError(f"{role} PNG could not be decoded") from error

    def load_mask(path_text: object, digest: object) -> np.ndarray:
        path = _verified_bound_file(path_text, digest, "CNN mask")
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode not in {"1", "L"} or image.size != (640, 480):
                    raise RGBRuntimeError("CNN mask must be an exact binary 640x480 PNG")
                image.load()
                raw = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        except (OSError, UnidentifiedImageError) as error:
            raise RGBRuntimeError("CNN mask PNG could not be decoded") from error
        if not set(int(value) for value in np.unique(raw)).issubset({0, 255}):
            raise RGBRuntimeError("CNN mask must contain only binary pixels")
        return raw == 255

    reference = load_rgb(
        locators[0]["reference_rgb_path"],
        locators[0]["reference_rgb_sha256"],
        "same-run pre-pour RGB",
    )
    frames = [
        load_rgb(locator["current_rgb_path"], locator["current_rgb_sha256"], "current RGB")
        for locator in locators
    ]
    if not np.array_equal(frames[0], reference):
        raise RGBRuntimeError("first current RGB differs from same-run pre-pour RGB")
    masks = [
        load_mask(locator["mask_path"], locator["mask_sha256"])
        for locator in locators
    ]
    roi, clamp = fillcam_masks()
    if any(np.any(mask & ~roi) or np.any(mask & clamp) for mask in masks):
        raise RGBRuntimeError("CNN mask violates accepted vial ROI/clamp exclusion")
    temporal = predict_fill_sequence(
        frames,
        reference_index=0,
        expected_reference_rgb_sha256=_array_sha256(reference),
        programmed_ml=programmed_ml,
    )
    measurements: list[RuntimeVisionMeasurement] = []
    case_id = entries[0].case_id
    for index, (entry, locator, frame, mask, temporal_item) in enumerate(
        zip(entries, locators, frames, masks, temporal, strict=True)
    ):
        surface: int | None = None
        height: int | None = None
        limit_codes: tuple[str, ...] = ()
        if index == 0:
            status = "empty_reference"
        elif not np.any(frame):
            status = "invalid_black_frame"
            limit_codes = ("invalid_black_frame",)
        elif not np.any(mask):
            status = "empty_mask_abstain"
            limit_codes = ("empty_mask_abstain",)
        else:
            occupied = np.flatnonzero(np.any(mask, axis=1))
            surface = int(occupied[0])
            bottom = int(occupied[-1])
            status = "visible_measured"
            if temporal_item.status == "clamp_hidden_inferred":
                inferred = temporal_item.surface_row
                if inferred is None or not 238 <= inferred <= 270:
                    raise RGBRuntimeError("accepted hidden-band inference is malformed")
                surface = inferred
                status = "clamp_hidden_inferred"
            height = bottom - surface + 1
        citation_payload = {
            "registry_sha256": registry.registry_sha256,
            "image_id": entry.image_id,
            "case_id": case_id,
            "frame_index": entry.frame_index,
            "reference_frame_index": entry.reference_frame_index,
            "current_rgb_sha256": locator["current_rgb_sha256"],
            "reference_rgb_sha256": locator["reference_rgb_sha256"],
            "mask_sha256": locator["mask_sha256"],
            "status": status,
            "surface_row": surface,
            "visible_height_px": height,
            "limit_codes": list(limit_codes),
        }
        measurements.append(
            RuntimeVisionMeasurement(
                image_id=entry.image_id,
                case_id=case_id,
                frame_index=entry.frame_index,
                status=status,
                surface_row=surface,
                visible_height_px=height,
                mask_sha256=str(locator["mask_sha256"]),
                limit_codes=limit_codes,
                citation_sha256=_canonical_sha256(citation_payload),
            )
        )
    return TrustedRuntimeVisionMeasurementsV1(
        tuple(measurements),
        registry.registry_sha256,
        _factory_token=_REAL_TRUST_TOKEN,
    )


def _regular_file(path: object, role: str) -> Path:
    try:
        candidate = Path(path)  # type: ignore[arg-type]
    except TypeError as error:
        raise RGBRuntimeError(f"{role} path is invalid") from error
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RGBRuntimeError(f"{role} must be an absolute regular symlink-free file")
    return candidate


def load_frozen_cnn(
    *,
    checkpoint_path: object,
    replay_path: object,
    review_path: object,
    expected_checkpoint_sha256: object,
    _loader: Callable[
        ...,
        tuple[
            object,
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ],
    ]
    | None = None,
) -> LoadedFrozenCNN:
    """Load the reviewed CNN only after exact checkpoint-byte verification."""

    checkpoint = _regular_file(checkpoint_path, "checkpoint")
    expected = _sha(expected_checkpoint_sha256, "expected checkpoint")
    observed = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if observed != expected:
        raise RGBRuntimeError("checkpoint SHA-256 mismatch")
    replay = _regular_file(replay_path, "replay")
    review = _regular_file(review_path, "review")
    if _loader is None:
        from .predicted_mask_materializer import (
            CNN_CHECKPOINT_SHA256,
            _load_frozen_cnn as accepted_loader,
        )

        if expected != CNN_CHECKPOINT_SHA256:
            raise RGBRuntimeError("checkpoint differs from the accepted frozen hash")
        loader = accepted_loader
    else:
        loader = _loader
    try:
        model, producer, config, accepted = loader(
            checkpoint_path=checkpoint,
            replay_path=replay,
            review_path=review,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RGBRuntimeError("frozen CNN lineage could not be loaded") from error
    if (
        not isinstance(producer, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(accepted, Mapping)
        or producer.get("model_sha256") != expected
    ):
        raise RGBRuntimeError("loaded CNN producer checkpoint binding differs")
    return LoadedFrozenCNN(model, producer, config, accepted, expected)


def predict_with_frozen_cnn(
    loaded: object, current_rgb: object, reference_rgb: object
) -> object:
    """Invoke the accepted six-channel predictor without a second mask path."""

    if not isinstance(loaded, LoadedFrozenCNN):
        raise TypeError("loaded must be a LoadedFrozenCNN")
    threshold = loaded.config.get("probability_threshold")
    if threshold != 0.5:
        raise RGBRuntimeError("frozen CNN probability threshold differs")
    current = _rgb(current_rgb, "current RGB")
    reference = _rgb(reference_rgb, "reference RGB")
    from .cnn_predictor import predict_pair

    return predict_pair(
        loaded.model,
        current,
        reference,
        threshold=0.5,
        device="cpu",
    )


def build_runtime_provenance(
    *,
    video_sha256: object,
    receipt_sha256: object,
    checkpoint_sha256: object,
    config_sha256: object,
    vial_records: object,
) -> dict[str, object]:
    """Build strict provenance for sources and per-vial selected evidence."""

    sources = {
        "video_sha256": _sha(video_sha256, "video"),
        "receipt_sha256": _sha(receipt_sha256, "receipt"),
        "checkpoint_sha256": _sha(checkpoint_sha256, "checkpoint"),
        "config_sha256": _sha(config_sha256, "config"),
        "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
        "code_sha256": runtime_source_sha256(),
    }
    records = _rows(vial_records, "vial provenance records")
    expected_fields = {
        "case_id",
        "reference_frame_index",
        "reference_rgb_sha256",
        "selected_frame_indices",
        "mask_sha256s",
        "measurement_citations",
    }
    normalized: list[dict[str, object]] = []
    case_ids: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise RGBRuntimeError("vial provenance fields are invalid")
        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise RGBRuntimeError("vial provenance case_id is invalid")
        case_ids.add(case_id)
        reference_index = _integer(raw["reference_frame_index"], "reference frame index")
        reference_hash = _sha(raw["reference_rgb_sha256"], "reference RGB")
        indices = raw["selected_frame_indices"]
        masks = raw["mask_sha256s"]
        citations = raw["measurement_citations"]
        if not all(isinstance(value, list) for value in (indices, masks, citations)):
            raise RGBRuntimeError("vial provenance evidence lists are invalid")
        assert isinstance(indices, list) and isinstance(masks, list) and isinstance(citations, list)
        if not indices or len(indices) != len(masks) or len(indices) != len(citations):
            raise RGBRuntimeError("vial provenance evidence counts differ")
        selected = [_integer(value, "selected frame index") for value in indices]
        mask_hashes = [_sha(value, "mask") for value in masks]
        if any(not isinstance(value, str) or not value for value in citations):
            raise RGBRuntimeError("measurement citations must be nonempty text")
        normalized.append(
            {
                "case_id": case_id,
                "reference_frame_index": reference_index,
                "reference_rgb_sha256": reference_hash,
                "selected_frame_indices": selected,
                "mask_sha256s": mask_hashes,
                "measurement_citations": list(citations),
            }
        )
    provenance = {
        "schema_version": "wp7-rgb-runtime-provenance-v1",
        "source_bindings": sources,
        "vials": normalized,
    }
    provenance["provenance_sha256"] = hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return provenance


__all__ = [
    "AdapterResult",
    "EmptyReference",
    "FillWindow",
    "LoadedFrozenCNN",
    "RUNTIME_CONFIG",
    "RUNTIME_CONFIG_SHA256",
    "RGBRuntimeError",
    "SelectedEndpoint",
    "SelectionResult",
    "StreamBout",
    "RuntimeVisionMeasurement",
    "TrustedRealRGBMaskEntryV1",
    "TrustedRealRGBMaskRegistryV1",
    "TrustedRuntimeVisionMeasurementsV1",
    "adapt_real_rgb_measurements",
    "bind_same_run_empty_references",
    "build_runtime_vision_measurements",
    "build_runtime_provenance",
    "build_trusted_real_rgb_mask_registry_v1",
    "compose_fill_windows",
    "detect_stream_bouts",
    "fail_closed_temporal_hidden_band_adapter",
    "load_frozen_cnn",
    "predict_with_frozen_cnn",
    "runtime_source_sha256",
    "select_dispense_endpoints",
]
