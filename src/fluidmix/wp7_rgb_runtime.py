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
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VISIBLE_STATUS = "visible_measured"
_MINIMUM_MASK_IOU = 0.90
_MAXIMUM_SURFACE_DELTA_PX = 2
_MAXIMUM_HEIGHT_DELTA_PX = 2
_TANK_IDS = frozenset({"t100", "t200", "t400"})
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
    "temporal_hidden_band": "fail_closed_adapter_pending_review",
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


def fail_closed_temporal_hidden_band_adapter() -> AdapterResult:
    """Refuse to reinterpret CNN masks across the hidden clamp band.

    The accepted RGB temporal inference consumes its own difference masks.  It
    cannot safely be applied to learned masks without a reviewed adapter, so
    this boundary preserves both existing semantics by abstaining.
    """

    return AdapterResult(
        "abstain", ("cnn_temporal_hidden_band_adapter_unavailable",)
    )


def adapt_real_rgb_measurements() -> AdapterResult:
    """Refuse to mint a trusted measurement from the twin-only registry API."""

    return AdapterResult(
        "abstain", ("trusted_real_rgb_measurement_adapter_unavailable",)
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
    "adapt_real_rgb_measurements",
    "bind_same_run_empty_references",
    "build_runtime_provenance",
    "compose_fill_windows",
    "detect_stream_bouts",
    "fail_closed_temporal_hidden_band_adapter",
    "load_frozen_cnn",
    "predict_with_frozen_cnn",
    "runtime_source_sha256",
    "select_dispense_endpoints",
]
